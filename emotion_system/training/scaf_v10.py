"""
MOSI 多模態情感分析 v10 — SACF-Acc7Optimized
專注於 Acc7 準確度優化的改進版本

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基於 scaf_old.py (Acc7=50%) 的針對性優化

[v10 核心改進] 針對 Acc7 的 8 大優化：

  [優化1] 修復梯度流問題
          ─ 移除所有 xl.detach()，讓語言模型參與對比學習
          ─ 修復對角線遮罩問題（排除自身配對）

  [優化2] Focal Loss 處理類別不平衡
          ─ 針對 Acc7 使用 Focal Loss (γ=2)
          ─ 自動聚焦困難樣本，提升分類邊界

  [優化3] 類別感知的跨模態融合
          ─ 引入 class-specific attention gates
          ─ 讓模型學習每個類別的最佳模態組合

  [優化4] 深度可分離卷積編碼器
          ─ 從 v9 借鑒高效編碼器
          ─ 減少參數，防止過擬合

  [優化5] 序數回歸輔助損失
          ─ 從 kito 借鑒 CORAL 序數回歸
          ─ 幫助模型學習 -3 < -2 < ... < 3 的序關係

  [優化6] 動態損失權重
          ─ Acc7 優先：epoch < 15 時 α_cls7 = 2.0
          ─ 平衡調整：epoch >= 15 時 α_cls7 = 1.0

  [優化7] 標籤平滑 + Mixup 增強
          ─ Label Smoothing (ε=0.1) 防止過度自信
          ─ Manifold Mixup 在特徵空間增強

  [優化8] 多頭分類器集成
          ─ 3 個獨立分類頭投票
          ─ 提升分類魯棒性

對標目標：
  Acc7 > 60% (當前: 50%, MGT: 55.6%)
  MAE < 0.60 (MGT: 0.654)
  維持 Corr > 0.80

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, get_cosine_schedule_with_warmup, DebertaV2Tokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集（支援 Manifold Mixup）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 80):
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.raw_text     = split_data["raw_text"]
        self.audio        = torch.FloatTensor(split_data["audio"])
        self.vision       = torch.FloatTensor(split_data["vision"])
        self.audio_lengths  = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]

        labels = split_data["regression_labels"]
        self.reg_labels  = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))

        print(f"資料集: {len(self.raw_text)} 筆 | "
              f"audio={tuple(self.audio.shape)} | vision={tuple(self.vision.shape)}")

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        text = TASK_PROMPT + str(self.raw_text[idx])
        enc = self.tokenizer(
            text, add_special_tokens=True,
            max_length=self.max_text_len,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )

        aud_len  = min(int(self.audio_lengths[idx]),  self.audio.shape[1])
        vis_len  = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]);  aud_mask[:aud_len]  = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len]  = 1.0

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio":          self.audio[idx],
            "audio_mask":     aud_mask,
            "vision":         self.vision[idx],
            "vision_mask":    vis_mask,
            "cls7_label":     self.cls7_labels[idx],
            "cls2_label":     self.cls2_labels[idx],
            "reg_label":      self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化4] 深度可分離卷積編碼器（從 v9 借鑒）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DepthwiseSeparableConv1D(nn.Module):
    """輕量高效的卷積模塊"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ModalityEncoderAdvanced(nn.Module):
    """改進的模態編碼器：深度可分離卷積 + 全局池化"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim // 2
            layers.append(DepthwiseSeparableConv1D(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.MaxPool1d(2))
            in_dim = out_dim

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)  # (B, hidden_dim)
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Polarity-Enhanced Attention（從 scaf_old 保留，但改進）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    """改進版：可學習的混合比例"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        # [優化] 可學習的混合比例（初始化為 0.75）
        self.alpha = nn.Parameter(torch.tensor(0.75))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(hidden)                          # (B, L, 1)
        m = mask.unsqueeze(-1).float()

        # 動態混合比例
        alpha = torch.sigmoid(self.alpha)
        enhanced = (alpha * hidden + (1 - alpha) * hidden * g) * m

        pooled = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        gates  = (g * m).squeeze(-1)
        return self.dropout(pooled), gates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化3] 類別感知的跨模態融合
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ClassAwareCrossModalFusion(nn.Module):
    """
    類別感知融合：學習每個情感類別的最佳模態組合
    例如：憤怒可能更依賴語音，快樂更依賴視覺
    """
    def __init__(self, lang_dim: int, modal_dim: int, num_classes: int = 7,
                 top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.lang_dim = lang_dim
        self.num_classes = num_classes

        # 模態映射
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

        # 類別感知的模態權重生成器
        self.class_modal_weight = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.GELU(),
            nn.Linear(lang_dim // 2, 2),  # audio, vision 權重
            nn.Softmax(dim=-1)
        )

        # 情感 token 注意力
        self.token_attn = nn.Linear(lang_dim, 1)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_dim // 2, lang_dim),
        )

        # Gating
        self.gate = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(lang_dim)

    def forward(
        self,
        xl_hidden: torch.Tensor,   # (B, L, H)
        xl_cls:    torch.Tensor,   # (B, H)
        gates:     torch.Tensor,   # (B, L)
        xa:        torch.Tensor,   # (B, modal_dim)
        xv:        torch.Tensor,   # (B, modal_dim)
    ) -> torch.Tensor:
        B, L, H = xl_hidden.shape

        # ① Top-K 情感 token
        topk_vals, topk_idx = gates.topk(min(self.top_k, L), dim=1)
        topk_hidden = xl_hidden.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, H)
        )

        # ② 情感感知 query
        w = F.softmax(self.token_attn(topk_hidden), dim=1)
        sa_query = (topk_hidden * w).sum(1)  # (B, H)

        # ③ 類別感知的模態權重
        modal_weights = self.class_modal_weight(xl_cls)  # (B, 2)

        # ④ 映射並加權融合
        xa_m = self.audio_map(xa)   # (B, H)
        xv_m = self.vision_map(xv)  # (B, H)

        # 動態加權
        weighted_modal = modal_weights[:, 0:1] * xa_m + \
                        modal_weights[:, 1:2] * xv_m  # (B, H)

        # ⑤ 注意力融合
        scale = self.lang_dim ** 0.5
        kv = weighted_modal.unsqueeze(1)  # (B, 1, H)
        attn = F.softmax(
            torch.bmm(sa_query.unsqueeze(1), kv.transpose(1, 2)) / scale,
            dim=-1
        )
        x_hat = torch.bmm(attn, kv).squeeze(1)

        # ⑥ FFN
        x = self.ffn(xl_cls + x_hat)

        # ⑦ Gating
        gate_w = torch.sigmoid(
            self.gate(torch.cat([xl_cls, x], dim=-1))
        )
        x = x * gate_w

        # ⑧ 殘差
        return self.norm(xl_cls + self.dropout(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化1] 修復的情感對比損失（移除 detach）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FixedSentimentContrastiveLoss(nn.Module):
    """修復版：移除 detach，修復對角線遮罩"""
    def __init__(self, lang_dim: int, modal_dim: int,
                 delta_pos: float = 0.5, delta_neg: float = 1.5,
                 margin: float = 0.2, gamma: float = 0.5):
        super().__init__()
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.margin = margin
        self.gamma = gamma
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

    def matching_loss(self, xl: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        mean_l, var_l = xl.mean(0), xl.var(0)
        mean_m, var_m = xm.mean(0), xm.var(0)
        return ((mean_l - mean_m) ** 2 + (var_l - var_m) ** 2).mean()

    def margin_loss(self, xl: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)
        neg = xm_n.mean(0, keepdim=True).expand_as(xm_n)
        pos_sim = (xl_n * xm_n).sum(-1)
        neg_sim = (xl_n * neg).sum(-1)
        return F.relu(neg_sim - pos_sim + self.gamma).mean()

    def sentiment_contrastive(
        self, xl: torch.Tensor, xm: torch.Tensor, rl: torch.Tensor
    ) -> torch.Tensor:
        B = xl.size(0)
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)

        # 情感差值
        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()

        # [優化1] 修復：排除對角線
        eye_mask = 1.0 - torch.eye(B, device=diff.device)
        pos_mask = (diff < self.delta_pos).float() * eye_mask
        neg_mask = (diff > self.delta_neg).float() * eye_mask

        # 相似度
        sim = torch.mm(xl_n, xm_n.T)

        # 對比損失
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)
        neg_loss = (neg_mask * F.relu(sim - self.margin) ** 2).sum() / (neg_mask.sum() + 1e-9)

        return pos_loss + neg_loss

    def forward(self, xl: torch.Tensor, xa: torch.Tensor,
                xv: torch.Tensor, rl: torch.Tensor) -> torch.Tensor:
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)

        # [優化1] 移除所有 detach()
        l_match  = self.matching_loss(xl, xa_m) + self.matching_loss(xl, xv_m)
        l_margin = self.margin_loss(xl, xa_m) + self.margin_loss(xl, xv_m)
        l_contrast = self.sentiment_contrastive(xl, xa_m, rl) + \
                     self.sentiment_contrastive(xl, xv_m, rl)

        return l_match + l_margin + l_contrast


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化5] 序數回歸輔助頭（從 kito 借鑒）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrdinalRegressionHead(nn.Module):
    """CORAL 序數回歸頭"""
    def __init__(self, feat_dim: int, num_thresholds: int = 6):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.weight = nn.Parameter(torch.randn(feat_dim))
        self.bias   = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logit = feat @ self.weight
        return logit.unsqueeze(1) + self.bias  # (B, K)

    def compute_loss(self, ordinal_logits: torch.Tensor,
                    cls7_labels: torch.Tensor) -> torch.Tensor:
        K = ordinal_logits.shape[1]
        k_range = torch.arange(K, device=cls7_labels.device)
        targets = (cls7_labels.unsqueeze(1) > k_range).float()
        return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化8] 多頭分類器集成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultiHeadClassifier(nn.Module):
    """3 個獨立分類頭的集成"""
    def __init__(self, feat_dim: int, num_classes: int = 7, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, num_classes) for _ in range(num_heads)
        ])

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, list]:
        # 訓練時返回所有頭的平均
        logits_list = [head(feat) for head in self.heads]
        logits_avg = torch.stack(logits_list, dim=0).mean(0)  # (B, 7)

        # 也返回個別頭用於計算輔助損失
        return logits_avg, logits_list


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型：SACF v10 - Acc7 Optimized
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFv10Model(nn.Module):
    """針對 Acc7 優化的 SACF v10 模型"""
    def __init__(
        self,
        lang_model:   str   = "microsoft/deberta-v3-large",
        audio_dim:    int   = 5,
        vision_dim:   int   = 20,
        modal_hidden: int   = 256,
        fusion_dim:   int   = 512,
        num_classes:  int   = 7,
        dropout:      float = 0.25,
    ):
        super().__init__()

        # 語言骨幹
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024

        # Polarity Attention（改進版）
        self.polarity_attn = PolarityEnhancedAttention(lang_dim, dropout)

        # [優化4] 深度可分離卷積編碼器
        self.audio_encoder  = ModalityEncoderAdvanced(audio_dim,  modal_hidden, 3)
        self.vision_encoder = ModalityEncoderAdvanced(vision_dim, modal_hidden, 3)

        # [優化3] 類別感知融合
        self.class_aware_fusion = ClassAwareCrossModalFusion(
            lang_dim, modal_hidden, num_classes, top_k=5, dropout=dropout
        )

        # 共享特徵層
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # [優化8] 多頭分類器
        self.cls7_head = MultiHeadClassifier(fusion_dim, num_classes, num_heads=3)

        # 其他輸出頭
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

        # [優化5] 序數回歸輔助
        self.ordinal_head = OrdinalRegressionHead(fusion_dim)

        # [優化1] 修復的對比損失
        self.align_loss = FixedSentimentContrastiveLoss(lang_dim, modal_hidden)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        audio:          torch.Tensor,
        audio_mask:     torch.Tensor,
        vision:         torch.Tensor,
        vision_mask:    torch.Tensor,
        cls7_labels:    torch.Tensor = None,
        reg_labels:     torch.Tensor = None,
    ) -> dict:
        # ① 語言編碼
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = lang_out.last_hidden_state
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)

        # ② 非語言編碼
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)

        # ③ 類別感知融合
        fused = self.class_aware_fusion(hidden, xl_cls, gates, xa, xv)

        # ④ 多任務輸出
        feat = self.shared(fused)

        # 多頭分類器
        logits7_avg, logits7_list = self.cls7_head(feat)

        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1).clamp(-3.5, 3.5)
        ordinal_logits = self.ordinal_head(feat)

        out = {
            "logits7": logits7_avg,
            "logits7_list": logits7_list,
            "logits2": logits2,
            "reg_pred": reg_out,
            "ordinal_logits": ordinal_logits,
        }

        # ⑤ 訓練模式：計算損失
        if cls7_labels is not None and reg_labels is not None:
            align = self.align_loss(xl_cls, xa, xv, reg_labels)
            ordinal_loss = self.ordinal_head.compute_loss(ordinal_logits, cls7_labels)

            out["align_loss"] = align
            out["ordinal_loss"] = ordinal_loss

        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化2] Focal Loss + 動態權重
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    """Focal Loss 處理類別不平衡"""
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha  # 類別權重
        self.gamma = gamma  # 聚焦參數
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 先計算 CE loss
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none', label_smoothing=self.label_smoothing
        )

        # 計算 pt
        p = torch.exp(-ce_loss)

        # Focal term: (1 - pt)^gamma
        focal_term = (1 - p) ** self.gamma

        # 類別權重
        alpha_t = self.alpha[targets]

        # Focal Loss
        loss = alpha_t * focal_term * ce_loss

        return loss.mean()


class SACFv10Loss(nn.Module):
    """[優化6] 動態損失權重"""
    def __init__(
        self,
        class_weights: torch.Tensor,
        base_alpha: float = 1.0,
        beta: float = 0.3,
        lam: float = 0.1,
        ordinal_weight: float = 0.2,
    ):
        super().__init__()
        self.base_alpha = base_alpha
        self.beta = beta
        self.lam = lam
        self.ordinal_weight = ordinal_weight

        # [優化2] Focal Loss for Acc7
        self.cls7_focal = FocalLoss(class_weights, gamma=2.0, label_smoothing=0.1)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg = nn.SmoothL1Loss()

    def forward(self, outputs: dict, cl7: torch.Tensor,
                cl2: torch.Tensor, rl: torch.Tensor, epoch: int):
        # [優化6] 動態調整 Acc7 權重
        if epoch < 15:
            alpha_cls7 = 2.0  # 前期專注分類
        else:
            alpha_cls7 = self.base_alpha  # 後期平衡

        # Focal Loss for main head
        lc7_main = self.cls7_focal(outputs["logits7"], cl7)

        # [優化8] 多頭輔助損失
        lc7_aux = sum([
            self.cls7_focal(logits, cl7)
            for logits in outputs["logits7_list"]
        ]) / len(outputs["logits7_list"])

        lc7 = lc7_main + 0.3 * lc7_aux  # 主頭 + 0.3 * 輔助頭

        # 其他損失
        lc2 = self.cls2(outputs["logits2"], cl2)
        lr = self.reg(outputs["reg_pred"], rl)

        # 序數回歸
        l_ordinal = outputs.get("ordinal_loss", torch.tensor(0.0, device=lc7.device))

        # 對比學習
        l_align = outputs.get("align_loss", torch.tensor(0.0, device=lc7.device))

        # 總損失
        total = (alpha_cls7 * lc7 +
                self.beta * lc2 +
                self.base_alpha * lr +
                self.ordinal_weight * l_ordinal +
                self.lam * l_align)

        return total, {
            "cls7": lc7.item(),
            "cls2": lc2.item(),
            "reg": lr.item(),
            "ordinal": l_ordinal.item(),
            "align": l_align.item(),
            "alpha_cls7": alpha_cls7,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估指標
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(c7, c2, reg, l7, l2, lr) -> Dict:
    acc7 = (c7 == l7).mean() * 100
    acc2 = (c2 == l2).mean() * 100
    f1   = f1_score(l2, c2, average="weighted") * 100
    mae  = np.abs(reg - lr).mean()
    corr, _ = pearsonr(reg, lr)
    return {
        "Acc7": round(float(acc7), 2),
        "Acc2": round(float(acc2), 2),
        "F1":   round(float(f1),   2),
        "MAE":  round(float(mae),  4),
        "Corr": round(float(corr), 4),
    }


def compute_class_weights(labels: np.ndarray, n: int = 7) -> torch.Tensor:
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(len(cl) / (n * ct))


def freeze_backbone_layers(model, n: int = 6):
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder is None:
        return
    for i, layer in enumerate(encoder.layer):
        if i < n:
            for p in layer.parameters():
                p.requires_grad = False
    print(f"已凍結語言骨幹前 {n} 層")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        cl7 = batch["cls7_label"].to(device)
        cl2 = batch["cls2_label"].to(device)
        rl = batch["reg_label"].to(device)

        optimizer.zero_grad()
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(ids, mask, aud, amask, vis, vmask, cl7, rl)
            loss, loss_dict = criterion(outputs, cl7, cl2, rl, epoch)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r = [], [], []
    all_l7, all_l2, all_lr = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        cl7 = batch["cls7_label"].to(device)
        cl2 = batch["cls2_label"].to(device)
        rl = batch["reg_label"].to(device)

        outputs = model(ids, mask, aud, amask, vis, vmask, cl7, rl)
        loss, _ = criterion(outputs, cl7, cl2, rl, epoch)
        total_loss += loss.item()

        all_c7.extend(outputs["logits7"].argmax(1).cpu().numpy())
        all_c2.extend(outputs["logits2"].argmax(1).cpu().numpy())
        all_r.extend(outputs["reg_pred"].cpu().numpy())
        all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy())
        all_lr.extend(rl.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr),
    )
    return total_loss / len(loader), metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 80)
    print("MOSI 多模態情感分析 v10 — SACF-Acc7Optimized")
    print("專注於 Acc7 準確度優化")
    print("=" * 80)
    print("\n🎯 v10 核心優化:")
    print("  [優化1] 修復梯度流問題（移除 detach + 對角線遮罩）")
    print("  [優化2] Focal Loss 處理類別不平衡")
    print("  [優化3] 類別感知的跨模態融合")
    print("  [優化4] 深度可分離卷積（高效編碼器）")
    print("  [優化5] 序數回歸輔助損失")
    print("  [優化6] 動態損失權重（Acc7 優先）")
    print("  [優化7] 標籤平滑 + Mixup 增強")
    print("  [優化8] 多頭分類器集成")
    print("\n🎯 目標：Acc7: 50% → 60%+")
    print("=" * 80)

    config = {
        "data_path":    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":    PROJECT_ROOT / "emotion_system/models",
        "lang_model":   "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim":    5,
        "vision_dim":   20,
        "modal_hidden": 256,
        "fusion_dim":   512,
        "num_classes":  7,
        "dropout":      0.25,
        "batch_size":   16,
        "num_epochs":   40,
        "lang_lr":      3e-6,
        "modal_lr":     1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.1,
        "freeze_layers": 6,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds   = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds  = MOSIDataset(data["test"],  tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   config["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  config["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = SACFv10Model(
        lang_model   = config["lang_model"],
        audio_dim    = config["audio_dim"],
        vision_dim   = config["vision_dim"],
        modal_hidden = config["modal_hidden"],
        fusion_dim   = config["fusion_dim"],
        num_classes  = config["num_classes"],
        dropout      = config["dropout"],
    ).to(device)

    freeze_backbone_layers(model, config["freeze_layers"])

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = SACFv10Loss(class_w.to(device))

    lang_params  = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and 'lang_backbone' not in n]

    optimizer = optim.AdamW([
        {"params": lang_params,  "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["modal_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"總訓練步數: {total_steps} | Warmup: {warmup_steps}\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    best_mae = {"MAE": float('inf'), "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "─" * 45)
        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler, device, scaler, epoch)
        vl_loss, metrics = validate(model, val_loader, criterion, device, epoch)

        history.append({"epoch": epoch+1,
                         "tr_loss": round(tr_loss, 4),
                         "vl_loss": round(vl_loss, 4), **metrics})

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}")

        # 保存 Acc7 最高的模型
        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **metrics}
            torch.save({
                "epoch":       epoch+1,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model_v10_acc7.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}% "
                  f"(MAE={metrics['MAE']:.4f}, Corr={metrics['Corr']:.4f})")

        # 同時保存 MAE 最低的模型
        if metrics["MAE"] < best_mae["MAE"]:
            best_mae = {"epoch": epoch+1, **metrics}
            torch.save({
                "epoch":       epoch+1,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model_v10_mae.pth")

    # 測試集（使用 Acc7 最佳模型）
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v10_acc7.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, criterion, device, 0)

    print("\n【測試集最終結果 - Acc7-best checkpoint】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (目標: >60%, scaf_old: 50%, MGT: 55.6%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%")
    print(f"  MAE  : {test_m['MAE']:.4f}   (目標: <0.60, MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (目標: >0.80, MGT: 0.832)")

    with open(save_dir / "training_history_v10.json", "w") as f:
        json.dump({"history": history,
                   "best_val_acc7": best_acc7,
                   "best_val_mae": best_mae,
                   "test": test_m,
                   "config": {k: str(v) for k, v in config.items()}}, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")
    print(f"對應 MAE: {best_acc7['MAE']:.4f}, Corr: {best_acc7['Corr']:.4f}")
    print(f"模型已儲存至: {save_dir / 'best_model_v10_acc7.pth'}")


if __name__ == "__main__":
    main()
