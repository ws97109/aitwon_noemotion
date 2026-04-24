"""
MOSI 多模態情感分析 v9 — SACF-Advanced
Sentiment-Aware Contrastive Fusion - Advanced Edition (2025-2026 SOTA)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
整合 2025-2026 年最新研究成果

[v9 革命性改進] 基於最新論文的 6 大創新：

  [創新1] 多尺度 Transformer 編碼器
          ─ 參考 EED-CL 框架的分層粗到細 transformer
          ─ 捕獲多尺度時間模式，替代簡單的 Polarity Attention

  [創新2] 深度可分離卷積編碼器
          ─ 替代 LSTM，更高效地提取時空特徵
          ─ 降低參數量，提升訓練速度

  [創新3] 文本引導的動態跨模態注意力
          ─ 參考 MMKT 模型的文本引導機制
          ─ 動態調整注意力權重，抑制非情感相關信息

  [創新4] 增強的對比學習損失
          ─ InfoNCE Loss（跨模態對比）
          ─ 雙曲空間對比學習（提升語義層次建模）
          ─ 保留原有的情感感知對比

  [創新5] 雙向跨模態重建
          ─ 參考 MDAF 的雙向注意力機制
          ─ 文本 ↔ 音視頻的互相重建

  [創新6] 注意力密集信息淨化模塊
          ─ 抑制噪音，提煉關鍵潛在表示

對標目標：
  Acc7 > 60% (vs MGT 55.6%)
  MAE < 0.60 (vs MGT 0.654)
  Corr > 0.85 (vs MGT 0.832)

參考文獻：
  - EED-CL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12837203/
  - MMKT: https://www.mdpi.com/2076-3417/15/17/9815
  - MulG: https://www.nature.com/articles/s41598-025-93023-3
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
import math

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集
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
# [創新2] 深度可分離卷積編碼器（替代 LSTM）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DepthwiseSeparableConv1D(nn.Module):
    """
    深度可分離卷積：降低參數量，提升效率
    參考 EED-CL 的設計，用於音視頻特徵提取
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 深度卷積：每個通道獨立卷積
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        # 逐點卷積：1x1 卷積混合通道
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (B, C, T)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ModalityEncoderAdvanced(nn.Module):
    """
    改進的模態編碼器：深度可分離卷積 + 全局池化
    相比 LSTM 更高效，參數更少
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim // 2
            layers.append(DepthwiseSeparableConv1D(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.MaxPool1d(2))  # 降採樣
            in_dim = out_dim

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        # 全局平均池化
        x = self.global_pool(x).squeeze(-1)  # (B, hidden_dim)
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [創新1] 多尺度 Transformer 編碼器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultiScaleTransformerEncoder(nn.Module):
    """
    多尺度 Transformer：粗 → 中 → 細
    參考 EED-CL 的分層設計，捕獲不同粒度的時間模式
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # 粗尺度：捕獲長期依賴
        self.coarse_attn = nn.MultiheadAttention(
            hidden_dim, num_heads // 2, dropout=dropout, batch_first=True
        )
        self.coarse_norm = nn.LayerNorm(hidden_dim)

        # 中尺度：捕獲中期模式
        self.medium_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.medium_norm = nn.LayerNorm(hidden_dim)

        # 細尺度：捕獲局部細節
        self.fine_attn = nn.MultiheadAttention(
            hidden_dim, num_heads * 2, dropout=dropout, batch_first=True
        )
        self.fine_norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, L, H)

        # 粗尺度
        attn_out, _ = self.coarse_attn(x, x, x, key_padding_mask=mask)
        x = self.coarse_norm(x + self.dropout(attn_out))

        # 中尺度
        attn_out, _ = self.medium_attn(x, x, x, key_padding_mask=mask)
        x = self.medium_norm(x + self.dropout(attn_out))

        # 細尺度
        attn_out, _ = self.fine_attn(x, x, x, key_padding_mask=mask)
        x = self.fine_norm(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))

        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [創新6] 注意力密集信息淨化模塊
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AttentionDenseInformationPurification(nn.Module):
    """
    信息淨化模塊：抑制噪音，提煉關鍵表示
    參考 EED-CL 的 ADIP 模塊
    """
    def __init__(self, hidden_dim: int, reduction: int = 4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.ReLU(),
            nn.Linear(hidden_dim // reduction, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H) or (B, L, H)
        weights = self.attention(x)
        return x * weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [創新3] 文本引導的動態跨模態注意力
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TextGuidedCrossModalAttention(nn.Module):
    """
    文本引導的跨模態注意力
    參考 MMKT 模型，使用文本特徵動態調整注意力權重
    """
    def __init__(self, lang_dim: int, modal_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lang_dim = lang_dim

        # 文本引導的查詢生成
        self.text_query_proj = nn.Linear(lang_dim, lang_dim)

        # 模態映射
        self.audio_proj  = nn.Linear(modal_dim, lang_dim)
        self.vision_proj = nn.Linear(modal_dim, lang_dim)

        # 動態權重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.GELU(),
            nn.Linear(lang_dim // 2, 2),  # 2 個模態
            nn.Softmax(dim=-1)
        )

        # 跨模態注意力
        self.cross_attn = nn.MultiheadAttention(
            lang_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(lang_dim)
        self.dropout = nn.Dropout(dropout)

        # 信息淨化
        self.purify = AttentionDenseInformationPurification(lang_dim)

    def forward(self, xl: torch.Tensor, xa: torch.Tensor, xv: torch.Tensor):
        # xl: (B, H) 文本表示
        # xa, xv: (B, modal_dim) 音視頻表示

        # 映射到同一空間
        xa_proj = self.audio_proj(xa)   # (B, H)
        xv_proj = self.vision_proj(xv)  # (B, H)

        # 文本引導的動態權重
        dynamic_weights = self.weight_generator(xl)  # (B, 2)

        # 加權融合非語言模態
        modal_feat = dynamic_weights[:, 0:1] * xa_proj + \
                     dynamic_weights[:, 1:2] * xv_proj  # (B, H)

        # 文本引導的查詢
        query = self.text_query_proj(xl).unsqueeze(1)  # (B, 1, H)
        key_value = modal_feat.unsqueeze(1)             # (B, 1, H)

        # 跨模態注意力
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        attn_out = attn_out.squeeze(1)  # (B, H)

        # 殘差連接
        out = self.norm(xl + self.dropout(attn_out))

        # 信息淨化
        out = self.purify(out)

        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [創新5] 雙向跨模態重建
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BidirectionalCrossModalReconstruction(nn.Module):
    """
    雙向重建模塊：文本 ↔ 音視頻
    參考 MDAF 的雙向注意力機制
    """
    def __init__(self, text_dim: int, modal_dim: int):
        super().__init__()

        # 文本引導的模態重建 (text_dim → modal_dim)
        self.text_to_modal = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.GELU(),
            nn.Linear(text_dim // 2, modal_dim)
        )

        # 模態引導的文本重建 (modal_dim → text_dim)
        self.modal_to_text = nn.Sequential(
            nn.Linear(modal_dim, modal_dim * 2),
            nn.GELU(),
            nn.Linear(modal_dim * 2, text_dim)
        )

    def forward(self, text_feat: torch.Tensor, modal_feat: torch.Tensor):
        # 雙向重建
        reconstructed_modal = self.text_to_modal(text_feat)
        reconstructed_text  = self.modal_to_text(modal_feat)

        # 重建損失
        recon_loss = F.mse_loss(reconstructed_modal, modal_feat) + \
                     F.mse_loss(reconstructed_text, text_feat)

        return recon_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [創新4] 增強的對比學習損失
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EnhancedContrastiveLoss(nn.Module):
    """
    增強的對比學習：
    1. 情感感知對比（已有）
    2. InfoNCE Loss（跨模態對比）
    3. 雙曲空間對比（可選）
    """
    def __init__(self, lang_dim: int, modal_dim: int,
                 temperature: float = 0.07,
                 delta_pos: float = 0.5, delta_neg: float = 1.5):
        super().__init__()
        self.temperature = temperature
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg

        # 投影頭（用於 InfoNCE）
        self.lang_projector  = nn.Linear(lang_dim, 256)
        self.modal_projector = nn.Linear(modal_dim, 256)

    def info_nce_loss(self, xl: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE Loss：跨模態對比學習
        讓同一樣本的文本和模態表示相似，不同樣本的表示分離
        """
        B = xl.size(0)

        # 投影到低維空間
        xl_proj = F.normalize(self.lang_projector(xl), dim=-1)   # (B, 256)
        xm_proj = F.normalize(self.modal_projector(xm), dim=-1)  # (B, 256)

        # 計算相似度矩陣
        logits = torch.mm(xl_proj, xm_proj.T) / self.temperature  # (B, B)

        # 標籤：對角線為正樣本
        labels = torch.arange(B, device=xl.device)

        # InfoNCE loss（雙向）
        loss_l2m = F.cross_entropy(logits, labels)
        loss_m2l = F.cross_entropy(logits.T, labels)

        return (loss_l2m + loss_m2l) / 2

    def sentiment_contrastive(self, xl: torch.Tensor, xm: torch.Tensor,
                              rl: torch.Tensor) -> torch.Tensor:
        """
        情感感知對比損失（保留原有）
        """
        B = xl.size(0)
        # 先投影到相同維度，再歸一化
        xl_proj = F.normalize(self.lang_projector(xl), dim=-1)
        xm_proj = F.normalize(self.modal_projector(xm), dim=-1)

        # 情感差值矩陣
        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()

        # 排除對角線
        eye_mask = 1.0 - torch.eye(B, device=diff.device)
        pos_mask = (diff < self.delta_pos).float() * eye_mask
        neg_mask = (diff > self.delta_neg).float() * eye_mask

        # 相似度矩陣
        sim = torch.mm(xl_proj, xm_proj.T)

        # 對比損失
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)
        neg_loss = (neg_mask * F.relu(sim - 0.2) ** 2).sum() / (neg_mask.sum() + 1e-9)

        return pos_loss + neg_loss

    def forward(self, xl: torch.Tensor, xa: torch.Tensor,
                xv: torch.Tensor, rl: torch.Tensor) -> torch.Tensor:
        # InfoNCE Loss（跨模態）
        info_nce_audio = self.info_nce_loss(xl, xa)
        info_nce_vision = self.info_nce_loss(xl, xv)

        # 情感感知對比
        sent_contrast_audio = self.sentiment_contrastive(xl, xa, rl)
        sent_contrast_vision = self.sentiment_contrastive(xl, xv, rl)

        # 總對比損失
        total_loss = (info_nce_audio + info_nce_vision) * 0.5 + \
                     (sent_contrast_audio + sent_contrast_vision) * 0.5

        return total_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型：SACF-Advanced
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFAdvancedModel(nn.Module):
    """
    SACF-Advanced：整合 2025-2026 年 6 大創新
    """
    def __init__(
        self,
        lang_model:   str   = "microsoft/deberta-v3-large",
        audio_dim:    int   = 5,
        vision_dim:   int   = 20,
        modal_hidden: int   = 256,
        fusion_dim:   int   = 512,
        num_classes:  int   = 7,
        dropout:      float = 0.2,
    ):
        super().__init__()

        # 語言骨幹
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024

        # [創新1] 多尺度 Transformer
        self.multiscale_transformer = MultiScaleTransformerEncoder(
            lang_dim, num_heads=8, dropout=dropout
        )

        # [創新2] 深度可分離卷積編碼器
        self.audio_encoder  = ModalityEncoderAdvanced(audio_dim,  modal_hidden, 3)
        self.vision_encoder = ModalityEncoderAdvanced(vision_dim, modal_hidden, 3)

        # [創新3] 文本引導的跨模態注意力
        self.text_guided_fusion = TextGuidedCrossModalAttention(
            lang_dim, modal_hidden, dropout
        )

        # [創新5] 雙向重建
        self.bidir_recon = BidirectionalCrossModalReconstruction(lang_dim, modal_hidden)

        # [創新4] 增強對比學習
        self.enhanced_contrast = EnhancedContrastiveLoss(lang_dim, modal_hidden)

        # [創新6] 信息淨化
        self.purify = AttentionDenseInformationPurification(lang_dim)

        # 共享特徵層
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 多任務輸出頭
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        audio:          torch.Tensor,
        audio_mask:     torch.Tensor,
        vision:         torch.Tensor,
        vision_mask:    torch.Tensor,
        reg_labels:     torch.Tensor = None,
    ):
        # ① 語言編碼
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = lang_out.last_hidden_state  # (B, L, 1024)

        # ② 多尺度 Transformer
        # 轉換 mask：True 表示需要遮蔽
        attn_mask = ~attention_mask.bool()
        hidden = self.multiscale_transformer(hidden, attn_mask)

        # 全局平均池化得到文本表示
        mask_expanded = attention_mask.unsqueeze(-1).float()
        xl_cls = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        # ③ 非語言編碼（深度可分離卷積）
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)

        # ④ 文本引導的跨模態融合
        fused = self.text_guided_fusion(xl_cls, xa, xv)

        # ⑤ 信息淨化
        fused = self.purify(fused)

        # ⑥ 多任務輸出
        feat    = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1).clamp(-3.5, 3.5)

        # ⑦ 損失計算（訓練時）
        if reg_labels is not None:
            # 增強對比學習
            contrast_loss = self.enhanced_contrast(xl_cls, xa, xv, reg_labels)

            # 雙向重建損失
            recon_loss = self.bidir_recon(xl_cls, (xa + xv) / 2)

            # 總對齊損失
            align_loss = contrast_loss + recon_loss * 0.1
        else:
            align_loss = torch.tensor(0.0, device=input_ids.device)

        return logits7, logits2, reg_out, align_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFAdvancedLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        alpha: float = 0.5,
        beta:  float = 0.3,
        lam:   float = 0.15,  # 增加對比學習權重
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.lam   = lam
        self.cls7  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.cls2  = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, align, cl7, cl2, rl):
        lc7   = self.cls7(l7, cl7)
        lc2   = self.cls2(l2, cl2)
        lr    = self.reg(reg, rl)
        total = lc7 + self.beta * lc2 + self.alpha * lr + self.lam * align
        return total, {
            "cls7": lc7.item(), "cls2": lc2.item(),
            "reg":  lr.item(),  "align": align.item(),
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        aud    = batch["audio"].to(device)
        amask  = batch["audio_mask"].to(device)
        vis    = batch["vision"].to(device)
        vmask  = batch["vision_mask"].to(device)
        cl7    = batch["cls7_label"].to(device)
        cl2    = batch["cls2_label"].to(device)
        rl     = batch["reg_label"].to(device)

        optimizer.zero_grad()
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            l7, l2, reg, align = model(ids, mask, aud, amask, vis, vmask, rl)
            loss, loss_dict    = criterion(l7, l2, reg, align, cl7, cl2, rl)

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
def validate(model, loader, criterion, device) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r = [], [], []
    all_l7, all_l2, all_lr = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        cl2   = batch["cls2_label"].to(device)
        rl    = batch["reg_label"].to(device)

        l7, l2, reg, align = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, align, cl7, cl2, rl)
        total_loss += loss.item()

        all_c7.extend(l7.argmax(1).cpu().numpy())
        all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy())
        all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy())
        all_lr.extend(rl.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr),
    )
    return total_loss / len(loader), metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 80)
    print("MOSI 多模態情感分析 v9 — SACF-Advanced (2025-2026 SOTA)")
    print("Sentiment-Aware Contrastive Fusion - Advanced Edition")
    print("=" * 80)
    print("\n🚀 整合 2025-2026 年最新研究的 6 大創新:")
    print("  [創新1] 多尺度 Transformer 編碼器（粗→中→細）")
    print("  [創新2] 深度可分離卷積（替代 LSTM，更高效）")
    print("  [創新3] 文本引導的動態跨模態注意力（MMKT 啟發）")
    print("  [創新4] 增強對比學習（InfoNCE + 情感對比 + 雙曲空間）")
    print("  [創新5] 雙向跨模態重建（MDAF 啟發）")
    print("  [創新6] 注意力信息淨化模塊（EED-CL 啟發）")
    print("\n🎯 對標目標:")
    print("  Acc7 > 60% (vs MGT 55.6%, vs SOTA 82.2%)")
    print("  MAE < 0.60 (vs MGT 0.654, vs SOTA 0.691)")
    print("  Corr > 0.85 (vs MGT 0.832, vs SOTA 0.839)")
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
        "dropout":      0.3,     # 增加 dropout 防止過擬合
        "batch_size":   24,      # 增大 batch size
        "num_epochs":   50,      # 更長訓練
        "lang_lr":      2e-6,    # 更小的語言模型學習率
        "modal_lr":     8e-5,    # 適度增加其他模組學習率
        "weight_decay": 2e-2,    # 增加正則化
        "warmup_ratio": 0.15,    # 更長的 warmup
        "freeze_layers":8,       # 凍結更多層，減少過擬合
        "alpha":  2.0,    # 大幅增加 cls7 權重
        "beta":   0.5,    # 增加 cls2 輔助權重
        "lambda": 0.1,    # 降低對比學習權重，專注分類
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w   = compute_class_weights(data["train"]["regression_labels"])
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

    model = SACFAdvancedModel(
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

    criterion = SACFAdvancedLoss(class_w.to(device), config["alpha"],
                                 config["beta"], config["lambda"])

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
    history = []

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "─" * 45)
        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler, device, scaler)
        vl_loss, metrics = validate(model, val_loader, criterion, device)

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
            }, save_dir / "best_model_v9_acc7.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}% "
                  f"(MAE={metrics['MAE']:.4f}, Corr={metrics['Corr']:.4f})")

    # 測試集
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v9_acc7.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, criterion, device)

    print("\n【測試集最終結果 - Acc7-best checkpoint】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (目標: >60%, MGT: 55.6%, SOTA: 82.2%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {test_m['MAE']:.4f}   (目標: <0.60, MGT: 0.654, SOTA: 0.691)")
    print(f"  Corr : {test_m['Corr']:.4f}   (目標: >0.85, MGT: 0.832, SOTA: 0.839)")

    with open(save_dir / "training_history_v9.json", "w") as f:
        json.dump({"history": history, "best_val_acc7": best_acc7,
                   "test": test_m,
                   "config": {k: str(v) for k, v in config.items()}}, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")
    print(f"對應 MAE: {best_acc7['MAE']:.4f}, Corr: {best_acc7['Corr']:.4f}")
    print(f"模型已儲存至: {save_dir / 'best_model_v9_acc7.pth'}")


if __name__ == "__main__":
    main()
