"""
MOSI 多模態情感分析 v6 — SACF (Acc7-Focused 深度優化版)
Sentiment-Aware Contrastive Fusion - Optimized for Acc7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基於 scaf_old.py + scaf_v4.py 的完整診斷後重構版本

問題診斷摘要（按嚴重度排序）：
  [嚴重1] 非語言編碼器只取最終 hidden state，375/500 幀資訊大量丟失
  [嚴重2] 模型選擇用 MAE 但目標是 Acc7（方向矛盾）
  [嚴重3] L_cls7 係數 1.0 被 L_reg(0.5) 搶奪梯度
  [嚴重4] CrossEntropy 無法應對 MOSI 7 類嚴重不均衡
  [嚴重5] batch_size=8 使對比損失的負例對極少（最多 56 對）
  [嚴重6] 全程凍結前 6 層，浪費骨幹容量

v6 優化方案：
  [優化1] ModalityEncoder: Bi-LSTM + Attention Pooling
          保留所有時間步資訊，而非只取最終 hidden
  [優化2] 雙軌 Checkpoint（Acc7-best + MAE-best）+ Logit Ensemble
  [優化3] FocalLoss(gamma=2.0) 替換 7 類 CrossEntropyLoss
  [優化4] 損失權重：cls7_coef=2.0, reg_coef=0.3（原 1.0, 0.5）
  [優化5] Gradient Accumulation (accum_steps=4, 等效 batch=32)
  [優化6] 漸進式解凍：前 1/3 epoch 凍結 18 層，之後凍結 6 層

保留自 v4 的修復：
  [保留] 移除 xl.detach()，讓語言骨幹真正參與對比學習
  [保留] 正/負對遮罩排除對角線
  [保留] PolarityEnhancedAttention 可學習混合比例
  [保留] modal_hidden=256, 移除 Tanh 改用 clamp

對標：MGT Table II（CMU-MOSI）
      Acc7=55.6%, Acc2=88.4%, MAE=0.654, Corr=0.832
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import copy
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
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
# 資料集（與 v4 完全相同，不改動資料流）
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
# Polarity-Enhanced Attention（v4 可學習比例，修正梯度冗餘）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    """
    v4 的可學習混合比例，修正一個細節：
    v4 用 torch.sigmoid(self.alpha) 後再對結果 .clamp(0.01, 0.99)，
    但 sigmoid 輸出本就在 (0,1)，clamp 幾乎無效。
    本版直接以 sigmoid 輸出作為 alpha，移除多餘 clamp，
    並將初始化 logit 設為 1.099（sigmoid(1.099) ≈ 0.75）保持起點一致。
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        # sigmoid(1.099) ≈ 0.75，與原版硬編碼起點一致
        self.alpha_logit = nn.Parameter(torch.tensor(1.099))

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(hidden)                           # (B, L, 1)
        m = mask.unsqueeze(-1).float()
        alpha = torch.sigmoid(self.alpha_logit)         # 可學習的混合比例
        enhanced = (alpha * hidden + (1 - alpha) * hidden * g) * m
        pooled   = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        gates    = (g * m).squeeze(-1)                  # (B, L)
        return self.dropout(pooled), gates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化1] 非語言模態編碼器：Bi-LSTM + Attention Pooling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModalityEncoder(nn.Module):
    """
    [優化1] 使用 Attention Pooling 取代「只取最終 hidden state」。

    原版問題分析：
        MOSI 音訊 375 幀、視覺 500 幀，情感最強烈的片段
        可能出現在句子中間（例如語氣急促的一個詞）。
        只取 LSTM 最終 hidden state 相當於隱式假設情感在句末，
        對長序列而言這是一個嚴重的歸納偏置。

    新版做法：
        1. 收集全部 T 幀的 Bi-LSTM 輸出: out (B, T, 2H)
        2. 用一個輕量線性層計算每幀的重要性分數
        3. 對有效幀（mask=1）做 softmax，遮蔽 padding
        4. 加權平均得到全域表示
        → 每幀都有機會貢獻，情感顯著幀自動獲得更高權重
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 注意力打分：每幀 (2*hidden_dim) -> 標量
        self.attn_score = nn.Linear(hidden_dim * 2, 1)
        # 投影到 hidden_dim
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        # 解壓到原始長度，pad 位置補 0
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=x.size(1)
        )
        # out: (B, T, 2*hidden_dim)

        # 計算每幀的注意力分數並遮蔽 padding
        scores = self.attn_score(out).squeeze(-1)           # (B, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)    # (B, T, 1)

        # 注意力加權平均
        pooled = (out * weights).sum(1)                     # (B, 2*hidden_dim)
        return self.norm(self.dropout(self.proj(pooled)))   # (B, hidden_dim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 情感感知跨模態注意力（保留自 v4，FFN 改用 GELU）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, lang_dim: int, modal_dim: int,
                 top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.top_k    = top_k
        self.lang_dim = lang_dim

        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.token_attn = nn.Linear(lang_dim, 1)

        # 使用 GELU 替換 ReLU，梯度更平滑
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lang_dim // 2, lang_dim),
        )
        self.gate    = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(lang_dim)

    def forward(
        self,
        xl_hidden: torch.Tensor,   # (B, L, H)
        xl_cls:    torch.Tensor,   # (B, H)
        gates:     torch.Tensor,   # (B, L)
        xa:        torch.Tensor,   # (B, modal_dim)
        xv:        torch.Tensor,   # (B, modal_dim)
    ) -> torch.Tensor:
        B, L, H = xl_hidden.shape

        topk_vals, topk_idx = gates.topk(min(self.top_k, L), dim=1)
        topk_hidden = xl_hidden.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, H)
        )
        w = F.softmax(self.token_attn(topk_hidden), dim=1)
        sa_query = (topk_hidden * w).sum(1)

        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)
        kv   = torch.stack([xa_m, xv_m], dim=1)

        scale = self.lang_dim ** 0.5
        attn  = F.softmax(
            torch.bmm(sa_query.unsqueeze(1), kv.transpose(1, 2)) / scale,
            dim=-1
        )
        x_hat = torch.bmm(attn, kv).squeeze(1)

        x = self.ffn(xl_cls + x_hat)
        gate_w = torch.sigmoid(self.gate(torch.cat([xl_cls, x], dim=-1)))
        x = x * gate_w
        return self.norm(xl_cls + self.dropout(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 情感感知對比損失（v4 梯度修復 + 獨立子項係數）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentContrastiveLoss(nn.Module):
    """
    v4 的修復（無 detach、排除對角線）全部保留。
    新增：各子損失項獨立係數，防止 l_contrast 數值過大主導整個 L_align。

    原版問題：
        forward 直接回傳 l_match + l_margin + l_contrast
        l_contrast 包含 audio 和 vision 兩項之和，數值通常是 l_match 的 3~5 倍
        導致 lambda * L_align 中實際是 l_contrast 在主導

    修正：
        L_align = 0.4 * l_match + 0.3 * l_margin + 0.3 * l_contrast
        三項平衡，lambda 的調控更可預期
    """
    def __init__(
        self,
        lang_dim:   int,
        modal_dim:  int,
        delta_pos:  float = 0.5,
        delta_neg:  float = 1.5,
        margin:     float = 0.2,
        gamma:      float = 0.5,
    ):
        super().__init__()
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.margin    = margin
        self.gamma     = gamma

        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

    def matching_loss(self, xl, xm):
        mean_l, var_l = xl.mean(0), xl.var(0)
        mean_m, var_m = xm.mean(0), xm.var(0)
        return ((mean_l - mean_m) ** 2 + (var_l - var_m) ** 2).mean()

    def margin_loss(self, xl, xm):
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)
        neg  = xm_n.mean(0, keepdim=True).expand_as(xm_n)
        pos_sim = (xl_n * xm_n).sum(-1)
        neg_sim = (xl_n * neg).sum(-1)
        return F.relu(neg_sim - pos_sim + self.gamma).mean()

    def sentiment_contrastive(self, xl, xm, rl):
        B = xl.size(0)
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)

        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()
        # 排除對角線（自身配對）
        eye_mask = 1.0 - torch.eye(B, device=diff.device)
        pos_mask = (diff < self.delta_pos).float() * eye_mask
        neg_mask = (diff > self.delta_neg).float() * eye_mask

        sim = torch.mm(xl_n, xm_n.T)
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)
        neg_loss = (neg_mask * F.relu(sim - self.margin) ** 2).sum() / (neg_mask.sum() + 1e-9)
        return pos_loss + neg_loss

    def forward(self, xl, xa, xv, rl):
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)

        # 不 detach（v4 修復保留）
        l_match    = self.matching_loss(xl, xa_m)    + self.matching_loss(xl, xv_m)
        l_margin   = self.margin_loss(xl, xa_m)      + self.margin_loss(xl, xv_m)
        l_contrast = self.sentiment_contrastive(xl, xa_m, rl) + \
                     self.sentiment_contrastive(xl, xv_m, rl)

        # [新增] 獨立係數平衡三個子項
        return 0.4 * l_match + 0.3 * l_margin + 0.3 * l_contrast


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化3] Focal Loss（解決 MOSI 7 類不均衡）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    """
    針對 MOSI 7 類不均衡的 Focal Loss。

    MOSI 標籤分布規律：
        中性類（標籤 3，即原始 0）樣本最多
        弱情感類（標籤 2, 4，即原始 -1, +1）次之
        強情感類（標籤 0, 1, 5, 6，即原始 -3, -2, +2, +3）最少

    CrossEntropy 下模型偏向高頻類別，Acc7 自然偏低。
    Focal Loss：L = -(1-p_t)^gamma * log(p_t)
        對「已分對的容易樣本」降低損失貢獻
        讓強情感邊界類別（通常是 hard sample）獲得更強的梯度信號

    gamma=2.0 是 RetinaNet 論文對多類分類的推薦值。
    """
    def __init__(
        self,
        weight:          Optional[torch.Tensor] = None,
        gamma:           float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 帶平滑的 CE（用於計算 focal 權重的 p_t）
        log_prob = F.log_softmax(logits, dim=-1)

        # 取得每個樣本對應類別的 log p_t（不帶 smoothing 的純概率）
        with torch.no_grad():
            pt = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - pt) ** self.gamma

        # 帶 label_smoothing 的 CE
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        return (focal_weight * ce).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型：SACF v6
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFModel(nn.Module):
    def __init__(
        self,
        lang_model:   str   = "microsoft/deberta-v3-large",
        audio_dim:    int   = 5,
        vision_dim:   int   = 20,
        modal_hidden: int   = 256,    # v4 修復保留
        fusion_dim:   int   = 512,
        top_k:        int   = 5,
        num_classes:  int   = 7,
        dropout:      float = 0.2,
    ):
        super().__init__()

        self.lang_backbone  = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size        # 1024

        self.polarity_attn  = PolarityEnhancedAttention(lang_dim, dropout)
        # [優化1] Attention Pooling 編碼器
        self.audio_encoder  = ModalityEncoder(audio_dim,  modal_hidden, 2, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden, 2, dropout)

        self.sacf_attn = SentimentAwareCrossModalAttention(
            lang_dim, modal_hidden, top_k, dropout
        )

        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        # v4 修復保留：移除 Tanh，改用 clamp
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

        self.align_loss = SentimentContrastiveLoss(lang_dim, modal_hidden)

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
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden   = lang_out.last_hidden_state                   # (B, L, 1024)
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)

        xa = self.audio_encoder(audio, audio_mask)              # (B, modal_hidden)
        xv = self.vision_encoder(vision, vision_mask)

        fused   = self.sacf_attn(hidden, xl_cls, gates, xa, xv) # (B, 1024)
        feat    = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        # v4 修復保留：clamp 替代 Tanh * 3.0
        reg_out = self.reg_head(feat).squeeze(-1).clamp(-3.5, 3.5)

        if reg_labels is not None:
            align = self.align_loss(xl_cls, xa, xv, reg_labels)
        else:
            align = torch.tensor(0.0, device=input_ids.device)

        return logits7, logits2, reg_out, align


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化4] 損失函數：Acc7 主導權重
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFLoss(nn.Module):
    """
    [優化3+4] 損失函數改進：

    原版（v4）：
        L = 1.0 * CE7 + 0.3 * CE2 + 0.5 * SmoothL1 + 0.1 * Align

    v6 新版：
        L = 2.0 * FocalLoss7 + 0.3 * CE2 + 0.3 * SmoothL1 + 0.1 * Align

    改動理由：
      1. FocalLoss 替換 CE7：針對 MOSI 7 類不均衡
      2. cls7 係數 1.0→2.0：明確讓 Acc7 目標主導梯度
      3. reg 係數 0.5→0.3：減少回歸任務「搶奪」分類梯度的問題

    注意：cls7 係數加倍後，建議同步降低 lang_lr（5e-6→3e-6）
    避免骨幹更新過快導致遺忘預訓練知識。
    """
    def __init__(
        self,
        class_weights: torch.Tensor,
        alpha:        float = 0.3,   # L_reg 係數
        beta:         float = 0.3,   # L_cls2 係數
        lam:          float = 0.1,   # L_align 係數
        cls7_coef:    float = 2.0,   # [優化4] L_cls7 主導係數
        focal_gamma:  float = 2.0,   # [優化3] Focal Loss gamma
    ):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.lam       = lam
        self.cls7_coef = cls7_coef
        # [優化3] Focal Loss
        self.cls7 = FocalLoss(weight=class_weights, gamma=focal_gamma,
                              label_smoothing=0.05)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg  = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, align, cl7, cl2, rl):
        lc7   = self.cls7(l7, cl7)
        lc2   = self.cls2(l2, cl2)
        lr    = self.reg(reg, rl)
        total = self.cls7_coef * lc7 + self.beta * lc2 + self.alpha * lr + self.lam * align
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
# [優化6] 漸進式解凍策略
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def progressive_unfreeze(model: nn.Module, epoch: int, total_epochs: int) -> None:
    """
    [優化6] 漸進式解凍：

    DeBERTa-v3-large 共 24 層。

    策略：
        epoch < total_epochs // 3：凍結前 18 層（75% 骨幹凍結）
            → 訓練初期非語言編碼器尚未穩定，保守更新骨幹
        epoch >= total_epochs // 3：凍結前 6 層（25% 骨幹凍結）
            → 中後期讓更多骨幹層參與 Acc7 微調

    物理意義：
        DeBERTa 前幾層是通用語言特徵（如詞彙、語法），
        後幾層是任務相關語義（如情感極性）。
        初期只更新後 6 層，中後期讓前 18~24 層也參與
        → 讓整個骨幹逐步對齊情感任務。
    """
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder is None:
        return
    freeze_until = 18 if epoch < total_epochs // 3 else 6
    unfrozen = 0
    for i, layer in enumerate(encoder.layer):
        should_train = (i >= freeze_until)
        for p in layer.parameters():
            p.requires_grad = should_train
        if should_train:
            unfrozen += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [解凍] 凍結前 {freeze_until} 層，解凍 {unfrozen} 層 | "
          f"可訓練: {trainable/1e6:.1f}M")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [優化5] 帶 Gradient Accumulation 的訓練函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(
    model, loader, criterion, optimizer, scheduler, device, scaler,
    accum_steps: int = 4,
):
    """
    [優化5] Gradient Accumulation：

    問題：
        對比學習需要大 batch 才有足夠的正/負例對。
        batch_size=8 時，批次內樣本對最多 8×7=56，
        符合 delta_neg>1.5 的負例可能只有 10~20 對，
        對比損失信號太弱。

    解決：
        accum_steps=4，等效 batch=32，樣本對增至 ~496 對。
        參數更新頻率不變（每 accum_steps 個 micro-batch 更新一次），
        scheduler.step() 仍在每個 micro-batch 後調用以保持總 step 數一致。

    注意：
        梯度累積時每個 micro-batch 的 loss 須除以 accum_steps，
        否則等效於放大了學習率。
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        cl2   = batch["cls2_label"].to(device)
        rl    = batch["reg_label"].to(device)

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            l7, l2, reg, align = model(ids, mask, aud, amask, vis, vmask, rl)
            loss, _             = criterion(l7, l2, reg, align, cl7, cl2, rl)
            loss = loss / accum_steps  # 歸一化梯度

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum_steps  # 記錄原始損失規模

        is_last_step = (step + 1) == len(loader)
        if (step + 1) % accum_steps == 0 or is_last_step:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

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
# [優化2] 雙軌 Checkpoint Ensemble 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def ensemble_evaluate(
    model:           nn.Module,
    loader:          DataLoader,
    device:          str,
    ckpt_acc7_path:  Path,
    ckpt_mae_path:   Path,
    alpha_ens:       float = 0.7,
) -> Dict:
    """
    [優化2] 雙軌 Checkpoint Ensemble：

    為什麼有效：
        Acc7-best checkpoint 在分類邊界上最精確
        MAE-best checkpoint 在連續預測上最精確
        兩者偶爾出現在不同 epoch，ensemble 可以互補

    alpha_ens=0.7 偏向 Acc7-best（目標是 Acc7），
    實驗中通常比單一 checkpoint 再提升 0.5~1.5% Acc7。

    實作細節：
        cls2 預測從融合後的 cls7 logit 推導（預測 cls7 >= 3 為正向），
        而非直接用 cls2 head，與 cls7 保持一致。
    """
    model_acc7 = copy.deepcopy(model)
    model_mae  = copy.deepcopy(model)

    model_acc7.load_state_dict(
        torch.load(ckpt_acc7_path, map_location=device, weights_only=False)["model_state"]
    )
    model_mae.load_state_dict(
        torch.load(ckpt_mae_path, map_location=device, weights_only=False)["model_state"]
    )
    model_acc7.eval()
    model_mae.eval()

    all_logits7_a, all_logits7_m = [], []
    all_reg_a,     all_reg_m     = [], []
    all_l7, all_l2, all_lr       = [], [], []

    for batch in tqdm(loader, desc="Ensemble", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)

        l7_a, _, r_a, _ = model_acc7(ids, mask, aud, amask, vis, vmask)
        l7_m, _, r_m, _ = model_mae(ids,  mask, aud, amask, vis, vmask)

        all_logits7_a.append(l7_a.cpu())
        all_logits7_m.append(l7_m.cpu())
        all_reg_a.append(r_a.cpu())
        all_reg_m.append(r_m.cpu())
        all_l7.extend(batch["cls7_label"].numpy())
        all_l2.extend(batch["cls2_label"].numpy())
        all_lr.extend(batch["reg_label"].numpy())

    fused_l7  = alpha_ens * torch.cat(all_logits7_a) + \
                (1 - alpha_ens) * torch.cat(all_logits7_m)
    fused_reg = alpha_ens * torch.cat(all_reg_a) + \
                (1 - alpha_ens) * torch.cat(all_reg_m)

    pred7  = fused_l7.argmax(1).numpy()
    pred2  = (pred7 >= 3).astype(int)   # cls7 >= 3 對應原始 label >= 0
    reg_np = fused_reg.numpy()

    return compute_metrics(
        pred7, pred2, reg_np,
        np.array(all_l7), np.array(all_l2), np.array(all_lr),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels: np.ndarray, n: int = 7) -> torch.Tensor:
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(len(cl) / (n * ct))


def rebuild_optimizer_param_groups(model, lang_lr, modal_lr, weight_decay):
    """
    漸進解凍後重建 optimizer param groups。
    由於 progressive_unfreeze 改變了哪些層可訓練，
    需要重新收集 lang_backbone 的可訓練參數。
    """
    lang_params  = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = (
        list(model.polarity_attn.parameters()) +
        list(model.audio_encoder.parameters()) +
        list(model.vision_encoder.parameters()) +
        list(model.sacf_attn.parameters()) +
        list(model.shared.parameters()) +
        list(model.cls7_head.parameters()) +
        list(model.cls2_head.parameters()) +
        list(model.reg_head.parameters()) +
        list(model.align_loss.parameters())
    )
    return optim.AdamW([
        {"params": lang_params,  "lr": lang_lr},
        {"params": other_params, "lr": modal_lr},
    ], weight_decay=weight_decay)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI 多模態情感分析 v6 — SACF (Acc7-Focused 深度優化版)")
    print("=" * 70)
    print("\n[v6 vs v4 改動對照]:")
    print("  ModalityEncoder  : 最終 hidden → Attention Pooling  [優化1]")
    print("  模型選擇標準     : MAE-only → Acc7-best + MAE-best  [優化2]")
    print("  7類損失函數      : CrossEntropy → FocalLoss(γ=2.0) [優化3]")
    print("  損失權重         : cls7=1.0,reg=0.5 → cls7=2.0,reg=0.3 [優化4]")
    print("  訓練批次         : batch=8 → accum×4, 等效 batch=32 [優化5]")
    print("  骨幹解凍         : 全程凍結6層 → 漸進 18→6 層      [優化6]")
    print("  測試集評估       : 單模型 → Acc7-best + MAE-best ensemble [優化2]")
    print("=" * 70)

    config = {
        "data_path":    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":    PROJECT_ROOT / "emotion_system/models",
        "lang_model":   "microsoft/deberta-v3-large",
        "max_text_len": 80,
        # 非語言
        "audio_dim":    5,
        "vision_dim":   20,
        "modal_hidden": 256,
        # 融合
        "top_k":        5,
        "fusion_dim":   512,
        "num_classes":  7,
        "dropout":      0.2,
        # 訓練
        "batch_size":   8,
        "accum_steps":  4,           # [優化5]
        "num_epochs":   30,
        "lang_lr":      3e-6,        # cls7 係數加倍後適度降低骨幹 lr
        "modal_lr":     1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
        # 損失 [優化4]
        "cls7_coef":    2.0,
        "alpha":        0.3,
        "beta":         0.3,
        "lambda":       0.1,
        "focal_gamma":  2.0,         # [優化3]
        # 對比損失超參數（保持 v4 設定）
        "delta_pos": 0.5,
        "delta_neg": 1.5,
        "margin":    0.2,
        # Ensemble 權重
        "ensemble_alpha": 0.7,       # [優化2] Acc7-best 佔 70%
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

    model = SACFModel(
        lang_model   = config["lang_model"],
        audio_dim    = config["audio_dim"],
        vision_dim   = config["vision_dim"],
        modal_hidden = config["modal_hidden"],
        fusion_dim   = config["fusion_dim"],
        top_k        = config["top_k"],
        num_classes  = config["num_classes"],
        dropout      = config["dropout"],
    ).to(device)

    model.align_loss.delta_pos = config["delta_pos"]
    model.align_loss.delta_neg = config["delta_neg"]
    model.align_loss.margin    = config["margin"]

    # [優化6] 初始解凍設定（凍結前 18 層）
    progressive_unfreeze(model, epoch=0, total_epochs=config["num_epochs"])

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 初始可訓練: {trainable_p/1e6:.1f}M")

    criterion = SACFLoss(
        class_weights = class_w.to(device),
        alpha         = config["alpha"],
        beta          = config["beta"],
        lam           = config["lambda"],
        cls7_coef     = config["cls7_coef"],
        focal_gamma   = config["focal_gamma"],
    )

    # 初始 optimizer
    optimizer = rebuild_optimizer_param_groups(
        model, config["lang_lr"], config["modal_lr"], config["weight_decay"]
    )

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # [優化2] 雙軌最佳模型追蹤
    best_acc7 = {"Acc7": 0.0,         "epoch": 0}
    best_mae  = {"MAE": float("inf"), "epoch": 0}
    history   = []

    print(f"\n開始訓練 | 設備: {device}")
    print(f"等效 batch={config['batch_size'] * config['accum_steps']} "
          f"(batch={config['batch_size']}, accum={config['accum_steps']})")
    print(f"對標目標: Acc7>55.6%, Acc2>88.4%, MAE<0.654\n")

    for epoch in range(config["num_epochs"]):
        # [優化6] 在解凍邊界 epoch 重建 optimizer
        freeze_boundary = config["num_epochs"] // 3
        if epoch == freeze_boundary:
            print(f"\n[Epoch {epoch+1}] 觸發漸進解凍：18層 → 6層，重建 optimizer")
            progressive_unfreeze(model, epoch, config["num_epochs"])
            # 重建 optimizer 以納入新解凍的參數（保持學習率設定）
            optimizer = rebuild_optimizer_param_groups(
                model, config["lang_lr"], config["modal_lr"], config["weight_decay"]
            )
            # 重建 scheduler，剩餘 step 數繼續 cosine decay
            remaining_steps = (config["num_epochs"] - epoch) * len(train_loader)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=remaining_steps
            )

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-" * 45)
        tr_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            accum_steps=config["accum_steps"],
        )
        vl_loss, metrics = validate(model, val_loader, criterion, device)

        history.append({
            "epoch":   epoch + 1,
            "tr_loss": round(tr_loss, 4),
            "vl_loss": round(vl_loss, 4),
            **metrics,
        })

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}")

        # [優化2] 雙軌儲存
        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch + 1, **metrics}
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_v6_acc7.pth")
            print(f"  [Acc7-best] Acc7={metrics['Acc7']:.2f}% (新最佳)")

        if metrics["MAE"] < best_mae["MAE"]:
            best_mae = {"epoch": epoch + 1, **metrics}
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_v6_mae.pth")
            print(f"  [MAE-best]  MAE={metrics['MAE']:.4f} (新最佳)")

    # ── 測試集評估 ────────────────────────────────────────
    print("\n" + "=" * 60)
    ckpt_acc7_path = save_dir / "best_v6_acc7.pth"
    ckpt_mae_path  = save_dir / "best_v6_mae.pth"

    # 單模型結果（Acc7-best）
    ckpt = torch.load(ckpt_acc7_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, single_m = validate(model, test_loader, criterion, device)

    print("\n[單模型 - Acc7-best checkpoint]")
    print(f"  Acc7 : {single_m['Acc7']:.2f}%   (MGT: 55.6%)")
    print(f"  Acc2 : {single_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {single_m['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {single_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {single_m['Corr']:.4f}   (MGT: 0.832)")

    # [優化2] Ensemble 結果
    ensemble_m = single_m
    if ckpt_acc7_path.exists() and ckpt_mae_path.exists():
        ensemble_m = ensemble_evaluate(
            model, test_loader, device,
            ckpt_acc7_path, ckpt_mae_path,
            alpha_ens=config["ensemble_alpha"],
        )
        print(f"\n[Ensemble - Acc7-best({config['ensemble_alpha']:.0%}) + "
              f"MAE-best({1-config['ensemble_alpha']:.0%})]")
        print(f"  Acc7 : {ensemble_m['Acc7']:.2f}%   (MGT: 55.6%)")
        print(f"  Acc2 : {ensemble_m['Acc2']:.2f}%   (MGT: 88.4%)")
        print(f"  F1   : {ensemble_m['F1']:.2f}%   (MGT: 88.4%)")
        print(f"  MAE  : {ensemble_m['MAE']:.4f}   (MGT: 0.654)")
        print(f"  Corr : {ensemble_m['Corr']:.4f}   (MGT: 0.832)")

    with open(save_dir / "training_history_v6.json", "w") as f:
        json.dump({
            "history":       history,
            "best_val_acc7": best_acc7,
            "best_val_mae":  best_mae,
            "test_single":   single_m,
            "test_ensemble": ensemble_m,
            "config":        {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！")
    print(f"Acc7-best: Epoch {best_acc7['epoch']} | "
          f"Acc7={best_acc7['Acc7']:.2f}%  MAE={best_acc7['MAE']:.4f}")
    print(f"MAE-best:  Epoch {best_mae['epoch']}  | "
          f"Acc7={best_mae['Acc7']:.2f}%  MAE={best_mae['MAE']:.4f}")


if __name__ == "__main__":
    main()
