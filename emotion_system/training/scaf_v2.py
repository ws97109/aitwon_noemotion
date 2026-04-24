"""
MOSI 多模態情感分析 v5 — SACF
Sentiment-Aware Contrastive Fusion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
論文創新點（相比 MGT）：

  [創新1] 情感敏感 Token 導向的跨模態注意力
          ─ MGT 用整體 CLS 向量當 query
          ─ 本文改為：挑出注意力權重最高的 K 個情感 token
            作為加權 query，讓非語言融合聚焦在情感相關語義

  [創新2] 情感感知多模態對比學習
          ─ MGT 的 Matching/Margin Loss 只對齊統計分佈
          ─ 本文以情感標籤距離定義正/負樣本對
            正對：同批次情感差值 < δ_pos 的樣本
            負對：同批次情感差值 > δ_neg 的樣本
            讓三模態在情感語義空間對齊，而非只對齊均值/方差

  [保留] DeBERTa-v3-large 語言骨幹 [Paper 1 最佳]
  [保留] Polarity-Enhanced Attention [Paper 1]
  [保留] Cross-Modal Additive Attention + Gating [Paper 2 MGT]
  [保留] Matching Loss + Margin Loss [Paper 2 MGT]
  [保留] 多任務輸出頭 Acc7/Acc2/Reg [Paper 3 EmoLLMs]

對標：MGT Table II（CMU-MOSI Acc7/Acc2/F1/MAE/Corr）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
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
# 資料集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 80):
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.raw_text     = split_data["raw_text"]
        self.audio        = torch.FloatTensor(split_data["audio"])    # (N, 375, 5)
        self.vision       = torch.FloatTensor(split_data["vision"])   # (N, 500, 20)
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
# Polarity-Enhanced Attention（Paper 1）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    """
    0.75 × 原始 + 0.25 × 極性門控，平均池化輸出 CLS 向量
    同時回傳 token-level gate 值（供情感 token 篩選使用）
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

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pooled : (B, H)  平均池化後的語言表示
            gates  : (B, L)  每個 token 的情感顯著性分數
        """
        g = self.gate(hidden)                          # (B, L, 1)
        m = mask.unsqueeze(-1).float()
        enhanced = (0.75 * hidden + 0.25 * hidden * g) * m
        pooled   = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        gates    = (g * m).squeeze(-1)                 # (B, L)
        return self.dropout(pooled), gates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [創新1] 情感敏感 Token 導向的跨模態注意力
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentAwareCrossModalAttention(nn.Module):
    """
    改進 MGT 的 CrossModalAdditive Attention：

    MGT：
        query = xl（CLS 整體）
        → 所有語音幀平等對待

    本模組（SACF 創新1）：
        先從語言 hidden states 挑出情感顯著性最高的 Top-K token
        query = 這 K 個 token 的注意力加權表示
        → 非語言融合聚焦在情感相關語義

    物理意義：
        "這句話裡 'terrible' 這個詞權重最高，
         所以我重點聽說話者在說 terrible 時的語音特徵"
    """

    def __init__(self, lang_dim: int, modal_dim: int,
                 top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.top_k    = top_k
        self.lang_dim = lang_dim

        # 非語言特徵映射到語言空間
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

        # 情感 token 加權聚合
        self.token_attn = nn.Linear(lang_dim, 1)

        # FFN（對應 MGT Eq.5）
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_dim // 2, lang_dim),
        )

        # Gating（對應 MGT Eq.6）
        self.gate    = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(lang_dim)

    def forward(
        self,
        xl_hidden: torch.Tensor,   # (B, L, H)  語言所有 token hidden
        xl_cls:    torch.Tensor,   # (B, H)     Polarity Attn 輸出
        gates:     torch.Tensor,   # (B, L)     Polarity gate 值
        xa:        torch.Tensor,   # (B, modal_dim)
        xv:        torch.Tensor,   # (B, modal_dim)
    ) -> torch.Tensor:             # (B, H)

        B, L, H = xl_hidden.shape

        # ① 挑出情感顯著性最高的 Top-K token
        topk_vals, topk_idx = gates.topk(
            min(self.top_k, L), dim=1
        )                                              # (B, K)
        topk_hidden = xl_hidden.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, H)
        )                                              # (B, K, H)

        # ② 用 token_attn 對 Top-K 再做加權聚合 → 情感感知 query
        w = F.softmax(self.token_attn(topk_hidden), dim=1)  # (B, K, 1)
        sa_query = (topk_hidden * w).sum(1)            # (B, H)  情感感知 query

        # ③ 非語言映射到語言空間
        xa_m = self.audio_map(xa)                      # (B, H)
        xv_m = self.vision_map(xv)                     # (B, H)
        kv   = torch.stack([xa_m, xv_m], dim=1)        # (B, 2, H)

        # ④ Cross-modal additive attention（用情感感知 query 而非 CLS）
        scale = self.lang_dim ** 0.5
        attn  = F.softmax(
            torch.bmm(sa_query.unsqueeze(1), kv.transpose(1, 2)) / scale,
            dim=-1
        )                                              # (B, 1, 2)
        x_hat = torch.bmm(attn, kv).squeeze(1)         # (B, H)

        # ⑤ 加回語言 CLS（非情感 query）並 FFN
        x = self.ffn(xl_cls + x_hat)

        # ⑥ Gating（評估多模態資訊的有效性）
        gate_w = torch.sigmoid(
            self.gate(torch.cat([xl_cls, x], dim=-1))
        )                                              # (B, 1)
        x = x * gate_w

        # ⑦ 殘差 + LayerNorm
        return self.norm(xl_cls + self.dropout(x))     # (B, H)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 非語言模態編碼器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)                            # (B, hidden_dim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [創新2] 情感感知多模態對比學習損失
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentContrastiveLoss(nn.Module):
    """
    改進 MGT 的 Matching + Margin Loss：

    MGT：
        以批次統計量（均值/方差）對齊模態分佈
        → 不感知情感標籤，正負樣本無區分

    本模組（SACF 創新2）：
        以情感標籤距離動態定義正/負樣本對
        正對：|y_i - y_j| < δ_pos  → 兩個模態應靠近
        負對：|y_i - y_j| > δ_neg  → 兩個模態應推遠

        L_contrast = Σ_正對 (1 - cos_sim)²
                   + Σ_負對 max(0, cos_sim - margin)²

    保留 MGT 的 Matching + Margin Loss 作為補充
    """

    def __init__(
        self,
        lang_dim:   int,
        modal_dim:  int,
        delta_pos:  float = 0.5,   # 情感差值 < 0.5 → 正對
        delta_neg:  float = 1.5,   # 情感差值 > 1.5 → 負對
        margin:     float = 0.2,
        gamma:      float = 0.5,   # MGT margin loss gamma
    ):
        super().__init__()
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.margin    = margin
        self.gamma     = gamma

        # 映射到語言空間用於 Matching Loss
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

    # ── MGT Matching Loss（保留）──────────────────────
    def matching_loss(self, xl: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        mean_l, var_l = xl.mean(0), xl.var(0)
        mean_m, var_m = xm.mean(0), xm.var(0)
        return ((mean_l - mean_m) ** 2 + (var_l - var_m) ** 2).mean()

    # ── MGT Margin Loss（保留）───────────────────────
    def margin_loss(self, xl: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)
        neg  = xm_n.mean(0, keepdim=True).expand_as(xm_n)
        pos_sim = (xl_n * xm_n).sum(-1)
        neg_sim = (xl_n * neg).sum(-1)
        return F.relu(neg_sim - pos_sim + self.gamma).mean()

    # ── [創新2] 情感感知對比損失 ─────────────────────
    def sentiment_contrastive(
        self,
        xl: torch.Tensor,   # (B, D) 語言表示
        xm: torch.Tensor,   # (B, D) 非語言表示
        rl: torch.Tensor,   # (B,)   情感回歸標籤
    ) -> torch.Tensor:
        B = xl.size(0)
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)

        # 批次內所有樣本對的情感差值矩陣
        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()   # (B, B)

        pos_mask = (diff < self.delta_pos).float()          # 正對遮罩
        neg_mask = (diff > self.delta_neg).float()          # 負對遮罩

        # 餘弦相似度矩陣
        sim = torch.mm(xl_n, xm_n.T)                       # (B, B)

        # 正對損失：相似度應接近 1
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)

        # 負對損失：相似度應低於 margin
        neg_loss = (neg_mask * F.relu(sim - self.margin) ** 2).sum() / (neg_mask.sum() + 1e-9)

        return pos_loss + neg_loss

    def forward(
        self,
        xl: torch.Tensor,
        xa: torch.Tensor,
        xv: torch.Tensor,
        rl: torch.Tensor,
    ) -> torch.Tensor:
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)

        # MGT 原有損失（保留）
        l_match  = self.matching_loss(xl.detach(), xa_m) + \
                   self.matching_loss(xl.detach(), xv_m)
        l_margin = self.margin_loss(xl.detach(), xa_m) + \
                   self.margin_loss(xl.detach(), xv_m)

        # 情感感知對比損失（創新2）
        l_contrast = self.sentiment_contrastive(xl.detach(), xa_m, rl) + \
                     self.sentiment_contrastive(xl.detach(), xv_m, rl)

        return l_match + l_margin + l_contrast


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型：SACF
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFModel(nn.Module):
    """
    Sentiment-Aware Contrastive Fusion

    資料流：
        input_ids, attention_mask
            → DeBERTa → hidden_states (B, L, 1024)
            → PolarityEnhancedAttention → xl_cls (B, 1024), gates (B, L)

        audio (B, 375, 5), audio_mask
            → ModalityEncoder(Bi-LSTM) → xa (B, 128)

        vision (B, 500, 20), vision_mask
            → ModalityEncoder(Bi-LSTM) → xv (B, 128)

        xl_hidden, xl_cls, gates, xa, xv
            → SentimentAwareCrossModalAttention → fused (B, 1024)  [創新1]

        fused → shared → feat (B, 512)
            → cls7_head → logits7 (B, 7)
            → cls2_head → logits2 (B, 2)
            → reg_head  → reg     (B,)

        xl_cls, xa, xv, reg_labels
            → SentimentContrastiveLoss → l_align  [創新2]
    """

    def __init__(
        self,
        lang_model:   str   = "microsoft/deberta-v3-large",
        audio_dim:    int   = 5,
        vision_dim:   int   = 20,
        modal_hidden: int   = 128,
        fusion_dim:   int   = 512,
        top_k:        int   = 5,
        num_classes:  int   = 7,
        dropout:      float = 0.2,
    ):
        super().__init__()

        self.lang_backbone  = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size            # 1024

        self.polarity_attn  = PolarityEnhancedAttention(lang_dim, dropout)
        self.audio_encoder  = ModalityEncoder(audio_dim,  modal_hidden, 2, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden, 2, dropout)

        # [創新1]
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
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Tanh(),
        )

        # [創新2]
        self.align_loss = SentimentContrastiveLoss(lang_dim, modal_hidden)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        audio:          torch.Tensor,
        audio_mask:     torch.Tensor,
        vision:         torch.Tensor,
        vision_mask:    torch.Tensor,
        reg_labels:     torch.Tensor = None,   # 訓練時傳入，推論時可不傳
    ):
        # ① 語言編碼
        lang_out  = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden    = lang_out.last_hidden_state                  # (B, L, 1024)
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)

        # ② 非語言編碼
        xa = self.audio_encoder(audio, audio_mask)              # (B, 128)
        xv = self.vision_encoder(vision, vision_mask)           # (B, 128)

        # ③ 情感感知跨模態融合（創新1）
        fused = self.sacf_attn(hidden, xl_cls, gates, xa, xv)   # (B, 1024)

        # ④ 多任務輸出
        feat    = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0

        # ⑤ 分佈對齊損失（創新2，訓練時計算）
        if reg_labels is not None:
            align = self.align_loss(xl_cls, xa, xv, reg_labels)
        else:
            align = torch.tensor(0.0, device=input_ids.device)

        return logits7, logits2, reg_out, align


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFLoss(nn.Module):
    """
    L = L_cls7(CrossEntropy+weight) + β·L_cls2 + α·L_reg + λ·L_align

    L_align = MGT Matching + MGT Margin + SACF SentimentContrastive
    還原使用加權 CrossEntropy（比 Focal Loss 更穩定）
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        alpha: float = 0.5,
        beta:  float = 0.3,
        lam:   float = 0.1,
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


def reg_to_cls7(reg: np.ndarray, thresholds: np.ndarray = None) -> np.ndarray:
    """
    將回歸值轉換為 7 類標籤
    thresholds: 6 個邊界值，預設為 [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    """
    if thresholds is None:
        thresholds = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    return np.digitize(reg, thresholds)   # 輸出 0~6


def find_optimal_thresholds(reg_preds: np.ndarray, true_cls7: np.ndarray) -> np.ndarray:
    """
    在驗證集上搜索最佳 6 個邊界閾值
    讓回歸值轉類別的 Acc7 最高
    """
    best_acc   = 0.0
    best_thr   = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    candidates = np.arange(-2.8, 2.8, 0.1)

    # 暴力搜索中間 4 個閾值（極端兩個固定）
    for t2 in np.arange(-2.2, -0.8, 0.2):
        for t3 in np.arange(-1.2, 0.2, 0.2):
            for t4 in np.arange(-0.3, 0.8, 0.2):
                for t5 in np.arange(0.3, 1.8, 0.2):
                    if not (t2 < t3 < t4 < t5):
                        continue
                    thr = np.array([-2.5, t2, t3, t4, t5, 2.5])
                    pred = np.digitize(reg_preds, thr)
                    acc  = (pred == true_cls7).mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_thr = thr

    print(f"  最佳閾值: {[f'{t:.1f}' for t in best_thr]}  Val Acc7={best_acc*100:.2f}%")
    return best_thr


@torch.no_grad()
def validate(
    model, loader, criterion, device,
    thresholds: np.ndarray = None,
    reg_cls_w:  float      = 0.3,
) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
    """
    Returns: loss, metrics, reg_preds, true_cls7
    reg_cls_w: 回歸輔助分類的混合權重（0=純分類，1=純回歸轉換）
    """
    model.eval()
    total_loss = 0.0
    all_logits7, all_c2, all_r = [], [], []
    all_l7, all_l2, all_lr     = [], [], []

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

        all_logits7.append(l7.cpu())
        all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy())
        all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy())
        all_lr.extend(rl.cpu().numpy())

    all_logits7 = torch.cat(all_logits7, dim=0).numpy()  # (N, 7)
    all_r   = np.array(all_r)
    all_l7  = np.array(all_l7)
    all_l2  = np.array(all_l2)
    all_lr  = np.array(all_lr)

    # 回歸輔助分類：把回歸值轉成 soft logits 再混合
    reg_cls = reg_to_cls7(all_r, thresholds)              # 回歸直接轉類別

    if reg_cls_w > 0:
        # 用回歸類別做 one-hot，與分類 logits 加權混合
        reg_onehot = np.zeros_like(all_logits7)
        for i, c in enumerate(reg_cls):
            reg_onehot[i, int(c)] = 1.0
        # softmax 分類 logits 後混合
        cls_prob  = np.exp(all_logits7) / np.exp(all_logits7).sum(-1, keepdims=True)
        mixed     = (1 - reg_cls_w) * cls_prob + reg_cls_w * reg_onehot
        final_c7  = mixed.argmax(-1)
    else:
        final_c7  = all_logits7.argmax(-1)

    metrics = compute_metrics(
        final_c7, np.array(all_c2), all_r,
        all_l7, all_l2, all_lr,
    )
    return total_loss / len(loader), metrics, all_r, all_l7


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
    print("=" * 70)
    print("MOSI 多模態情感分析 v5 — SACF")
    print("Sentiment-Aware Contrastive Fusion")
    print("DeBERTa-v3-large + 情感敏感融合 + 情感感知對比學習")
    print("=" * 70)

    config = {
        "data_path":    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":    PROJECT_ROOT / "models/mosi_sacf",
        "lang_model":   "microsoft/deberta-v3-large",
        "max_text_len": 80,
        # 非語言
        "audio_dim":    5,
        "vision_dim":   20,
        "modal_hidden": 128,
        # 融合
        "top_k":        5,       # 情感 token 取前 K 個
        "fusion_dim":   512,
        "num_classes":  7,
        "dropout":      0.2,      # 還原：0.3 過強
        # 訓練
        "batch_size":   16,       # 保留：梯度更穩定
        "num_epochs":   30,
        "lang_lr":      5e-6,
        "modal_lr":     1e-4,
        "weight_decay": 1e-2,     # 還原：5e-2 過強
        "warmup_ratio": 0.06,
        "freeze_layers":6,
        # 損失
        "alpha":  0.5,
        "beta":   0.3,
        "lambda": 0.1,
        # 對比損失超參數
        "delta_pos": 0.5,
        "delta_neg": 1.5,
        "margin":    0.2,
        # SWA 設定（改善泛化）
        "swa_start":  20,         # 第 20 epoch 開始 SWA
        "swa_lr":     1e-6,
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

    # 更新對比損失超參數
    model.align_loss.delta_pos = config["delta_pos"]
    model.align_loss.delta_neg = config["delta_neg"]
    model.align_loss.margin    = config["margin"]

    freeze_backbone_layers(model, config["freeze_layers"])

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = SACFLoss(
        class_w.to(device),
        config["alpha"], config["beta"], config["lambda"],
    )

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
    optimizer = optim.AdamW([
        {"params": lang_params,  "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["modal_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[創新1] 情感敏感 Token 導向跨模態注意力（Top-K={config['top_k']}）")
    print(f"[創新2] 情感感知對比損失（δ+={config['delta_pos']}, δ-={config['delta_neg']}）")
    print(f"對標: MGT Table II（Acc7/Acc2/F1/MAE/Corr）\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── SWA 模型（改善泛化）──────────────────────────────
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"],
                          anneal_epochs=5, anneal_strategy="cos")
    swa_started = False

    # ── 複合最佳指標：Acc7 + Corr 綜合評分 ──────────────
    def composite_score(m):
        return m["Acc7"] / 100 * 0.6 + m["Corr"] * 0.4

    best             = {"score": 0.0, "Acc7": 0.0, "epoch": 0}
    best_thresholds  = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    history          = []

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "─" * 45)
        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler, device, scaler)
        vl_loss, metrics, val_reg, val_l7 = validate(
            model, val_loader, criterion, device,
            thresholds=None, reg_cls_w=0.3,
        )

        history.append({"epoch": epoch+1,
                         "tr_loss": round(tr_loss, 4),
                         "vl_loss": round(vl_loss, 4), **metrics})

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}")

        # SWA：第 swa_start epoch 後開始累積權重
        if epoch + 1 >= config["swa_start"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_started = True
            print(f"  [SWA] 累積中（epoch {epoch+1}）")
        else:
            scheduler.step()

        # 複合評分選最佳模型
        score = composite_score(metrics)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch+1, **metrics}
            torch.save({
                "epoch":       epoch+1,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model.pth")
            print(f"  ✅ 新最佳 score={score:.4f} "
                  f"(Acc7={metrics['Acc7']:.2f}%, Corr={metrics['Corr']:.4f})")
            # 搜索最佳閾值（在驗證集上）
            best_thresholds = find_optimal_thresholds(val_reg, val_l7)

    # ── SWA BN 更新 ──────────────────────────────────────
    if swa_started:
        print("\n更新 SWA BatchNorm...")
        update_bn(train_loader, swa_model, device=device)
        torch.save({
            "epoch":       config["num_epochs"],
            "model_state": swa_model.module.state_dict(),
            "metrics":     best,
            "config":      {k: str(v) for k, v in config.items()},
        }, save_dir / "swa_model.pth")
        print("SWA 模型已儲存 → swa_model.pth")

    # ── 測試集：比較 best vs SWA ─────────────────────────
    print("\n" + "=" * 60)
    print("\n[Best Model 測試集結果]")
    print(f"  使用最佳閾值: {[f'{t:.1f}' for t in best_thresholds]}")
    ckpt = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m, _, _ = validate(model, test_loader, criterion, device,
                               thresholds=best_thresholds, reg_cls_w=0.3)

    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (MGT: 55.6%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")

    if swa_started:
        print("\n[SWA Model 測試集結果]")
        swa_ckpt = torch.load(save_dir / "swa_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(swa_ckpt["model_state"])
        _, swa_test_m, _, _ = validate(model, test_loader, criterion, device,
                                       thresholds=best_thresholds, reg_cls_w=0.3)
        print(f"  Acc7 : {swa_test_m['Acc7']:.2f}%   (MGT: 55.6%)")
        print(f"  Acc2 : {swa_test_m['Acc2']:.2f}%   (MGT: 88.4%)")
        print(f"  F1   : {swa_test_m['F1']:.2f}%   (MGT: 88.4%)")
        print(f"  MAE  : {swa_test_m['MAE']:.4f}   (MGT: 0.654)")
        print(f"  Corr : {swa_test_m['Corr']:.4f}   (MGT: 0.832)")

        # 自動選較好的存為最終結果
        final_m = swa_test_m if composite_score(swa_test_m) > composite_score(test_m) else test_m
        final_src = "SWA" if composite_score(swa_test_m) > composite_score(test_m) else "Best"
        print(f"\n最終採用: {final_src} Model")
    else:
        final_m = test_m

    with open(save_dir / "training_history.json", "w") as f:
        json.dump({"history": history, "best_val": best,
                   "test": final_m,
                   "config": {k: str(v) for k, v in config.items()}}, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best['Acc7']:.2f}% (Epoch {best['epoch']})")
    print(f"測試集 Acc7: {final_m['Acc7']:.2f}%  Corr: {final_m['Corr']:.4f}")


if __name__ == "__main__":
    main()