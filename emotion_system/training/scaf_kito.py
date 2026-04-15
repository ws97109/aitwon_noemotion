"""
MOSI 多模態情感分析 v6 — SACF v6
Sentiment-Aware Contrastive Fusion v6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在 v5（SACF）基礎上引入四項改進：

  [改進1] 多尺度非語言特徵編碼器（MultiScaleModalityEncoder）
          ─ v5：單一 Bi-LSTM 取最終 hidden state（128維）
          ─ v6：TCN（dilation=1,2,4）+ Bi-LSTM + 注意力池化（256維）
            捕捉不同時間尺度的情感動態

  [改進2] 層次化兩階段跨模態融合
          ─ v5：單次 L→(A,V) 獨立融合
          ─ v6：Stage1 A↔V 雙向跨模態注意力（AudioVisualCrossAttention）
                Stage2 L→AV 情感感知融合（SentimentAwareCrossModalAttention）
            讓語言模態接收已互融合的非語言資訊

  [改進3] 動態損失權重（DynamicLossWeighting）
          ─ v5：固定 α=0.5, β=0.3, λ=0.1
          ─ v6：Uncertainty Weighting（Kendall et al. 2018）
            5個可訓練 σ 參數自動調整各任務損失權重

  [改進4] 序數回歸輔助任務頭（OrdinalRegressionHead）
          ─ v5：純回歸頭（Linear→GELU→Linear→Tanh）
          ─ v6：主回歸頭 + 序數輔助頭（CORAL 設計）
            學習情感強度的序數關係（-3 < -2 < ... < 3）

  [保留] DeBERTa-v3-large 語言骨幹
  [保留] PolarityEnhancedAttention
  [保留] SentimentContrastiveLoss（情感感知對比學習）
  [保留] 多任務輸出頭 Acc7/Acc2/Reg

對標：MGT Table II（CMU-MOSI Acc7=55.6%, Acc2=88.4%, MAE=0.654, Corr=0.832）
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
from typing import Dict, Tuple
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, get_cosine_schedule_with_warmup, DebertaV2Tokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent


def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集（繼承自 v5）
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
# Polarity-Enhanced Attention（繼承自 v5）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    """0.75×原始 + 0.25×極性門控，回傳 CLS 向量與 token gates"""
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
        g = self.gate(hidden)                          # (B, L, 1)
        m = mask.unsqueeze(-1).float()
        enhanced = (0.75 * hidden + 0.25 * hidden * g) * m
        pooled   = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        gates    = (g * m).squeeze(-1)                 # (B, L)
        return self.dropout(pooled), gates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SentimentContrastiveLoss（繼承自 v5）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentContrastiveLoss(nn.Module):
    """情感感知多模態對比學習損失（v5 創新2，保留）"""
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

    def matching_loss(self, xl: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        mean_l, var_l = xl.mean(0), xl.var(0)
        mean_m, var_m = xm.mean(0), xm.var(0)
        return ((mean_l - mean_m) ** 2 + (var_l - var_m) ** 2).mean()

    def margin_loss(self, xl: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)
        neg  = xm_n.mean(0, keepdim=True).expand_as(xm_n)
        pos_sim = (xl_n * xm_n).sum(-1)
        neg_sim = (xl_n * neg).sum(-1)
        return F.relu(neg_sim - pos_sim + self.gamma).mean()

    def sentiment_contrastive(
        self, xl: torch.Tensor, xm: torch.Tensor, rl: torch.Tensor
    ) -> torch.Tensor:
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)
        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()
        pos_mask = (diff < self.delta_pos).float()
        neg_mask = (diff > self.delta_neg).float()
        sim = torch.mm(xl_n, xm_n.T)
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)
        neg_loss = (neg_mask * F.relu(sim - self.margin) ** 2).sum() / (neg_mask.sum() + 1e-9)
        return pos_loss + neg_loss

    def forward(
        self, xl: torch.Tensor, xa: torch.Tensor,
        xv: torch.Tensor, rl: torch.Tensor,
    ) -> torch.Tensor:
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)
        l_match  = self.matching_loss(xl.detach(), xa_m) + \
                   self.matching_loss(xl.detach(), xv_m)
        l_margin = self.margin_loss(xl.detach(), xa_m) + \
                   self.margin_loss(xl.detach(), xv_m)
        l_contrast = self.sentiment_contrastive(xl.detach(), xa_m, rl) + \
                     self.sentiment_contrastive(xl.detach(), xv_m, rl)
        return l_match + l_margin + l_contrast



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [改進1] TCNBlock — 膨脹因果卷積基本單元
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TCNBlock(nn.Module):
    """
    單一膨脹因果卷積塊

    架構：
        Conv1d(dilation=d, padding=causal) → LayerNorm → GELU → Dropout
        → Conv1d(dilation=d, padding=causal) → LayerNorm → GELU → Dropout
        → 殘差連接（1×1 Conv 調整維度）

    因果性：使用左側 padding（(kernel-1)*dilation），確保位置 t 只依賴 t 及之前的輸入。
    """
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        kernel_size: int = 3,
        dilation:   int = 1,
        dropout:    float = 0.1,
    ):
        super().__init__()
        # 因果 padding：只在左側填充
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            input_dim, output_dim, kernel_size,
            dilation=dilation, padding=0,
        )
        self.norm1 = nn.LayerNorm(output_dim)

        self.conv2 = nn.Conv1d(
            output_dim, output_dim, kernel_size,
            dilation=dilation, padding=0,
        )
        self.norm2 = nn.LayerNorm(output_dim)

        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 殘差投影（維度不同時）
        self.residual_proj = (
            nn.Conv1d(input_dim, output_dim, 1)
            if input_dim != output_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            out: (B, T, output_dim)
        """
        # 轉為 (B, C, T) 做 Conv1d
        residual = x.transpose(1, 2)                   # (B, C_in, T)
        h = residual

        # 第一層因果卷積
        h = F.pad(h, (self.pad, 0))                    # 左側 padding
        h = self.conv1(h)                              # (B, C_out, T)
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.dropout(h)

        # 第二層因果卷積
        h = F.pad(h, (self.pad, 0))
        h = self.conv2(h)                              # (B, C_out, T)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.dropout(h)

        # 殘差連接
        out = h + self.residual_proj(residual)         # (B, C_out, T)
        return out.transpose(1, 2)                     # (B, T, output_dim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [改進1] MultiScaleModalityEncoder — 多尺度非語言特徵編碼器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultiScaleModalityEncoder(nn.Module):
    """
    多尺度非語言特徵編碼器

    架構：
        TCN 分支：TCNBlock(d=1) → TCNBlock(d=2) → TCNBlock(d=4) → 注意力池化
        Bi-LSTM 分支：2層雙向 LSTM → 注意力池化
        concat → Linear → LayerNorm → output (B, output_dim=256)
    """
    def __init__(
        self,
        input_dim:       int,
        tcn_hidden:      int   = 128,
        lstm_hidden:     int   = 128,
        output_dim:      int   = 256,
        num_lstm_layers: int   = 2,
        tcn_dilations:   list  = None,
        dropout:         float = 0.2,
    ):
        super().__init__()
        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4]

        # TCN 分支
        tcn_layers = []
        in_dim = input_dim
        for d in tcn_dilations:
            tcn_layers.append(TCNBlock(in_dim, tcn_hidden, kernel_size=3,
                                       dilation=d, dropout=dropout))
            in_dim = tcn_hidden
        self.tcn = nn.Sequential(*tcn_layers)
        self.tcn_attn_score = nn.Linear(tcn_hidden, 1)

        # Bi-LSTM 分支
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden, num_layers=num_lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.lstm_proj = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.lstm_attn_score = nn.Linear(lstm_hidden, 1)

        # 融合投影
        self.proj = nn.Sequential(
            nn.Linear(tcn_hidden + lstm_hidden, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _attention_pool(
        self,
        seq:   torch.Tensor,   # (B, T, D)
        mask:  torch.Tensor,   # (B, T)
        score_layer: nn.Linear,
    ) -> torch.Tensor:         # (B, D)
        score = score_layer(seq).squeeze(-1)           # (B, T)
        score = score.masked_fill(mask == 0, float('-inf'))
        # 若 mask 全零，避免 softmax 產生 NaN
        all_zero = (mask.sum(dim=1) == 0).unsqueeze(1)  # (B, 1)
        score = score.masked_fill(all_zero.expand_as(score), 0.0)
        weight = F.softmax(score, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (seq * weight).sum(dim=1)               # (B, D)

    def forward(
        self,
        x:    torch.Tensor,    # (B, T, input_dim)
        mask: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:         # (B, output_dim)
        # NaN/Inf 保護
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # TCN 分支
        tcn_out = self.tcn(x)                          # (B, T, tcn_hidden)
        tcn_feat = self._attention_pool(tcn_out, mask, self.tcn_attn_score)

        # Bi-LSTM 分支
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=x.size(1)
        )                                              # (B, T, lstm_hidden*2)
        lstm_out = self.lstm_proj(lstm_out)            # (B, T, lstm_hidden)
        lstm_feat = self._attention_pool(lstm_out, mask, self.lstm_attn_score)

        # 拼接並投影
        combined = torch.cat([tcn_feat, lstm_feat], dim=-1)  # (B, tcn+lstm)
        return self.dropout(self.proj(combined))       # (B, output_dim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [改進2 Stage1] AudioVisualCrossAttention — 音訊-視覺雙向跨模態注意力
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AudioVisualCrossAttention(nn.Module):
    """
    音訊-視覺雙向跨模態注意力（Stage1）

    A→V：以音訊為 query，視覺為 key/value
    V→A：以視覺為 query，音訊為 key/value
    殘差連接後拼接，Linear(512→256) + LayerNorm 輸出 xav
    """
    def __init__(
        self,
        modal_dim: int   = 256,
        num_heads: int   = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.modal_dim = modal_dim
        self.scale     = modal_dim ** 0.5

        # A→V 方向
        self.q_a = nn.Linear(modal_dim, modal_dim)
        self.k_v = nn.Linear(modal_dim, modal_dim)
        self.v_v = nn.Linear(modal_dim, modal_dim)

        # V→A 方向
        self.q_v = nn.Linear(modal_dim, modal_dim)
        self.k_a = nn.Linear(modal_dim, modal_dim)
        self.v_a = nn.Linear(modal_dim, modal_dim)

        self.proj    = nn.Linear(modal_dim * 2, modal_dim)
        self.norm    = nn.LayerNorm(modal_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        xa: torch.Tensor,   # (B, modal_dim)
        xv: torch.Tensor,   # (B, modal_dim)
    ) -> torch.Tensor:      # (B, modal_dim)  xav
        # A→V：音訊 query 注意視覺
        q_a = self.q_a(xa).unsqueeze(1)                # (B, 1, D)
        k_v = self.k_v(xv).unsqueeze(1)                # (B, 1, D)
        v_v = self.v_v(xv).unsqueeze(1)                # (B, 1, D)
        attn_av = F.softmax(
            torch.bmm(q_a, k_v.transpose(1, 2)) / self.scale, dim=-1
        )                                              # (B, 1, 1)
        xa_enhanced = xa + self.dropout(
            torch.bmm(attn_av, v_v).squeeze(1)
        )                                              # (B, D)

        # V→A：視覺 query 注意音訊
        q_v = self.q_v(xv).unsqueeze(1)
        k_a = self.k_a(xa).unsqueeze(1)
        v_a = self.v_a(xa).unsqueeze(1)
        attn_va = F.softmax(
            torch.bmm(q_v, k_a.transpose(1, 2)) / self.scale, dim=-1
        )
        xv_enhanced = xv + self.dropout(
            torch.bmm(attn_va, v_a).squeeze(1)
        )                                              # (B, D)

        # 拼接並投影
        xav = self.norm(self.proj(
            torch.cat([xa_enhanced, xv_enhanced], dim=-1)
        ))                                             # (B, modal_dim)
        return xav


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [改進2 Stage2] SentimentAwareCrossModalAttention（v6 更新版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentAwareCrossModalAttention(nn.Module):
    """
    情感敏感 Token 導向跨模態注意力（v6 更新）

    v5 vs v6：
        v5: key/value = [xa_map, xv_map]（獨立音訊+視覺）
        v6: key/value = xav_map（Stage1 互融合後的 AV 表示）
    """
    def __init__(
        self,
        lang_dim:  int,
        modal_dim: int,
        top_k:     int   = 5,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.top_k    = top_k
        self.lang_dim = lang_dim

        # AV 融合表示映射到語言空間（v6：單一 av_map）
        self.av_map = nn.Linear(modal_dim, lang_dim)

        # 情感 token 加權聚合
        self.token_attn = nn.Linear(lang_dim, 1)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_dim // 2, lang_dim),
        )

        # Gating
        self.gate    = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(lang_dim)

    def forward(
        self,
        xl_hidden: torch.Tensor,   # (B, L, lang_dim)
        xl_cls:    torch.Tensor,   # (B, lang_dim)
        gates:     torch.Tensor,   # (B, L)
        xav:       torch.Tensor,   # (B, modal_dim)  Stage1 輸出
    ) -> torch.Tensor:             # (B, lang_dim)

        B, L, H = xl_hidden.shape

        # ① Top-K 情感 token 選取
        topk_vals, topk_idx = gates.topk(min(self.top_k, L), dim=1)
        topk_hidden = xl_hidden.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, H)
        )                                              # (B, K, H)

        # ② 情感感知 query
        w = F.softmax(self.token_attn(topk_hidden), dim=1)
        sa_query = (topk_hidden * w).sum(1)            # (B, H)

        # ③ AV 映射到語言空間（v6：單一 xav）
        xav_m = self.av_map(xav)                       # (B, H)
        kv    = xav_m.unsqueeze(1)                     # (B, 1, H)

        # ④ Cross-modal attention
        scale = self.lang_dim ** 0.5
        attn  = F.softmax(
            torch.bmm(sa_query.unsqueeze(1), kv.transpose(1, 2)) / scale,
            dim=-1
        )                                              # (B, 1, 1)
        x_hat = torch.bmm(attn, kv).squeeze(1)         # (B, H)

        # ⑤ FFN
        x = self.ffn(xl_cls + x_hat)

        # ⑥ Gating
        gate_w = torch.sigmoid(
            self.gate(torch.cat([xl_cls, x], dim=-1))
        )
        x = x * gate_w

        # ⑦ 殘差 + LayerNorm
        return self.norm(xl_cls + self.dropout(x))     # (B, H)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [改進3] DynamicLossWeighting — 不確定性動態損失權重
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DynamicLossWeighting(nn.Module):
    """
    基於不確定性的動態損失權重
    參考：Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)

    L_weighted_i = L_i * exp(-s_i) + s_i，s_i = clamp(log_sigma_i, -3, 3)
    """
    def __init__(self, task_names: list):
        super().__init__()
        self.task_names = task_names
        # 每個任務一個可訓練的 log_sigma 純量，初始化為 0（σ=1）
        for name in task_names:
            self.register_parameter(
                f"log_sigma_{name}",
                nn.Parameter(torch.zeros(1))
            )

    def forward(
        self,
        losses: dict,   # {"cls7": L7, "cls2": L2, "reg": Lr, "ordinal": Lo, "align": La}
    ) -> Tuple[torch.Tensor, dict]:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        weight_dict = {}

        for name in self.task_names:
            if name not in losses:
                continue
            s_i = getattr(self, f"log_sigma_{name}").clamp(-3, 3)
            L_i = losses[name]
            L_weighted = L_i * torch.exp(-s_i) + s_i
            total = total + L_weighted
            weight_dict[name] = torch.exp(-s_i).item()

        return total, weight_dict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [改進4] OrdinalRegressionHead — 序數回歸輔助任務頭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrdinalRegressionHead(nn.Module):
    """
    序數回歸輔助任務頭（CORAL 設計）

    輸出 K-1=6 個累積機率 P(y>k)，k ∈ {0,...,5}
    訓練：BCE(P(y>k), 1[y_true>k])
    推論：pred = Σ_k P(y>k) - 3 ∈ [-3, 3]
    """
    def __init__(
        self,
        feat_dim:       int,
        num_thresholds: int   = 6,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.dropout = nn.Dropout(dropout)
        # 共享權重，各閾值獨立偏置（CORAL）
        self.weight = nn.Parameter(torch.randn(feat_dim))
        self.bias   = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, feat_dim)
        Returns:
            ordinal_logits: (B, num_thresholds)  原始 logits（未過 sigmoid）
        """
        feat = self.dropout(feat)
        logit = feat @ self.weight                     # (B,)
        return logit.unsqueeze(1) + self.bias          # (B, K)  raw logits

    def compute_loss(
        self,
        ordinal_logits: torch.Tensor,  # (B, K)  raw logits
        cls7_labels:    torch.Tensor,  # (B,)  值域 [0,6]
    ) -> torch.Tensor:
        """序數 BCEWithLogits 損失（autocast 安全）"""
        K = ordinal_logits.shape[1]
        k_range = torch.arange(K, device=cls7_labels.device)
        targets = (cls7_labels.unsqueeze(1) > k_range).float()  # (B, K)
        return F.binary_cross_entropy_with_logits(ordinal_logits, targets)

    @staticmethod
    def predict(ordinal_logits: torch.Tensor) -> torch.Tensor:
        """
        raw logits → 連續情感強度
        先 sigmoid，再單調修正，pred = Σ P(y>k) - 3
        """
        ordinal_probs = torch.sigmoid(ordinal_logits)
        p = ordinal_probs.clone()
        K = p.shape[-1]
        for k in range(K - 2, -1, -1):
            p[..., k] = torch.max(p[..., k], p[..., k + 1])
        return p.sum(dim=-1) - 3.0                     # (B,)  ∈ [-3, 3]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型：SACFv6Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFv6Model(nn.Module):
    """
    SACF v6 主模型

    資料流：
        text → DeBERTa → PolarityEnhancedAttention → xl_cls, gates
        audio → MultiScaleModalityEncoder → xa (B,256)
        vision → MultiScaleModalityEncoder → xv (B,256)
        xa, xv → AudioVisualCrossAttention → xav (B,256)  [Stage1]
        xl_hidden, xl_cls, gates, xav → SentimentAwareCrossModalAttention → fused  [Stage2]
        fused → Shared → feat
            → cls7_head, cls2_head, reg_head, ordinal_head
        xl_cls, xa, xv → SentimentContrastiveLoss → l_align
        {l_cls7, l_cls2, l_reg, l_ordinal, l_align} → DynamicLossWeighting → l_total
    """
    def __init__(
        self,
        lang_model:   str   = "microsoft/deberta-v3-large",
        audio_dim:    int   = 5,
        vision_dim:   int   = 20,
        modal_hidden: int   = 256,
        fusion_dim:   int   = 512,
        top_k:        int   = 5,
        num_classes:  int   = 7,
        dropout:      float = 0.2,
    ):
        super().__init__()

        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size            # 1024

        self.polarity_attn  = PolarityEnhancedAttention(lang_dim, dropout)

        # [改進1] 多尺度非語言編碼器
        self.audio_encoder  = MultiScaleModalityEncoder(
            audio_dim,  tcn_hidden=modal_hidden//2, lstm_hidden=modal_hidden//2,
            output_dim=modal_hidden, dropout=dropout
        )
        self.vision_encoder = MultiScaleModalityEncoder(
            vision_dim, tcn_hidden=modal_hidden//2, lstm_hidden=modal_hidden//2,
            output_dim=modal_hidden, dropout=dropout
        )

        # [改進2] 層次化融合
        self.av_cross_attn = AudioVisualCrossAttention(modal_hidden, dropout=dropout)
        self.sacf_attn     = SentimentAwareCrossModalAttention(
            lang_dim, modal_hidden, top_k, dropout
        )

        # 共享投影
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 輸出頭
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Tanh(),
        )

        # [改進4] 序數回歸輔助頭
        self.ordinal_head = OrdinalRegressionHead(fusion_dim, dropout=dropout)

        # 對比損失（繼承 v5）
        self.align_loss = SentimentContrastiveLoss(lang_dim, modal_hidden)

        # [改進3] 動態損失權重
        self.dynamic_loss = DynamicLossWeighting(
            ["cls7", "cls2", "reg", "ordinal", "align"]
        )


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
        # NaN/Inf 保護
        audio  = torch.nan_to_num(audio,  nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)

        # ① 語言編碼
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = lang_out.last_hidden_state                     # (B, L, 1024)
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)

        # ② 多尺度非語言編碼（改進1）
        xa = self.audio_encoder(audio, audio_mask)              # (B, 256)
        xv = self.vision_encoder(vision, vision_mask)           # (B, 256)

        # ③ Stage1：A↔V 互融合（改進2）
        xav = self.av_cross_attn(xa, xv)                        # (B, 256)

        # ④ Stage2：L→AV 情感感知融合（改進2）
        fused = self.sacf_attn(hidden, xl_cls, gates, xav)      # (B, 1024)

        # ⑤ 多任務輸出
        feat         = self.shared(fused)
        logits7      = self.cls7_head(feat)
        logits2      = self.cls2_head(feat)
        reg_pred     = self.reg_head(feat).squeeze(-1) * 3.0
        ordinal_probs = self.ordinal_head(feat)                 # (B, 6) logits

        out = {
            "logits7":       logits7,
            "logits2":       logits2,
            "reg_pred":      reg_pred,
            "ordinal_probs": ordinal_probs,
        }

        # ⑥ 訓練模式：計算損失
        if cls7_labels is not None and reg_labels is not None:
            cls2_labels = (reg_labels >= 0).long()

            l_cls7    = F.cross_entropy(logits7, cls7_labels, label_smoothing=0.1)
            l_cls2    = F.cross_entropy(logits2, cls2_labels, label_smoothing=0.05)
            l_reg     = F.smooth_l1_loss(reg_pred, reg_labels)
            l_ordinal = self.ordinal_head.compute_loss(ordinal_probs, cls7_labels)
            l_align   = self.align_loss(xl_cls, xa, xv, reg_labels)

            total_loss, weight_dict = self.dynamic_loss({
                "cls7":    l_cls7,
                "cls2":    l_cls2,
                "reg":     l_reg,
                "ordinal": l_ordinal,
                "align":   l_align,
            })

            out["total_loss"]  = total_loss
            out["loss_dict"]   = {
                "cls7":    l_cls7.item(),
                "cls2":    l_cls2.item(),
                "reg":     l_reg.item(),
                "ordinal": l_ordinal.item(),
                "align":   l_align.item(),
            }
            out["weight_dict"] = weight_dict
        else:
            # 推論模式
            out["ordinal_pred"] = OrdinalRegressionHead.predict(ordinal_probs)

        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估指標（繼承自 v5）
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
# 訓練 / 驗證（v6 更新版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        rl    = batch["reg_label"].to(device)

        optimizer.zero_grad()
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            out  = model(ids, mask, aud, amask, vis, vmask, cl7, rl)
            loss = out["total_loss"]

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
def validate(model, loader, device) -> Tuple[float, Dict]:
    model.eval()
    all_c7, all_c2, all_r = [], [], []
    all_l7, all_l2, all_lr = [], [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        rl    = batch["reg_label"].to(device)

        # 推論模式（不傳標籤）
        out = model(ids, mask, aud, amask, vis, vmask)

        # 計算簡易驗證損失（不含動態權重）
        l_cls7 = F.cross_entropy(out["logits7"], cl7)
        l_reg  = F.smooth_l1_loss(out["reg_pred"], rl)
        total_loss += (l_cls7 + l_reg).item()

        all_c7.extend(out["logits7"].argmax(1).cpu().numpy())
        all_c2.extend(out["logits2"].argmax(1).cpu().numpy())
        all_r.extend(out["reg_pred"].cpu().numpy())
        all_l7.extend(cl7.cpu().numpy())
        all_l2.extend((rl >= 0).long().cpu().numpy())
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
    print("=" * 70)
    print("MOSI 多模態情感分析 v6 — SACF v6")
    print("多尺度編碼 + 層次化融合 + 動態損失 + 序數回歸")
    print("=" * 70)

    config = {
        "data_path":    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":    PROJECT_ROOT / "emotion_system/models",
        "lang_model":   "microsoft/deberta-v3-large",
        "max_text_len": 80,
        # 非語言
        "audio_dim":    5,
        "vision_dim":   20,
        "modal_hidden": 256,    # v6: 256（v5: 128）
        # 融合
        "top_k":        5,
        "fusion_dim":   512,
        "num_classes":  7,
        "dropout":      0.2,
        # 訓練
        "batch_size":   8,
        "num_epochs":   30,
        "lang_lr":      5e-6,
        "modal_lr":     1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
        "freeze_layers": 6,
        # 對比損失（繼承 v5）
        "delta_pos":    0.5,
        "delta_neg":    1.5,
        "margin":       0.2,
    }

    set_seed(42)

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

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

    model = SACFv6Model(
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

    # 參數分組：語言骨幹 vs 其他
    lang_params  = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = (
        list(model.polarity_attn.parameters()) +
        list(model.audio_encoder.parameters()) +
        list(model.vision_encoder.parameters()) +
        list(model.av_cross_attn.parameters()) +
        list(model.sacf_attn.parameters()) +
        list(model.shared.parameters()) +
        list(model.cls7_head.parameters()) +
        list(model.cls2_head.parameters()) +
        list(model.reg_head.parameters()) +
        list(model.ordinal_head.parameters()) +
        list(model.align_loss.parameters()) +
        list(model.dynamic_loss.parameters())
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
    print(f"[改進1] 多尺度非語言編碼器（TCN+LSTM，modal_hidden={config['modal_hidden']}）")
    print(f"[改進2] 層次化兩階段跨模態融合（A↔V → L→AV）")
    print(f"[改進3] 動態損失權重（Uncertainty Weighting）")
    print(f"[改進4] 序數回歸輔助任務頭（CORAL）")
    print(f"對標: MGT Table II（Acc7=55.6%, Acc2=88.4%, MAE=0.654, Corr=0.832）\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best    = {"MAE": float('inf'), "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "─" * 45)
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        vl_loss, metrics = validate(model, val_loader, device)

        history.append({"epoch": epoch+1,
                         "tr_loss": round(tr_loss, 4),
                         "vl_loss": round(vl_loss, 4), **metrics})

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}")

        if metrics["MAE"] < best["MAE"]:
            best = {"epoch": epoch+1, **metrics}
            torch.save({
                "epoch":       epoch+1,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model_v6.pth")
            print(f"  ✅ 新最佳 MAE={metrics['MAE']:.4f}")

    # 測試集
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v6.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, device)

    print("\n【測試集最終結果（對標 MGT Table II）】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (MGT: 55.6%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")

    with open(save_dir / "training_history_v6.json", "w") as f:
        json.dump({"history": history, "best_val": best,
                   "test": test_m,
                   "config": {k: str(v) for k, v in config.items()}}, f, indent=2)

    print(f"\n完成！最佳 Val MAE: {best['MAE']:.4f} (Epoch {best['epoch']})")


if __name__ == "__main__":
    main()
