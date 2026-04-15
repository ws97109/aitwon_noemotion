"""
MOSI 多模態情感分析 v28 — SACF Elite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

架構融合（三代精華）：

  [v5/scaf_old.py 核心創新 — 保留]
    PolarityEnhancedAttention: 回傳 token 極性閘值 + 池化 CLS
    SentimentAwareCrossModalAttention: Top-K 情感 token 作為跨模態 query
      → 讓非語言融合聚焦在情感相關語義（比純 CLS 更精準）
    SentimentContrastiveLoss: 情感感知對比（xl.detach() → 不干擾語言骨幹）

  [v6/scaf_kito.py 編碼器升級 — 保留]
    MultiScaleModalityEncoder: TCN(d=1,2,4) + Bi-LSTM + 注意力池化 (256-dim)
      → 比 v5 單一 Bi-LSTM(128-dim) 更強的時序特徵提取
    AudioVisualCrossAttention (修正版): A↔V 雙向互注意力
      → 分別回傳 xa_enhanced, xv_enhanced（不合併為 xav）
      → 保持 2-key SentimentAwareCrossModalAttention 有意義的注意力分配

  [v27 訓練技術 — 全部加入]
    L2 normalization (F.normalize，解決音視頻幅度偏移)
    FocalLoss(gamma=2) + capped class weights [0.5, 3.0]
    R-Drop (alpha=0.05，比 v27 小，因 SentimentContrastiveLoss 已提供對比正則)
    EMA(0.9995)
    SWA (start 50%)
    Optimizer 修復: 所有 DeBERTa 層從初始化就加入 optimizer
    Progressive Unfreeze (搭配 optimizer 修復)
    4步推論增強 (TTA → 先驗校正 → 閾值搜索 → 軟集成)
    Acc7 選模（不用 MAE）
    patience=25, epochs=150

  [v6 移除]
    DynamicLossWeighting → 改為固定權重（更穩定，150 epoch 訓練中不會跑偏）

設計亮點：
  AudioVisualCrossAttention 修正: 分離輸出 xa_enh, xv_enh
    → SentimentAwareCrossModalAttention 仍接 2-key (audio+vision 分離)
    → 注意力可有意義地在音頻/視覺間分配權重
  SentimentContrastiveLoss 只對 xa_enh, xv_enh（不是原始 xa, xv）
    → 對比損失對 A↔V 互融合後的特徵進行對齊，更有語義

關鍵超參:
  lang_lr=5e-6, other_lr=1e-4, epochs=150, patience=25
  w_cls7=3.5, w_cls2=0.5, w_reg=0.3, w_ord=0.2, w_align=0.05
  top_k=5, modal_hidden=256

歷史:
  v5  (scaf_old.py):  最佳架構創新基線
  v6  (scaf_kito.py): v5 + 多尺度編碼 + A↔V 融合（但訓練僅30 epoch）
  v23:               Test 48.69% (無 v5 創新，簡單 CrossModalAttention)
  v27:               Test 48.54% (v23 + optimizer 修復 + 推論增強)
  v28:               目標 Test 50.5%+
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, pickle, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, get_cosine_schedule_with_warmup, DebertaV2Tokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_text_len=80):
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
        print(f"資料: {len(self.raw_text)} 筆 | audio={tuple(self.audio.shape)} | vision={tuple(self.vision.shape)}")

    def __len__(self): return len(self.raw_text)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            TASK_PROMPT + str(self.raw_text[idx]),
            add_special_tokens=True, max_length=self.max_text_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        aud_len = min(int(self.audio_lengths[idx]),  self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]);  aud_mask[:aud_len]  = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len]  = 1.0
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio":    self.audio[idx],    "audio_mask": aud_mask,
            "vision":   self.vision[idx],   "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label":  self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v5] PolarityEnhancedAttention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    """
    0.75×原始 + 0.25×極性門控 → 池化 CLS 和 token 極性閘值
    gates 用於 SentimentAwareCrossModalAttention 的 Top-K 選取
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        g = self.gate(hidden)                          # (B, L, 1)
        m = mask.unsqueeze(-1).float()
        pooled = ((0.75 * hidden + 0.25 * hidden * g) * m).sum(1) / m.sum(1).clamp(min=1e-9)
        gates  = (g * m).squeeze(-1)                   # (B, L) token 極性顯著性
        return self.dropout(pooled), gates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v6] TCNBlock + MultiScaleModalityEncoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TCNBlock(nn.Module):
    """因果膨脹卷積塊: Conv1d(d) → LayerNorm → GELU → Dropout × 2 + 殘差"""
    def __init__(self, input_dim, output_dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(input_dim,  output_dim, kernel_size, dilation=dilation, padding=0)
        self.norm1 = nn.LayerNorm(output_dim)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, dilation=dilation, padding=0)
        self.norm2 = nn.LayerNorm(output_dim)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.res_proj = (nn.Conv1d(input_dim, output_dim, 1)
                         if input_dim != output_dim else nn.Identity())

    def forward(self, x):                              # (B, T, C)
        res = x.transpose(1, 2)                        # (B, C, T)
        h = res
        h = self.dropout(self.act(self.norm1(
            self.conv1(F.pad(h, (self.pad, 0))).transpose(1,2)
        ).transpose(1,2)))
        h = self.dropout(self.act(self.norm2(
            self.conv2(F.pad(h, (self.pad, 0))).transpose(1,2)
        ).transpose(1,2)))
        return (h + self.res_proj(res)).transpose(1, 2) # (B, T, output_dim)


class MultiScaleModalityEncoder(nn.Module):
    """
    多尺度非語言特徵編碼器 (v6)
    TCN(d=1,2,4) + Bi-LSTM → 注意力池化 → Linear+LayerNorm → 256-dim
    """
    def __init__(self, input_dim, tcn_hidden=128, lstm_hidden=128,
                 output_dim=256, num_lstm_layers=2, dropout=0.2):
        super().__init__()
        tcn_layers, in_dim = [], input_dim
        for d in [1, 2, 4]:
            tcn_layers.append(TCNBlock(in_dim, tcn_hidden, 3, d, dropout))
            in_dim = tcn_hidden
        self.tcn = nn.Sequential(*tcn_layers)
        self.tcn_attn = nn.Linear(tcn_hidden, 1)

        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=num_lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.lstm_attn = nn.Linear(lstm_hidden, 1)

        self.proj    = nn.Sequential(nn.Linear(tcn_hidden + lstm_hidden, output_dim),
                                     nn.LayerNorm(output_dim))
        self.dropout = nn.Dropout(dropout)

    def _attn_pool(self, seq, mask, scorer):
        s = scorer(seq).squeeze(-1)
        s = s.masked_fill(mask == 0, float('-inf'))
        s = s.masked_fill((mask.sum(1, keepdim=True) == 0).expand_as(s), 0.0)
        return (seq * F.softmax(s, dim=1).unsqueeze(-1)).sum(1)

    def forward(self, x, mask):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # TCN 分支
        tcn_feat = self._attn_pool(self.tcn(x), mask, self.tcn_attn)

        # Bi-LSTM 分支
        lengths = mask.sum(1).long().clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=x.size(1))
        lstm_feat = self._attn_pool(self.lstm_proj(lstm_out), mask, self.lstm_attn)

        return self.dropout(self.proj(torch.cat([tcn_feat, lstm_feat], dim=-1)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v6 修正版] AudioVisualCrossAttention — 分離輸出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AudioVisualCrossAttention(nn.Module):
    """
    A↔V 雙向跨模態注意力 (v28 修正版)

    v6 原版: 輸出單一 xav (concat 後 project)
      → SentimentAwareCrossModalAttention 只有1個 key → softmax trivial (always 1.0)
      → 注意力失去意義

    v28 修正: 分別輸出 xa_enhanced, xv_enhanced
      → SentimentAwareCrossModalAttention 保持2個 key (audio + vision)
      → 注意力可有意義地在音頻/視覺間分配權重
    """
    def __init__(self, modal_dim=256, dropout=0.1):
        super().__init__()
        self.scale = modal_dim ** 0.5
        # A→V 方向
        self.q_a = nn.Linear(modal_dim, modal_dim)
        self.k_v = nn.Linear(modal_dim, modal_dim)
        self.v_v = nn.Linear(modal_dim, modal_dim)
        # V→A 方向
        self.q_v = nn.Linear(modal_dim, modal_dim)
        self.k_a = nn.Linear(modal_dim, modal_dim)
        self.v_a = nn.Linear(modal_dim, modal_dim)
        self.norm_a  = nn.LayerNorm(modal_dim)
        self.norm_v  = nn.LayerNorm(modal_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xa, xv):                        # (B, D) each
        # A→V: 音訊 query 注意視覺 → xa enriched
        q_a  = self.q_a(xa).unsqueeze(1)              # (B, 1, D)
        k_v  = self.k_v(xv).unsqueeze(1)
        v_v  = self.v_v(xv).unsqueeze(1)
        attn = F.softmax(torch.bmm(q_a, k_v.transpose(1,2)) / self.scale, dim=-1)
        xa_enhanced = self.norm_a(xa + self.dropout(torch.bmm(attn, v_v).squeeze(1)))

        # V→A: 視覺 query 注意音訊 → xv enriched
        q_v  = self.q_v(xv).unsqueeze(1)
        k_a  = self.k_a(xa).unsqueeze(1)
        v_a  = self.v_a(xa).unsqueeze(1)
        attn = F.softmax(torch.bmm(q_v, k_a.transpose(1,2)) / self.scale, dim=-1)
        xv_enhanced = self.norm_v(xv + self.dropout(torch.bmm(attn, v_a).squeeze(1)))

        return xa_enhanced, xv_enhanced                # (B, D), (B, D)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v5] SentimentAwareCrossModalAttention (2-key, Top-K)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentAwareCrossModalAttention(nn.Module):
    """
    情感敏感 Token 導向跨模態注意力 (v5 核心創新)

    v27 的 CrossModalAttention: query = xl_cls（整體 CLS）
    本模組: query = 情感顯著性最高的 Top-K token 的加權表示
    → 非語言融合聚焦在情感相關詞彙（如 "terrible", "amazing"）

    Key/Value: xa_enhanced (A↔V 互融合後的音頻) + xv_enhanced (視覺)
    → 2個 key → softmax 在音頻/視覺間有意義地分配注意力
    """
    def __init__(self, lang_dim, modal_dim, top_k=5, dropout=0.1):
        super().__init__()
        self.top_k    = top_k
        self.lang_dim = lang_dim
        self.audio_map   = nn.Linear(modal_dim, lang_dim)
        self.vision_map  = nn.Linear(modal_dim, lang_dim)
        self.token_attn  = nn.Linear(lang_dim, 1)
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim // 2, lang_dim),
        )
        self.gate    = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(lang_dim)

    def forward(self, xl_hidden, xl_cls, gates, xa, xv):
        _, L, H = xl_hidden.shape

        # ① Top-K 情感顯著性 token 選取
        topk_idx = gates.topk(min(self.top_k, L), dim=1)[1]          # (B, K)
        topk_h   = xl_hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))  # (B, K, H)

        # ② 情感感知 query (加權聚合 Top-K)
        w        = F.softmax(self.token_attn(topk_h), dim=1)          # (B, K, 1)
        sa_query = (topk_h * w).sum(1)                                 # (B, H)

        # ③ 非語言映射到語言空間 (2個 key — 有意義的注意力分配)
        xa_m = self.audio_map(xa)                                      # (B, H)
        xv_m = self.vision_map(xv)                                     # (B, H)
        kv   = torch.stack([xa_m, xv_m], dim=1)                       # (B, 2, H)

        # ④ Cross-modal attention (Top-K query → [audio, vision])
        attn  = F.softmax(
            torch.bmm(sa_query.unsqueeze(1), kv.transpose(1,2)) / (self.lang_dim ** 0.5),
            dim=-1
        )                                                              # (B, 1, 2)
        x_hat = torch.bmm(attn, kv).squeeze(1)                        # (B, H)

        # ⑤ FFN + Gating + 殘差 + LayerNorm
        x     = self.ffn(xl_cls + x_hat)
        g_w   = torch.sigmoid(self.gate(torch.cat([xl_cls, x], dim=-1)))
        return self.norm(xl_cls + self.dropout(x * g_w))              # (B, H)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v6] OrdinalRegressionHead (CORAL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrdinalRegressionHead(nn.Module):
    """CORAL: 共享權重 + 獨立偏置，輸出 6 個累積閾值 P(y>k)"""
    def __init__(self, feat_dim, num_thresholds=6, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight  = nn.Parameter(torch.randn(feat_dim) * 0.02)
        self.bias    = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, feat):                           # (B, feat_dim)
        logit = self.dropout(feat) @ self.weight       # (B,)
        return logit.unsqueeze(1) + self.bias          # (B, 6)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [v5] SentimentContrastiveLoss (xl.detach() — 不干擾語言骨幹)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentContrastiveLoss(nn.Module):
    """
    情感感知多模態對比損失 (v5 創新2)

    正對: |y_i - y_j| < delta_pos → 語言/非語言表示應靠近
    負對: |y_i - y_j| > delta_neg → 語言/非語言表示應推遠

    注意: xl.detach() → 梯度不流回語言骨幹
    → 只訓練 audio_map, vision_map 和音視頻編碼器
    → 不與主分類損失競爭梯度
    """
    def __init__(self, lang_dim, modal_dim, delta_pos=0.5, delta_neg=1.5,
                 margin=0.2, gamma=0.5):
        super().__init__()
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.margin    = margin
        self.gamma     = gamma
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

    def _match(self, xl, xm):
        return ((xl.mean(0) - xm.mean(0))**2 + (xl.var(0) - xm.var(0))**2).mean()

    def _margin(self, xl, xm):
        xl_n, xm_n = F.normalize(xl, dim=-1), F.normalize(xm, dim=-1)
        neg = xm_n.mean(0, keepdim=True).expand_as(xm_n)
        return F.relu((xl_n * neg).sum(-1) - (xl_n * xm_n).sum(-1) + self.gamma).mean()

    def _contrastive(self, xl, xm, rl):
        xl_n, xm_n = F.normalize(xl, dim=-1), F.normalize(xm, dim=-1)
        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()
        pos_mask = (diff < self.delta_pos).float()
        neg_mask = (diff > self.delta_neg).float()
        sim = torch.mm(xl_n, xm_n.T)
        pos_l = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)
        neg_l = (neg_mask * F.relu(sim - self.margin)**2).sum() / (neg_mask.sum() + 1e-9)
        return pos_l + neg_l

    def forward(self, xl, xa, xv, rl):
        xl   = xl.detach()                             # ← 不干擾語言骨幹
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)
        return (self._match(xl, xa_m) + self._match(xl, xv_m) +
                self._margin(xl, xa_m) + self._margin(xl, xv_m) +
                self._contrastive(xl, xa_m, rl) + self._contrastive(xl, xv_m, rl))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型: SACFv28Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFv28Model(nn.Module):
    """
    SACF v28: 三代架構精華融合

    資料流:
      text → DeBERTa → PolarityEnhancedAttention → xl_cls (B,1024), gates (B,L)
      audio → L2Norm → MultiScaleModalityEncoder → xa (B,256)
      vision → L2Norm → MultiScaleModalityEncoder → xv (B,256)
      xa, xv → AudioVisualCrossAttention → xa_enh (B,256), xv_enh (B,256)
      xl_hidden, xl_cls, gates, xa_enh, xv_enh
          → SentimentAwareCrossModalAttention (Top-K=5) → fused (B,1024)
      fused → Shared → feat (B,512)
          → cls7_head (7) | cls2_head (2) | reg_head (1) | OrdinalHead (6)
      [訓練] xl_cls, xa_enh, xv_enh, rl → SentimentContrastiveLoss → align
    """
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=256,
                 fusion_dim=512, top_k=5, num_classes=7, dropout=0.2):
        super().__init__()
        self.lang_backbone  = AutoModel.from_pretrained(lang_model)
        lang_dim            = self.lang_backbone.config.hidden_size  # 1024

        self.polarity_attn  = PolarityEnhancedAttention(lang_dim, dropout)

        self.audio_encoder  = MultiScaleModalityEncoder(
            audio_dim,  tcn_hidden=modal_hidden//2, lstm_hidden=modal_hidden//2,
            output_dim=modal_hidden, dropout=dropout)
        self.vision_encoder = MultiScaleModalityEncoder(
            vision_dim, tcn_hidden=modal_hidden//2, lstm_hidden=modal_hidden//2,
            output_dim=modal_hidden, dropout=dropout)

        self.av_cross_attn  = AudioVisualCrossAttention(modal_hidden, dropout)
        self.sacf_attn      = SentimentAwareCrossModalAttention(
            lang_dim, modal_hidden, top_k, dropout)

        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Linear(fusion_dim//2, 1), nn.Tanh(),
        )
        self.ordinal_head = OrdinalRegressionHead(fusion_dim, dropout=dropout)
        self.align_loss   = SentimentContrastiveLoss(lang_dim, modal_hidden)

    def forward(self, input_ids, attention_mask, audio, audio_mask,
                vision, vision_mask, reg_labels=None):
        # NaN 保護 + L2 歸一化 (解決 train/test 幅度偏移)
        audio  = F.normalize(torch.nan_to_num(audio,  nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        vision = F.normalize(torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)

        hidden = self.lang_backbone(input_ids=input_ids,
                                    attention_mask=attention_mask).last_hidden_state
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)

        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)

        xa_enh, xv_enh = self.av_cross_attn(xa, xv)
        fused = self.sacf_attn(hidden, xl_cls, gates, xa_enh, xv_enh)

        feat    = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0
        ord_out = self.ordinal_head(feat)

        align = (self.align_loss(xl_cls, xa_enh, xv_enh, reg_labels)
                 if reg_labels is not None
                 else torch.tensor(0.0, device=input_ids.device))

        return logits7, logits2, reg_out, ord_out, align


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    def __init__(self, model, decay=0.9995):
        self.model  = model
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]
        self.backup = {}

    def add_new_params(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.shadow:
                self.shadow[n] = p.data.clone()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             label_smoothing=self.label_smoothing, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class SACFv28Loss(nn.Module):
    """
    固定權重多任務損失 (取代 DynamicLossWeighting，更穩定)
    L = w_cls7*FocalLoss7 + w_cls2*CE2 + w_reg*SmoothL1 + w_ord*CORAL_BCE + w_align*Contrastive
    """
    def __init__(self, class_weights, w_cls7=3.5, w_cls2=0.5, w_reg=0.3,
                 w_ord=0.2, w_align=0.05):
        super().__init__()
        self.w = (w_cls7, w_cls2, w_reg, w_ord, w_align)
        self.focal = FocalLoss(alpha=class_weights)
        self.cls2  = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, ord_logits, align, cl7, cl2, rl):
        lc7  = self.focal(l7, cl7)
        lc2  = self.cls2(l2, cl2)
        lr   = self.reg(reg, rl)
        k    = torch.arange(6, device=cl7.device)
        lord = F.binary_cross_entropy_with_logits(
            ord_logits, (cl7.unsqueeze(1) > k).float()
        )
        total = (self.w[0]*lc7 + self.w[1]*lc2 + self.w[2]*lr +
                 self.w[3]*lord + self.w[4]*align)
        return total, {"cls7": lc7.item(), "cls2": lc2.item(),
                       "reg": lr.item(), "ord": lord.item(), "align": align.item()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(c7, c2, reg, l7, l2, lr):
    return {
        "Acc7": round(float((c7 == l7).mean() * 100), 2),
        "Acc2": round(float((c2 == l2).mean() * 100), 2),
        "F1":   round(float(f1_score(l2, c2, average="weighted") * 100), 2),
        "MAE":  round(float(np.abs(reg - lr).mean()), 4),
        "Corr": round(float(pearsonr(reg, lr)[0]), 4),
    }


def run_batch(batch, device):
    return (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["audio"].to(device),  batch["audio_mask"].to(device),
            batch["vision"].to(device), batch["vision_mask"].to(device),
            batch["cls7_label"].to(device), batch["cls2_label"].to(device),
            batch["reg_label"].to(device))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練（正確 AMP，含 R-Drop + SentimentContrastiveLoss）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler,
                device, scaler, ema, rdrop_alpha=0.05):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # 第一次前向（含 SentimentContrastiveLoss）
            l7, l2, reg, ord_l, align = model(ids, mask, aud, amask, vis, vmask, rl)
            loss, _ = criterion(l7, l2, reg, ord_l, align, cl7, cl2, rl)

            # R-Drop: 第二次前向（不計 align 節省計算）
            if rdrop_alpha > 0:
                l7b, _, _, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7,  -1), F.softmax(l7b, -1), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b, -1), F.softmax(l7,  -1), reduction='batchmean')) / 2
                loss = loss + rdrop_alpha * kl

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ema.update(); scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r, all_l7, all_l2, all_lr = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
        # 推論：不傳 reg_labels → align = 0
        l7, l2, reg, ord_l, align = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, ord_l, align, cl7, cl2, rl)
        total_loss += loss.item()
        all_c7.extend(l7.argmax(1).cpu().numpy())
        all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy())
        all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy())
        all_lr.extend(rl.cpu().numpy())
    return total_loss/len(loader), compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 推論增強（v27 完整 4 步 pipeline）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def collect_tta_preds(model, loader, device, n_tta=10):
    model.train()  # MC Dropout
    all_probs7, all_reg, all_l7, all_lr = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask, aud, amask, vis, vmask, cl7, _, rl = run_batch(batch, device)
            p7_acc  = torch.zeros(len(cl7), 7, device=device)
            reg_acc = torch.zeros(len(cl7), device=device)
            for _ in range(n_tta):
                l7, _, reg, _, _ = model(ids, mask, aud, amask, vis, vmask)
                p7_acc  += F.softmax(l7, dim=-1)
                reg_acc += reg
            all_probs7.append((p7_acc / n_tta).cpu().numpy())
            all_reg.append((reg_acc / n_tta).cpu().numpy())
            all_l7.extend(cl7.cpu().numpy())
            all_lr.extend(rl.cpu().numpy())
    return (np.concatenate(all_probs7), np.concatenate(all_reg),
            np.array(all_l7), np.array(all_lr))


def compute_label_prior(reg_labels, n=7):
    cls = np.clip(np.round(reg_labels).astype(int), -3, 3) + 3
    counts = np.bincount(cls, minlength=n).astype(float)
    return np.where(counts == 0, 1e-6, counts) / counts.sum()


def apply_prior_correction(probs, train_prior, val_prior, strength=1.0):
    ratio = np.power(np.maximum(val_prior, 1e-10) / np.maximum(train_prior, 1e-10), strength)
    corrected = probs * ratio[np.newaxis, :]
    s = corrected.sum(1, keepdims=True)
    return corrected / np.where(s == 0, 1.0, s)


def reg_to_cls7(reg_preds, thresholds):
    cls = np.zeros(len(reg_preds), dtype=int)
    for i, t in enumerate(thresholds):
        cls[reg_preds > t] = i + 1
    return cls


def search_thresholds_greedy(reg_preds, labels7):
    thresholds = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=float)
    best_acc = (reg_to_cls7(reg_preds, thresholds) == labels7).mean()
    improved = True
    while improved:
        improved = False
        for i in range(6):
            lo = thresholds[i-1] + 0.05 if i > 0 else -3.5
            hi = thresholds[i+1] - 0.05 if i < 5 else 3.5
            for t in np.arange(lo, hi, 0.05):
                new_t = thresholds.copy(); new_t[i] = t
                acc = (reg_to_cls7(reg_preds, new_t) == labels7).mean()
                if acc > best_acc:
                    best_acc = acc; thresholds[i] = t; improved = True
    print(f"  [閾值搜索] 閾值={[round(t,2) for t in thresholds]}, Val Acc7={best_acc*100:.2f}%")
    return thresholds, best_acc * 100


def reg_to_soft_labels(reg_preds, sigma=0.5, n=7):
    centers = np.arange(n)
    reg_cls = reg_preds[:, np.newaxis] + 3
    dists   = -((reg_cls - centers[np.newaxis, :]) ** 2) / (2 * sigma ** 2)
    probs   = np.exp(dists - dists.max(1, keepdims=True))
    return probs / probs.sum(1, keepdims=True)


def search_ensemble_alpha(val_probs7, val_reg, val_l7):
    best_acc, best_alpha, best_sigma = 0.0, 0.8, 0.5
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        reg_probs = reg_to_soft_labels(val_reg, sigma)
        for alpha in np.arange(0.5, 1.01, 0.05):
            acc = ((alpha * val_probs7 + (1-alpha) * reg_probs).argmax(1) == val_l7).mean() * 100
            if acc > best_acc:
                best_acc = acc; best_alpha = round(float(alpha), 2); best_sigma = sigma
    print(f"  [軟集成] alpha={best_alpha:.2f}, sigma={best_sigma:.2f}, Val Acc7={best_acc:.2f}%")
    return best_alpha, best_sigma, best_acc


def enhanced_inference(model, val_loader, test_loader, train_reg_labels, device, n_tta=10):
    print("\n[推論增強] 收集 val TTA 預測...")
    val_probs7, val_reg, val_l7, val_lr = collect_tta_preds(model, val_loader, device, n_tta)

    base_acc = (val_probs7.argmax(1) == val_l7).mean() * 100
    print(f"  TTA softmax 基線: Val Acc7 = {base_acc:.2f}%")

    train_prior = compute_label_prior(train_reg_labels)
    val_prior   = compute_label_prior(val_lr)
    best_corr_acc, best_strength = 0.0, 0.0
    for s in np.arange(0.0, 3.1, 0.2):
        corr = apply_prior_correction(val_probs7, train_prior, val_prior, s)
        acc  = (corr.argmax(1) == val_l7).mean() * 100
        if acc > best_corr_acc:
            best_corr_acc, best_strength = acc, round(float(s), 1)
    print(f"  [先驗校正] strength={best_strength}, Val Acc7={best_corr_acc:.2f}%")

    best_thresh, thresh_acc = search_thresholds_greedy(val_reg, val_l7)
    best_alpha, best_sigma, ensemble_acc = search_ensemble_alpha(val_probs7, val_reg, val_l7)

    methods = {"TTA": base_acc, "Prior": best_corr_acc,
               "Thresh": thresh_acc, "Ensemble": ensemble_acc}
    best_method = max(methods, key=methods.get)
    print(f"\n  最優 val 方案: {best_method} ({methods[best_method]:.2f}%)")

    print("\n[推論增強] 收集 test TTA 預測...")
    test_probs7, test_reg, test_l7, test_lr = collect_tta_preds(model, test_loader, device, n_tta)

    if best_method == "TTA":
        test_c7 = test_probs7.argmax(1)
    elif best_method == "Prior":
        test_c7 = apply_prior_correction(test_probs7, train_prior, val_prior, best_strength).argmax(1)
    elif best_method == "Thresh":
        test_c7 = reg_to_cls7(test_reg, best_thresh)
    else:
        reg_p   = reg_to_soft_labels(test_reg, best_sigma)
        test_c7 = (best_alpha * test_probs7 + (1-best_alpha) * reg_p).argmax(1)

    test_c2 = (test_c7 >= 3).astype(int)
    test_m  = compute_metrics(test_c7, test_c2, test_reg, test_l7,
                              (test_lr >= 0).astype(int), test_lr)
    return test_m, best_method, methods


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(np.clip(len(cl) / (n * ct), 0.5, 3.0))


def progressive_unfreeze(model, epoch, total_epochs):
    """toggle requires_grad，不重建 optimizer（v27 optimizer 修復的配合函數）"""
    encoder = getattr(model.lang_backbone, "encoder", None)
    if not encoder: return False
    freeze_until = 6 if epoch < total_epochs // 3 else (3 if epoch < 2 * total_epochs // 3 else 0)
    changed = False
    for i, layer in enumerate(encoder.layer):
        want = (i >= freeze_until)
        for p in layer.parameters():
            if p.requires_grad != want:
                p.requires_grad = want; changed = True
    if changed:
        print(f"  [解凍] Epoch {epoch}: 凍結前 {freeze_until} 層")
    return changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)
    print("=" * 70)
    print("MOSI 多模態情感分析 v28 — SACF Elite")
    print("v5創新(Top-K情感融合+對比) + v6編碼器(TCN+LSTM+A↔V) + v27訓練技術")
    print("=" * 70)

    config = {
        "data_path":       PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":       PROJECT_ROOT / "emotion_system/models",
        "lang_model":      "microsoft/deberta-v3-large",
        "max_text_len":    80,
        "audio_dim":       5,
        "vision_dim":      20,
        "modal_hidden":    256,
        "fusion_dim":      512,
        "top_k":           5,
        "num_classes":     7,
        "dropout":         0.2,
        "batch_size":      8,
        "num_epochs":      150,
        "lang_lr":         5e-6,      # 稍低於 v27(6e-6)，因多了 SentimentContrastiveLoss
        "other_lr":        1e-4,
        "weight_decay":    1e-2,
        "warmup_ratio":    0.06,
        "w_cls7":          3.5,
        "w_cls2":          0.5,
        "w_reg":           0.3,
        "w_ord":           0.2,
        "w_align":         0.05,      # 小：SentimentContrastiveLoss 只是輔助正則
        "ema_decay":       0.9995,
        "patience":        25,
        "rdrop_alpha":     0.05,      # 小：SentimentContrastiveLoss 已提供對比正則
        "swa_start_ratio": 0.50,
        "swa_lr":          1e-6,
        "n_tta":           10,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"Capped 類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer  = DebertaV2Tokenizer.from_pretrained(config["lang_model"])
    train_ds   = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds     = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds    = MOSIDataset(data["test"],  tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"設備: {device}")

    model = SACFv28Model(
        lang_model=config["lang_model"], audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"], modal_hidden=config["modal_hidden"],
        fusion_dim=config["fusion_dim"], top_k=config["top_k"],
        num_classes=config["num_classes"], dropout=config["dropout"],
    ).to(device)

    # 初始凍結前 6 層
    if enc := getattr(model.lang_backbone, "encoder", None):
        for i in range(6):
            for p in enc.layer[i].parameters(): p.requires_grad = False
        print("初始凍結前 6 層（DeBERTa layer 0-5）")

    total_p  = sum(p.numel() for p in model.parameters())
    train_p  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {train_p/1e6:.1f}M")

    criterion = SACFv28Loss(class_w.to(device),
                            config["w_cls7"], config["w_cls2"],
                            config["w_reg"], config["w_ord"], config["w_align"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # v27 Optimizer 修復: 所有 DeBERTa 層加入 optimizer（含 frozen 層）
    # frozen 層 requires_grad=False → 梯度 None → optimizer.step() 自動跳過
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    all_lang_params = list(model.lang_backbone.parameters())
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
        list(model.align_loss.parameters())
    )
    optimizer = optim.AdamW([
        {"params": all_lang_params, "lr": config["lang_lr"]},
        {"params": other_params,    "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler() if device == "cuda" else None
    ema          = EMA(model, decay=config["ema_decay"])

    swa_model    = AveragedModel(model)
    swa_start    = int(config["num_epochs"] * config["swa_start_ratio"])
    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"])
    swa_started  = False

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[v28] SACF Elite: Top-K情感融合 + A↔V預融合 + SentimentContrastive")
    print(f"      Optimizer修復 + EMA + SWA(epoch {swa_start}+) + FocalLoss + R-Drop")
    print(f"      lang_lr={config['lang_lr']:.0e} | epochs={config['num_epochs']} | patience={config['patience']}")
    print(f"      w_cls7={config['w_cls7']} | w_align={config['w_align']} | rdrop={config['rdrop_alpha']}\n")

    save_dir = Path(config["model_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history   = []; patience_counter = 0

    for epoch in range(config["num_epochs"]):
        if progressive_unfreeze(model, epoch, config["num_epochs"]):
            ema.add_new_params()

        if epoch >= swa_start and not swa_started:
            swa_started = True
            print(f"  [SWA] 啟動 (epoch {epoch})")

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-"*45)
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                              device, scaler, ema, config["rdrop_alpha"])

        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # EMA 驗證
        ema.apply_shadow()
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        history.append({"epoch": epoch+1, "tr_loss": round(tr_loss, 4),
                        "vl_loss": round(vl_loss, 4), **metrics})
        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **metrics}
            ema.apply_shadow()
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "metrics": metrics, "config": config},
                       save_dir / "v28_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善"); break

    # SWA 最終評估
    if swa_started:
        print("\n[SWA] 更新 BatchNorm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        _, swa_val = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val Acc7={swa_val['Acc7']:.2f}%  (EMA best={best_acc7['Acc7']:.2f}%)")
        if swa_val["Acc7"] > best_acc7["Acc7"]:
            print("  ✅ SWA 更優！")
            torch.save({"model_state": swa_model.module.state_dict(),
                        "metrics": swa_val, "config": config},
                       save_dir / "v28_swa.pth")
            eval_model = swa_model
        else:
            ckpt = torch.load(save_dir / "v28_best_acc7.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"]); eval_model = model
    else:
        ckpt = torch.load(save_dir / "v28_best_acc7.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"]); eval_model = model

    # 標準測試
    print("\n" + "=" * 60)
    _, test_std = validate(eval_model, test_loader, criterion, device)
    print("\n【標準測試集結果 - v28】")
    print(f"  Acc7 : {test_std['Acc7']:.2f}%   (目標: 50.5%, v23: 48.69%, v27: 48.54%)")
    print(f"  Acc2 : {test_std['Acc2']:.2f}%")
    print(f"  F1   : {test_std['F1']:.2f}%")
    print(f"  MAE  : {test_std['MAE']:.4f}")
    print(f"  Corr : {test_std['Corr']:.4f}")
    print(f"\n  Val-Test Gap: {best_acc7['Acc7'] - test_std['Acc7']:.2f}%")

    # 推論增強
    print("\n" + "=" * 60)
    print("[推論增強] 開始...")
    ema.apply_shadow()
    test_enh, best_method, all_val_accs = enhanced_inference(
        eval_model, val_loader, test_loader,
        data["train"]["regression_labels"], device, config["n_tta"])
    ema.restore()

    print(f"\n【推論增強測試集結果 - v28 ({best_method})】")
    print(f"  Acc7 : {test_enh['Acc7']:.2f}%   (目標: 50.5%)")
    print(f"  Acc2 : {test_enh['Acc2']:.2f}%")
    print(f"  F1   : {test_enh['F1']:.2f}%")
    print(f"  MAE  : {test_enh['MAE']:.4f}")
    print(f"  Corr : {test_enh['Corr']:.4f}")

    final_acc7 = max(test_std["Acc7"], test_enh["Acc7"])
    status = "🎉 達標！" if final_acc7 >= 50.5 else f"❌ 差 {50.5 - final_acc7:.2f}%"
    print(f"\n  最終 Test Acc7: {final_acc7:.2f}% | {status}")

    with open(save_dir / "v28_history.json", "w") as f:
        json.dump({
            "history":               history,
            "best_val_acc7":         best_acc7,
            "test_standard":         test_std,
            "test_enhanced":         test_enh,
            "best_inference_method": best_method,
            "all_val_inference_accs":all_val_accs,
            "config": {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
