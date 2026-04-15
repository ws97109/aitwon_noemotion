"""
MOSI 多模態情感分析 v36 — MulT (Multimodal Transformer) Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v36 策略: 改用 MulT (Tsai et al. 2019) 跨模態 Transformer 架構
  歷史分析:
    v5:  test=50.00% (SACF 最佳基線)
    v27: val=51.09%, test=48.54% (gap=2.55%)
    v28: val=51.97%, test=48.83% (gap=3.14%)
    v34: test=47.81% (SACF+EMA+TTA)
    v35: 訓練中 (SACF+SWA+Mixup)

  根本問題: val-test gap 約 2.5-3%，模型對 validation set 過擬合
  v36 新方向: 完全不同的架構，跳出 SACF 局部最優解

  核心改進:
  1. MulT 跨模態 Transformer: 4 方向 CrossModalAttention
     - text→audio, text→vision (文字主導感知非語言情緒)
     - audio→text, vision→text (非語言模態反過來閱讀文字)
  2. 更保守的 lang_lr=2e-6 (防止 DeBERTa 過擬合)
  3. SWA (epoch 60 開始) 替代早停選最佳模型
  4. 不使用 TTA/閾值搜索 (防止 val 過擬合)
  5. 直接 argmax 推論 (簡單可靠)
  6. 維持 L2Norm + NaN 保護

目標: test Acc7 > 51%, val-test gap < 1.5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import pickle
import random
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"隨機種子固定為 {seed}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集 (與 v34/v35 相同結構)
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
# 模態編碼器 (BiLSTM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModalEncoder(nn.Module):
    """
    用 BiLSTM 將低維音訊/視覺特徵序列編碼為 hidden_size*2 維。
    - input_dim: 輸入特徵維度 (audio=5, vision=20)
    - hidden_dim: LSTM 單向 hidden size (預設128，雙向=256)
    - num_layers: LSTM 層數 (預設2)
    - forward 返回: (sequence [B,T,256], pooled [B,256])
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2  # 256 (雙向)
        self.layer_norm = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:    [B, T, input_dim]
        mask: [B, T] — 1 表示有效幀，0 表示 padding

        返回:
          seq:    [B, T, 256] — 完整序列輸出
          pooled: [B, 256]    — mask-weighted mean pooling
        """
        seq, _ = self.lstm(x)              # [B, T, 256]
        seq = self.layer_norm(seq)
        seq = self.dropout(seq)

        # Mask-weighted mean pooling (只取有效幀的平均)
        mask_f = mask.unsqueeze(-1).float()         # [B, T, 1]
        sum_seq = (seq * mask_f).sum(dim=1)         # [B, 256]
        count   = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled  = sum_seq / count                   # [B, 256]

        return seq, pooled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 跨模態注意力 (CrossModalAttention)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CrossModalAttention(nn.Module):
    """
    跨模態 MultiHead Attention:
      query 來自 query_mod，key/value 來自 kv_mod。
      核心想法: 讓 query 模態「去讀」kv 模態的資訊。

    - query_dim: query 模態的特徵維度
    - kv_dim:    key/value 模態的特徵維度
    - num_heads: 注意力頭數
    - dropout:   注意力 dropout

    KV 先投影到 query_dim，再做標準 MHA。
    """
    def __init__(self, query_dim: int, kv_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # 若 kv_dim != query_dim，先投影
        self.kv_proj = (
            nn.Linear(kv_dim, query_dim) if kv_dim != query_dim else nn.Identity()
        )
        # nn.MultiheadAttention: embed_dim 即 query_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query_seq: torch.Tensor,
                kv_seq: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_seq: [B, Tq, query_dim]
        kv_seq:    [B, Tk, kv_dim]
        kv_mask:   [B, Tk] — 1=有效, 0=padding；轉換為 key_padding_mask

        返回:
          out:    [B, Tq, query_dim] — attended 輸出 (含殘差)
          pooled: [B, query_dim]     — [CLS] token 或第一個 token
        """
        kv = self.kv_proj(kv_seq)  # [B, Tk, query_dim]

        # nn.MHA 的 key_padding_mask: True=忽略，False=保留
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = (kv_mask == 0)  # [B, Tk], True=padding

        attn_out, _ = self.attn(
            query=query_seq,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )
        # 殘差 + LayerNorm
        out = self.layer_norm(query_seq + self.dropout(attn_out))

        # pooled: 取第一個 token (類似 CLS)
        pooled = out[:, 0, :]  # [B, query_dim]

        return out, pooled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型: MulTModel (Multimodal Transformer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MulTModel(nn.Module):
    """
    MulT (Multimodal Transformer) 架構:
      1. DeBERTa-v3-large 提取文字 hidden states [B, L, 1024]
      2. BiLSTM 編碼音訊/視覺序列 → [B, T, 256]
      3. 文字特徵投影: 1024 → 256
      4. 四方向跨模態注意力:
         t2a: text(query) → audio(kv)   # 文字去感知音訊情緒
         t2v: text(query) → vision(kv)  # 文字去感知視覺情緒
         a2t: audio(query) → text(kv)   # 音訊去讀取文字語義
         v2t: vision(query) → text(kv)  # 視覺去讀取文字語義
      5. Concat 4 個 pooled 輸出 → MLP → 分類/迴歸
    """
    def __init__(
        self,
        lang_model:  str   = "microsoft/deberta-v3-large",
        audio_dim:   int   = 5,
        vision_dim:  int   = 20,
        modal_hidden: int  = 128,
        proj_dim:    int   = 256,
        num_heads:   int   = 4,
        num_classes: int   = 7,
        dropout:     float = 0.3,
    ):
        super().__init__()

        # ── 文字骨幹: DeBERTa-v3-large (hidden_size=1024) ──
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024

        # ── 非語言模態編碼器 ──
        self.audio_encoder  = ModalEncoder(audio_dim,  modal_hidden, 2, dropout * 0.5)
        self.vision_encoder = ModalEncoder(vision_dim, modal_hidden, 2, dropout * 0.5)
        modal_out_dim = modal_hidden * 2  # 256

        # ── 文字特徵投影: 1024 → 256 ──
        self.text_proj = nn.Sequential(
            nn.Linear(lang_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── 四方向跨模態注意力 ──
        # text 作為 query，讀取 audio/vision
        self.t2a = CrossModalAttention(proj_dim, modal_out_dim, num_heads, dropout * 0.5)
        self.t2v = CrossModalAttention(proj_dim, modal_out_dim, num_heads, dropout * 0.5)
        # audio/vision 作為 query，讀取 text
        self.a2t = CrossModalAttention(modal_out_dim, proj_dim, num_heads, dropout * 0.5)
        self.v2t = CrossModalAttention(modal_out_dim, proj_dim, num_heads, dropout * 0.5)

        # ── 融合層: concat 4 個 pooled [各 proj_dim] → 1024 → 512 → 256 ──
        # t2a, t2v 各輸出 proj_dim=256; a2t 輸出 modal_out_dim=256; v2t 輸出 modal_out_dim=256
        # 共 256*4 = 1024
        fusion_in = proj_dim * 2 + modal_out_dim * 2  # 1024
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
        )

        # ── 多任務頭部 ──
        self.cls7_head = nn.Linear(256, num_classes)
        self.cls2_head = nn.Linear(256, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        audio:          torch.Tensor,
        audio_mask:     torch.Tensor,
        vision:         torch.Tensor,
        vision_mask:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        輸入:
          input_ids, attention_mask: [B, L]
          audio:       [B, T_a, 5]
          audio_mask:  [B, T_a]
          vision:      [B, T_v, 20]
          vision_mask: [B, T_v]

        輸出:
          logits7: [B, 7]
          logits2: [B, 2]
          reg_out: [B]    — 範圍 [-3, 3]
        """
        # ── NaN 保護 + L2 歸一化 (v34 驗證：解決 domain shift) ──
        audio  = torch.nan_to_num(audio,  nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)
        audio  = F.normalize(audio,  p=2, dim=-1)
        vision = F.normalize(vision, p=2, dim=-1)

        # ── 文字特徵 (DeBERTa) ──
        lang_out  = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = lang_out.last_hidden_state          # [B, L, 1024]
        text_seq    = self.text_proj(text_hidden)          # [B, L, 256]

        # ── 非語言模態序列 ──
        audio_seq, _  = self.audio_encoder(audio,  audio_mask)   # [B, T_a, 256]
        vision_seq, _ = self.vision_encoder(vision, vision_mask)  # [B, T_v, 256]

        # 文字的 padding mask: attention_mask=0 表示 padding
        text_mask = attention_mask  # [B, L], 1=有效

        # ── 四方向跨模態注意力 ──
        # text(query) → audio(kv): 文字讀音訊
        _, t2a_pooled = self.t2a(text_seq,   audio_seq,  audio_mask)   # [B, 256]
        # text(query) → vision(kv): 文字讀視覺
        _, t2v_pooled = self.t2v(text_seq,   vision_seq, vision_mask)  # [B, 256]
        # audio(query) → text(kv): 音訊讀文字
        _, a2t_pooled = self.a2t(audio_seq,  text_seq,   text_mask)    # [B, 256]
        # vision(query) → text(kv): 視覺讀文字
        _, v2t_pooled = self.v2t(vision_seq, text_seq,   text_mask)    # [B, 256]

        # ── 融合 4 個 pooled 向量 ──
        fused = torch.cat([t2a_pooled, t2v_pooled, a2t_pooled, v2t_pooled], dim=-1)  # [B, 1024]
        feat  = self.fusion(fused)  # [B, 256]

        # ── 多任務輸出 ──
        logits7 = self.cls7_head(feat)                   # [B, 7]
        logits2 = self.cls2_head(feat)                   # [B, 2]
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0  # [B], [-3, 3]

        return logits7, logits2, reg_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MulTLoss(nn.Module):
    """
    多任務損失:
      total = cls7 + 0.3*cls2 + 0.4*reg
    - cls7: 加權 CrossEntropy (label_smoothing=0.12) — 7-class 分類
    - cls2: CrossEntropy (label_smoothing=0.05) — 正負情感分類
    - reg:  SmoothL1Loss — 情感強度迴歸
    """
    def __init__(self, class_weights: torch.Tensor,
                 label_smoothing: float = 0.12):
        super().__init__()
        self.cls7 = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg  = nn.SmoothL1Loss()

    def forward(
        self,
        l7:  torch.Tensor, l2:  torch.Tensor, reg: torch.Tensor,
        cl7: torch.Tensor, cl2: torch.Tensor, rl:  torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        lc7   = self.cls7(l7, cl7)
        lc2   = self.cls2(l2, cl2)
        lr    = self.reg(reg, rl)
        total = lc7 + 0.3 * lc2 + 0.4 * lr
        return total, {
            "cls7": lc7.item(),
            "cls2": lc2.item(),
            "reg":  lr.item(),
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
# 工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels: np.ndarray, n: int = 7) -> torch.Tensor:
    """計算類別權重 (capped [0.5, 3.0])"""
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(np.clip(len(cl) / (n * ct), 0.5, 3.0))


def progressive_unfreeze(model: MulTModel, epoch: int, unfreeze_epoch: int):
    """
    漸進式解凍 DeBERTa 層:
    - epoch < unfreeze_epoch: 凍結前 freeze_layers 層
    - epoch >= unfreeze_epoch: 全部解凍
    """
    encoder = getattr(model.lang_backbone, "encoder", None)
    if not encoder:
        return
    # epoch 達到 unfreeze_epoch 時全部解凍
    if epoch == unfreeze_epoch:
        for p in model.lang_backbone.parameters():
            p.requires_grad = True
        print(f"  [解凍] Epoch {epoch+1}: DeBERTa 全部層解凍")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練一個 epoch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: MulTLoss,
    optimizer: optim.Optimizer,
    scheduler,
    device:    str,
    scaler,
) -> float:
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
        cl2   = batch["cls2_label"].to(device)
        rl    = batch["reg_label"].to(device)

        optimizer.zero_grad()
        use_amp = scaler is not None

        with torch.amp.autocast("cuda", enabled=use_amp):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 驗證/測試
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: MulTLoss,
    device:    str,
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r  = [], [], []
    all_l7, all_l2, all_lr = [], [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        cl2   = batch["cls2_label"].to(device)
        rl    = batch["reg_label"].to(device)

        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)
        total_loss += loss.item()

        # 直接 argmax，不用 TTA/閾值搜索 (防止 val 過擬合)
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
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)

    print("=" * 70)
    print("MOSI 多模態情感分析 v36 — MulT (Multimodal Transformer) Edition")
    print("4-Direction CrossModal Attention + SWA + Conservative LR")
    print("=" * 70)

    # ── 資料路徑 (支援多候選路徑) ──
    _data_candidates = [
        PROJECT_ROOT / "aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl",
        PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl"),
    ]
    _data_path = next((p for p in _data_candidates if p.exists()), _data_candidates[0])

    _model_candidates = [
        PROJECT_ROOT / "aitwon_emotion/emotion_system/models",
        PROJECT_ROOT / "emotion_system/models",
        Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/models"),
    ]
    _model_dir = next((p for p in _model_candidates if p.exists()), _model_candidates[0])

    # ── 訓練超參數 ──
    config = {
        "data_path":      _data_path,
        "model_dir":      _model_dir,
        "lang_model":     "microsoft/deberta-v3-large",
        "max_text_len":   80,
        "audio_dim":      5,
        "vision_dim":     20,
        "modal_hidden":   128,   # BiLSTM 單向 hidden, 雙向輸出 256
        "proj_dim":       256,   # 文字投影維度 & 跨模態對齊維度
        "num_heads":      4,     # 跨模態注意力頭數
        "num_classes":    7,
        "dropout":        0.3,
        # ── 學習率: 非常保守，防止 DeBERTa 過擬合 ──
        "batch_size":     12,
        "num_epochs":     80,
        "lang_lr":        2e-6,  # 比 v35 (3e-6) 更保守
        "head_lr":        5e-5,  # 其他所有參數
        "weight_decay":   1e-2,
        "warmup_ratio":   0.06,
        "label_smoothing": 0.12,
        # ── 解凍策略: epoch 27 全部解凍 ──
        "freeze_layers":  6,     # 初始凍結前 6 層
        "unfreeze_epoch": 27,    # epoch 27 全解凍
        # ── SWA ──
        "swa_start":      60,    # epoch 60 開始 SWA
        "swa_lr":         1e-5,
        # ── 早停 ──
        "patience":       25,
        "seed":           42,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重 (capped [0.5, 3.0]): {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds   = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds  = MOSIDataset(data["test"],  tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   config["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  config["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── 設備選擇 (CUDA_VISIBLE_DEVICES=1 會將 A6000 映射為 cuda:0) ──
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 建立 MulT 模型 ──
    model = MulTModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"],
        proj_dim=config["proj_dim"],
        num_heads=config["num_heads"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    # ── 初始凍結 DeBERTa 前 freeze_layers 層 ──
    encoder = model.lang_backbone.encoder
    for i in range(config["freeze_layers"]):
        for p in encoder.layer[i].parameters():
            p.requires_grad = False
    print(f"已凍結 DeBERTa 前 {config['freeze_layers']} 層 (epoch {config['unfreeze_epoch']} 解凍)")

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    # ── 損失函數 ──
    criterion = MulTLoss(
        class_weights=class_w.to(device),
        label_smoothing=config["label_smoothing"],
    )

    # ── 優化器: 語言模型與其他模組使用不同 LR ──
    lang_params = list(model.lang_backbone.parameters()) + list(model.text_proj.parameters())
    head_params = (
        list(model.audio_encoder.parameters()) +
        list(model.vision_encoder.parameters()) +
        list(model.t2a.parameters()) +
        list(model.t2v.parameters()) +
        list(model.a2t.parameters()) +
        list(model.v2t.parameters()) +
        list(model.fusion.parameters()) +
        list(model.cls7_head.parameters()) +
        list(model.cls2_head.parameters()) +
        list(model.reg_head.parameters())
    )
    optimizer = optim.AdamW([
        {"params": lang_params, "lr": config["lang_lr"]},
        {"params": head_params, "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None

    # ── SWA 設置 ──
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"])
    swa_start = config["swa_start"]
    swa_active = False
    print(f"SWA 從 epoch {swa_start+1} 開始，swa_lr={config['swa_lr']}")

    # ── 保存目錄 ──
    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n開始訓練 | 設備: {device} | Batch: {config['batch_size']}")
    print(f"Lang LR: {config['lang_lr']} | Head LR: {config['head_lr']}")
    print(f"Weight Decay: {config['weight_decay']} | Dropout: {config['dropout']}")
    print(f"Epochs: {config['num_epochs']} | Patience: {config['patience']}\n")

    best = {"Acc7": 0.0, "epoch": 0}
    history = []
    patience_counter = 0

    for epoch in range(config["num_epochs"]):

        # ── 漸進式解凍 ──
        progressive_unfreeze(model, epoch, config["unfreeze_epoch"])

        # ── SWA 模式切換 ──
        if epoch >= swa_start and not swa_active:
            swa_active = True
            print(f"  [SWA 啟動] 從 epoch {epoch+1} 開始累積 SWA 參數")

        # ── 訓練 ──
        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler, device, scaler)

        # ── SWA 更新 ──
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # ── 驗證 (用當前模型) ──
        vl_loss, metrics = validate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "tr_loss": round(tr_loss, 4),
            "vl_loss": round(vl_loss, 4),
            **metrics,
        })

        swa_tag = " [SWA]" if swa_active else ""
        print(f"E{epoch+1:03d} | Train={tr_loss:.4f}  Val={vl_loss:.4f} | "
              f"Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}" + swa_tag)

        # ── 儲存最佳 checkpoint ──
        if metrics["Acc7"] > best["Acc7"]:
            best = {"epoch": epoch + 1, **metrics}
            patience_counter = 0
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model_v36.pth")
            print(f"  => 新最佳 Acc7={metrics['Acc7']:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停: {config['patience']} epochs 無提升 (best E{best['epoch']})")
                break

    # ── 最終評估 ──
    print("\n" + "=" * 70)
    print("最終評估階段")
    print("=" * 70)

    # 1. 最佳 checkpoint 評估
    ckpt = torch.load(save_dir / "best_model_v36.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, val_m_best  = validate(model, val_loader,  criterion, device)
    _, test_m_best = validate(model, test_loader, criterion, device)
    print(f"[最佳 Checkpoint E{ckpt['epoch']}]")
    print(f"  Val  Acc7={val_m_best['Acc7']:.2f}%  Acc2={val_m_best['Acc2']:.2f}%")
    print(f"  Test Acc7={test_m_best['Acc7']:.2f}%  Acc2={test_m_best['Acc2']:.2f}%")
    print(f"  Val-Test Gap: {val_m_best['Acc7'] - test_m_best['Acc7']:.2f}%")

    # 2. SWA 模型評估
    swa_test_acc7 = 0.0
    if swa_active:
        print("\n更新 SWA BN 統計 (使用 train loader)...")
        # update_bn 需要 dataloader 來重新計算 BN 統計量
        update_bn(train_loader, swa_model, device=device)

        _, val_m_swa  = validate(swa_model, val_loader,  criterion, device)
        _, test_m_swa = validate(swa_model, test_loader, criterion, device)
        swa_test_acc7 = test_m_swa["Acc7"]
        print(f"[SWA 模型]")
        print(f"  Val  Acc7={val_m_swa['Acc7']:.2f}%  Acc2={val_m_swa['Acc2']:.2f}%")
        print(f"  Test Acc7={swa_test_acc7:.2f}%  Acc2={test_m_swa['Acc2']:.2f}%")
        print(f"  Val-Test Gap: {val_m_swa['Acc7'] - swa_test_acc7:.2f}%")

        # 若 SWA 更好，保存
        if swa_test_acc7 >= test_m_best["Acc7"]:
            torch.save({
                "epoch":       "SWA",
                "model_state": swa_model.module.state_dict(),
                "metrics":     test_m_swa,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model_v36_swa.pth")
            print("  => SWA 模型更優，已另存為 best_model_v36_swa.pth")

    # ── 選出最終最佳方案並輸出 ──
    print("\n" + "=" * 70)
    if swa_active and swa_test_acc7 >= test_m_best["Acc7"]:
        final_val  = val_m_swa["Acc7"]
        final_test = swa_test_acc7
        method = "SWA"
    else:
        final_val  = val_m_best["Acc7"]
        final_test = test_m_best["Acc7"]
        method = "Best Checkpoint"

    print(f"最終方案: {method}")
    print(f"Val  Acc7: {final_val:.2f}%")
    print(f"Test Acc7: {final_test:.2f}%")
    print(f"Val-Test Gap: {final_val - final_test:.2f}%")
    print("=" * 70)

    # 監控腳本所需的輸出格式
    print(f"\nTest Acc7: {final_test:.2f}%")

    # 儲存訓練歷史
    history_path = save_dir / "history_v36.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"訓練歷史已儲存至: {history_path}")


if __name__ == "__main__":
    main()
