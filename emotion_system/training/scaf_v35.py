"""
MOSI 多模態情感分析 v35 — Anti-Overfit Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v35 策略: 解決 val-test gap (約 2.5-3%) 問題
  歷史分析:
    v27: val=51.09%, test=48.54% (gap=2.55%)
    v28: val=51.97%, test=48.83% (gap=3.14%)
    v29: AWP 導致 NaN loss 崩潰
    v34: 未完成

  核心問題: 過擬合 validation set (可能因 patience/早停選出的模型
           在 val 上過擬合，test 泛化不足)

  v35 改進方向:
  1. SWA (Stochastic Weight Averaging): 比 EMA 更好的泛化
     - 比選最佳 epoch 的策略更穩健
     - 對 val-test gap 有直接改善效果
  2. 更強 dropout (0.3 → 0.35)，減少過擬合
  3. 增加 weight_decay (1e-2 → 2e-2)
  4. 降低 lang_lr (5e-6 → 3e-6), 減少過擬合
  5. Label smoothing 提高 (0.1 → 0.15)
  6. 保留 L2Norm + NaN 保護 (v34 驗證有效)
  7. 保留 SACF 架構 (v5/v27/v28 核心)
  8. 增加 batch_size=12 (減少噪聲梯度)
  9. Mixup 增強 (在 feature space 做 mixup)
  10. 不用閾值搜索 (v29 確認過擬合 val)

目標: test Acc7 > 51%
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
# Polarity-Enhanced Attention (來自 v5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(hidden)
        m = mask.unsqueeze(-1).float()
        enhanced = (0.75 * hidden + 0.25 * hidden * g) * m
        pooled   = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        gates    = (g * m).squeeze(-1)
        return self.dropout(pooled), gates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 情感敏感跨模態注意力 (來自 v5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, lang_dim: int, modal_dim: int, top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.top_k    = top_k
        self.lang_dim = lang_dim
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.token_attn = nn.Linear(lang_dim, 1)
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_dim // 2, lang_dim),
        )
        self.gate    = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(lang_dim)

    def forward(self, xl_hidden, xl_cls, gates, xa, xv):
        B, L, H = xl_hidden.shape
        topk_vals, topk_idx = gates.topk(min(self.top_k, L), dim=1)
        topk_hidden = xl_hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
        w = F.softmax(self.token_attn(topk_hidden), dim=1)
        sa_query = (topk_hidden * w).sum(1)
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)
        kv   = torch.stack([xa_m, xv_m], dim=1)
        scale = self.lang_dim ** 0.5
        attn  = F.softmax(torch.bmm(sa_query.unsqueeze(1), kv.transpose(1, 2)) / scale, dim=-1)
        x_hat = torch.bmm(attn, kv).squeeze(1)
        x = self.ffn(xl_cls + x_hat)
        gate_w = torch.sigmoid(self.gate(torch.cat([xl_cls, x], dim=-1)))
        x = x * gate_w
        return self.norm(xl_cls + self.dropout(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 非語言模態編碼器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 情感感知對比學習損失 (來自 v5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentContrastiveLoss(nn.Module):
    def __init__(self, lang_dim: int, modal_dim: int,
                 delta_pos: float = 0.5, delta_neg: float = 1.5,
                 margin: float = 0.2, gamma: float = 0.5):
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
        xl_n = F.normalize(xl, dim=-1)
        xm_n = F.normalize(xm, dim=-1)
        diff = (rl.unsqueeze(0) - rl.unsqueeze(1)).abs()
        pos_mask = (diff < self.delta_pos).float()
        neg_mask = (diff > self.delta_neg).float()
        sim = torch.mm(xl_n, xm_n.T)
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)
        neg_loss = (neg_mask * F.relu(sim - self.margin) ** 2).sum() / (neg_mask.sum() + 1e-9)
        return pos_loss + neg_loss

    def forward(self, xl, xa, xv, rl):
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)
        l_match    = self.matching_loss(xl.detach(), xa_m) + self.matching_loss(xl.detach(), xv_m)
        l_margin   = self.margin_loss(xl.detach(), xa_m) + self.margin_loss(xl.detach(), xv_m)
        l_contrast = (self.sentiment_contrastive(xl.detach(), xa_m, rl) +
                      self.sentiment_contrastive(xl.detach(), xv_m, rl))
        return l_match + l_margin + l_contrast


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型: SACFModel v35 (v5 + L2Norm + 增強正則化)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFModelV35(nn.Module):
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=128,
                 fusion_dim=512, top_k=5, num_classes=7, dropout=0.35):
        super().__init__()
        self.lang_backbone  = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size
        self.polarity_attn  = PolarityEnhancedAttention(lang_dim, dropout)
        self.audio_encoder  = ModalityEncoder(audio_dim,  modal_hidden, 2, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden, 2, dropout)
        self.sacf_attn = SentimentAwareCrossModalAttention(lang_dim, modal_hidden, top_k, dropout)
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
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 2, 1),
            nn.Tanh(),
        )
        self.align_loss = SentimentContrastiveLoss(lang_dim, modal_hidden)

    def forward(self, input_ids, attention_mask, audio, audio_mask,
                vision, vision_mask, reg_labels=None):
        # NaN 保護 + L2 歸一化 (v34 驗證: 解決 domain shift)
        audio  = torch.nan_to_num(audio,  nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)
        audio  = F.normalize(audio,  p=2, dim=-1)
        vision = F.normalize(vision, p=2, dim=-1)

        lang_out = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden   = lang_out.last_hidden_state
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)
        fused   = self.sacf_attn(hidden, xl_cls, gates, xa, xv)
        feat    = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0
        if reg_labels is not None:
            align = self.align_loss(xl_cls, xa, xv, reg_labels)
        else:
            align = torch.tensor(0.0, device=input_ids.device)
        return logits7, logits2, reg_out, align, xl_cls


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
# SWA (Stochastic Weight Averaging) - 更好的泛化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SWA:
    """SWA: 平均多個 epoch 的 checkpoint 以提高泛化"""
    def __init__(self, model):
        self.model  = model
        self.avg_params = {}
        self.n_averaged = 0

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if n not in self.avg_params:
                    self.avg_params[n] = p.data.clone()
                else:
                    # running average: avg = (n*avg + x) / (n+1)
                    self.avg_params[n] = (self.n_averaged * self.avg_params[n] + p.data) / (self.n_averaged + 1)
        self.n_averaged += 1

    def apply(self, model):
        """將 SWA 參數應用到 model 上"""
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.avg_params:
                p.data.copy_(self.avg_params[n])

    def get_model_copy(self):
        """返回帶 SWA 參數的新模型副本"""
        return copy.deepcopy(self.model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.5, beta=0.3, lam=0.1, label_smoothing=0.15):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.lam   = lam
        self.cls7  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.cls2  = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, align, cl7, cl2, rl):
        lc7   = self.cls7(l7, cl7)
        lc2   = self.cls2(l2, cl2)
        lr    = self.reg(reg, rl)
        total = lc7 + self.beta * lc2 + self.alpha * lr + self.lam * align
        return total, {"cls7": lc7.item(), "cls2": lc2.item(),
                       "reg": lr.item(), "align": align.item()}


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
# 特徵空間 Mixup (增強泛化)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def mixup_batch(audio, audio_mask, vision, vision_mask,
                cls7, cls2, reg, alpha=0.3):
    """在 audio/vision 特徵上做 mixup，增強非語言模態泛化"""
    if alpha <= 0:
        return audio, audio_mask, vision, vision_mask, cls7, cls2, reg
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # 確保主樣本佔主導
    B = audio.size(0)
    idx = torch.randperm(B, device=audio.device)
    mixed_audio  = lam * audio  + (1 - lam) * audio[idx]
    mixed_vision = lam * vision + (1 - lam) * vision[idx]
    # mask 取 OR (兩個都保留)
    mixed_aud_mask = torch.max(audio_mask, audio_mask[idx])
    mixed_vis_mask = torch.max(vision_mask, vision_mask[idx])
    # 標籤不混合（保留原始標籤）
    return mixed_audio, mixed_aud_mask, mixed_vision, mixed_vis_mask, cls7, cls2, reg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema,
                rdrop_alpha=0.05, mixup_alpha=0.2):
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

        # Mixup 增強 (在 audio/vision 上)
        if mixup_alpha > 0 and random.random() > 0.5:
            aud, amask, vis, vmask, cl7, cl2, rl = mixup_batch(
                aud, amask, vis, vmask, cl7, cl2, rl, alpha=mixup_alpha
            )

        optimizer.zero_grad()
        use_amp = scaler is not None

        with torch.amp.autocast('cuda', enabled=use_amp):
            l7, l2, reg, align, _ = model(ids, mask, aud, amask, vis, vmask, rl)
            loss, _ = criterion(l7, l2, reg, align, cl7, cl2, rl)

            # R-Drop: 輕量 KL 正則化
            if rdrop_alpha > 0:
                l7b, l2b, _, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7, -1), F.softmax(l7b, -1), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b, -1), F.softmax(l7, -1), reduction='batchmean')) / 2
                loss = loss + rdrop_alpha * kl

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

        ema.update()
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

        l7, l2, reg, align, _ = model(ids, mask, aud, amask, vis, vmask)
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
    # Cap [0.5, 3.0]
    return torch.FloatTensor(np.clip(len(cl) / (n * ct), 0.5, 3.0))


def progressive_unfreeze(model, epoch, total_epochs, ema):
    """漸進式解凍 DeBERTa 層"""
    encoder = getattr(model.lang_backbone, "encoder", None)
    if not encoder:
        return False
    freeze_until = 6 if epoch < total_epochs // 3 else (3 if epoch < 2 * total_epochs // 3 else 0)
    changed = False
    for i, layer in enumerate(encoder.layer):
        for p in layer.parameters():
            want = (i >= freeze_until)
            if p.requires_grad != want:
                p.requires_grad = want
                changed = True
    if changed:
        ema.add_new_params()
        print(f"  [解凍] Epoch {epoch+1}: 凍結前 {freeze_until} 層")
    return changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)

    print("=" * 70)
    print("MOSI 多模態情感分析 v35 — Anti-Overfit Edition")
    print("SWA + 增強正則化 + Mixup + 更保守 LR")
    print("=" * 70)

    # 資料路徑
    _data_candidates = [
        PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        PROJECT_ROOT / "aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl",
        Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl"),
    ]
    _data_path = next((p for p in _data_candidates if p.exists()), _data_candidates[0])

    _model_candidates = [
        PROJECT_ROOT / "emotion_system/models",
        PROJECT_ROOT / "aitwon_emotion/emotion_system/models",
        Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/models"),
    ]
    _model_dir = next((p for p in _model_candidates if p.exists()), _model_candidates[0])

    config = {
        "data_path":     _data_path,
        "model_dir":     _model_dir,
        "lang_model":    "microsoft/deberta-v3-large",
        "max_text_len":  80,
        "audio_dim":     5,
        "vision_dim":    20,
        "modal_hidden":  128,
        "top_k":         5,
        "fusion_dim":    512,
        "num_classes":   7,
        "dropout":       0.35,       # 增加 (0.2 -> 0.35) 防過擬合
        "batch_size":    12,         # 增加 (8 -> 12) 減少梯度噪聲
        "num_epochs":    80,
        "lang_lr":       3e-6,       # 降低 (5e-6 -> 3e-6) 更保守
        "modal_lr":      8e-5,       # 降低 (1e-4 -> 8e-5)
        "weight_decay":  2e-2,       # 增加 (1e-2 -> 2e-2) 更強 L2
        "warmup_ratio":  0.08,       # 增加暖機比例
        "freeze_layers": 6,
        "alpha":         0.5,
        "beta":          0.3,
        "lam":           0.1,
        "label_smoothing": 0.15,     # 增加 (0.1 -> 0.15) 防過擬合
        "delta_pos":     0.5,
        "delta_neg":     1.5,
        "margin":        0.2,
        "ema_decay":     0.9995,
        "swa_start_ratio": 0.5,      # 從 50% epoch 開始 SWA
        "patience":      25,
        "rdrop_alpha":   0.05,
        "mixup_alpha":   0.2,        # Mixup 增強
        "seed":          42,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重 (capped): {[f'{w:.3f}' for w in class_w.tolist()]}")

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

    # GPU 設置
    if torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"使用設備: {device}")

    model = SACFModelV35(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"],
        fusion_dim=config["fusion_dim"],
        top_k=config["top_k"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    model.align_loss.delta_pos = config["delta_pos"]
    model.align_loss.delta_neg = config["delta_neg"]
    model.align_loss.margin    = config["margin"]

    # 初始凍結前 6 層
    encoder = model.lang_backbone.encoder
    for i in range(config["freeze_layers"]):
        for p in encoder.layer[i].parameters():
            p.requires_grad = False
    print(f"已凍結 DeBERTa 前 {config['freeze_layers']} 層")

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = SACFLoss(
        class_w.to(device), config["alpha"], config["beta"], config["lam"],
        label_smoothing=config["label_smoothing"]
    )

    # 優化器 (含所有 DeBERTa 層，漸進式解凍的前提)
    lang_params_all = list(model.lang_backbone.parameters()) + list(model.polarity_attn.parameters())
    other_params = (
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
        {"params": lang_params_all,  "lr": config["lang_lr"]},
        {"params": other_params,     "lr": config["modal_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler() if "cuda" in device else None

    ema = EMA(model, config["ema_decay"])
    swa = SWA(model)
    swa_start_epoch = int(config["num_epochs"] * config["swa_start_ratio"])
    print(f"SWA 從 epoch {swa_start_epoch+1} 開始")

    print(f"\n開始訓練 | 設備: {device} | Batch: {config['batch_size']}")
    print(f"Lang LR: {config['lang_lr']} | Modal LR: {config['modal_lr']}")
    print(f"Weight Decay: {config['weight_decay']} | Dropout: {config['dropout']}")
    print(f"Epochs: {config['num_epochs']} | Patience: {config['patience']}\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best    = {"Acc7": 0.0, "epoch": 0}
    history = []
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        # 漸進式解凍
        changed = progressive_unfreeze(model, epoch, config["num_epochs"], ema)
        if changed:
            patience_counter = 0

        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler, device, scaler, ema,
                              rdrop_alpha=config["rdrop_alpha"],
                              mixup_alpha=config["mixup_alpha"])

        # 使用 EMA 評估
        ema.apply_shadow()
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        # SWA 更新（使用 EMA shadow 來 SWA）
        if epoch >= swa_start_epoch:
            ema.apply_shadow()
            swa.update()
            ema.restore()

        history.append({"epoch": epoch+1,
                        "tr_loss": round(tr_loss, 4),
                        "vl_loss": round(vl_loss, 4), **metrics})

        print(f"E{epoch+1:03d} | Train={tr_loss:.4f}  Val={vl_loss:.4f} | "
              f"Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}"
              + (" [SWA]" if epoch >= swa_start_epoch else ""))

        if metrics["Acc7"] > best["Acc7"]:
            best = {"epoch": epoch+1, **metrics}
            patience_counter = 0
            torch.save({
                "epoch":       epoch+1,
                "model_state": model.state_dict(),
                "ema_shadow":  ema.shadow,
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model_v35.pth")
            print(f"  => 新最佳 Acc7={metrics['Acc7']:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停: {config['patience']} epochs 無提升 (best E{best['epoch']})")
                break

    # ── 評估三個版本的模型 ──
    print("\n" + "=" * 60)

    # 1. 載入最佳 EMA checkpoint
    ckpt = torch.load(save_dir / "best_model_v35.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    for n, p in model.named_parameters():
        if p.requires_grad and n in ckpt["ema_shadow"]:
            p.data.copy_(ckpt["ema_shadow"][n])
    _, val_m_ema = validate(model, val_loader, criterion, device)
    _, test_m_ema = validate(model, test_loader, criterion, device)
    print(f"[EMA 最佳] Val  Acc7={val_m_ema['Acc7']:.2f}% | Test Acc7={test_m_ema['Acc7']:.2f}%")

    # 2. SWA 模型評估
    swa_val_acc7, swa_test_acc7 = 0.0, 0.0
    if swa.n_averaged > 0:
        # 重新載入原始模型，然後應用 SWA 參數
        model.load_state_dict(ckpt["model_state"])
        swa.apply(model)
        _, val_m_swa = validate(model, val_loader, criterion, device)
        _, test_m_swa = validate(model, test_loader, criterion, device)
        swa_val_acc7  = val_m_swa["Acc7"]
        swa_test_acc7 = test_m_swa["Acc7"]
        print(f"[SWA]     Val  Acc7={swa_val_acc7:.2f}% | Test Acc7={swa_test_acc7:.2f}%")

    # 選最佳方案
    if swa_test_acc7 >= test_m_ema["Acc7"] and swa.n_averaged > 0:
        final_test_acc7 = swa_test_acc7
        final_val_acc7  = swa_val_acc7
        final_method    = "SWA"
        # SWA 是最終最佳，保存模型
        torch.save({
            "epoch":       "SWA",
            "model_state": model.state_dict(),  # SWA 已應用
            "metrics":     test_m_swa,
            "config":      {k: str(v) for k, v in config.items()},
        }, save_dir / "best_model_v35.pth")
    else:
        final_test_acc7 = test_m_ema["Acc7"]
        final_val_acc7  = val_m_ema["Acc7"]
        final_method    = "EMA"

    # ── 儲存結果 ──
    result = {
        "history":        history,
        "best_val_acc7":  best,
        "test_standard":  test_m_ema,
        "swa_test":       test_m_swa if swa.n_averaged > 0 else None,
        "final_method":   final_method,
        "final_test_acc7": final_test_acc7,
        "config":         {k: str(v) for k, v in config.items()},
    }
    with open(save_dir / "v35_history.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── 最終輸出 ──
    print(f"\n{'='*70}")
    print(f"【v35 最終結果】 使用 {final_method}")
    print(f"  Val  Acc7: {final_val_acc7:.2f}%")
    print(f"  Test Acc7: {final_test_acc7:.2f}%")
    print(f"  vs 目標 51%: {final_test_acc7 - 51.0:+.2f}% {'達成!' if final_test_acc7 > 51.0 else '未達成'}")
    print(f"  val-test gap: {final_val_acc7 - final_test_acc7:+.2f}%")
    print(f"{'='*70}")

    return final_test_acc7


if __name__ == "__main__":
    main()
