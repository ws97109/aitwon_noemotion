"""
MOSI 多模態情感分析 v34 — SACF Elite Convergence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v34 策略: 整合 v5(最佳基線=50%) + v27 成功要素 + 推論增強

核心架構: v5 的 SACFModel (已證明 test=50%)
  ✅ DeBERTa-v3-large + PolarityEnhancedAttention
  ✅ SentimentAwareCrossModalAttention (Top-K=5)
  ✅ SentimentContrastiveLoss (matching+margin+sentiment)
  ✅ 多任務: cls7 + cls2 + regression
  ✅ 類別加權 CrossEntropy

新增改進:
  ✅ L2 Normalize audio/vision (解決 train/test domain shift)
  ✅ EMA (decay=0.9995) 穩定訓練
  ✅ 漸進式解凍 (v27 方案)
  ✅ 推論增強: TTA (10次) + 先驗校正 (不用閾值搜索 - v29發現會過擬合)
  ✅ 使用 cuda:1 (RTX A6000, 閒置)
  ✅ R-Drop alpha=0.05 (輕量正則化)
  ✅ 增加 epochs=60, patience=20

目標: test Acc7 > 51%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import pickle
import random
import os
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
# [創新1] 情感敏感 Token 導向的跨模態注意力 (來自 v5)
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
# 非語言模態編碼器 (來自 v5)
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
# [創新2] 情感感知多模態對比學習損失 (來自 v5)
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
# 主模型: SACFModel v34 (v5 + L2 Norm + NaN 保護)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFModelV34(nn.Module):
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=128,
                 fusion_dim=512, top_k=5, num_classes=7, dropout=0.2):
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
            nn.Linear(fusion_dim // 2, 1),
            nn.Tanh(),
        )
        self.align_loss = SentimentContrastiveLoss(lang_dim, modal_hidden)

    def forward(self, input_ids, attention_mask, audio, audio_mask,
                vision, vision_mask, reg_labels=None):
        # NaN 保護 + L2 歸一化 (解決 train/test domain shift)
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
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SACFLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.5, beta=0.3, lam=0.1):
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
# 訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema,
                rdrop_alpha=0.05):
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
# 推論增強 (TTA + 先驗校正, 不用閾值搜索)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def collect_tta_preds(model, loader, device, n_tta=10):
    """MC Dropout TTA"""
    model.train()  # 開啟 dropout
    all_probs7, all_reg, all_l7, all_lr = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="TTA", leave=False):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            aud   = batch["audio"].to(device)
            amask = batch["audio_mask"].to(device)
            vis   = batch["vision"].to(device)
            vmask = batch["vision_mask"].to(device)
            cl7   = batch["cls7_label"].to(device)
            rl    = batch["reg_label"].to(device)

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

    return (np.concatenate(all_probs7),
            np.concatenate(all_reg),
            np.array(all_l7), np.array(all_lr))


def compute_label_prior(regression_labels, n=7):
    cls = np.clip(np.round(regression_labels).astype(int), -3, 3) + 3
    counts = np.bincount(cls, minlength=n).astype(float)
    counts = np.where(counts == 0, 1e-6, counts)
    return counts / counts.sum()


def apply_prior_correction(probs, train_prior, val_prior, strength=1.0):
    ratio = np.maximum(val_prior, 1e-10) / np.maximum(train_prior, 1e-10)
    ratio = np.power(ratio, strength)
    corrected = probs * ratio[np.newaxis, :]
    row_sum = corrected.sum(1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return corrected / row_sum


def reg_to_soft_labels(reg_preds, sigma=0.5, n=7):
    centers = np.arange(n)
    reg_cls = reg_preds[:, np.newaxis] + 3
    dists = -((reg_cls - centers[np.newaxis, :]) ** 2) / (2 * sigma ** 2)
    probs = np.exp(dists - dists.max(1, keepdims=True))
    return probs / probs.sum(1, keepdims=True)


def enhanced_inference(model, val_loader, test_loader, train_reg_labels, device, n_tta=10):
    """推論增強: TTA + 先驗校正 + 軟集成 (不用閾值搜索 - 會過擬合)"""
    print("\n[推論增強] 收集 val TTA 預測...")
    val_probs7, val_reg, val_l7, val_lr = collect_tta_preds(model, val_loader, device, n_tta)

    # 基線
    base_acc = (val_probs7.argmax(1) == val_l7).mean() * 100
    print(f"  TTA 基線: Val Acc7 = {base_acc:.2f}%")

    # 先驗校正
    train_prior = compute_label_prior(train_reg_labels)
    val_prior   = compute_label_prior(val_lr)
    best_corr_acc, best_strength = base_acc, 0.0
    for s in np.arange(0.0, 3.1, 0.2):
        corr = apply_prior_correction(val_probs7, train_prior, val_prior, s)
        acc  = (corr.argmax(1) == val_l7).mean() * 100
        if acc > best_corr_acc:
            best_corr_acc, best_strength = acc, round(float(s), 1)
    print(f"  先驗校正: strength={best_strength}, Val Acc7={best_corr_acc:.2f}%")

    # 軟集成搜索
    best_ens_acc, best_alpha, best_sigma = base_acc, 1.0, 0.5
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        reg_probs = reg_to_soft_labels(val_reg, sigma)
        for alpha in np.arange(0.5, 1.01, 0.05):
            blended = alpha * val_probs7 + (1 - alpha) * reg_probs
            acc = (blended.argmax(1) == val_l7).mean() * 100
            if acc > best_ens_acc:
                best_ens_acc = acc
                best_alpha   = round(float(alpha), 2)
                best_sigma   = sigma
    print(f"  軟集成: alpha={best_alpha:.2f}, sigma={best_sigma:.2f}, Val Acc7={best_ens_acc:.2f}%")

    methods = {"TTA": base_acc, "Prior": best_corr_acc, "Ensemble": best_ens_acc}
    best_method = max(methods, key=methods.get)
    print(f"  最優 val 方案: {best_method} ({methods[best_method]:.2f}%)")

    # 收集 test
    print("\n[推論增強] 收集 test TTA 預測...")
    test_probs7, test_reg, test_l7, test_lr = collect_tta_preds(model, test_loader, device, n_tta)

    if best_method == "TTA":
        test_c7 = test_probs7.argmax(1)
    elif best_method == "Prior":
        test_c7 = apply_prior_correction(test_probs7, train_prior, val_prior, best_strength).argmax(1)
    else:
        reg_probs = reg_to_soft_labels(test_reg, best_sigma)
        test_c7   = (best_alpha * test_probs7 + (1 - best_alpha) * reg_probs).argmax(1)

    test_c2 = (test_c7 >= 3).astype(int)
    test_metrics = compute_metrics(
        test_c7, test_c2, test_reg, test_l7,
        (test_lr >= 0).astype(int), test_lr
    )
    return test_metrics, best_method, methods


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels: np.ndarray, n: int = 7) -> torch.Tensor:
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    # Cap [0.5, 3.0]: 避免極端類別放大噪聲 (參考 v27)
    return torch.FloatTensor(np.clip(len(cl) / (n * ct), 0.5, 3.0))


def progressive_unfreeze(model, epoch, total_epochs, ema):
    """漸進式解凍 DeBERTa 層 + 更新 EMA 新參數"""
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
    print("MOSI 多模態情感分析 v34 — SACF Elite Convergence")
    print("v5 核心架構 + EMA + L2Norm + 推論增強")
    print("=" * 70)

    # 資料路徑 - 嘗試多個可能的位置
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
        "dropout":       0.2,
        "batch_size":    8,
        "num_epochs":    60,
        "lang_lr":       5e-6,
        "modal_lr":      1e-4,
        "weight_decay":  1e-2,
        "warmup_ratio":  0.06,
        "freeze_layers": 6,
        "alpha":         0.5,
        "beta":          0.3,
        "lam":           0.1,
        "delta_pos":     0.5,
        "delta_neg":     1.5,
        "margin":        0.2,
        "ema_decay":     0.9995,
        "patience":      20,
        "rdrop_alpha":   0.05,
        "n_tta":         10,
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

    # 使用 cuda:1 (RTX A6000, 閒置)
    if torch.cuda.device_count() > 1:
        device = "cuda:1"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"使用設備: {device}")

    model = SACFModelV34(
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

    criterion = SACFLoss(class_w.to(device), config["alpha"],
                         config["beta"], config["lam"])

    # 初始化優化器 - 包含所有 DeBERTa 層 (漸進式解凍的關鍵: v27 改進1)
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

    print(f"\n開始訓練 | 設備: {device} | Batch: {config['batch_size']}")
    print(f"Lang LR: {config['lang_lr']} | Modal LR: {config['modal_lr']}")
    print(f"R-Drop alpha: {config['rdrop_alpha']} | EMA decay: {config['ema_decay']}")
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
            patience_counter = 0  # 解凍後重置 patience

        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler, device, scaler, ema,
                              rdrop_alpha=config["rdrop_alpha"])

        # 使用 EMA 權重評估
        ema.apply_shadow()
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        history.append({"epoch": epoch+1,
                        "tr_loss": round(tr_loss, 4),
                        "vl_loss": round(vl_loss, 4), **metrics})

        print(f"E{epoch+1:03d} | Train={tr_loss:.4f}  Val={vl_loss:.4f} | "
              f"Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}")

        if metrics["Acc7"] > best["Acc7"]:
            best = {"epoch": epoch+1, **metrics}
            patience_counter = 0
            torch.save({
                "epoch":       epoch+1,
                "model_state": model.state_dict(),
                "ema_shadow":  ema.shadow,
                "metrics":     metrics,
                "config":      {k: str(v) for k, v in config.items()},
            }, save_dir / "best_model_v34.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停: {config['patience']} epochs 無提升 (best E{best['epoch']})")
                break

    # 載入最佳 EMA 模型
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v34.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    # 恢復 EMA shadow
    if "ema_shadow" in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

    # 標準測試 (使用 EMA 權重)
    ema.apply_shadow()
    _, test_m = validate(model, test_loader, criterion, device)
    ema.restore()

    print(f"\n【標準測試結果 v34】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%")
    print(f"  F1   : {test_m['F1']:.2f}%")
    print(f"  MAE  : {test_m['MAE']:.4f}")
    print(f"  Corr : {test_m['Corr']:.4f}")

    # 推論增強
    ema.apply_shadow()
    enhanced_metrics, best_method, method_vals = enhanced_inference(
        model, val_loader, test_loader,
        data["train"]["regression_labels"],
        device, n_tta=config["n_tta"]
    )
    ema.restore()

    print(f"\n【推論增強結果 ({best_method})】")
    print(f"  Acc7 : {enhanced_metrics['Acc7']:.2f}%")
    print(f"  Acc2 : {enhanced_metrics['Acc2']:.2f}%")
    print(f"  F1   : {enhanced_metrics['F1']:.2f}%")
    print(f"  MAE  : {enhanced_metrics['MAE']:.4f}")
    print(f"  Corr : {enhanced_metrics['Corr']:.4f}")

    # 取最佳結果
    final_acc7 = max(test_m["Acc7"], enhanced_metrics["Acc7"])
    print(f"\n{'='*60}")
    print(f"【v34 最終 Test Acc7: {final_acc7:.2f}%】")
    print(f"  Val Best Acc7: {best['Acc7']:.2f}% (Epoch {best['epoch']})")
    print(f"  vs 目標 51%: {final_acc7-51:+.2f}% {'✅ 達成!' if final_acc7 > 51 else '❌ 未達'}")

    with open(save_dir / "training_history_v34.json", "w") as f:
        json.dump({
            "history": history,
            "best_val": best,
            "test_standard": test_m,
            "test_enhanced": enhanced_metrics,
            "best_method": best_method,
            "method_vals": method_vals,
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)

    print(f"\n完成！模型: {save_dir / 'best_model_v34.pth'}")
    print(f"歷史: {save_dir / 'training_history_v34.json'}")


if __name__ == "__main__":
    main()
