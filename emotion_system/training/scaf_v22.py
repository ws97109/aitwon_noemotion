"""
MOSI 多模態情感分析 v22 — 全員共識版
四位 Agent 深度討論後的最終方案

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
討論關鍵結論：

【v20 失敗根因】
- 過度修改架構（BiLSTM 取代 Transformer，過度正則化疊加）
- 一次性應用太多改變導致欠擬合

【kito 核心問題（全員共識）】
1. MAE 選模 → 目標是 Acc7，選模標準錯誤是最大問題
2. AudioVisualCrossAttention 退化：對 1 個 token 做 softmax=1.0，等於廢掉
3. DynamicLossWeighting 可能自動壓低 cls7 權重，傷害 Acc7
4. 只訓練 30 epochs 明顯不足

【v22 最終規格（四位 agent 全員簽署）】
架構 = v15 完整保留（Transformer 256dim + PolarityAttn + CrossModalAttn + FocalLoss）
     + OrdinalRegressionHead (CORAL, 唯一從 kito 借用的模塊)

訓練 = 保留 v15 超參（lang_lr=5e-6, bs=8, dropout=0.2, other_lr=1e-4）
     + 修復 optimizer 重建 bug（不重建，只 toggle requires_grad）
     + EMA (decay=0.999)
     + Early Stopping (patience=10)
     + set_seed(42)
     + NaN protection（音視覺輸入）
     + class_weights cap [0.5, 3.0]（修正 Class0 權重 7.64 放大噪聲）

損失 = 3.0*cls7(FocalLoss) + 0.5*cls2 + 0.3*reg(SmoothL1) + 0.3*ordinal(BCE)
選模 = Acc7（非 MAE）

預期：
- Test Acc7 >= 50.5%（v15: 47.81%, v17: 48.54%）
- Val-Test Gap < 6%（v15: 8.09%）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, pickle, random, os
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集（v15 原版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 80):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.raw_text = split_data["raw_text"]
        self.audio = torch.FloatTensor(split_data["audio"])
        self.vision = torch.FloatTensor(split_data["vision"])
        self.audio_lengths = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]

        labels = split_data["regression_labels"]
        self.reg_labels = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))
        print(f"資料: {len(self.raw_text)} 筆")

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
        aud_len = min(int(self.audio_lengths[idx]), self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]); aud_mask[:aud_len] = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx],
            "audio_mask": aud_mask,
            "vision": self.vision[idx],
            "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label": self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v15 完整架構（不修改）
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

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        g = self.gate(hidden)
        m = mask.unsqueeze(-1).float()
        enhanced = (0.75 * hidden + 0.25 * hidden * g) * m
        pooled = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        return self.dropout(pooled)


class TransformerModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, _ = x.shape
        x = self.proj_in(x) + self.pos_enc[:, :T, :]
        key_pad_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_pad_mask)
        scores = self.attn_score(x).squeeze(-1)
        scores = scores.masked_fill(key_pad_mask, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(1)


class CrossModalAttention(nn.Module):
    def __init__(self, lang_dim, modal_dim, dropout=0.1):
        super().__init__()
        self.lang_dim = lang_dim
        self.audio_map = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim // 2, lang_dim),
        )
        self.gate = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(lang_dim)

    def forward(self, xl, xa, xv):
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)
        kv = torch.stack([xa_m, xv_m], dim=1)
        scale = self.lang_dim ** 0.5
        attn = F.softmax(torch.bmm(xl.unsqueeze(1), kv.transpose(1, 2)) / scale, dim=-1)
        x_hat = torch.bmm(attn, kv).squeeze(1)
        x = self.ffn(xl + x_hat)
        gate_w = torch.sigmoid(self.gate(torch.cat([xl, x], dim=-1)))
        x = x * gate_w
        return self.norm(xl + self.dropout(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 唯一從 kito 借用：OrdinalRegressionHead（CORAL）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrdinalRegressionHead(nn.Module):
    """CORAL 序數回歸輔助頭，學習 -3 < -2 < ... < 3 的序數關係"""
    def __init__(self, feat_dim: int, num_thresholds: int = 6, dropout: float = 0.1):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(feat_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        feat = self.dropout(feat)
        logit = feat @ self.weight  # (B,)
        return logit.unsqueeze(1) + self.bias  # (B, 6)

    def compute_loss(self, ordinal_logits, cls7_labels):
        K = ordinal_logits.shape[1]
        k_range = torch.arange(K, device=cls7_labels.device)
        targets = (cls7_labels.unsqueeze(1) > k_range).float()
        return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
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
        """解凍新層後，將新參數加入 EMA shadow"""
        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.shadow:
                self.shadow[n] = p.data.clone()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型（v15 + OrdinalHead）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HybridModel(nn.Module):
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=256,
                 fusion_dim=512, num_classes=7, dropout=0.2):
        super().__init__()
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size

        self.polarity_attn = PolarityEnhancedAttention(lang_dim, dropout)
        self.audio_encoder = TransformerModalEncoder(audio_dim, modal_hidden, 2, 4, dropout)
        self.vision_encoder = TransformerModalEncoder(vision_dim, modal_hidden, 2, 4, dropout)
        self.cross_modal = CrossModalAttention(lang_dim, modal_hidden, dropout)

        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.GELU(),
            nn.Linear(fusion_dim // 2, 1), nn.Tanh(),
        )
        # 唯一從 kito 借用的模塊
        self.ordinal_head = OrdinalRegressionHead(fusion_dim, dropout=dropout)

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        # NaN 保護（data-engineer 建議）
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)

        lang_out = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = lang_out.last_hidden_state
        xl_cls = self.polarity_attn(hidden, attention_mask)

        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)
        fused = self.cross_modal(xl_cls, xa, xv)

        feat = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0
        ordinal_logits = self.ordinal_head(feat)

        return logits7, logits2, reg_out, ordinal_logits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失（固定權重，不用動態權重）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class HybridLoss(nn.Module):
    def __init__(self, class_weights, w_cls7=3.0, w_cls2=0.5, w_reg=0.3, w_ord=0.3):
        super().__init__()
        self.w_cls7 = w_cls7; self.w_cls2 = w_cls2
        self.w_reg = w_reg;   self.w_ord = w_ord
        self.focal = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
        self.cls2  = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, ordinal_logits, cl7, cl2, rl):
        lc7 = self.focal(l7, cl7)
        lc2 = self.cls2(l2, cl2)
        lr  = self.reg(reg, rl)
        lord = F.binary_cross_entropy_with_logits(
            ordinal_logits,
            (cl7.unsqueeze(1) > torch.arange(6, device=cl7.device)).float()
        )
        total = self.w_cls7 * lc7 + self.w_cls2 * lc2 + self.w_reg * lr + self.w_ord * lord
        return total, {"cls7": lc7.item(), "cls2": lc2.item(), "reg": lr.item(), "ord": lord.item()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(c7, c2, reg, l7, l2, lr):
    acc7 = (c7 == l7).mean() * 100
    acc2 = (c2 == l2).mean() * 100
    f1   = f1_score(l2, c2, average="weighted") * 100
    mae  = np.abs(reg - lr).mean()
    corr, _ = pearsonr(reg, lr)
    return {"Acc7": round(float(acc7), 2), "Acc2": round(float(acc2), 2),
            "F1": round(float(f1), 2), "MAE": round(float(mae), 4), "Corr": round(float(corr), 4)}


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema):
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
        with torch.cuda.amp.autocast(enabled=use_amp):
            l7, l2, reg, ord_logits = model(ids, mask, aud, amask, vis, vmask)
            loss, _ = criterion(l7, l2, reg, ord_logits, cl7, cl2, rl)

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
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r, all_l7, all_l2, all_lr = [], [], [], [], [], []
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

        l7, l2, reg, ord_logits = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, ord_logits, cl7, cl2, rl)
        total_loss += loss.item()

        all_c7.extend(l7.argmax(1).cpu().numpy())
        all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy())
        all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy())
        all_lr.extend(rl.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr)
    )
    return total_loss / len(loader), metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7):
    """evaluation-analyst + data-engineer 建議：cap [0.5, 3.0] 避免極端權重放大噪聲"""
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    raw = len(cl) / (n * ct)
    return torch.FloatTensor(np.clip(raw, 0.5, 3.0))  # Class0: 7.64→3.0


def progressive_unfreeze_fixed(model, epoch, total_epochs):
    """修復版：不重建 optimizer，只 toggle requires_grad"""
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder is None:
        return False

    if epoch < total_epochs // 3:
        freeze_until = 6
    elif epoch < 2 * total_epochs // 3:
        freeze_until = 3
    else:
        freeze_until = 0

    changed = False
    for i, layer in enumerate(encoder.layer):
        should_freeze = (i < freeze_until)
        for p in layer.parameters():
            if p.requires_grad == should_freeze:
                p.requires_grad = not should_freeze
                changed = True

    if changed:
        print(f"  [解凍] Epoch {epoch}: 凍結前 {freeze_until} 層")
    return changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)
    print("=" * 70)
    print("MOSI 多模態情感分析 v22 — 四位 Agent 全員共識版")
    print("v15 架構 + OrdinalHead + EMA + 修復 bug + Acc7 選模")
    print("=" * 70)

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
        "dropout":      0.2,
        "batch_size":   8,
        "num_epochs":   80,
        "lang_lr":      5e-6,
        "other_lr":     1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.1,
        "w_cls7":       3.0,
        "w_cls2":       0.5,
        "w_reg":        0.3,
        "w_ord":        0.3,
        "ema_decay":    0.999,
        "patience":     10,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"Capped 類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds   = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds  = MOSIDataset(data["test"],  tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    model = HybridModel(
        lang_model=config["lang_model"], audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"], modal_hidden=config["modal_hidden"],
        fusion_dim=config["fusion_dim"], num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    # 初始凍結前 6 層
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder:
        for i in range(6):
            for p in encoder.layer[i].parameters():
                p.requires_grad = False
        print("初始凍結前 6 層")

    total_p    = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = HybridLoss(
        class_weights=class_w.to(device),
        w_cls7=config["w_cls7"], w_cls2=config["w_cls2"],
        w_reg=config["w_reg"],   w_ord=config["w_ord"],
    )

    lang_params  = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = (
        list(model.polarity_attn.parameters()) +
        list(model.audio_encoder.parameters()) +
        list(model.vision_encoder.parameters()) +
        list(model.cross_modal.parameters()) +
        list(model.shared.parameters()) +
        list(model.cls7_head.parameters()) +
        list(model.cls2_head.parameters()) +
        list(model.reg_head.parameters()) +
        list(model.ordinal_head.parameters())
    )

    optimizer = optim.AdamW([
        {"params": lang_params,  "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None
    ema       = EMA(model, decay=config["ema_decay"])

    print(f"\n開始訓練 | 設備: {device}")
    print("[v22] v15完整架構 + OrdinalHead + EMA + bug修復 + Acc7選模 + capped weights\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history   = []
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        # 修復版漸進解凍（不重建 optimizer）
        if progressive_unfreeze_fixed(model, epoch, config["num_epochs"]):
            ema.add_new_params()  # 新解凍的層加入 EMA

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-" * 45)
        tr_loss = train_epoch(model, train_loader, criterion,
                              optimizer, scheduler, device, scaler, ema)

        # 用 EMA 模型驗證
        ema.apply_shadow()
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        history.append({"epoch": epoch+1, "tr_loss": round(tr_loss, 4),
                         "vl_loss": round(vl_loss, 4), **metrics})

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        # 選模依據：Acc7（非 MAE）
        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **metrics}
            ema.apply_shadow()
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "metrics": metrics, "config": config},
                       save_dir / "v22_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善")
                break

    # 測試集
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "v22_best_acc7.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, criterion, device)

    print("\n【測試集結果 - v22（四位 Agent 全員共識版）】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (目標: 50.5%, v17: 48.54%, v15: 47.81%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")
    print(f"\n  Val-Test Gap: {best_acc7['Acc7'] - test_m['Acc7']:.2f}% (v15: 8.09%)")

    with open(save_dir / "v22_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc7": best_acc7, "test": test_m,
                   "config": {k: str(v) for k, v in config.items()}}, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
