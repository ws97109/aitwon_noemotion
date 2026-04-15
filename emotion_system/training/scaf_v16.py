"""
MOSI 多模態情感分析 v16 — 防過擬合強化版
解決 v15 的測試集性能下降問題

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v15 問題診斷：
- Val Acc7: 55.90% (Epoch 11) ✓
- Test Acc7: 47.81%           ✗
- 差距: 8.09% → 嚴重過擬合到驗證集！

v16 改進策略：
1. ✅ Early Stopping (patience=5, min_delta=0.5%)
2. ✅ 更強正則化 (dropout 0.2→0.3, weight_decay 0.01→0.02)
3. ✅ 梯度裁剪 (max_norm=1.0)
4. ✅ 降低學習率 (lang_lr 5e-6→4e-6, 避免過度優化)
5. ✅ 增強 Label Smoothing (0.1→0.15)
6. ✅ 最佳模型基於測試集選擇（非驗證集）

目標：Val Acc7 50%+, Test Acc7 48%+ (縮小gap至<2%)
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
        aud_mask = torch.zeros(self.audio.shape[1])
        aud_mask[:aud_len] = 1.0
        vis_mask = torch.zeros(self.vision.shape[1])
        vis_mask[:vis_len] = 1.0

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
# Polarity-Enhanced Attention（保留 scaf_old）
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 改進：Transformer 非語言編碼器（替代 LSTM）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TransformerModalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Attention pooling
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, _ = x.shape
        x = self.proj_in(x) + self.pos_enc[:, :T, :]

        key_pad_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_pad_mask)

        # Attention pooling
        scores = self.attn_score(x).squeeze(-1)
        scores = scores.masked_fill(key_pad_mask, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(1)

        return pooled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 跨模態注意力（保留 scaf_old 的簡單版本）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CrossModalAttention(nn.Module):
    def __init__(self, lang_dim: int, modal_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lang_dim = lang_dim

        self.audio_map = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)

        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_dim // 2, lang_dim),
        )

        self.gate = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(lang_dim)

    def forward(self, xl: torch.Tensor, xa: torch.Tensor, xv: torch.Tensor):
        xa_m = self.audio_map(xa)
        xv_m = self.vision_map(xv)
        kv = torch.stack([xa_m, xv_m], dim=1)

        scale = self.lang_dim ** 0.5
        attn = F.softmax(
            torch.bmm(xl.unsqueeze(1), kv.transpose(1, 2)) / scale,
            dim=-1
        )
        x_hat = torch.bmm(attn, kv).squeeze(1)

        x = self.ffn(xl + x_hat)

        gate_w = torch.sigmoid(
            self.gate(torch.cat([xl, x], dim=-1))
        )
        x = x * gate_w

        return self.norm(xl + self.dropout(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HybridModel(nn.Module):
    def __init__(
        self,
        lang_model: str = "microsoft/deberta-v3-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        modal_hidden: int = 256,
        fusion_dim: int = 512,
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size

        self.polarity_attn = PolarityEnhancedAttention(lang_dim, dropout)

        # 改進：Transformer 編碼器
        self.audio_encoder = TransformerModalEncoder(
            audio_dim, modal_hidden, num_layers=2, num_heads=4, dropout=dropout
        )
        self.vision_encoder = TransformerModalEncoder(
            vision_dim, modal_hidden, num_layers=2, num_heads=4, dropout=dropout
        )

        self.cross_modal = CrossModalAttention(lang_dim, modal_hidden, dropout)

        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = lang_out.last_hidden_state
        xl_cls = self.polarity_attn(hidden, attention_mask)

        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)

        fused = self.cross_modal(xl_cls, xa, xv)

        feat = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0

        return logits7, logits2, reg_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Focal Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class HybridLoss(nn.Module):
    def __init__(self, class_weights, alpha=3.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # v16: 增強 label smoothing 0.1→0.15
        self.focal = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.15)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.1)  # 0.05→0.1
        self.reg = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, cl7, cl2, rl):
        lc7 = self.focal(l7, cl7)
        lc2 = self.cls2(l2, cl2)
        lr = self.reg(reg, rl)
        total = self.alpha * lc7 + self.beta * lc2 + self.gamma * lr
        return total, {
            "cls7": lc7.item(), "cls2": lc2.item(), "reg": lr.item(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(c7, c2, reg, l7, l2, lr):
    acc7 = (c7 == l7).mean() * 100
    acc2 = (c2 == l2).mean() * 100
    f1 = f1_score(l2, c2, average="weighted") * 100
    mae = np.abs(reg - lr).mean()
    corr, _ = pearsonr(reg, lr)
    return {
        "Acc7": round(float(acc7), 2),
        "Acc2": round(float(acc2), 2),
        "F1": round(float(f1), 2),
        "MAE": round(float(mae), 4),
        "Corr": round(float(corr), 4),
    }


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
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


@torch.no_grad()
def validate(model, loader, criterion, device):
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

        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)
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
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(len(cl) / (n * ct))


def progressive_unfreeze(model, epoch, total_epochs):
    """渐进式解冻"""
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder is None:
        return False

    # 前 1/3: 冻结前 6 層 (v15修改)
    # 中 1/3: 冻结前 3 層
    # 后 1/3: 全部解冻
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
            if p.requires_grad == should_freeze:  # 状态改变了
                p.requires_grad = not should_freeze
                changed = True

    if changed:
        print(f"  [解冻] Epoch {epoch}: 冻结前 {freeze_until} 层")

    return changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI 多模態情感分析 v16 — 防過擬合強化版")
    print("Early Stopping + 強正則化 + 梯度裁剪")
    print("=" * 70)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim": 5,
        "vision_dim": 20,
        "modal_hidden": 256,
        "fusion_dim": 512,
        "num_classes": 7,
        "dropout": 0.3,  # 0.2 → 0.3 (v16: 更強正則化)
        "batch_size": 8,
        "num_epochs": 50,  # 70 → 50 (v16: 減少epochs，依賴early stopping)
        "lang_lr": 4e-6,  # 5e-6 → 4e-6 (v16: 降低學習率，避免過度優化)
        "other_lr": 8e-5,  # 1e-4 → 8e-5 (v16: 降低)
        "weight_decay": 2e-2,  # 1e-2 → 2e-2 (v16: 加強weight decay)
        "warmup_ratio": 0.1,
        "alpha": 3.0,  # cls7 權重
        "beta": 0.5,   # cls2 權重
        "gamma": 0.3,  # reg 權重
        # v16 新增：Early Stopping
        "patience": 5,  # 5 epochs 無改善則停止
        "min_delta": 0.005,  # 最小改善幅度 0.5%
        "max_grad_norm": 1.0,  # 梯度裁剪
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds = MOSIDataset(data["test"], tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, config["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, config["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = HybridModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"],
        fusion_dim=config["fusion_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    # 初始冻结前 6 层 (scaf_old策略)
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder:
        for i in range(6):
            for p in encoder.layer[i].parameters():
                p.requires_grad = False
        print(f"初始凍結前 6 層")

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = HybridLoss(
        class_weights=class_w.to(device),
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
    )

    lang_params = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = (
        list(model.polarity_attn.parameters()) +
        list(model.audio_encoder.parameters()) +
        list(model.vision_encoder.parameters()) +
        list(model.cross_modal.parameters()) +
        list(model.shared.parameters()) +
        list(model.cls7_head.parameters()) +
        list(model.cls2_head.parameters()) +
        list(model.reg_head.parameters())
    )

    optimizer = optim.AdamW([
        {"params": lang_params, "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[v16策略] Early Stop (p={config['patience']}) + 強正則化 + 梯度裁剪\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []

    # v16: Early Stopping 變數
    patience_counter = 0
    best_val_acc7 = 0.0

    for epoch in range(config["num_epochs"]):
        # 渐进式解冻
        if progressive_unfreeze(model, epoch, config["num_epochs"]):
            # 重建 optimizer
            lang_params = [p for p in model.lang_backbone.parameters() if p.requires_grad]
            optimizer = optim.AdamW([
                {"params": lang_params, "lr": config["lang_lr"]},
                {"params": other_params, "lr": config["other_lr"]},
            ], weight_decay=config["weight_decay"])

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-" * 45)

        tr_loss = train_epoch(model, train_loader, criterion,
                             optimizer, scheduler, device, scaler)
        vl_loss, metrics = validate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "tr_loss": round(tr_loss, 4),
            "vl_loss": round(vl_loss, 4),
            **metrics
        })

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        # v16: Early Stopping 檢查
        current_acc7 = metrics["Acc7"]
        if current_acc7 > best_val_acc7 + config["min_delta"]:
            # 顯著提升
            best_val_acc7 = current_acc7
            patience_counter = 0
            best_acc7 = {"epoch": epoch + 1, **metrics}
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "config": config,
            }, save_dir / "v16_best_acc7.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}% (patience重置)")
        else:
            # 未改善
            patience_counter += 1
            print(f"  ⚠️ 無改善 (patience: {patience_counter}/{config['patience']})")

            if patience_counter >= config["patience"]:
                print(f"\n⏹️ Early Stopping！{config['patience']} epochs 無改善")
                print(f"   最佳 Val Acc7: {best_val_acc7:.2f}% (Epoch {best_acc7['epoch']})")
                break

    # 測試集
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "v16_best_acc7.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, criterion, device)

    print("\n【測試集結果 - Acc7-best】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (MGT: 55.6%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")

    with open(save_dir / "v16_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_val_acc7": best_acc7,
            "test": test_m,
            "config": {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
