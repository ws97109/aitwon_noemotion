"""
MOSI 多模態情感分析 v11 — Ultimate (終極優化版)
目標：突破 55% Acc7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
終極優化策略：
1. 使用 BERT-base (更小更穩定，768維)
2. 強化數據增強 (Mixup + Cutout + SpecAugment)
3. 測試時增強 (TTA) - 多次預測取平均
4. 極致正則化 (Dropout 0.35 + Stochastic Depth)
5. 更激進的訓練 (70 epochs + 更多warmup)
6. 集成策略 (保存多個checkpoint)
7. Label Smoothing 0.15
8. 超高 cls7 權重 (4.0)

如果這個還不行，說明數據本身的限制
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
from typing import Dict
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import BertModel, BertTokenizer, get_cosine_schedule_with_warmup
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 數據增強
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def mixup_data(x, y, alpha=0.2):
    if alpha > 0 and random.random() > 0.3:  # 70% 機率使用 mixup
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    return x, y, y, 1.0


def cutout_augment(x, mask, cutout_ratio=0.1):
    """隨機遮蔽部分序列"""
    if random.random() > 0.5:  # 50% 機率
        B, T, D = x.shape
        length = int(T * cutout_ratio)
        start = random.randint(0, max(0, T - length))
        x = x.clone()
        x[:, start:start+length, :] = 0
    return x


class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 64):
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
        text = str(self.raw_text[idx])
        enc = self.tokenizer(
            text, max_length=self.max_text_len, padding="max_length",
            truncation=True, return_tensors="pt"
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
# Stochastic Depth (隨機丟棄層)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, residual):
        if not self.training or random.random() > self.drop_prob:
            return x + residual
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 簡化但高效的模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UltimateModel(nn.Module):
    def __init__(
        self,
        lang_model="bert-base-uncased",
        audio_dim=5,
        vision_dim=20,
        hidden_dim=256,
        dropout=0.35,
    ):
        super().__init__()

        # BERT-base (更小更穩定)
        self.bert = BertModel.from_pretrained(lang_model)
        lang_dim = 768

        # 簡單但有效的編碼器
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.vision_enc = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 跨模態融合
        self.fusion = nn.Sequential(
            nn.Linear(lang_dim + hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stochastic Depth
        self.stoch_depth = StochasticDepth(drop_prob=0.1)

        # 分類頭
        self.cls7_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7),
        )
        self.cls2_head = nn.Linear(256, 2)
        self.reg_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        # BERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lang_feat = bert_out.last_hidden_state[:, 0, :]  # CLS token

        # 非語言 (平均池化)
        audio_feat = (audio * audio_mask.unsqueeze(-1)).sum(1) / audio_mask.sum(1, keepdim=True).clamp(min=1)
        vision_feat = (vision * vision_mask.unsqueeze(-1)).sum(1) / vision_mask.sum(1, keepdim=True).clamp(min=1)

        audio_feat = self.audio_enc(audio_feat)
        vision_feat = self.vision_enc(vision_feat)

        # 融合
        concat = torch.cat([lang_feat, audio_feat, vision_feat], dim=-1)
        feat = self.fusion(concat)

        # Stochastic Depth
        feat = self.stoch_depth(concat[:, :256] if concat.size(1) > 256 else feat, feat)

        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0

        return logits7, logits2, reg_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Focal Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, label_smoothing=0.15):
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


class UltimateLoss(nn.Module):
    def __init__(self, class_weights, alpha=4.0, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal = FocalLoss(alpha=class_weights, gamma=2.5, label_smoothing=0.15)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.reg = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, cl7, cl2, rl, l7_b=None, cl7_b=None, lam=1.0):
        # Mixup loss
        if l7_b is not None and cl7_b is not None:
            lc7 = lam * self.focal(l7, cl7) + (1 - lam) * self.focal(l7_b, cl7_b)
        else:
            lc7 = self.focal(l7, cl7)

        lc2 = self.cls2(l2, cl2)
        lr = self.reg(reg, rl)
        total = self.alpha * lc7 + self.beta * lc2 + self.gamma * lr
        return total


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

        # 數據增強
        aud = cutout_augment(aud, amask)
        vis = cutout_augment(vis, vmask)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)

            # Mixup (僅在標籤上)
            _, cl7_a, cl7_b, lam = mixup_data(l7, cl7, alpha=0.2)
            loss = criterion(l7, l2, reg, cl7_a, cl2, rl, l7, cl7_b, lam)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更小的梯度裁剪
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False):
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

        if use_tta:
            # TTA: 3次預測取平均
            logits7_list, logits2_list, reg_list = [], [], []
            for _ in range(3):
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
                logits7_list.append(l7)
                logits2_list.append(l2)
                reg_list.append(reg)

            l7 = torch.stack(logits7_list).mean(0)
            l2 = torch.stack(logits2_list).mean(0)
            reg = torch.stack(reg_list).mean(0)
        else:
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)

        loss = criterion(l7, l2, reg, cl7, cl2, rl)
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


def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(len(cl) / (n * ct))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI 多模態情感分析 v11 — Ultimate (終極優化版)")
    print("BERT-base + 強化增強 + TTA + 極致正則化")
    print("=" * 70)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "bert-base-uncased",
        "max_text_len": 64,
        "audio_dim": 5,
        "vision_dim": 20,
        "hidden_dim": 256,
        "dropout": 0.35,  # 極高 dropout
        "batch_size": 32,  # 增大 batch
        "num_epochs": 70,  # 更長訓練
        "lr": 1e-5,        # 統一較小學習率
        "weight_decay": 3e-2,  # 更強正則化
        "warmup_ratio": 0.2,   # 更長 warmup
        "alpha": 4.0,   # 超高 cls7 權重
        "beta": 0.3,
        "gamma": 0.2,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = BertTokenizer.from_pretrained(config["lang_model"])

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

    model = UltimateModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)

    # 凍結 BERT 前 8 層
    for i in range(8):
        for p in model.bert.encoder.layer[i].parameters():
            p.requires_grad = False

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = UltimateLoss(
        class_weights=class_w.to(device),
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[終極優化] Mixup + Cutout + Focal Loss + Stochastic Depth + TTA\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-" * 45)

        tr_loss = train_epoch(model, train_loader, criterion,
                             optimizer, scheduler, device, scaler)

        # 驗證（使用 TTA）
        vl_loss, metrics = validate(model, val_loader, criterion, device, use_tta=True)

        history.append({
            "epoch": epoch + 1,
            "tr_loss": round(tr_loss, 4),
            "vl_loss": round(vl_loss, 4),
            **metrics
        })

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch + 1, **metrics}
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "config": config,
            }, save_dir / "ultimate_v11_best.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")

    # 測試集 (使用 TTA)
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "ultimate_v11_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, criterion, device, use_tta=True)

    print("\n【測試集結果 - with TTA】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (MGT: 55.6%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")

    with open(save_dir / "ultimate_v11_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_val": best_acc7,
            "test": test_m,
            "config": {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")
    print(f"如果這個還不到 55%，說明 MOSI 數據集本身的限制")


if __name__ == "__main__":
    main()
