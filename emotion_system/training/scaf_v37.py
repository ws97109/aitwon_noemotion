"""
MOSI 多模態情感分析 v37 — Text-Only Strong Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v37 假設: 音訊(dim=5)和視覺(dim=20)維度太低，可能帶入雜訊
          純文字 DeBERTa-v3-large 可能表現更好

問題診斷: v34/v35/v36 均卡在 47.81% (328/686)
  - val Acc7 達 50-51%，但 test 卡住 → 過擬合 val 分佈
  - 三版架構不同但結果相同 → 可能是 audio/vision 特徵在拖累

v37 策略:
  ✅ 純文字 (無 audio, 無 vision)
  ✅ DeBERTa-v3-large CLS token 直接分類
  ✅ 多任務: 7class + 2class + regression
  ✅ 分層學習率: lang=3e-6, head=5e-5
  ✅ 標籤平滑 0.1
  ✅ 多種子集成 (42, 123, 777)
  ✅ 不凍結任何層 (直接 fine-tune)
  ✅ 去除 TASK_PROMPT (避免干擾分類任務)

目標: test Acc7 > 51%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

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
from typing import Dict, Tuple, List
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 資料路徑（嘗試多個候選）
_DATA_CANDIDATES = [
    PROJECT_ROOT / "aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl",
    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
    Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])

_MODEL_CANDIDATES = [
    PROJECT_ROOT / "aitwon_emotion/emotion_system/models",
    PROJECT_ROOT / "emotion_system/models",
]
MODEL_DIR = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[0])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集（純文字版）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSITextDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 128):
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len
        self.raw_text     = split_data["raw_text"]

        labels = split_data["regression_labels"]
        self.reg_labels  = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))

        print(f"資料集: {len(self.raw_text)} 筆 (文字長度上限={max_text_len})")

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        # 不加 TASK_PROMPT，直接使用原始文字
        text = str(self.raw_text[idx])
        enc = self.tokenizer(
            text, add_special_tokens=True,
            max_length=self.max_text_len,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "cls7_label":     self.cls7_labels[idx],
            "cls2_label":     self.cls2_labels[idx],
            "reg_label":      self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模型：純文字 DeBERTa + 多任務頭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TextOnlyModel(nn.Module):
    def __init__(self, lang_model: str = "microsoft/deberta-v3-large",
                 num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(lang_model)
        hidden_dim = self.backbone.config.hidden_size  # 1024

        # 簡單 attention pooling (比只用 CLS 更好)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # 共享特徵層
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 三個任務頭
        self.cls7_head = nn.Linear(256, num_classes)
        self.cls2_head = nn.Linear(256, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, H]

        # Attention pooling
        scores = self.attn_pool(hidden).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        pooled = (hidden * weights).sum(1)  # [B, H]

        feat    = self.shared(pooled)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0  # [-3, 3]

        return logits7, logits2, reg_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor,
                 label_smoothing: float = 0.1,
                 alpha: float = 0.4, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.cls7  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.cls2  = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, cl7, cl2, rl):
        lc7 = self.cls7(l7, cl7)
        lc2 = self.cls2(l2, cl2)
        lr  = self.reg(reg, rl)
        total = lc7 + self.beta * lc2 + self.alpha * lr
        return total, {"cls7": lc7.item(), "cls2": lc2.item(), "reg": lr.item()}


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


def compute_class_weights(labels: np.ndarray, n: int = 7) -> torch.Tensor:
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(np.clip(len(cl) / (n * ct), 0.5, 3.0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練一個 Epoch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        cl7  = batch["cls7_label"].to(device)
        cl2  = batch["cls2_label"].to(device)
        rl   = batch["reg_label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask)
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
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        cl7  = batch["cls7_label"].to(device)
        cl2  = batch["cls2_label"].to(device)
        rl   = batch["reg_label"].to(device)

        l7, l2, reg = model(ids, mask)
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
        np.array(all_l7), np.array(all_l2), np.array(all_lr),
    )
    return total_loss / len(loader), metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 單一模型訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_single_model(seed: int, train_loader, val_loader, test_loader,
                       class_weights, device, config) -> Tuple[float, float, np.ndarray]:
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"訓練 Seed={seed}")
    print(f"{'='*60}")

    model = TextOnlyModel(
        lang_model=config["lang_model"],
        dropout=config["dropout"]
    ).to(device)

    criterion = MultiTaskLoss(
        class_weights.to(device),
        label_smoothing=config["label_smoothing"],
        alpha=config["alpha"], beta=config["beta"]
    )

    # 分層學習率
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.attn_pool.parameters()) + \
                      list(model.shared.parameters()) + \
                      list(model.cls7_head.parameters()) + \
                      list(model.cls2_head.parameters()) + \
                      list(model.reg_head.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": config["lang_lr"]},
        {"params": head_params,     "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema = EMA(model, decay=0.9995)

    best_acc7    = 0.0
    best_epoch   = 0
    best_state   = None
    patience_cnt = 0

    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, ema)

        # EMA 評估
        ema.apply_shadow()
        _, val_metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        acc7 = val_metrics["Acc7"]
        print(f"E{epoch+1:02d} | Loss={train_loss:.4f} | Val Acc7={acc7:.2f}%", end="")

        if acc7 > best_acc7:
            best_acc7  = acc7
            best_epoch = epoch + 1
            patience_cnt = 0
            # 儲存 EMA 權重
            ema.apply_shadow()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ema.restore()
            print(f" ✅ 新最佳 Acc7={acc7:.2f}%")
        else:
            patience_cnt += 1
            print()
            if patience_cnt >= config["patience"]:
                print(f"  早停 (patience={config['patience']})")
                break

    # 載入最佳模型，評估測試集
    model.load_state_dict(best_state)
    model.to(device)
    _, test_metrics = validate(model, test_loader, criterion, device)

    print(f"  [Seed {seed}] Val={best_acc7:.2f}% (E{best_epoch}) | Test={test_metrics['Acc7']:.2f}%")

    # 收集 test softmax 機率（用於集成）
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            l7, _, _ = model(ids, mask)
            all_probs.append(F.softmax(l7, dim=-1).cpu().numpy())
    test_probs = np.concatenate(all_probs, axis=0)

    return best_acc7, test_metrics["Acc7"], test_probs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主程式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI v37 — Text-Only DeBERTa + Multi-Seed Ensemble")
    print("=" * 70)
    print("假設: 純文字可能比低維度多模態特徵更好泛化")
    print("=" * 70)

    config = {
        "lang_model":      "microsoft/deberta-v3-large",
        "batch_size":      16,
        "num_epochs":      50,
        "lang_lr":         3e-6,
        "head_lr":         5e-5,
        "weight_decay":    1e-2,
        "dropout":         0.2,
        "label_smoothing": 0.10,
        "alpha":           0.4,   # reg 損失權重
        "beta":            0.3,   # cls2 損失權重
        "patience":        15,
        "seeds":           [42, 123, 777],
    }

    print(f"\n載入資料: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSITextDataset(data["train"], tokenizer, max_text_len=128)
    val_ds   = MOSITextDataset(data["valid"], tokenizer, max_text_len=128)
    test_ds  = MOSITextDataset(data["test"],  tokenizer, max_text_len=128)

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")

    # 類別權重（根據訓練集計算）
    train_labels = data["train"]["regression_labels"]
    class_weights = compute_class_weights(train_labels)
    print(f"類別權重: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # 取測試集真實標籤
    test_labels_np = np.array(data["test"]["regression_labels"])
    test_cls7_true = np.clip(np.round(test_labels_np).astype(int), -3, 3) + 3
    test_cls2_true = (test_labels_np >= 0).astype(int)

    # 多種子訓練
    all_test_probs = []
    seed_results   = []

    for seed in config["seeds"]:
        val_acc, test_acc, test_probs = train_single_model(
            seed, train_loader, val_loader, test_loader,
            class_weights, device, config
        )
        all_test_probs.append(test_probs)
        seed_results.append((seed, val_acc, test_acc))

    # 集成（平均機率）
    ensemble_probs = np.mean(all_test_probs, axis=0)
    ens_preds = ensemble_probs.argmax(1)
    ens_acc7  = (ens_preds == test_cls7_true).mean() * 100
    ens_acc2  = ((ens_preds >= 3).astype(int) == test_cls2_true).mean() * 100
    ens_f1    = f1_score(test_cls2_true, (ens_preds >= 3).astype(int), average="weighted") * 100

    # 最終結果
    print(f"\n{'='*70}")
    print(f"【v37 最終結果】Text-Only + Multi-Seed Ensemble")
    print(f"{'='*70}")
    print(f"\n個別模型成績:")
    for seed, val_acc, test_acc in seed_results:
        print(f"  Seed {seed:3d}: Val={val_acc:.2f}% | Test={test_acc:.2f}%")

    print(f"\n集成結果 ({len(config['seeds'])} 種子):")
    print(f"  Test Acc7: {ens_acc7:.2f}%")
    print(f"  Test Acc2: {ens_acc2:.2f}%")
    print(f"  Test F1:   {ens_f1:.2f}%")
    print(f"  vs 目標 51%: {ens_acc7 - 51.0:+.2f}% {'✓ 達標！' if ens_acc7 > 51.0 else '✗ 未達標'}")

    # 監控腳本需要的格式
    print(f"\nTest Acc7: {ens_acc7:.2f}%")

    # 儲存結果
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    import json
    history = {
        "version": "v37",
        "strategy": "text_only_multi_seed_ensemble",
        "seeds": config["seeds"],
        "seed_results": [{"seed": s, "val": v, "test": t} for s, v, t in seed_results],
        "ensemble_acc7": round(ens_acc7, 2),
        "ensemble_acc2": round(ens_acc2, 2),
        "ensemble_f1":   round(ens_f1,   2),
    }
    with open(MODEL_DIR / "history_v37.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"結果已儲存至: {MODEL_DIR / 'history_v37.json'}")


if __name__ == "__main__":
    main()
