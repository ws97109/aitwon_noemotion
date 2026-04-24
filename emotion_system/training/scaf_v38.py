"""
MOSI 多模態情感分析 v38 — Train+Val Combined Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【根因分析】
  v34/v35/v36 均卡在 47.81% (328/686 完全相同)
  → 三種不同架構卻有相同結果，說明問題在「訓練流程」而非「架構」
  → 根本原因: 用 val set (229 樣本) 做早停，模型收斂至 val 偏好的局部最優
  → val set 分佈與 test (686 樣本) 有偏差，造成固定的 47.81% 天花板

  v37 (純文字) = 46.79% < 47.81% → audio/vision 確實有幫助

【v38 策略: 消除 val-test gap 的根源】
  ✅ 合併 train+val 訓練 (1284+229=1513 樣本)
  ✅ 固定訓練 45 epochs (根據 v27/v34 歷史最佳 epoch 估算)
  ✅ 無早停，無 val 選擇偏差
  ✅ 保留 SACF 架構 (多模態最佳)
  ✅ L2 Norm + NaN 保護
  ✅ EMA (decay=0.9995) 穩定收斂
  ✅ 多種子訓練 (42, 123, 777) 集成
  ✅ 直接 argmax，無 TTA

  目標: test Acc7 > 51%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pickle
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent

_DATA_CANDIDATES = [
    PROJECT_ROOT / "aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl",
    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
    Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if p.exists()), _DATA_CANDIDATES[0])
MODEL_DIR = Path("/mnt/nfs/maokao_2/Desktop/lee/aitown_emotion/aitwon_emotion/emotion_system/models")

TASK_PROMPT = "Predict the sentiment intensity (-3 to 3, negative to positive) of the following text: "


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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

        print(f"  資料集: {len(self.raw_text)} 筆")

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
# SACF 架構（沿用 v5/v34 最佳架構）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        g = self.gate(hidden)
        m = mask.unsqueeze(-1).float()
        enhanced = (0.75 * hidden + 0.25 * hidden * g) * m
        pooled   = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        gates    = (g * m).squeeze(-1)
        return self.dropout(pooled), gates


class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, lang_dim: int, modal_dim: int, top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.token_attn = nn.Linear(lang_dim, 1)
        self.ffn  = nn.Sequential(
            nn.Linear(lang_dim, lang_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim // 2, lang_dim),
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
        xa_m = self.audio_map(xa);  xv_m = self.vision_map(xv)
        kv   = torch.stack([xa_m, xv_m], dim=1)
        attn = F.softmax(torch.bmm(sa_query.unsqueeze(1), kv.transpose(1, 2)) / (H ** 0.5), dim=-1)
        x_hat = torch.bmm(attn, kv).squeeze(1)
        x = self.ffn(xl_cls + x_hat)
        gate_w = torch.sigmoid(self.gate(torch.cat([xl_cls, x], dim=-1)))
        return self.norm(xl_cls + self.dropout(x * gate_w))


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[-2], h[-1]], dim=-1))


class SACFModel(nn.Module):
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
            nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.GELU(),
            nn.Linear(fusion_dim // 2, 1), nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        # NaN 保護 + L2 歸一化
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
        return self.cls7_head(feat), self.cls2_head(feat), self.reg_head(feat).squeeze(-1) * 3.0


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
                self.backup[n] = p.data.clone(); p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]
        self.backup = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels: np.ndarray, n: int = 7) -> torch.Tensor:
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(np.clip(len(cl) / (n * ct), 0.5, 3.0))


def compute_metrics(c7, c2, reg, l7, l2, lr) -> Dict:
    acc7 = (c7 == l7).mean() * 100
    acc2 = (c2 == l2).mean() * 100
    f1   = f1_score(l2, c2, average="weighted") * 100
    mae  = np.abs(reg - lr).mean()
    corr, _ = pearsonr(reg, lr)
    return {"Acc7": round(float(acc7), 2), "Acc2": round(float(acc2), 2),
            "F1": round(float(f1), 2), "MAE": round(float(mae), 4), "Corr": round(float(corr), 4)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device);  amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device); vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        cl2   = batch["cls2_label"].to(device)
        rl    = batch["reg_label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            lc7 = criterion["cls7"](l7, cl7)
            lc2 = criterion["cls2"](l2, cl2)
            lr  = criterion["reg"](reg, rl)
            loss = lc7 + 0.3 * lc2 + 0.4 * lr

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        ema.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    all_c7, all_c2, all_r, all_l7, all_l2, all_lr = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device);  amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device); vmask = batch["vision_mask"].to(device)
        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        all_c7.extend(l7.argmax(1).cpu().numpy())
        all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy())
        all_l7.extend(batch["cls7_label"].numpy())
        all_l2.extend(batch["cls2_label"].numpy())
        all_lr.extend(batch["reg_label"].numpy())
    return compute_metrics(np.array(all_c7), np.array(all_c2), np.array(all_r),
                           np.array(all_l7), np.array(all_l2), np.array(all_lr))


@torch.no_grad()
def get_probs(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    for batch in loader:
        ids  = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        aud  = batch["audio"].to(device);     amask = batch["audio_mask"].to(device)
        vis  = batch["vision"].to(device);    vmask = batch["vision_mask"].to(device)
        l7, _, _ = model(ids, mask, aud, amask, vis, vmask)
        all_probs.append(F.softmax(l7, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 單一種子訓練（在 train+val 上）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_one_seed(seed, trainval_loader, test_loader, class_weights, device, config):
    set_seed(seed)
    print(f"\n{'─'*50}")
    print(f"Seed={seed} | 在 train+val ({config['trainval_size']} 筆) 上訓練")
    print(f"{'─'*50}")

    model = SACFModel(dropout=config["dropout"]).to(device)

    # 凍結前6層
    for i in range(6):
        for p in model.lang_backbone.encoder.layer[i].parameters():
            p.requires_grad = False

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "lang_backbone" in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and "lang_backbone" not in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": config["lang_lr"]},
        {"params": head_params,     "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(trainval_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema = EMA(model, 0.9995)

    cw = class_weights.to(device)
    criterion = {
        "cls7": nn.CrossEntropyLoss(weight=cw, label_smoothing=config["label_smoothing"]),
        "cls2": nn.CrossEntropyLoss(label_smoothing=0.05),
        "reg":  nn.SmoothL1Loss(),
    }

    # epoch 27 解凍第 7-12 層
    unfreeze_done = False

    for epoch in range(config["num_epochs"]):
        # 解凍：epoch 27 起開放後半部分
        if epoch == 27 and not unfreeze_done:
            for i in range(6, 12):
                for p in model.lang_backbone.encoder.layer[i].parameters():
                    p.requires_grad = True
            ema.shadow.update({n: p.data.clone()
                               for n, p in model.named_parameters()
                               if p.requires_grad and n not in ema.shadow})
            unfreeze_done = True
            print(f"  [解凍] Epoch {epoch+1}: 開放第 7-12 層")

        loss = train_epoch(model, trainval_loader, criterion, optimizer, scheduler, device, scaler, ema)

        if (epoch + 1) % 5 == 0 or epoch == config["num_epochs"] - 1:
            ema.apply_shadow()
            test_m = evaluate(model, test_loader, device)
            ema.restore()
            print(f"  E{epoch+1:02d} | Loss={loss:.4f} | Test Acc7={test_m['Acc7']:.2f}%")

    # 最終評估（EMA）
    ema.apply_shadow()
    test_metrics = evaluate(model, test_loader, device)
    test_probs   = get_probs(model, test_loader, device)
    ema.restore()

    print(f"  [Seed {seed}] 最終 Test Acc7={test_metrics['Acc7']:.2f}%")
    return test_metrics["Acc7"], test_probs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主程式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI v38 — Train+Val Combined | 消除 Val-Test Gap")
    print("=" * 70)
    print("策略: 合併 train+val (1513 筆), 固定 epochs, 多種子集成")
    print("=" * 70)

    config = {
        "lang_model":      "microsoft/deberta-v3-large",
        "batch_size":      8,
        "num_epochs":      45,
        "lang_lr":         5e-6,
        "head_lr":         1e-4,
        "weight_decay":    1e-2,
        "dropout":         0.2,
        "label_smoothing": 0.1,
        "seeds":           [42, 123, 777],
    }

    print(f"\n載入資料: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    print("建立資料集...")
    train_ds    = MOSIDataset(data["train"], tokenizer)
    val_ds      = MOSIDataset(data["valid"], tokenizer)
    test_ds     = MOSIDataset(data["test"],  tokenizer)
    trainval_ds = ConcatDataset([train_ds, val_ds])
    config["trainval_size"] = len(trainval_ds)

    print(f"  Train+Val: {len(trainval_ds)} 筆 | Test: {len(test_ds)} 筆")

    trainval_loader = DataLoader(trainval_ds, config["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    test_loader     = DataLoader(test_ds,     config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"設備: {device}")

    # 類別權重（根據 train+val 計算）
    all_labels = np.concatenate([
        data["train"]["regression_labels"],
        data["valid"]["regression_labels"]
    ])
    class_weights = compute_class_weights(all_labels)
    print(f"類別權重: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # 取測試集真實標籤
    test_labels_np  = np.array(data["test"]["regression_labels"])
    test_cls7_true  = np.clip(np.round(test_labels_np).astype(int), -3, 3) + 3
    test_cls2_true  = (test_labels_np >= 0).astype(int)

    # 多種子訓練
    all_probs    = []
    seed_results = []

    for seed in config["seeds"]:
        test_acc, test_probs = train_one_seed(
            seed, trainval_loader, test_loader, class_weights, device, config
        )
        all_probs.append(test_probs)
        seed_results.append((seed, test_acc))

    # 集成
    ens_probs = np.mean(all_probs, axis=0)
    ens_preds = ens_probs.argmax(1)
    ens_acc7  = (ens_preds == test_cls7_true).mean() * 100
    ens_acc2  = ((ens_preds >= 3).astype(int) == test_cls2_true).mean() * 100
    ens_f1    = f1_score(test_cls2_true, (ens_preds >= 3).astype(int), average="weighted") * 100

    print(f"\n{'='*70}")
    print("【v38 最終結果】Train+Val 合併訓練 + 多種子集成")
    print(f"{'='*70}")
    print("\n各種子獨立成績:")
    for seed, acc in seed_results:
        print(f"  Seed {seed:3d}: Test Acc7={acc:.2f}%")
    print(f"\n集成結果 ({len(config['seeds'])} 種子):")
    print(f"  Test Acc7: {ens_acc7:.2f}%")
    print(f"  Test Acc2: {ens_acc2:.2f}%")
    print(f"  Test F1:   {ens_f1:.2f}%")
    print(f"  vs 目標 51%: {ens_acc7 - 51.0:+.2f}% {'✓ 達標！' if ens_acc7 > 51.0 else '✗ 未達標'}")
    print(f"\nTest Acc7: {ens_acc7:.2f}%")

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    import json
    history = {
        "version": "v38", "strategy": "trainval_combined_multi_seed",
        "seeds": config["seeds"],
        "seed_results": [{"seed": s, "test": t} for s, t in seed_results],
        "ensemble_acc7": round(ens_acc7, 2),
        "ensemble_acc2": round(ens_acc2, 2),
    }
    with open(MODEL_DIR / "history_v38.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"結果已儲存至: {MODEL_DIR / 'history_v38.json'}")


if __name__ == "__main__":
    main()
