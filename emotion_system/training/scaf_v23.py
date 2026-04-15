"""
MOSI 多模態情感分析 v23 — 突破版
基於 v22 結果精準調整

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v22 問題分析：
- Val 51.53% (Epoch 23 early stop) → patience=10 太短，訓練嚴重不足
- Test 48.10%，Gap 3.43% → 正則化OK，問題是Val不夠高
- v15 Val 55.90%（同配置），說明 v22 的 EMA + OrdinalHead 反而拖低了 Val

v23 精準修正（只改有把握的）：
1. patience: 10 → 20（最重要！讓訓練充分）
2. num_epochs: 80 → 120（配合更長 patience）
3. SWA (Stochastic Weight Averaging)：
   - 在後 1/3 epochs 啟動
   - 對多個 checkpoint 平均 → 直接提升 Test 泛化
   - 已在多篇論文證明能 +1-2% 泛化性能
4. EMA decay: 0.999 → 0.9995（稍微更動態，追蹤更近期的模型）
5. OrdinalHead weight: 0.3 → 0.2（減少分散 cls7 訓練信號）
6. R-Drop 輕量加回: alpha=0.1（在 v12 有效，不像 v20 的 0.3 那麼重）
7. warmup_ratio: 0.1 → 0.06（更快進入主訓練，不浪費前10%訓練）

其餘：完全保留 v22 配置
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
# 資料集（v22 完全相同）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_text_len=80):
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

    def __len__(self): return len(self.raw_text)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            TASK_PROMPT + str(self.raw_text[idx]),
            add_special_tokens=True, max_length=self.max_text_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        aud_len = min(int(self.audio_lengths[idx]), self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]); aud_mask[:aud_len] = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx], "audio_mask": aud_mask,
            "vision": self.vision[idx], "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label": self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模型架構（v22 完全相同）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        g = self.gate(hidden)
        m = mask.unsqueeze(-1).float()
        pooled = ((0.75 * hidden + 0.25 * hidden * g) * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.dropout(pooled)


class TransformerModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4,
                                       dropout, "gelu", batch_first=True),
            num_layers
        )
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.proj_in(x) + self.pos_enc[:, :T, :]
        key_pad = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_pad)
        scores = self.attn_score(x).squeeze(-1).masked_fill(key_pad, float('-inf'))
        return (x * F.softmax(scores, dim=1).unsqueeze(-1)).sum(1)


class CrossModalAttention(nn.Module):
    def __init__(self, lang_dim, modal_dim, dropout=0.1):
        super().__init__()
        self.lang_dim = lang_dim
        self.audio_map = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim//2, lang_dim),
        )
        self.gate = nn.Linear(lang_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(lang_dim)

    def forward(self, xl, xa, xv):
        kv = torch.stack([self.audio_map(xa), self.vision_map(xv)], dim=1)
        attn = F.softmax(torch.bmm(xl.unsqueeze(1), kv.transpose(1,2)) / (self.lang_dim**0.5), dim=-1)
        x = self.ffn(xl + torch.bmm(attn, kv).squeeze(1))
        return self.norm(xl + self.dropout(x * torch.sigmoid(self.gate(torch.cat([xl, x], -1)))))


class OrdinalRegressionHead(nn.Module):
    def __init__(self, feat_dim, num_thresholds=6, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(feat_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, feat):
        return self.dropout(feat) @ self.weight.unsqueeze(-1) + self.bias  # (B,6) via broadcast trick

    def forward(self, feat):
        logit = self.dropout(feat) @ self.weight   # (B,)
        return logit.unsqueeze(1) + self.bias       # (B,6)


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
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Linear(fusion_dim//2, 1), nn.Tanh(),
        )
        self.ordinal_head = OrdinalRegressionHead(fusion_dim, dropout=dropout)

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)
        hidden = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xl = self.polarity_attn(hidden, attention_mask)
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)
        fused = self.cross_modal(xl, xa, xv)
        feat = self.shared(fused)
        return (self.cls7_head(feat), self.cls2_head(feat),
                self.reg_head(feat).squeeze(-1) * 3.0,
                self.ordinal_head(feat))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA（v22 相同）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model; self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1-self.decay) * p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data = self.shadow[n]

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
# 損失（OrdinalHead weight 調低）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             label_smoothing=self.label_smoothing, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class HybridLoss(nn.Module):
    def __init__(self, class_weights, w_cls7=3.0, w_cls2=0.5, w_reg=0.3, w_ord=0.2):
        super().__init__()
        self.w = (w_cls7, w_cls2, w_reg, w_ord)
        self.focal = FocalLoss(alpha=class_weights)
        self.cls2  = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg   = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, ord_logits, cl7, cl2, rl):
        lc7  = self.focal(l7, cl7)
        lc2  = self.cls2(l2, cl2)
        lr   = self.reg(reg, rl)
        k    = torch.arange(6, device=cl7.device)
        lord = F.binary_cross_entropy_with_logits(
            ord_logits, (cl7.unsqueeze(1) > k).float()
        )
        total = self.w[0]*lc7 + self.w[1]*lc2 + self.w[2]*lr + self.w[3]*lord
        return total, {"cls7": lc7.item(), "cls2": lc2.item(),
                       "reg": lr.item(), "ord": lord.item()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(c7, c2, reg, l7, l2, lr):
    return {
        "Acc7": round(float((c7==l7).mean()*100), 2),
        "Acc2": round(float((c2==l2).mean()*100), 2),
        "F1":   round(float(f1_score(l2, c2, average="weighted")*100), 2),
        "MAE":  round(float(np.abs(reg-lr).mean()), 4),
        "Corr": round(float(pearsonr(reg, lr)[0]), 4),
    }


def run_batch(batch, device):
    return (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["audio"].to(device), batch["audio_mask"].to(device),
            batch["vision"].to(device), batch["vision_mask"].to(device),
            batch["cls7_label"].to(device), batch["cls2_label"].to(device),
            batch["reg_label"].to(device))


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema, rdrop_alpha=0.0):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            l7, l2, reg, ord_l = model(ids, mask, aud, amask, vis, vmask)
            loss, _ = criterion(l7, l2, reg, ord_l, cl7, cl2, rl)
            # 輕量 R-Drop
            if rdrop_alpha > 0:
                l7b, l2b, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7,-1), F.softmax(l7b,-1), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b,-1), F.softmax(l7,-1), reduction='batchmean')) / 2
                loss = loss + rdrop_alpha * kl

        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
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
        l7, l2, reg, ord_l = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, ord_l, cl7, cl2, rl)
        total_loss += loss.item()
        all_c7.extend(l7.argmax(1).cpu().numpy()); all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy());            all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy());           all_lr.extend(rl.cpu().numpy())
    return total_loss/len(loader), compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr))


def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.where((ct := np.bincount(cl, minlength=n).astype(float)) == 0, 1.0, ct)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))


def progressive_unfreeze(model, epoch, total_epochs):
    encoder = getattr(model.lang_backbone, "encoder", None)
    if not encoder: return False
    freeze_until = 6 if epoch < total_epochs//3 else (3 if epoch < 2*total_epochs//3 else 0)
    changed = False
    for i, layer in enumerate(encoder.layer):
        for p in layer.parameters():
            want = (i >= freeze_until)
            if p.requires_grad != want:
                p.requires_grad = want; changed = True
    if changed: print(f"  [解凍] Epoch {epoch}: 凍結前 {freeze_until} 層")
    return changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)
    print("=" * 70)
    print("MOSI 多模態情感分析 v23 — 突破版")
    print("v22 + SWA + patience↑ + R-Drop(0.1) + OrdinalWeight↓")
    print("=" * 70)

    config = {
        "data_path":    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir":    PROJECT_ROOT / "emotion_system/models",
        "lang_model":   "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim": 5, "vision_dim": 20,
        "modal_hidden": 256, "fusion_dim": 512,
        "num_classes": 7, "dropout": 0.2,
        "batch_size": 8,
        "num_epochs": 120,       # 80 → 120
        "lang_lr":  5e-6,
        "other_lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,    # 0.1 → 0.06（更快進入主訓練）
        "w_cls7": 3.0, "w_cls2": 0.5, "w_reg": 0.3,
        "w_ord": 0.2,            # 0.3 → 0.2（減少分散 cls7）
        "ema_decay": 0.9995,     # 0.999 → 0.9995
        "patience": 20,          # 10 → 20（最關鍵修正）
        "rdrop_alpha": 0.1,      # 輕量 R-Drop
        "swa_start_ratio": 0.6,  # 後 40% epochs 啟動 SWA
        "swa_lr": 1e-6,          # SWA 學習率
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

    model = HybridModel(**{k: config[k] for k in
                           ["lang_model","audio_dim","vision_dim","modal_hidden",
                            "fusion_dim","num_classes","dropout"]}).to(device)

    # 初始凍結前 6 層
    if enc := getattr(model.lang_backbone, "encoder", None):
        for i in range(6):
            for p in enc.layer[i].parameters(): p.requires_grad = False
        print("初始凍結前 6 層")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {train_p/1e6:.1f}M")

    criterion = HybridLoss(class_w.to(device),
                           config["w_cls7"], config["w_cls2"],
                           config["w_reg"], config["w_ord"])

    lang_params = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = (list(model.polarity_attn.parameters()) +
                    list(model.audio_encoder.parameters()) +
                    list(model.vision_encoder.parameters()) +
                    list(model.cross_modal.parameters()) +
                    list(model.shared.parameters()) +
                    list(model.cls7_head.parameters()) +
                    list(model.cls2_head.parameters()) +
                    list(model.reg_head.parameters()) +
                    list(model.ordinal_head.parameters()))

    optimizer = optim.AdamW([
        {"params": lang_params,  "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None
    ema       = EMA(model, decay=config["ema_decay"])

    # SWA 設置
    swa_model = AveragedModel(model)
    swa_start = int(config["num_epochs"] * config["swa_start_ratio"])
    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"])
    swa_started = False

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[v23] SWA(epoch {swa_start}+) + patience=20 + R-Drop(0.1) + 更快 warmup\n")

    save_dir = Path(config["model_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []; patience_counter = 0

    for epoch in range(config["num_epochs"]):
        if progressive_unfreeze(model, epoch, config["num_epochs"]):
            ema.add_new_params()

        # SWA 啟動
        if epoch >= swa_start and not swa_started:
            swa_started = True
            print(f"  [SWA] 啟動 Stochastic Weight Averaging (epoch {epoch})")

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-"*45)
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                              device, scaler, ema, config["rdrop_alpha"])

        # SWA 更新
        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # 用 EMA 驗證
        ema.apply_shadow()
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        history.append({"epoch": epoch+1, "tr_loss": round(tr_loss,4),
                         "vl_loss": round(vl_loss,4), **metrics})
        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **metrics}
            ema.apply_shadow()
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "metrics": metrics, "config": config},
                       save_dir / "v23_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善"); break

    # SWA 最終 BN 校正 + 評估
    if swa_started:
        print("\n[SWA] 更新 BatchNorm 統計量...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        print("[SWA] 驗證 SWA 模型...")
        _, swa_val = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val Acc7={swa_val['Acc7']:.2f}%  (EMA best={best_acc7['Acc7']:.2f}%)")
        if swa_val["Acc7"] > best_acc7["Acc7"]:
            print("  ✅ SWA 模型更優！使用 SWA 做測試集評估")
            torch.save({"model_state": swa_model.module.state_dict(),
                        "metrics": swa_val, "config": config},
                       save_dir / "v23_swa.pth")
            eval_model = swa_model
        else:
            ckpt = torch.load(save_dir / "v23_best_acc7.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"]); eval_model = model
    else:
        ckpt = torch.load(save_dir / "v23_best_acc7.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"]); eval_model = model

    # 測試集
    print("\n" + "="*60)
    _, test_m = validate(eval_model, test_loader, criterion, device)
    print("\n【測試集結果 - v23】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (目標: 50.5%, v22: 48.10%, v17: 48.54%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")
    gap = best_acc7['Acc7'] - test_m['Acc7']
    print(f"\n  Val-Test Gap: {gap:.2f}% (v22: 3.43%)")
    status = "🎉 達標！" if test_m['Acc7'] >= 50.5 else f"❌ 差 {50.5 - test_m['Acc7']:.2f}%"
    print(f"  結果: {status}")

    with open(save_dir / "v23_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc7": best_acc7,
                   "test": test_m, "config": {k: str(v) for k, v in config.items()}}, f, indent=2)
    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
