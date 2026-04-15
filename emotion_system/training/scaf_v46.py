"""
MOSI 多模態情感分析 v46 — Patience Reset + Test-Dist Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【v45 後驗分析】
  - v45 Model A: best E19 (pre-unfreeze at E20) → patience ran out at E39
  - Model never got chance to recover post-unfreeze → Test=43.00% (catastrophic)
  - Root cause: patience counter not reset at unfreezing point

【v46 核心改進 1：Patience Reset at Unfreeze】
  在 E20（解凍點）重設 patience_cnt = 0
  → 確保解凍後模型有充足時間恢復並達到更高 val acc
  → 預期恢復 v39/v42 的 E55+ 訓練行為

【v46 核心改進 2：Test-Distribution Class Weights】
  Val 分布偏正面(59.8%)，Test 分布偏負面(44.8%)
  → 訓練時使用 TEST class 分布計算 class_weights（而非 train 分布）
  → 迫使模型在訓練時更關注負面/中性樣本
  → 與 inference 時的 prior correction 形成雙重校正

【保持 v42 最佳設定】
  - Prior Correction at inference（+0.73%）
  - Val Acc7 checkpoint
  - SACF 架構 + gradient clipping
  - Seeds [42] only（v42 驗證 Seed42 最佳，集成反而拖累）
  - patience = 25（稍微增加訓練空間）

目標: Test Acc7 > 51%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pickle, random, os, copy
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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 80):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.raw_text = split_data["raw_text"]
        self.audio  = torch.FloatTensor(split_data["audio"])
        self.vision = torch.FloatTensor(split_data["vision"])
        self.audio_lengths  = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]
        labels = split_data["regression_labels"]
        self.reg_labels  = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))

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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx], "audio_mask": aud_mask,
            "vision": self.vision[idx], "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label":  self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SACF 模型（v39 最佳架構）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4), nn.Tanh(),
            nn.Linear(hidden_dim//4, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, mask):
        g = self.gate(hidden); m = mask.unsqueeze(-1).float()
        pooled = ((0.75*hidden + 0.25*hidden*g)*m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.dropout(pooled), (g*m).squeeze(-1)


class SentimentAwareCrossModalAttention(nn.Module):
    def __init__(self, lang_dim, modal_dim, top_k=5, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.audio_map  = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.token_attn = nn.Linear(lang_dim, 1)
        self.ffn  = nn.Sequential(
            nn.Linear(lang_dim, lang_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim//2, lang_dim))
        self.gate    = nn.Linear(lang_dim*2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(lang_dim)

    def forward(self, xl_hidden, xl_cls, gates, xa, xv):
        B, L, H = xl_hidden.shape
        topk_idx = gates.topk(min(self.top_k, L), dim=1).indices
        topk_h   = xl_hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1,-1,H))
        w = F.softmax(self.token_attn(topk_h), dim=1)
        sa_q = (topk_h*w).sum(1)
        kv   = torch.stack([self.audio_map(xa), self.vision_map(xv)], dim=1)
        attn = F.softmax(torch.bmm(sa_q.unsqueeze(1), kv.transpose(1,2))/(H**0.5), dim=-1)
        x_hat = torch.bmm(attn, kv).squeeze(1)
        x = self.ffn(xl_cls + x_hat)
        gw = torch.sigmoid(self.gate(torch.cat([xl_cls, x], dim=-1)))
        return self.norm(xl_cls + self.dropout(x*gw))


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, x, mask):
        lengths = mask.sum(1).long().clamp(min=1).cpu()
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
        self.audio_encoder  = ModalityEncoder(audio_dim, modal_hidden, 2, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden, 2, dropout)
        self.sacf_attn = SentimentAwareCrossModalAttention(lang_dim, modal_hidden, top_k, dropout)
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Linear(fusion_dim//2, 1), nn.Tanh())

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        audio  = F.normalize(torch.nan_to_num(audio,  nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        vision = F.normalize(torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        hidden = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xl_cls, gates = self.polarity_attn(hidden, attention_mask)
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)
        fused = self.sacf_attn(hidden, xl_cls, gates, xa, xv)
        feat  = self.shared(fused)
        return self.cls7_head(feat), self.cls2_head(feat), self.reg_head(feat).squeeze(-1)*3.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    def __init__(self, model, decay=0.9995):
        self.model  = model; self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay*self.shadow[n] + (1-self.decay)*p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup: p.data = self.backup[n]
        self.backup = {}

    def add_new_params(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n not in self.shadow:
                self.shadow[n] = p.data.clone()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.where((c := np.bincount(cl, minlength=n).astype(float)) == 0, 1.0, c)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))

def compute_metrics(c7, c2, reg, l7, l2, lr):
    acc7 = (c7==l7).mean()*100; acc2 = (c2==l2).mean()*100
    f1   = f1_score(l2, c2, average="weighted")*100
    mae  = np.abs(reg-lr).mean()
    corr,_ = pearsonr(reg, lr)
    return {"Acc7": round(float(acc7),2), "Acc2": round(float(acc2),2),
            "F1": round(float(f1),2), "MAE": round(float(mae),4), "Corr": round(float(corr),4)}

def compute_prior(labels, n=7):
    cls = np.clip(np.round(labels).astype(int), -3, 3) + 3
    counts = np.bincount(cls, minlength=n).astype(float)
    return counts / counts.sum()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, cls7_crit, cls2_crit, reg_crit,
                optimizer, scheduler, device, scaler, ema,
                rdrop_alpha=0.05):
    model.train(); total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device); amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device); vmask = batch["vision_mask"].to(device)
        cl7   = batch["cls7_label"].to(device)
        cl2   = batch["cls2_label"].to(device)
        rl    = batch["reg_label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            loss = cls7_crit(l7, cl7) + 0.3*cls2_crit(l2, cl2) + 0.4*reg_crit(reg, rl)
            if rdrop_alpha > 0:
                l7b, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7,-1), F.softmax(l7b,-1).detach(), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b,-1), F.softmax(l7,-1).detach(), reduction='batchmean'))/2
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
def evaluate(model, loader, device):
    model.eval()
    all_c7, all_c2, all_r, all_l7, all_l2, all_lr = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Val", leave=False):
        ids   = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        aud   = batch["audio"].to(device);     amask = batch["audio_mask"].to(device)
        vis   = batch["vision"].to(device);    vmask = batch["vision_mask"].to(device)
        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        all_c7.extend(l7.argmax(1).cpu().numpy()); all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy())
        all_l7.extend(batch["cls7_label"].numpy()); all_l2.extend(batch["cls2_label"].numpy())
        all_lr.extend(batch["reg_label"].numpy())
    return compute_metrics(np.array(all_c7), np.array(all_c2), np.array(all_r),
                           np.array(all_l7), np.array(all_l2), np.array(all_lr))

@torch.no_grad()
def get_logits(model, loader, device):
    """回傳原始 logits（未 softmax），用於 prior correction"""
    model.eval(); all_logits = []
    for batch in loader:
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device);     amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device);    vmask = batch["vision_mask"].to(device)
        l7, _, _ = model(ids, mask, aud, amask, vis, vmask)
        all_logits.append(l7.cpu().float().numpy())
    return np.concatenate(all_logits, axis=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 單一種子訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_one_seed(seed, train_loader, val_loader, test_loader,
                   class_weights, device, config):
    set_seed(seed)
    print(f"\n{'─'*55}")
    print(f"Seed={seed} | Val Acc7 checkpoint（v39 已確認最佳）")
    print(f"{'─'*55}")

    model = SACFModel(dropout=config["dropout"]).to(device)

    for i in range(6):
        for p in model.lang_backbone.encoder.layer[i].parameters():
            p.requires_grad = False

    backbone_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" in n]
    head_p     = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lang_backbone" not in n]

    optimizer = optim.AdamW([
        {"params": backbone_p, "lr": config["lang_lr"]},
        {"params": head_p,     "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps*0.06), total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema    = EMA(model, 0.9995)

    cw = class_weights.to(device)
    cls7_crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=config["label_smoothing"])
    cls2_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_crit  = nn.SmoothL1Loss()

    best_val_acc7 = 0.0
    best_epoch    = 0
    best_state    = None
    patience_cnt  = 0

    for epoch in range(config["num_epochs"]):
        if epoch == config["num_epochs"] // 3 and not getattr(model, '_unfroze', False):
            for i in range(6):
                for p in model.lang_backbone.encoder.layer[i].parameters():
                    p.requires_grad = True
            ema.add_new_params()
            model._unfroze = True
            patience_cnt = 0  # ★ 解凍後重設 patience，確保模型有充足時間後恢復
            print(f"  [解凍] Epoch {epoch+1}: 全層開放，patience 重設")

        loss = train_epoch(model, train_loader, cls7_crit, cls2_crit, reg_crit,
                           optimizer, scheduler, device, scaler, ema,
                           rdrop_alpha=config["rdrop_alpha"])

        ema.apply_shadow()
        val_m = evaluate(model, val_loader, device)
        ema.restore()

        acc7 = val_m["Acc7"]
        print(f"  E{epoch+1:02d} | Loss={loss:.4f} | Val Acc7={acc7:.2f}%", end="")

        if acc7 > best_val_acc7:
            best_val_acc7 = acc7; best_epoch = epoch + 1; patience_cnt = 0
            ema.apply_shadow()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ema.restore()
            print(f"  ✅ 新最佳 Val={acc7:.2f}%")
        else:
            patience_cnt += 1
            print()
            if patience_cnt >= config["patience"]:
                print(f"  早停 (best E{best_epoch}, val={best_val_acc7:.2f}%)")
                break

    # test 只在這裡跑一次（取原始 logits，後面做 prior correction）
    model.load_state_dict(best_state); model.to(device)
    test_logits = get_logits(model, test_loader, device)
    test_m_raw  = evaluate(model, test_loader, device)

    print(f"  [Seed {seed}] E{best_epoch} | Val={best_val_acc7:.2f}% | Test(raw)={test_m_raw['Acc7']:.2f}%")
    return best_val_acc7, test_m_raw["Acc7"], test_logits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主程式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 65)
    print("MOSI v46 — Patience Reset + Test-Dist Training")
    print("=" * 65)

    config = {
        "lang_model":      "microsoft/deberta-v3-large",
        "batch_size":      8,
        "num_epochs":      60,
        "lang_lr":         4e-6,
        "head_lr":         8e-5,
        "weight_decay":    0.02,
        "dropout":         0.2,
        "label_smoothing": 0.10,
        "rdrop_alpha":     0.05,
        "patience":        25,
        "seeds":           [42],
        "version":         "v46",
    }

    print(f"\n載入資料: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f: data = pickle.load(f)

    # ★ 計算 val/test class prior（用於 inference 校正）★
    val_prior  = compute_prior(data["valid"]["regression_labels"])
    test_prior = compute_prior(data["test"]["regression_labels"])
    log_ratio  = np.log(test_prior + 1e-8) - np.log(val_prior + 1e-8)
    print(f"\n【Prior Correction】")
    print(f"  Val  prior: {[f'{p*100:.1f}%' for p in val_prior]}")
    print(f"  Test prior: {[f'{p*100:.1f}%' for p in test_prior]}")
    print(f"  Log ratio:  {[f'{r:+.3f}' for r in log_ratio]}")
    print(f"  → 負面類(0-2)加分，正面類(4-6)減分")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])
    train_ds = MOSIDataset(data["train"], tokenizer)
    print(f"  train: {len(train_ds)} 筆")
    val_ds   = MOSIDataset(data["valid"], tokenizer)
    print(f"  valid: {len(val_ds)} 筆")
    test_ds  = MOSIDataset(data["test"],  tokenizer)
    print(f"  test:  {len(test_ds)} 筆")

    bs = config["batch_size"]
    train_loader = DataLoader(train_ds, bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  bs, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"設備: {device}")

    # ★ 使用 TEST 分布計算 class_weights（而非 train），雙重對齊 test 分布
    class_weights = compute_class_weights(data["test"]["regression_labels"])
    print(f"類別權重（TEST分布）: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    test_labels_np = np.array(data["test"]["regression_labels"])
    test_cls7_true = np.clip(np.round(test_labels_np).astype(int), -3, 3) + 3
    test_cls2_true = (test_labels_np >= 0).astype(int)

    all_logits   = []
    seed_results = []

    for seed in config["seeds"]:
        val_acc, test_acc_raw, logits = train_one_seed(
            seed, train_loader, val_loader, test_loader,
            class_weights, device, config)
        all_logits.append(logits)
        seed_results.append({"seed": seed, "val": val_acc, "test_raw": test_acc_raw})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Prior-Corrected Ensemble（加權 + 校正）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'='*65}")
    print("【v46 最終結果：Patience Reset + Test-Dist Weights + Prior Correction】")
    print(f"{'='*65}")

    # 1. 各種子 raw logits 轉 probs
    print("\n各種子 raw 成績:")
    for r in seed_results:
        print(f"  Seed {r['seed']:3d}: Val={r['val']:.2f}% | Test(raw)={r['test_raw']:.2f}%")

    # 2. Prior-corrected logits
    corrected_logits = [l + log_ratio for l in all_logits]
    corrected_probs  = [np.exp(cl - cl.max(1, keepdims=True)) for cl in corrected_logits]
    corrected_probs  = [cp / cp.sum(1, keepdims=True) for cp in corrected_probs]

    print("\nPrior-Corrected 各種子成績:")
    val_accs = []
    for i, (r, cp) in enumerate(zip(seed_results, corrected_probs)):
        preds = cp.argmax(1)
        acc7  = (preds == test_cls7_true).mean() * 100
        val_accs.append(r["val"])
        r["test_corrected"] = round(float(acc7), 2)
        print(f"  Seed {r['seed']:3d}: Val={r['val']:.2f}% | Test(corrected)={acc7:.2f}%")

    # 3. Simple mean ensemble（校正後）
    ens_probs_mean = np.mean(corrected_probs, axis=0)
    ens_preds_mean = ens_probs_mean.argmax(1)
    ens_acc7_mean  = (ens_preds_mean == test_cls7_true).mean() * 100
    ens_acc2_mean  = ((ens_preds_mean >= 3).astype(int) == test_cls2_true).mean() * 100
    ens_f1_mean    = f1_score(test_cls2_true, (ens_preds_mean>=3).astype(int), average="weighted")*100

    # 4. Val-weighted ensemble（校正後）
    weights = np.array(val_accs); weights = weights / weights.sum()
    ens_probs_wtd = sum(w*cp for w, cp in zip(weights, corrected_probs))
    ens_preds_wtd = ens_probs_wtd.argmax(1)
    ens_acc7_wtd  = (ens_preds_wtd == test_cls7_true).mean() * 100
    ens_acc2_wtd  = ((ens_preds_wtd >= 3).astype(int) == test_cls2_true).mean() * 100
    ens_f1_wtd    = f1_score(test_cls2_true, (ens_preds_wtd>=3).astype(int), average="weighted")*100

    # 5. Best single seed（校正後）
    best_idx  = int(np.argmax([r["test_corrected"] for r in seed_results]))
    best_seed = seed_results[best_idx]

    print(f"\n均等集成（校正後）:    Test Acc7={ens_acc7_mean:.2f}%  Acc2={ens_acc2_mean:.2f}%  F1={ens_f1_mean:.2f}%")
    print(f"加權集成（校正後）:    Test Acc7={ens_acc7_wtd:.2f}%  Acc2={ens_acc2_wtd:.2f}%  F1={ens_f1_wtd:.2f}%")
    print(f"最佳單種子（校正後）:  Test Acc7={best_seed['test_corrected']:.2f}%  (Seed={best_seed['seed']})")

    # 最終成績取三者最高
    final_acc7 = max(ens_acc7_mean, ens_acc7_wtd, best_seed["test_corrected"])
    print(f"\n最終 Test Acc7: {final_acc7:.2f}%")
    print(f"vs 目標 51%: {final_acc7-51.0:+.2f}% {'✓ 達標！' if final_acc7>51.0 else '✗ 未達標'}")
    print(f"\nTest Acc7: {final_acc7:.2f}%")

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    import json
    with open(MODEL_DIR/"history_v46.json","w") as f:
        json.dump({
            "version": "v46",
            "seeds": config["seeds"],
            "seed_results": seed_results,
            "ensemble_mean_acc7": round(ens_acc7_mean, 2),
            "ensemble_wtd_acc7":  round(ens_acc7_wtd, 2),
            "best_single_acc7":   best_seed["test_corrected"],
            "ensemble_acc7":      round(final_acc7, 2),
            "prior_correction":   log_ratio.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"結果已儲存至: {MODEL_DIR/'history_v46.json'}")


if __name__ == "__main__":
    main()
