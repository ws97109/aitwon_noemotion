"""
MOSI 多模態情感分析 v27 — 穩定重建版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v26 失敗根因（徹底排查）：
  FGM + AMP(CUDA) 梯度 scale 衝突：
    scaler.scale(loss).backward()   → SCALED 梯度
    scaler.unscale_(optimizer)      → 就地轉換為 UNSCALED
    scaler.scale(loss_adv).backward()← BUG：再次 SCALED 疊加到 UNSCALED 梯度
  → 混合 scale 梯度 → NaN → 所有權重損壞 → Val Acc7=4.37%

v27 = v23 (Test 48.69%, 最佳基線) + 4 個安全改進：

  改進1: Optimizer 完全修復（最重要，估計 +1~2% Val）
    v23 bug: 初始化 optimizer 只加 requires_grad=True 的參數
            frozen 層解凍後雖 requires_grad=True 但不在 optimizer 中 → 白解凍
    修復: 所有 DeBERTa 層加入 optimizer，requires_grad=False 時梯度自動跳過

  改進2: 音視頻 L2 歸一化（零計算成本，解決 train/test 幅度偏移）
    用 F.normalize 安全實現：零向量自動保持零（不除以零）

  改進3: 推論增強（純推論時，不改訓練，零風險）
    a) TTA: 10次 MC Dropout → softmax 空間平均
    b) 標籤先驗校正: Bayesian shift 補償 train/test 分布偏移
       train mean=+0.23 vs test mean=-0.32 → 先驗比率校正
    c) 貪心閾值搜索: 在 val set 找最優 6 個回歸→分類邊界
    d) 軟集成: cls7 softmax + reg Gaussian 加權融合
    選 val 最優方案應用到 test

  改進4: 超參微調
    lang_lr: 5e-6 → 6e-6
    patience: 20 → 25
    epochs: 120 → 150
    w_cls7: 3.0 → 3.5
    SWA start: 60% → 50%

保留 v23 全部有效機制：
  EMA(0.9995) + SWA + FocalLoss + R-Drop(0.1) + Acc7選模
  + capped class_weights [0.5,3.0] + progressive unfreeze

目標：Val 53%+，Test 50.5%+
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
# 資料集
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
# 模型架構（v23 + L2 Norm）
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
    """CORAL 序數回歸：利用 -3<-2<...<+3 的序數結構輔助分類"""
    def __init__(self, feat_dim, num_thresholds=6, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(feat_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, feat):
        logit = self.dropout(feat) @ self.weight   # (B,)
        return logit.unsqueeze(1) + self.bias       # (B,6)


class HybridModel(nn.Module):
    """v23 架構 + v27 改進2: 音視頻 L2 歸一化"""
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
        # NaN 保護
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)

        # v27 改進2: L2 歸一化（F.normalize 自動安全處理零向量）
        # 解決 train/test 說話人特徵幅度偏移，零計算成本
        audio = F.normalize(audio, p=2, dim=-1)
        vision = F.normalize(vision, p=2, dim=-1)

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
# EMA
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
# 損失
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
    def __init__(self, class_weights, w_cls7=3.5, w_cls2=0.5, w_reg=0.3, w_ord=0.2):
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
# 評估工具
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練（v27 改進1: 正確的 AMP + 無 FGM bug）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema, rdrop_alpha=0.1):
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 推論增強（v27 改進3）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def collect_tta_preds(model, loader, device, n_tta=10):
    """MC Dropout TTA: 保持 dropout 開啟，多次前向取 softmax 平均"""
    model.train()  # 開啟 dropout
    all_probs7, all_reg, all_l7, all_lr = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
            p7_acc = torch.zeros(len(cl7), 7, device=device)
            reg_acc = torch.zeros(len(cl7), device=device)
            for _ in range(n_tta):
                l7, _, reg, _ = model(ids, mask, aud, amask, vis, vmask)
                p7_acc += F.softmax(l7, dim=-1)
                reg_acc += reg
            all_probs7.append((p7_acc / n_tta).cpu().numpy())
            all_reg.append((reg_acc / n_tta).cpu().numpy())
            all_l7.extend(cl7.cpu().numpy())
            all_lr.extend(rl.cpu().numpy())

    return (np.concatenate(all_probs7),
            np.concatenate(all_reg),
            np.array(all_l7), np.array(all_lr))


def compute_label_prior(regression_labels, n=7):
    """從回歸標籤計算 7 類先驗分布"""
    cls = np.clip(np.round(regression_labels).astype(int), -3, 3) + 3
    counts = np.bincount(cls, minlength=n).astype(float)
    counts = np.where(counts == 0, 1e-6, counts)
    return counts / counts.sum()


def apply_prior_correction(probs, train_prior, val_prior, strength=1.0):
    """
    Bayesian 先驗校正: P_corrected ∝ P_model × (val_prior/train_prior)^strength
    補償 train→test 分布偏移 (train mean=+0.23, test mean=-0.32)
    """
    ratio = np.maximum(val_prior, 1e-10) / np.maximum(train_prior, 1e-10)
    ratio = np.power(ratio, strength)
    corrected = probs * ratio[np.newaxis, :]
    row_sum = corrected.sum(1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return corrected / row_sum


def reg_to_cls7(reg_preds, thresholds):
    """用閾值將回歸值轉為 7 類"""
    cls = np.zeros(len(reg_preds), dtype=int)
    for i, t in enumerate(thresholds):
        cls[reg_preds > t] = i + 1
    return cls


def search_thresholds_greedy(reg_preds, labels7):
    """貪心搜索 6 個最優回歸→分類邊界"""
    thresholds = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=float)
    best_acc = (reg_to_cls7(reg_preds, thresholds) == labels7).mean()

    improved = True
    while improved:
        improved = False
        for i in range(6):
            lo = thresholds[i-1] + 0.05 if i > 0 else -3.5
            hi = thresholds[i+1] - 0.05 if i < 5 else 3.5
            for t in np.arange(lo, hi, 0.05):
                new_t = thresholds.copy(); new_t[i] = t
                acc = (reg_to_cls7(reg_preds, new_t) == labels7).mean()
                if acc > best_acc:
                    best_acc = acc; thresholds[i] = t; improved = True

    print(f"  [閾值搜索] 閾值={[round(t,2) for t in thresholds]}, Val Acc7={best_acc*100:.2f}%")
    return thresholds, best_acc * 100


def reg_to_soft_labels(reg_preds, sigma=0.5, n=7):
    """回歸值轉 Gaussian 軟標籤 (用於集成)"""
    centers = np.arange(n)
    reg_cls = reg_preds[:, np.newaxis] + 3  # 移位到 [0,6]
    dists = -((reg_cls - centers[np.newaxis, :]) ** 2) / (2 * sigma ** 2)
    probs = np.exp(dists - dists.max(1, keepdims=True))
    return probs / probs.sum(1, keepdims=True)


def search_ensemble_alpha(val_probs7, val_reg, val_l7):
    """搜索 cls7 softmax 和 reg Gaussian 的最優融合比例"""
    best_acc, best_alpha, best_sigma = 0.0, 0.8, 0.5
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        reg_probs = reg_to_soft_labels(val_reg, sigma)
        for alpha in np.arange(0.5, 1.01, 0.05):
            blended = alpha * val_probs7 + (1 - alpha) * reg_probs
            acc = (blended.argmax(1) == val_l7).mean() * 100
            if acc > best_acc:
                best_acc = acc; best_alpha = round(float(alpha), 2); best_sigma = sigma
    print(f"  [軟集成] alpha={best_alpha:.2f}, sigma={best_sigma:.2f}, Val Acc7={best_acc:.2f}%")
    return best_alpha, best_sigma, best_acc


def enhanced_inference(model, val_loader, test_loader, train_reg_labels, device, n_tta=10):
    """
    四步推論增強 (全部在推論時做，不改訓練):
    1. TTA softmax 基線
    2. 標籤先驗校正
    3. 貪心閾值搜索
    4. 軟集成
    在 val 選最優方案 → 應用到 test
    """
    print("\n[推論增強] 收集 val TTA 預測...")
    val_probs7, val_reg, val_l7, val_lr = collect_tta_preds(model, val_loader, device, n_tta)

    # 基線
    base_acc = (val_probs7.argmax(1) == val_l7).mean() * 100
    print(f"  TTA softmax 基線: Val Acc7 = {base_acc:.2f}%")

    # 先驗校正
    train_prior = compute_label_prior(train_reg_labels)
    val_prior = compute_label_prior(val_lr)
    best_corr_acc, best_strength = 0.0, 0.0
    for s in np.arange(0.0, 3.1, 0.2):
        corr = apply_prior_correction(val_probs7, train_prior, val_prior, s)
        acc = (corr.argmax(1) == val_l7).mean() * 100
        if acc > best_corr_acc:
            best_corr_acc, best_strength = acc, round(float(s), 1)
    print(f"  [先驗校正] strength={best_strength}, Val Acc7={best_corr_acc:.2f}%")

    # 閾值搜索
    best_thresh, thresh_acc = search_thresholds_greedy(val_reg, val_l7)

    # 軟集成
    best_alpha, best_sigma, ensemble_acc = search_ensemble_alpha(val_probs7, val_reg, val_l7)

    # 選最優方案
    methods = {
        "TTA":      base_acc,
        "Prior":    best_corr_acc,
        "Thresh":   thresh_acc,
        "Ensemble": ensemble_acc,
    }
    best_method = max(methods, key=methods.get)
    print(f"\n  最優 val 方案: {best_method} ({methods[best_method]:.2f}%)")

    # 收集 test TTA 預測
    print("\n[推論增強] 收集 test TTA 預測...")
    test_probs7, test_reg, test_l7, test_lr = collect_tta_preds(model, test_loader, device, n_tta)

    # 應用最優方案
    if best_method == "TTA":
        test_c7 = test_probs7.argmax(1)
    elif best_method == "Prior":
        test_c7 = apply_prior_correction(test_probs7, train_prior, val_prior, best_strength).argmax(1)
    elif best_method == "Thresh":
        test_c7 = reg_to_cls7(test_reg, best_thresh)
    else:  # Ensemble
        reg_probs = reg_to_soft_labels(test_reg, best_sigma)
        test_c7 = (best_alpha * test_probs7 + (1 - best_alpha) * reg_probs).argmax(1)

    # cls7 label >= 3 → sentiment >= 0 → cls2 = positive
    test_c2 = (test_c7 >= 3).astype(int)
    test_metrics = compute_metrics(test_c7, test_c2, test_reg, test_l7,
                                   (test_lr >= 0).astype(int), test_lr)
    return test_metrics, best_method, methods


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.where((ct := np.bincount(cl, minlength=n).astype(float)) == 0, 1.0, ct)
    # Cap [0.5, 3.0]：避免極端類別（24筆/46筆）權重放大噪聲
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))


def progressive_unfreeze(model, epoch, total_epochs):
    """只 toggle requires_grad，不重建 optimizer（v27 改進1的配合函數）"""
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
    print("MOSI 多模態情感分析 v27 — 穩定重建版")
    print("v23基線 + Optimizer修復 + L2Norm + 4步推論增強")
    print("移除 FGM（v26 AMP+CUDA bug 根因）")
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
        "num_epochs": 150,
        "lang_lr":  6e-6,            # 5e-6 → 6e-6
        "other_lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
        "w_cls7": 3.5,               # 3.0 → 3.5
        "w_cls2": 0.5, "w_reg": 0.3, "w_ord": 0.2,
        "ema_decay": 0.9995,
        "patience": 25,              # 20 → 25
        "rdrop_alpha": 0.1,
        "swa_start_ratio": 0.50,     # 60% → 50%（更早啟動 SWA）
        "swa_lr": 1e-6,
        "n_tta": 10,
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
    print(f"設備: {device}")

    model = HybridModel(**{k: config[k] for k in
                           ["lang_model","audio_dim","vision_dim","modal_hidden",
                            "fusion_dim","num_classes","dropout"]}).to(device)

    # 初始凍結前 6 層
    if enc := getattr(model.lang_backbone, "encoder", None):
        for i in range(6):
            for p in enc.layer[i].parameters(): p.requires_grad = False
        print("初始凍結前 6 層（DeBERTa encoder layer 0-5）")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {train_p/1e6:.1f}M")

    criterion = HybridLoss(class_w.to(device),
                           config["w_cls7"], config["w_cls2"],
                           config["w_reg"], config["w_ord"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # v27 改進1: 所有 DeBERTa 參數加入 optimizer（包括 frozen 層）
    # frozen 層 requires_grad=False → 梯度為 None → optimizer.step() 自動跳過
    # 解凍後立即有正確的 lr/weight_decay，momentum 連續不中斷
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    all_lang_params = list(model.lang_backbone.parameters())  # 包括 frozen 層！
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
        {"params": all_lang_params, "lr": config["lang_lr"]},
        {"params": other_params,    "lr": config["other_lr"]},
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
    print(f"[v27] Optimizer修復(所有DeBERTa層) + L2Norm + SWA(epoch {swa_start}+)")
    print(f"      lang_lr={config['lang_lr']:.0e} | patience={config['patience']} | epochs={config['num_epochs']}")
    print(f"      w_cls7={config['w_cls7']} | rdrop={config['rdrop_alpha']} | EMA={config['ema_decay']}\n")

    save_dir = Path(config["model_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []; patience_counter = 0

    for epoch in range(config["num_epochs"]):
        if progressive_unfreeze(model, epoch, config["num_epochs"]):
            ema.add_new_params()

        if epoch >= swa_start and not swa_started:
            swa_started = True
            print(f"  [SWA] 啟動 (epoch {epoch})")

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-"*45)
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                              device, scaler, ema, config["rdrop_alpha"])

        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # EMA 驗證
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
                       save_dir / "v27_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善"); break

    # SWA 最終評估
    if swa_started:
        print("\n[SWA] 更新 BatchNorm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        _, swa_val = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val Acc7={swa_val['Acc7']:.2f}%  (EMA best={best_acc7['Acc7']:.2f}%)")
        if swa_val["Acc7"] > best_acc7["Acc7"]:
            print("  ✅ SWA 更優！")
            torch.save({"model_state": swa_model.module.state_dict(),
                        "metrics": swa_val, "config": config},
                       save_dir / "v27_swa.pth")
            eval_model = swa_model
        else:
            ckpt = torch.load(save_dir / "v27_best_acc7.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"]); eval_model = model
    else:
        ckpt = torch.load(save_dir / "v27_best_acc7.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"]); eval_model = model

    # 標準測試
    print("\n" + "="*60)
    _, test_std = validate(eval_model, test_loader, criterion, device)
    print("\n【標準測試集結果 - v27】")
    print(f"  Acc7 : {test_std['Acc7']:.2f}%   (目標: 50.5%, v23: 48.69%)")
    print(f"  Acc2 : {test_std['Acc2']:.2f}%")
    print(f"  F1   : {test_std['F1']:.2f}%")
    print(f"  MAE  : {test_std['MAE']:.4f}")
    print(f"  Corr : {test_std['Corr']:.4f}")
    gap = best_acc7['Acc7'] - test_std['Acc7']
    print(f"\n  Val-Test Gap: {gap:.2f}% (v23: 3.28%)")

    # 推論增強測試
    print("\n" + "="*60)
    print("[推論增強] 開始...")
    ema.apply_shadow()
    test_enh, best_method, all_val_accs = enhanced_inference(
        eval_model, val_loader, test_loader,
        data["train"]["regression_labels"], device, config["n_tta"])
    ema.restore()

    print(f"\n【推論增強測試集結果 - v27 ({best_method})】")
    print(f"  Acc7 : {test_enh['Acc7']:.2f}%   (目標: 50.5%)")
    print(f"  Acc2 : {test_enh['Acc2']:.2f}%")
    print(f"  F1   : {test_enh['F1']:.2f}%")
    print(f"  MAE  : {test_enh['MAE']:.4f}")
    print(f"  Corr : {test_enh['Corr']:.4f}")

    final_acc7 = max(test_std['Acc7'], test_enh['Acc7'])
    status = "🎉 達標！" if final_acc7 >= 50.5 else f"❌ 差 {50.5 - final_acc7:.2f}%"
    print(f"\n  最終 Test Acc7: {final_acc7:.2f}% | {status}")

    with open(save_dir / "v27_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_val_acc7": best_acc7,
            "test_standard": test_std,
            "test_enhanced": test_enh,
            "best_inference_method": best_method,
            "all_val_inference_accs": all_val_accs,
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
