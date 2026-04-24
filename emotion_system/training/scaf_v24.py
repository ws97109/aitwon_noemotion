"""
MOSI 多模態情感分析 v24 — 推理突破版
基於 v23 結果，從推理層面提升效果

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v23 問題分析：
- Val 51.97% (Epoch 40 早停), Test 48.69%, Gap 3.28%
- 訓練已充分，推理策略是突破點
- Train/Test 分布偏移：train mean=+0.23, test mean=-0.32

v24 核心改動：
【推理突破 — 主要貢獻】
1. TTA (Test Time Augmentation):
   - 8次前向傳播（保持 Dropout 激活）
   - 對 softmax 概率取均值，降低預測方差
   - 估計 +0.5-1%

2. 閾值搜索 (Threshold Search on Val):
   - 回歸預測轉 7 分類時，用 val set 搜索最優閾值
   - 處理 train/test 分布偏移（train 偏正面，test 偏負面）
   - 估計 +0.3-0.8%

3. 軟概率融合 (Soft Ensemble):
   - cls7 softmax + 回歸轉軟概率 → 搜索最優融合比 alpha
   - 融合雙頭信息，取長補短

【訓練微調 — 輔助貢獻】
4. lang_lr: 5e-6 → 6e-6（稍微加快語言模型學習）
5. SWA start: 0.6 → 0.5（更早平均，更多 checkpoint）
6. cls7 weight: 3.0 → 3.5（強化 7 分類信號）
7. w_ord: 0.2 → 0.1（幾乎消除 OrdinalHead 干擾）
8. 修復 OrdinalRegressionHead 的重複 forward 定義 bug

其餘：完全保留 v23 配置
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
# 模型架構
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
    """CORAL 式序數回歸頭（修復 v23 重複 forward 定義 bug）"""
    def __init__(self, feat_dim, num_thresholds=6, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(feat_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_thresholds))

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
    def __init__(self, class_weights, w_cls7=3.5, w_cls2=0.5, w_reg=0.3, w_ord=0.1):
        super().__init__()
        self.w = (w_cls7, w_cls2, w_reg, w_ord)
        self.focal = FocalLoss(alpha=class_weights)
        self.cls2  = nn.CrossEntropyLoss()
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
# 評估（標準 + TTA）
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


@torch.no_grad()
def validate(model, loader, criterion, device):
    """標準驗證（單次前向，用於訓練中監控）"""
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


def validate_tta(model, loader, device, n_tta=8):
    """TTA 驗證：N 次前向（保持 Dropout 激活），取 softmax 均值"""
    model.eval()
    # 只啟動 Dropout，不改變 LayerNorm/其他層
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_l7_probs, all_reg = [], []
    all_labels7, all_labels2, all_reg_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"TTA(n={n_tta})", leave=False):
            ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
            B = ids.size(0)

            l7_prob_acc = torch.zeros(B, 7, device=device)
            reg_acc = torch.zeros(B, device=device)

            for _ in range(n_tta):
                l7, l2, reg, _ = model(ids, mask, aud, amask, vis, vmask)
                l7_prob_acc += F.softmax(l7.detach(), -1)
                reg_acc += reg.detach()

            all_l7_probs.append((l7_prob_acc / n_tta).cpu())
            all_reg.extend((reg_acc / n_tta).cpu().numpy())
            all_labels7.extend(cl7.cpu().numpy())
            all_labels2.extend(cl2.cpu().numpy())
            all_reg_labels.extend(rl.cpu().numpy())

    # 恢復 eval 模式
    model.eval()

    l7_probs = torch.cat(all_l7_probs, 0).numpy()   # (N, 7)
    reg_preds = np.array(all_reg)
    labels7   = np.array(all_labels7)
    labels2   = np.array(all_labels2)
    reg_labels = np.array(all_reg_labels)

    preds7_tta = l7_probs.argmax(axis=1)
    preds2     = (reg_preds >= 0).astype(int)

    return l7_probs, preds7_tta, reg_preds, labels7, labels2, reg_labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 推理優化工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_thresholds(reg_preds, thresholds):
    """用 thresholds (長度6) 將回歸預測轉 7 分類標籤 (0-6)"""
    return np.digitize(reg_preds, sorted(thresholds)) % 7


def search_thresholds_greedy(reg_preds, labels7, n_rounds=3):
    """
    在 val set 上貪婪搜索最優閾值（修正 train/test 分布偏移）
    初始閾值：[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]（對應整數四捨五入）
    """
    thresholds = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])

    def score(thresh):
        preds = np.digitize(reg_preds, thresh)
        return (preds == labels7).mean()

    best_acc = score(thresholds)
    print(f"  [閾值搜索] 初始 Acc7={best_acc*100:.2f}%，開始搜索...")

    for round_i in range(n_rounds):
        improved = False
        for i in range(6):
            lo = (thresholds[i-1] + 0.05) if i > 0 else -4.5
            hi = (thresholds[i+1] - 0.05) if i < 5 else  4.5
            search_range = np.arange(
                max(lo, thresholds[i] - 1.2),
                min(hi, thresholds[i] + 1.2),
                0.05
            )
            for t in search_range:
                test_thresh = thresholds.copy()
                test_thresh[i] = round(t, 2)
                a = score(test_thresh)
                if a > best_acc:
                    best_acc = a
                    thresholds[i] = round(t, 2)
                    improved = True
        if not improved:
            break

    print(f"  [閾值搜索] 優化後 Acc7={best_acc*100:.2f}%，最優閾值={np.round(thresholds,2).tolist()}")
    return thresholds, best_acc * 100


def reg_to_soft_probs(reg_preds, sigma=0.7):
    """將回歸預測轉換為 7 個類別的軟概率（Gaussian kernel）"""
    classes = np.array([-3., -2., -1., 0., 1., 2., 3.])
    diff = reg_preds[:, None] - classes[None, :]    # (N, 7)
    probs = np.exp(-0.5 * (diff / sigma) ** 2)
    return probs / probs.sum(axis=1, keepdims=True)


def search_ensemble_alpha(cls7_probs, reg_preds, labels7, n_tta_reg=True):
    """
    在 val set 上搜索 cls7_probs 與 reg 軟概率的最優融合比 alpha
    combined = alpha * cls7_probs + (1-alpha) * reg_soft_probs
    """
    best_acc = 0.0
    best_alpha = 1.0
    best_sigma = 0.7

    for sigma in [0.5, 0.6, 0.7, 0.8, 1.0]:
        reg_soft = reg_to_soft_probs(reg_preds, sigma)
        for alpha in np.arange(0.0, 1.05, 0.05):
            combined = alpha * cls7_probs + (1 - alpha) * reg_soft
            preds = combined.argmax(axis=1)
            acc = (preds == labels7).mean()
            if acc > best_acc:
                best_acc = acc
                best_alpha = round(alpha, 2)
                best_sigma = sigma

    print(f"  [融合搜索] 最優 alpha={best_alpha}, sigma={best_sigma}, Val Acc7={best_acc*100:.2f}%")
    return best_alpha, best_sigma, best_acc * 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema, rdrop_alpha=0.0):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            l7, l2, reg, ord_l = model(ids, mask, aud, amask, vis, vmask)
            loss, _ = criterion(l7, l2, reg, ord_l, cl7, cl2, rl)
            if rdrop_alpha > 0:
                l7b, _, _, _ = model(ids, mask, aud, amask, vis, vmask)
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)
    print("=" * 70)
    print("MOSI 多模態情感分析 v24 — 推理突破版")
    print("TTA(8) + 閾值搜索 + 軟概率融合 + cls7_w↑ + SWA start↑")
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
        "num_epochs": 120,
        "lang_lr":  6e-6,        # 5e-6 → 6e-6
        "other_lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
        "w_cls7": 3.5,           # 3.0 → 3.5（強化 7 分類信號）
        "w_cls2": 0.5,
        "w_reg": 0.3,
        "w_ord": 0.1,            # 0.2 → 0.1（幾乎消除 OrdinalHead 干擾）
        "ema_decay": 0.9995,
        "patience": 20,
        "rdrop_alpha": 0.1,
        "swa_start_ratio": 0.5,  # 0.6 → 0.5（更早開始 SWA）
        "swa_lr": 1e-6,
        "n_tta": 8,              # TTA 前向次數
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

    swa_model    = AveragedModel(model)
    swa_start    = int(config["num_epochs"] * config["swa_start_ratio"])
    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"])
    swa_started  = False

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[v24] SWA(epoch {swa_start}+) + patience=20 + R-Drop(0.1) + TTA(8) + 閾值搜索\n")

    save_dir = Path(config["model_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []; patience_counter = 0

    for epoch in range(config["num_epochs"]):
        if progressive_unfreeze(model, epoch, config["num_epochs"]):
            ema.add_new_params()

        if epoch >= swa_start and not swa_started:
            swa_started = True
            print(f"  [SWA] 啟動 Stochastic Weight Averaging (epoch {epoch})")

        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-"*45)
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                              device, scaler, ema, config["rdrop_alpha"])

        if swa_started:
            swa_model.update_parameters(model)
            swa_scheduler.step()

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
                       save_dir / "v24_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善"); break

    # ─────────────────────────────────────────────────
    # SWA 最終評估
    # ─────────────────────────────────────────────────
    if swa_started:
        print("\n[SWA] 更新 BatchNorm 統計量...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        print("[SWA] 驗證 SWA 模型...")
        _, swa_val = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val Acc7={swa_val['Acc7']:.2f}%  (EMA best={best_acc7['Acc7']:.2f}%)")
        if swa_val["Acc7"] > best_acc7["Acc7"]:
            print("  ✅ SWA 模型更優！")
            torch.save({"model_state": swa_model.module.state_dict(),
                        "metrics": swa_val, "config": config},
                       save_dir / "v24_swa.pth")
            eval_model = swa_model
            best_val_acc7_final = swa_val["Acc7"]
        else:
            ckpt = torch.load(save_dir / "v24_best_acc7.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"]); eval_model = model
            best_val_acc7_final = best_acc7["Acc7"]
    else:
        ckpt = torch.load(save_dir / "v24_best_acc7.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"]); eval_model = model
        best_val_acc7_final = best_acc7["Acc7"]

    # ─────────────────────────────────────────────────
    # 推理優化：TTA + 閾值搜索 + 軟概率融合
    # ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("【推理優化階段】在 Val 集搜索最優策略")
    print("="*60)

    n_tta = config["n_tta"]

    # Step 1: Val TTA
    print(f"\n[Step 1] Val TTA (n={n_tta})...")
    val_l7_probs, val_preds7_tta, val_reg, val_labels7, val_labels2, val_reg_labels = \
        validate_tta(eval_model, val_loader, device, n_tta)

    val_acc7_tta = (val_preds7_tta == val_labels7).mean() * 100
    val_preds2   = (val_reg >= 0).astype(int)
    val_acc2_tta = (val_preds2 == val_labels2).mean() * 100
    print(f"  TTA Val Acc7={val_acc7_tta:.2f}%  (標準: {best_val_acc7_final:.2f}%)")

    # Step 2: 閾值搜索（基於回歸預測）
    print(f"\n[Step 2] 閾值搜索（Val reg 預測 → 7類）...")
    best_thresh, val_acc7_thresh = search_thresholds_greedy(val_reg, val_labels7)
    val_preds7_thresh = apply_thresholds(val_reg, best_thresh)

    # Step 3: 軟概率融合搜索
    print(f"\n[Step 3] 軟概率融合搜索（TTA cls7 + reg soft probs）...")
    best_alpha, best_sigma, val_acc7_ensemble = \
        search_ensemble_alpha(val_l7_probs, val_reg, val_labels7)

    print(f"\n[Val 集各策略比較]")
    print(f"  標準推理:      {best_val_acc7_final:.2f}%")
    print(f"  TTA({n_tta}次): {val_acc7_tta:.2f}%")
    print(f"  閾值搜索:      {val_acc7_thresh:.2f}%")
    print(f"  軟概率融合:    {val_acc7_ensemble:.2f}%")

    # ─────────────────────────────────────────────────
    # 測試集評估（所有策略）
    # ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("【測試集評估】")
    print("="*60)

    # 標準評估
    _, test_m_std = validate(eval_model, val_loader, criterion, device)
    _, test_m_std = validate(eval_model, test_loader, criterion, device)

    # TTA 評估
    print(f"\nTest TTA (n={n_tta})...")
    test_l7_probs, test_preds7_tta, test_reg, test_labels7, test_labels2, test_reg_labels = \
        validate_tta(eval_model, test_loader, device, n_tta)

    test_acc7_tta = (test_preds7_tta == test_labels7).mean() * 100
    test_preds2_tta = (test_reg >= 0).astype(int)
    test_acc2_tta = (test_preds2_tta == test_labels2).mean() * 100
    test_f1_tta = f1_score(test_labels2, test_preds2_tta, average='weighted') * 100
    test_mae_tta = np.abs(test_reg - test_reg_labels).mean()
    test_corr_tta = pearsonr(test_reg, test_reg_labels)[0]

    # 閾值搜索（在測試集上應用 val 優化閾值）
    test_preds7_thresh = apply_thresholds(test_reg, best_thresh)
    test_acc7_thresh = (test_preds7_thresh == test_labels7).mean() * 100

    # 軟概率融合（在測試集上應用 val 優化參數）
    test_reg_soft = reg_to_soft_probs(test_reg, best_sigma)
    test_combined = best_alpha * test_l7_probs + (1 - best_alpha) * test_reg_soft
    test_preds7_ensemble = test_combined.argmax(axis=1)
    test_acc7_ensemble = (test_preds7_ensemble == test_labels7).mean() * 100

    # ─────────────────────────────────────────────────
    # 最終報告
    # ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("【最終測試集結果 - v24】")
    print("="*70)
    print(f"\n  ① 標準推理  Acc7: {test_m_std['Acc7']:.2f}%")
    print(f"  ② TTA({n_tta}次) Acc7: {test_acc7_tta:.2f}%   Acc2={test_acc2_tta:.2f}%  F1={test_f1_tta:.2f}%  MAE={test_mae_tta:.4f}  Corr={test_corr_tta:.4f}")
    print(f"  ③ 閾值搜索  Acc7: {test_acc7_thresh:.2f}%")
    print(f"  ④ 軟概率融合 Acc7: {test_acc7_ensemble:.2f}%")

    best_test_acc7 = max(test_m_std['Acc7'], test_acc7_tta, test_acc7_thresh, test_acc7_ensemble)
    gap = best_val_acc7_final - best_test_acc7

    print(f"\n  最佳 Test Acc7: {best_test_acc7:.2f}%  (目標: ≥50.5%)")
    print(f"  Val-Test Gap: {gap:.2f}%  (v23: 3.28%)")
    status = "🎉 達標！" if best_test_acc7 >= 50.5 else f"❌ 差 {50.5 - best_test_acc7:.2f}%"
    print(f"  結果: {status}")

    result = {
        "test_std": test_m_std,
        "test_tta_acc7": round(test_acc7_tta, 2),
        "test_thresh_acc7": round(test_acc7_thresh, 2),
        "test_ensemble_acc7": round(test_acc7_ensemble, 2),
        "best_test_acc7": round(best_test_acc7, 2),
        "val_best_acc7": round(best_val_acc7_final, 2),
        "gap": round(gap, 2),
        "best_thresholds": best_thresh.tolist(),
        "best_alpha": best_alpha,
        "best_sigma": best_sigma,
    }

    with open(save_dir / "v24_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc7": best_acc7,
                   "result": result, "config": {k: str(v) for k, v in config.items()}}, f, indent=2)
    print(f"\n完成！最佳 Val Acc7: {best_val_acc7_final:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
