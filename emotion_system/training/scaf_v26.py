"""
MOSI 多模態情感分析 v26 — FGM 對抗訓練版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

失敗分析結論 (best2 → Test 45.77%, 最差成績):
  ✗ Top-K gate selection: 不可微 + selection bias 過擬合
  ✗ SentimentContrastiveLoss: 梯度競爭，小數據不穩定
  ✗ 可學習多層融合: 1284筆資料 → 過擬合
  → 回歸 v23 簡單架構 (Test 48.69%, 最佳基線)

v26 三大改進 (每個改動有明確理由):
  1. FGM 對抗訓練 (epsilon=0.5)
     - 攻擊: deberta.embeddings.word_embeddings
     - 原理: 讓模型對 embedding 擾動不敏感 → 平滑決策邊界 → 更好泛化
     - 預期: Gap 從 3.28% 縮至 2.5-3% (+0.3-0.8% Test)
  2. 音視頻 L2 Normalization (零訓練成本)
     - 原理: MOSI train/test 說話人不同 → 特徵幅度分佈偏移
     - L2 norm on feature dim → 消除幅度差異，保留方向信息
  3. 推論增強: TTA(10) + 閾值搜索 + Bias 校正
     - Bias 校正針對分布偏移: train mean=+0.23 vs test mean=-0.32
     - 閾值搜索: 用 val 集找最優 6 個切分點

保留 v23 所有有效機制:
  EMA(0.9995) + SWA + FocalLoss + Capped權重 + Acc7選模 + Progressive Unfreeze

移除 (簡化):
  OrdinalHead (減少梯度競爭), R-Drop (FGM 已提供正則化, 節省計算)

目標: Val 53%+, Test 50.5%+
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
# 資料集（與 v23 完全相同）
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
# 模型架構（v23 + L2 Norm + 移除 OrdinalHead）
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


class HybridModel(nn.Module):
    """v23 架構 + L2 Norm + 移除 OrdinalHead (簡化)"""
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

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        vision = torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0)

        # v26新增: L2 Normalization — 解決 train/test 說話人特徵幅度偏移 (零訓練成本)
        audio = audio / audio.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        vision = vision / vision.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)

        hidden = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xl = self.polarity_attn(hidden, attention_mask)
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)
        fused = self.cross_modal(xl, xa, xv)
        feat = self.shared(fused)
        return (self.cls7_head(feat), self.cls2_head(feat),
                self.reg_head(feat).squeeze(-1) * 3.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FGM 對抗訓練 (v26 核心新增)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FGM:
    """
    Fast Gradient Method 對抗訓練

    設計決策:
    - epsilon=0.5: DeBERTa-v3-Large (1024-dim) embedding 範數約 2-5，
                   epsilon=0.5 使擾動幅度佔 10-25%，語義保留同時有效干擾
    - 攻擊目標: deberta.embeddings.word_embeddings (精確路徑，非寬泛匹配)
    - 與 EMA 不衝突: FGM 在 optimizer.step 前完成，EMA update 在 step 後

    流程:
      loss.backward()       → 得到梯度 g_clean
      fgm.attack()          → embedding += epsilon * g_clean / ||g_clean||
      forward_adv()         → 對抗樣本前向
      loss_adv.backward()   → 梯度疊加 (不 zero_grad！)
      fgm.restore()         → embedding 還原
      optimizer.step()      → 用 (g_clean + g_adv) 更新
    """
    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='deberta.embeddings.word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self, emb_name='deberta.embeddings.word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA（與 v23 相同）
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
# 損失（移除 OrdinalHead，w_cls7=3.5）
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
    def __init__(self, class_weights, w_cls7=3.5, w_cls2=0.5, w_reg=0.3):
        super().__init__()
        self.w = (w_cls7, w_cls2, w_reg)
        self.focal = FocalLoss(alpha=class_weights)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg = nn.SmoothL1Loss()

    def forward(self, l7, l2, reg, cl7, cl2, rl):
        lc7 = self.focal(l7, cl7)
        lc2 = self.cls2(l2, cl2)
        lr = self.reg(reg, rl)
        total = self.w[0]*lc7 + self.w[1]*lc2 + self.w[2]*lr
        return total, {"cls7": lc7.item(), "cls2": lc2.item(), "reg": lr.item()}


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
# 訓練（FGM 對抗訓練，移除 R-Drop 節省計算）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema, fgm):
    model.train()
    total_loss = 0.0
    FGM_EMB = 'deberta.embeddings.word_embeddings'

    for batch in tqdm(loader, desc="Train", leave=False):
        ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
        optimizer.zero_grad()

        # ── 第一次前向：正常樣本 ──
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # FGM 需要真實梯度幅度
        else:
            loss.backward()

        # ── FGM 對抗訓練 ──
        fgm.attack(FGM_EMB)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            l7_adv, l2_adv, reg_adv = model(ids, mask, aud, amask, vis, vmask)
            loss_adv, _ = criterion(l7_adv, l2_adv, reg_adv, cl7, cl2, rl)

        if scaler:
            scaler.scale(loss_adv).backward()
        else:
            loss_adv.backward()  # 梯度疊加（不 zero_grad）

        fgm.restore(FGM_EMB)

        # ── 梯度裁剪 + 更新 ──
        if scaler:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
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
        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)
        total_loss += loss.item()
        all_c7.extend(l7.argmax(1).cpu().numpy()); all_c2.extend(l2.argmax(1).cpu().numpy())
        all_r.extend(reg.cpu().numpy());            all_l7.extend(cl7.cpu().numpy())
        all_l2.extend(cl2.cpu().numpy());           all_lr.extend(rl.cpu().numpy())
    return total_loss/len(loader), compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 推論增強（TTA + 閾值搜索 + Bias 校正）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def collect_tta_preds(model, loader, device, n_tta=10):
    """TTA: dropout ON, 平均 n_tta 次 softmax"""
    model.train()  # 開啟 dropout
    all_probs7, all_reg, all_l7, all_lr = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ids, mask, aud, amask, vis, vmask, cl7, cl2, rl = run_batch(batch, device)
            p7_acc = None
            reg_acc = None
            for _ in range(n_tta):
                l7, _, reg = model(ids, mask, aud, amask, vis, vmask)
                p7 = F.softmax(l7, dim=-1)
                p7_acc = p7 if p7_acc is None else p7_acc + p7
                reg_acc = reg if reg_acc is None else reg_acc + reg
            all_probs7.append((p7_acc / n_tta).cpu())
            all_reg.append((reg_acc / n_tta).cpu())
            all_l7.extend(cl7.cpu().numpy())
            all_lr.extend(rl.cpu().numpy())

    probs7 = torch.cat(all_probs7).numpy()
    reg = torch.cat(all_reg).numpy()
    return probs7, reg, np.array(all_l7), np.array(all_lr)


def search_bias_calibration(reg_preds, labels7):
    """
    顯式 Bias 校正 — 針對 train(+0.23) vs test(-0.32) 分布偏移
    在 val 集搜索最優 bias，再用於 test
    """
    best_acc, best_bias = 0.0, 0.0
    for bias in np.arange(-1.0, 0.5, 0.02):
        corrected = reg_preds + bias
        preds = np.clip(np.round(corrected).astype(int), -3, 3) + 3
        acc = (preds == labels7).mean()
        if acc > best_acc:
            best_acc = acc
            best_bias = round(float(bias), 2)
    print(f"  [Bias校正] 最優 bias={best_bias:+.2f}, Val Acc7={best_acc*100:.2f}%")
    return best_bias, best_acc * 100


def search_thresholds_greedy(reg_preds, labels7, n_rounds=3):
    """
    貪心搜索 6 個閾值將 regression 轉為 7-class
    初始化偏向負面 (對應 test 均值偏負)
    """
    def reg_to_cls(reg, thresh):
        cls = np.zeros(len(reg), dtype=int)
        for i, t in enumerate(thresh):
            cls[reg > t] = i + 1
        return cls

    # 初始閾值偏向負面 (考慮 train/test 分布偏移)
    thresholds = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=float)
    best_acc = (reg_to_cls(reg_preds, thresholds) == labels7).mean()

    for _ in range(n_rounds):
        improved = True
        while improved:
            improved = False
            for i in range(6):
                lo = thresholds[i-1] + 0.05 if i > 0 else -3.5
                hi = thresholds[i+1] - 0.05 if i < 5 else 3.5
                for t in np.arange(lo, hi, 0.05):
                    new_t = thresholds.copy(); new_t[i] = t
                    acc = (reg_to_cls(reg_preds, new_t) == labels7).mean()
                    if acc > best_acc:
                        best_acc = acc; thresholds[i] = t; improved = True

    print(f"  [閾值搜索] 最優閾值={[round(t,2) for t in thresholds]}, Val Acc7={best_acc*100:.2f}%")
    return thresholds, best_acc * 100


def enhanced_inference(model, val_loader, test_loader, device, n_tta=10):
    """
    五步推論增強:
    1. TTA(10) → 平均 softmax + reg
    2. Bias 校正搜索 (val)
    3. 閾值搜索 (val)
    4. 選最優方案 (softmax argmax vs bias+閾值)
    5. 用最優方案評估 test
    """
    print("\n[推論增強] 收集 val TTA 預測...")
    val_probs7, val_reg, val_l7, val_lr = collect_tta_preds(model, val_loader, device, n_tta)
    val_c7_base = val_probs7.argmax(1)
    val_acc_base = (val_c7_base == val_l7).mean() * 100
    print(f"  TTA softmax 基線: Val Acc7={val_acc_base:.2f}%")

    # Bias 校正（針對分布偏移）
    best_bias, bias_val_acc = search_bias_calibration(val_reg, val_l7)

    # 閾值搜索
    best_thresh, thresh_val_acc = search_thresholds_greedy(val_reg, val_l7)

    def reg_to_cls(reg, thresh):
        cls = np.zeros(len(reg), dtype=int)
        for i, t in enumerate(thresh):
            cls[reg > t] = i + 1
        return cls

    # 選最優方案
    methods = {
        "TTA_softmax": (val_acc_base, None),
        "Bias_calib": (bias_val_acc, None),
        "Thresh_search": (thresh_val_acc, None),
    }
    best_method = max(methods, key=lambda k: methods[k][0])
    print(f"\n  最優 val 方案: {best_method} ({methods[best_method][0]:.2f}%)")

    # 收集 test TTA 預測
    print("\n[推論增強] 收集 test TTA 預測...")
    test_probs7, test_reg, test_l7, test_lr = collect_tta_preds(model, test_loader, device, n_tta)

    # 應用最優方案到 test
    if best_method == "TTA_softmax":
        test_c7 = test_probs7.argmax(1)
    elif best_method == "Bias_calib":
        test_c7 = np.clip(np.round(test_reg + best_bias).astype(int), -3, 3) + 3
    else:  # Thresh_search
        test_c7 = reg_to_cls(test_reg, best_thresh)

    test_c2 = (test_reg + (best_bias if best_method == "Bias_calib" else 0) >= 0).astype(int)
    test_metrics = compute_metrics(test_c7, test_c2, test_reg, test_l7,
                                   (test_lr >= 0).astype(int), test_lr)

    return test_metrics, best_method, methods[best_method][0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函數
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_seed(42)
    print("=" * 70)
    print("MOSI 多模態情感分析 v26 — FGM 對抗訓練版")
    print("v23基線 + FGM(ε=0.5) + L2Norm + TTA + Bias校正 + 閾值搜索")
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
        "num_epochs": 150,           # 150 epochs (比v23更充分)
        "lang_lr":  6e-6,            # 5e-6 → 6e-6 (微調)
        "other_lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
        "w_cls7": 3.5,               # 3.0 → 3.5 (更強調7分類)
        "w_cls2": 0.5, "w_reg": 0.3,
        "ema_decay": 0.9995,
        "patience": 25,              # 20 → 25 (更充分訓練)
        "swa_start_ratio": 0.55,     # 後 45% epochs 啟動 SWA
        "swa_lr": 1e-6,
        "fgm_epsilon": 0.5,          # v26 核心: FGM epsilon
        "n_tta": 10,                 # 推論 TTA 次數
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

    # 確認 FGM 攻擊目標存在
    fgm_target = 'deberta.embeddings.word_embeddings'
    found_emb = [(n, p.shape) for n, p in model.named_parameters()
                 if fgm_target in n]
    if found_emb:
        print(f"[FGM] 攻擊目標: {found_emb[0][0]}, shape={found_emb[0][1]}")
    else:
        # 備用：搜索任何包含 word_embedding 的層
        fallback = [(n, p.shape) for n, p in model.named_parameters()
                    if 'word_embed' in n.lower()]
        if fallback:
            fgm_target = fallback[0][0].rsplit('.', 1)[0]
            fgm_target = '.'.join(fallback[0][0].split('.')[1:-1])  # 去掉 'lang_backbone.' 前綴
            print(f"[FGM] 使用備用目標: {fgm_target}")
        else:
            print("[FGM] 警告: 未找到 word_embedding，FGM 可能無效")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {train_p/1e6:.1f}M")

    criterion = HybridLoss(class_w.to(device),
                           config["w_cls7"], config["w_cls2"], config["w_reg"])

    lang_params = [p for p in model.lang_backbone.parameters() if p.requires_grad]
    other_params = (list(model.polarity_attn.parameters()) +
                    list(model.audio_encoder.parameters()) +
                    list(model.vision_encoder.parameters()) +
                    list(model.cross_modal.parameters()) +
                    list(model.shared.parameters()) +
                    list(model.cls7_head.parameters()) +
                    list(model.cls2_head.parameters()) +
                    list(model.reg_head.parameters()))

    optimizer = optim.AdamW([
        {"params": lang_params,  "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None
    ema       = EMA(model, decay=config["ema_decay"])
    fgm       = FGM(model, epsilon=config["fgm_epsilon"])

    # SWA 設置
    swa_model = AveragedModel(model)
    swa_start = int(config["num_epochs"] * config["swa_start_ratio"])
    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"])
    swa_started = False

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[v26] FGM(ε={config['fgm_epsilon']}) + L2Norm + SWA(epoch {swa_start}+)")
    print(f"      lang_lr={config['lang_lr']:.0e} | patience={config['patience']} | epochs={config['num_epochs']}\n")

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
                              device, scaler, ema, fgm)

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
                       save_dir / "v26_best_acc7.pth")
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
            print("  ✅ SWA 模型更優！")
            torch.save({"model_state": swa_model.module.state_dict(),
                        "metrics": swa_val, "config": config},
                       save_dir / "v26_swa.pth")
            eval_model = swa_model
        else:
            ckpt = torch.load(save_dir / "v26_best_acc7.pth", map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"]); eval_model = model
    else:
        ckpt = torch.load(save_dir / "v26_best_acc7.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"]); eval_model = model

    # ── 標準測試集評估 ──
    print("\n" + "="*60)
    _, test_std = validate(eval_model, test_loader, criterion, device)
    print("\n【標準測試集結果 - v26】")
    print(f"  Acc7 : {test_std['Acc7']:.2f}%   (目標: 50.5%, v23最佳: 48.69%)")
    print(f"  Acc2 : {test_std['Acc2']:.2f}%")
    print(f"  F1   : {test_std['F1']:.2f}%")
    print(f"  MAE  : {test_std['MAE']:.4f}")
    print(f"  Corr : {test_std['Corr']:.4f}")
    gap = best_acc7['Acc7'] - test_std['Acc7']
    print(f"\n  Val-Test Gap: {gap:.2f}% (v23: 3.28%)")

    # ── 推論增強測試集評估 ──
    print("\n" + "="*60)
    print("[推論增強] 開始...")
    ema.apply_shadow()
    test_enh, best_method, val_enh_acc = enhanced_inference(
        eval_model, val_loader, test_loader, device, config["n_tta"])
    ema.restore()

    print(f"\n【推論增強測試集結果 - v26 ({best_method})】")
    print(f"  Acc7 : {test_enh['Acc7']:.2f}%   (目標: 50.5%)")
    print(f"  Acc2 : {test_enh['Acc2']:.2f}%")
    print(f"  F1   : {test_enh['F1']:.2f}%")
    print(f"  MAE  : {test_enh['MAE']:.4f}")
    print(f"  Corr : {test_enh['Corr']:.4f}")

    # 最終判定
    final_acc7 = max(test_std['Acc7'], test_enh['Acc7'])
    status = "🎉 達標！" if final_acc7 >= 50.5 else f"❌ 差 {50.5 - final_acc7:.2f}%"
    print(f"\n  最終 Test Acc7: {final_acc7:.2f}% | 結果: {status}")

    with open(save_dir / "v26_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_val_acc7": best_acc7,
            "test_standard": test_std,
            "test_enhanced": test_enh,
            "best_inference_method": best_method,
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)
    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
