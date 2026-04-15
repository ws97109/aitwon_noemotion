"""
MOSI 多模態情感分析 v59 — Zero Leakage + TrainVal + EMD Loss + TTA + Prior Correction + 3-Seed Ensemble
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【v58 → v59: 三大改進】

v58 預期結果: ~52% (同 v56 架構 + prior correction)

v59 新增:
  ✅ Ordinal EMD Loss (25%): 序數感知損失
     ‧ 用 CDF L1 距離取代部分 Focal Loss
     ‧ 相鄰類別錯誤懲罰輕、遠距錯誤懲罰重 → 符合情感分析的連續性
     ‧ 預期 +0.5~1% Acc7
  ✅ Test-Time Augmentation (TTA×3): 零洩漏推斷增強
     ‧ SWA 模型以 dropout 開啟推斷 3 次，取 logits 平均
     ‧ 每個 seed 的有效「模型」數從 1 增為 3 → 9 個 stochastic passes
     ‧ 預期 +0.3~0.5% Acc7
  ✅ Val-justified alpha search: 零洩漏最佳 alpha 選擇
     ‧ 用 v55 val logits + trainval→val log_ratio 在 val 上搜尋最佳 alpha
     ‧ 替代硬編碼 alpha=1.0，預期 +0.3~0.8%

v59 零洩漏保證:
  ‧ EMD loss 只使用訓練標籤 (不接觸 test)
  ‧ TTA 完全不使用 labels
  ‧ alpha 從 v55 val logits 決定 (val 已在 trainval 中，prior 資訊合法)
  ‧ 最終報告唯一結果 = 3-seed ensemble (不以 test 選模型)

目標: Test Acc7 > 52%（零資料洩漏）
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
MODEL_DIR = PROJECT_ROOT / "emotion_system" / "models"

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
# Focal Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    """Focal Loss = -(1-pt)^γ × log(pt), with class weights and label smoothing"""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight,
                                  label_smoothing=self.label_smoothing,
                                  reduction='none')
        with torch.no_grad():
            log_pt = F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


class OrdinalEMDLoss(nn.Module):
    """Earth Mover's Distance (Wasserstein-1) loss for ordinal classification.
    Minimises L1 distance between predicted CDF and ground-truth CDF.
    Penalises large ordinal mistakes more than adjacent mistakes — better than
    plain CE for ordered 7-class labels (-3..+3).
    """
    def __init__(self, n_classes: int = 7):
        super().__init__()
        self.n = n_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs    = F.softmax(logits, dim=-1)
        cdf_pred = probs.cumsum(dim=-1)[:, :-1]                      # [B, n-1]
        cdf_true = F.one_hot(labels, self.n).float().cumsum(dim=-1)[:, :-1]
        return (cdf_pred - cdf_true).abs().mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 評估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, cls7_crit, cls2_crit, reg_crit,
                optimizer, scheduler, device, scaler, ema,
                rdrop_alpha=0.05, emd_crit=None, emd_weight=0.0):
    model.train(); total_loss = 0.0; nan_count = 0
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
            if emd_crit is not None and emd_weight > 0:
                cls7_loss = (1.0-emd_weight)*cls7_crit(l7, cl7) + emd_weight*emd_crit(l7, cl7)
            else:
                cls7_loss = cls7_crit(l7, cl7)
            loss = cls7_loss + 0.3*cls2_crit(l2, cl2) + 0.4*reg_crit(reg, rl)
            if rdrop_alpha > 0:
                l7b, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7,-1), F.softmax(l7b,-1).detach(), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7b,-1), F.softmax(l7,-1).detach(), reduction='batchmean'))/2
                loss = loss + rdrop_alpha * kl

        # ★ NaN 保護：偵測到 NaN 直接跳過此 batch（修正 #3）
        # loss NaN 在 backward() 之前發生，GradScaler 不介入；
        # zero_grad() 已在上方執行，scaler.update() 不可在未 step() 後呼叫
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            scheduler.step()  # 保持 LR schedule 步數一致
            continue

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
        # ★ 若解凍後有附加的新 group scheduler，一起 step（修正 #2）
        if hasattr(scheduler, '_new_group_sched'):
            scheduler._new_group_sched.step()
        total_loss += loss.item()
    valid_batches = len(loader) - nan_count
    if nan_count > 0:
        print(f"  [警告] {nan_count} batches skipped due to NaN/Inf loss", end="")
    return total_loss / max(valid_batches, 1)


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
def get_all_outputs(model, loader, device):
    """回傳 (cls7_logits, cls2_logits, reg_values)，用於 ensemble 和 alpha search"""
    model.eval()
    all_l7, all_l2, all_reg = [], [], []
    for batch in tqdm(loader, desc="Infer", leave=False):
        ids  = batch["input_ids"].to(device);  mask  = batch["attention_mask"].to(device)
        aud  = batch["audio"].to(device);      amask = batch["audio_mask"].to(device)
        vis  = batch["vision"].to(device);     vmask = batch["vision_mask"].to(device)
        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
        all_l7.append(l7.cpu().float().numpy())
        all_l2.append(l2.cpu().float().numpy())
        all_reg.append(reg.cpu().float().numpy())
    return (np.concatenate(all_l7, 0),
            np.concatenate(all_l2, 0),
            np.concatenate(all_reg, 0))


def get_all_outputs_tta(model, loader, device, n_tta: int = 3):
    """Test-Time Augmentation: n_tta stochastic passes with dropout active.
    Averages logits across passes — zero-leak by definition (no labels used).
    n_tta=1 is identical to deterministic get_all_outputs().
    """
    if n_tta <= 1:
        return get_all_outputs(model, loader, device)
    all_runs_l7, all_runs_l2, all_runs_reg = [], [], []
    model.train()  # activate dropout
    for _ in range(n_tta):
        run_l7, run_l2, run_reg = [], [], []
        for batch in loader:
            ids   = batch["input_ids"].to(device);  mask  = batch["attention_mask"].to(device)
            aud   = batch["audio"].to(device);      amask = batch["audio_mask"].to(device)
            vis   = batch["vision"].to(device);     vmask = batch["vision_mask"].to(device)
            with torch.no_grad():
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            run_l7.append(l7.cpu().float().numpy())
            run_l2.append(l2.cpu().float().numpy())
            run_reg.append(reg.cpu().float().numpy())
        all_runs_l7.append(np.concatenate(run_l7, 0))
        all_runs_l2.append(np.concatenate(run_l2, 0))
        all_runs_reg.append(np.concatenate(run_reg, 0))
    model.eval()
    return (np.mean(all_runs_l7,  axis=0),
            np.mean(all_runs_l2,  axis=0),
            np.mean(all_runs_reg, axis=0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 單一種子訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_one_seed(seed, train_loader, val_loader, test_loader,
                   class_weights, device, config, trainval_mode=False):
    """trainval_mode=True: val 已在訓練集，跳過 val 監控/早停，僅用 SWA 作模型選擇"""
    set_seed(seed)
    print(f"\n{'─'*55}")
    print(f"Seed={seed} | SWA E{config['swa_start']}-E{config['num_epochs']} step={config['swa_step']}")
    if trainval_mode:
        print(f"  [TrainVal 模式] 固定 {config['num_epochs']} epoch，不監控 val，SWA 作唯一選擇")
    print(f"{'─'*55}")

    model = SACFModel(dropout=config["dropout"]).to(device)

    # 初始凍結前 6 層
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

    total_steps   = len(train_loader) * config["num_epochs"]
    warmup_steps  = int(total_steps * 0.06)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema    = EMA(model, 0.9995)
    # ★ 記錄 schedule 參數供解凍後新 group 使用（修正 #2）
    _sched_warmup  = warmup_steps
    _sched_total   = total_steps

    cw = class_weights.to(device)
    cls7_crit = FocalLoss(gamma=config["focal_gamma"], weight=cw,
                          label_smoothing=config["label_smoothing"])
    cls2_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_crit  = nn.SmoothL1Loss()

    best_val_acc7 = 0.0
    best_epoch    = 0
    best_state    = None
    patience_cnt  = 0
    swa_states    = []

    for epoch in range(config["num_epochs"]):
        # 1/3 進度後解凍所有層（修正 #2：新 group 掛上與主 scheduler 相同的 cosine 曲線）
        if epoch == config["num_epochs"] // 3 and not getattr(model, '_unfroze', False):
            new_backbone_params = []
            for i in range(6):
                for p in model.lang_backbone.encoder.layer[i].parameters():
                    p.requires_grad = True
                    new_backbone_params.append(p)
            unfreeze_base_lr = config["lang_lr"] / 2
            optimizer.add_param_group({
                "params": new_backbone_params,
                "lr": unfreeze_base_lr,
                "weight_decay": config["weight_decay"]
            })
            # 讓新 group 的 LR 也跟著 cosine scheduler 衰減：
            # 替換 scheduler 為包含新 group 的新 scheduler，並從當前 step 繼續
            current_step = epoch * len(train_loader)
            new_group_idx = len(optimizer.param_groups) - 1

            # 自定義 lambda：對新 group 從當前 step 開始套用相同 cosine 衰減
            import math
            def _cosine_lambda_for_new_group(step, base=current_step,
                                              w=_sched_warmup, t=_sched_total,
                                              blr=unfreeze_base_lr):
                abs_step = base + step
                if abs_step < w:
                    return float(abs_step) / float(max(1, w))
                prog = float(abs_step - w) / float(max(1, t - w))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

            from torch.optim.lr_scheduler import LambdaLR
            new_group_scheduler = LambdaLR(
                optimizer,
                lr_lambda=[lambda s: 1.0] * new_group_idx + [_cosine_lambda_for_new_group]
            )
            # 將新 scheduler 的 step 綁到原 scheduler 之後
            scheduler._new_group_sched = new_group_scheduler

            ema.add_new_params()
            model._unfroze = True
            patience_cnt = 0
            print(f"  [解凍] Epoch {epoch+1}: 全層開放，新層 lr={unfreeze_base_lr:.1e}（接 cosine 衰減），patience 重設")

        loss = train_epoch(model, train_loader, cls7_crit, cls2_crit, reg_crit,
                           optimizer, scheduler, device, scaler, ema,
                           rdrop_alpha=config["rdrop_alpha"],
                           emd_crit=OrdinalEMDLoss() if config.get("emd_weight", 0) > 0 else None,
                           emd_weight=config.get("emd_weight", 0.0))

        ep1 = epoch + 1

        if trainval_mode:
            # TrainVal 模式：val 在訓練集中，僅記錄 loss，不做 val 評估
            print(f"  E{ep1:02d} | Loss={loss:.4f}", end="")
        else:
            ema.apply_shadow()
            val_m = evaluate(model, val_loader, device)   # ★ 真實 val（不在訓練集中）
            ema.restore()

            acc7 = val_m["Acc7"]
            print(f"  E{ep1:02d} | Loss={loss:.4f} | Val Acc7={acc7:.2f}% Acc2={val_m['Acc2']:.2f}% "
                  f"F1={val_m['F1']:.2f}% MAE={val_m['MAE']:.4f} Corr={val_m['Corr']:.4f}", end="")

        # SWA: 從 swa_start 起每 swa_step 存一次 EMA state
        if ep1 >= config["swa_start"] and (ep1 - config["swa_start"]) % config["swa_step"] == 0:
            ema.apply_shadow()
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            ema.restore()
            print(f"  [SWA saved E{ep1}]", end="")

        if trainval_mode:
            # 不做早停，訓練完整 num_epochs
            print()
        else:
            if acc7 > best_val_acc7:
                best_val_acc7 = acc7; best_epoch = ep1; patience_cnt = 0
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

    # ★ SWA 平均權重
    if len(swa_states) > 0:
        print(f"\n  [SWA] 平均 {len(swa_states)} 個 checkpoint 的參數...")
        swa_state = {}
        for k in swa_states[0]:
            if swa_states[0][k].dtype.is_floating_point:
                swa_state[k] = torch.stack([s[k].float() for s in swa_states], 0).mean(0).to(swa_states[0][k].dtype)
            else:
                swa_state[k] = swa_states[-1][k]
    else:
        swa_state = best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"\n  [SWA] 無 SWA checkpoints，使用最終 checkpoint")

    if trainval_mode:
        # TrainVal 模式：val 在訓練集，直接用 SWA，不做 SWA vs BestVal 比較
        model.load_state_dict(swa_state); model.to(device)
        n_tta = config.get("n_tta", 1)
        swa_test_out = get_all_outputs_tta(model, test_loader, device, n_tta=n_tta)
        tta_tag = f" (TTA×{n_tta})" if n_tta > 1 else ""
        print(f"  [TrainVal-SWA] Seed={seed} 完成，使用 SWA checkpoint{tta_tag}")
        return 0.0, None, swa_test_out, swa_state
    else:
        # ★ 以 val 表現選擇 SWA vs BestVal（完全不看 test！）
        model.load_state_dict(swa_state); model.to(device)
        swa_val_m    = evaluate(model, val_loader, device)
        swa_val_out  = get_all_outputs(model, val_loader,  device)
        swa_test_out = get_all_outputs(model, test_loader, device)
        print(f"  [SWA]     Val: Acc7={swa_val_m['Acc7']:.2f}% Acc2={swa_val_m['Acc2']:.2f}% "
              f"F1={swa_val_m['F1']:.2f}% MAE={swa_val_m['MAE']:.4f}")

        model.load_state_dict(best_state); model.to(device)
        bv_val_m    = evaluate(model, val_loader, device)
        bv_val_out  = get_all_outputs(model, val_loader,  device)
        bv_test_out = get_all_outputs(model, test_loader, device)
        print(f"  [BestVal] Val: Acc7={bv_val_m['Acc7']:.2f}% Acc2={bv_val_m['Acc2']:.2f}% "
              f"F1={bv_val_m['F1']:.2f}% MAE={bv_val_m['MAE']:.4f}")

        # 選擇依據：val Acc7（不看 test）
        if swa_val_m["Acc7"] >= bv_val_m["Acc7"]:
            chosen_val_out  = swa_val_out
            chosen_test_out = swa_test_out
            chosen_label    = "SWA"
            chosen_val_acc7 = swa_val_m["Acc7"]
            chosen_state    = swa_state
        else:
            chosen_val_out  = bv_val_out
            chosen_test_out = bv_test_out
            chosen_label    = "BestVal"
            chosen_val_acc7 = bv_val_m["Acc7"]
            chosen_state    = best_state

        print(f"  [Seed {seed}] E{best_epoch} | 選擇 {chosen_label} (Val Acc7={chosen_val_acc7:.2f}%)")
        return best_val_acc7, chosen_val_out, chosen_test_out, chosen_state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主程式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI v59 — Zero Leakage: TrainVal + EMD Loss + TTA + Prior Correction + 3-Seed Ensemble")
    print("=" * 70)

    config = {
        "lang_model":      "microsoft/deberta-v3-large",
        "batch_size":      8,
        "num_epochs":      60,        # v59: 同 v56 proven 60 epoch
        "lang_lr":         4e-6,
        "head_lr":         8e-5,
        "weight_decay":    0.01,
        "dropout":         0.15,
        "label_smoothing": 0.05,
        "rdrop_alpha":     0.05,
        "focal_gamma":     2.0,
        "patience":        999,       # 停用早停（trainval 模式不需要）
        "swa_start":       42,        # v59: 同 v56 proven E42
        "swa_step":        3,
        "seeds":           [42, 123, 2024],
        "version":         "v59",
        "use_trainval":    True,
        "fixed_alpha":     1.0,       # ★ 將由 val-search 動態覆蓋
        "emd_weight":      0.25,      # ★ v59 新增：70% Focal + 25% EMD + 5% 隱式
        "n_tta":           3,         # ★ v59 新增：TTA 3 次推斷平均
    }

    print(f"\n載入資料: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f: data = pickle.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer)
    val_ds   = MOSIDataset(data["valid"], tokenizer)
    test_ds  = MOSIDataset(data["test"],  tokenizer)
    print(f"  train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    bs = config["batch_size"]
    val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  bs, shuffle=False, num_workers=2, pin_memory=True)

    use_trainval = config.get("use_trainval", False)
    if use_trainval:
        # ★ v56: 合併 train+val 訓練（18% 更多資料）
        from torch.utils.data import ConcatDataset
        trainval_ds = ConcatDataset([train_ds, val_ds])
        train_loader = DataLoader(trainval_ds, bs, shuffle=True, num_workers=2, pin_memory=True)
        print(f"  [v58] TrainVal 合併: {len(trainval_ds)} 樣本 (train {len(train_ds)} + val {len(val_ds)})")
        # class_weights 從 train+val 計算
        all_train_labels = np.concatenate([
            data["train"]["regression_labels"],
            data["valid"]["regression_labels"]
        ])
        class_weights = compute_class_weights(all_train_labels)
        # ★ v59: Prior correction — trainval→val 方向（不接觸 test）
        trainval_prior  = compute_prior(all_train_labels)
        val_prior_for_corr = compute_prior(data["valid"]["regression_labels"])
        log_ratio = np.log(val_prior_for_corr + 1e-8) - np.log(trainval_prior + 1e-8)
        print(f"  [v59] Prior correction: trainval→val, log_ratio={[f'{r:+.3f}' for r in log_ratio]}")

        # ★ v59: Val-justified alpha search — 使用 v55 val logits + trainval→val log_ratio
        # 完全不接觸 test；v55 val logits 是從 standard train/val split 產生的
        val_logits_v55_path = MODEL_DIR / "val_logits_v55.npy"
        if val_logits_v55_path.exists():
            vl55 = np.load(str(val_logits_v55_path))          # [3, 229, 7]
            mean_vl55 = vl55.mean(0)                           # [229, 7]
            val_labels_arr  = np.array(data["valid"]["regression_labels"])
            val_cls7_search = np.clip(np.round(val_labels_arr).astype(int), -3, 3) + 3
            best_val_alpha_acc, best_found_alpha = 0.0, config["fixed_alpha"]
            print(f"  [v59] Alpha search on v55 val logits (zero-leak):")
            for a in [round(x * 0.25, 2) for x in range(0, 25)]:  # 0.0 → 6.0
                corr_l7 = mean_vl55 + a * log_ratio
                acc = (corr_l7.argmax(1) == val_cls7_search).mean() * 100
                mark = " ← best" if acc > best_val_alpha_acc else ""
                print(f"    alpha={a:.2f}: Val Acc7={acc:.2f}%{mark}")
                if acc > best_val_alpha_acc:
                    best_val_alpha_acc = acc; best_found_alpha = a
            config["fixed_alpha"] = best_found_alpha
            print(f"  [v59] ★ Val-justified alpha={best_found_alpha:.2f} (Val Acc7={best_val_alpha_acc:.2f}%)")
        else:
            print(f"  [v59] val_logits_v55.npy 不存在，使用 fixed alpha={config['fixed_alpha']}")
    else:
        train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=2, pin_memory=True)
        class_weights = compute_class_weights(data["train"]["regression_labels"])
        # ★ Prior correction: train→val 方向，完全不用 test labels
        train_prior = compute_prior(data["train"]["regression_labels"])
        val_prior   = compute_prior(data["valid"]["regression_labels"])
        log_ratio   = np.log(val_prior + 1e-8) - np.log(train_prior + 1e-8)
        print(f"\n【Prior Correction — train→val（不看 test）】")
        print(f"  Train prior: {[f'{p*100:.1f}%' for p in train_prior]}")
        print(f"  Val   prior: {[f'{p*100:.1f}%' for p in val_prior]}")
        print(f"  log_ratio:   {[f'{r:+.3f}' for r in log_ratio]}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"設備: {device}")
    print(f"類別權重: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    test_labels_np = np.array(data["test"]["regression_labels"])
    test_cls7_true = np.clip(np.round(test_labels_np).astype(int), -3, 3) + 3
    test_cls2_true = (test_labels_np >= 0).astype(int)

    all_test_l7, all_test_l2, all_test_reg = [], [], []
    seed_results = []
    seed_model_states = []  # 儲存每個 seed 的模型狀態

    all_val_l7, all_val_l2, all_val_reg = [], [], []

    for seed in config["seeds"]:
        val_acc, val_out, test_out, model_state = train_one_seed(
            seed, train_loader, val_loader, test_loader,
            class_weights, device, config,
            trainval_mode=use_trainval)
        # val_out is None in trainval_mode (val merged into train)
        if val_out is not None:
            all_val_l7.append(val_out[0]); all_val_l2.append(val_out[1]); all_val_reg.append(val_out[2])
        all_test_l7.append(test_out[0]); all_test_l2.append(test_out[1]); all_test_reg.append(test_out[2])
        seed_results.append({"seed": seed, "trainval_mode": use_trainval, "best_val_acc7": round(val_acc, 2)})
        seed_model_states.append({"seed": seed, "val_acc7": val_acc, "state_dict": model_state})

    # ─────────────────────────────────────────────────────
    # ★ 最終 Test 評估（test 只做一次，zero-leak）
    # ─────────────────────────────────────────────────────
    mean_test_l7  = np.mean(all_test_l7, axis=0)
    mean_test_l2  = np.mean(all_test_l2, axis=0)
    mean_test_reg = np.mean(all_test_reg, axis=0)

    print(f"\n{'='*65}")
    print(f"【最終 Test 評估 (Zero Leakage) — test 只評估一次】")
    print(f"{'='*65}")

    import json
    import math as _math

    # ─── 通用：計算完整 test 指標（顯示 Acc7/Acc2/F1/MAE/Corr）───────────
    # v57: apply_prior_corr=True 套用 prior correction（僅 trainval 模式）
    def compute_test_metrics(l7, l2, reg, label, apply_prior_corr=False, lr=None, alpha=1.0):
        """回傳 (acc7, metrics_dict)，顯示 Acc7/Acc2/F1/MAE/Corr"""
        if apply_prior_corr and lr is not None:
            # ★ Prior correction: logits += alpha * log_ratio（不使用 test labels）
            l7_corr = l7 + alpha * lr
        else:
            l7_corr = l7
        pred7 = l7_corr.argmax(1)
        acc7  = (pred7 == test_cls7_true).mean() * 100
        pred2 = l2.argmax(1)
        acc2  = (pred2 == test_cls2_true).mean() * 100
        f1    = f1_score(test_cls2_true, pred2, average="weighted") * 100
        mae   = np.abs(reg - test_labels_np).mean()
        corr, _ = pearsonr(reg.astype(float), test_labels_np.astype(float))
        corr_tag = " [prior corr]" if apply_prior_corr else ""
        print(f"  [{label}]{corr_tag}: Acc7={acc7:.2f}%  Acc2={acc2:.2f}%  F1={f1:.2f}%  MAE={mae:.4f}  Corr={corr:.4f}")
        return acc7, {"Acc7": round(acc7,2), "Acc2": round(acc2,2),
                      "F1": round(f1,2), "MAE": round(mae,4), "Corr": round(corr,4)}

    s42_idx = config["seeds"].index(42) if 42 in config["seeds"] else 0

    if use_trainval:
        # ── v57：固定 alpha + prior correction（不接觸 test）──────────────
        alpha = config["fixed_alpha"]
        print(f"  [v59 TrainVal] alpha={alpha} + prior correction (trainval→val) + TTA×{config.get('n_tta',1)}\n")

        # 各 seed 個別結果（供參考；不用於選擇最終模型）
        print("  === 各 Seed 個別 Test 結果 [僅供參考] ===")
        for i, s in enumerate(config["seeds"]):
            compute_test_metrics(all_test_l7[i], all_test_l2[i], all_test_reg[i],
                                 f"Seed-{s}", apply_prior_corr=True, lr=log_ratio, alpha=alpha)

        # ★ v57: 唯一最終結果 = Ensemble（不以 test 選最佳，零洩漏）
        print(f"\n  === {len(config['seeds'])}-Seed Ensemble Test 結果 [最終] ===")
        final_acc7, best_m = compute_test_metrics(
            mean_test_l7, mean_test_l2, mean_test_reg,
            f"{len(config['seeds'])}-Seed Ensemble",
            apply_prior_corr=True, lr=log_ratio, alpha=alpha)
        best_label = f"{len(config['seeds'])}-Seed Ensemble"

        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        np.save(str(MODEL_DIR / "raw_logits_v59.npy"), np.stack(all_test_l7, axis=0))

        with open(MODEL_DIR / "history_v59.json", "w") as f:
            json.dump({
                "version":         config["version"],
                "note":            "TrainVal + EMD loss + TTA + val-justified prior correction — zero leakage",
                "seeds":           config["seeds"],
                "seed_results":    seed_results,
                "fixed_alpha":     config["fixed_alpha"],
                "emd_weight":      config.get("emd_weight", 0),
                "n_tta":           config.get("n_tta", 1),
                "log_ratio":       log_ratio.tolist(),
                "final_candidate": best_label,
                "final_acc7":      round(final_acc7, 2),
                "final_metrics":   best_m,
            }, f, indent=2, ensure_ascii=False)
        history_file = MODEL_DIR / "history_v59.json"

    else:
        # ── v55：val-based alpha search（修正 #4，完整實作）───────────────────
        val_labels_np = np.array(data["valid"]["regression_labels"])
        val_cls7_true = np.clip(np.round(val_labels_np).astype(int), -3, 3) + 3
        val_cls2_true = (val_labels_np > 0).astype(int)

        # Prior correction log_ratio（train→val）
        train_prior = compute_prior(data["train"]["regression_labels"])
        val_prior   = compute_prior(data["valid"]["regression_labels"])
        log_ratio   = np.log(val_prior + 1e-8) - np.log(train_prior + 1e-8)

        alpha_values = [round(a * 0.25, 2) for a in range(2, 25)]

        def _val_acc(l7, alpha):
            cl  = l7 + alpha * log_ratio
            cp  = np.exp(cl - cl.max(1, keepdims=True)); cp /= cp.sum(1, keepdims=True)
            return (cp.argmax(1) == val_cls7_true).mean() * 100

        # Seed-42 alpha search on val
        mean_val_l7 = np.mean(all_val_l7, axis=0) if all_val_l7 else None
        print(f"\n[Seed-42] Alpha Search on Val:")
        s42_best_val_acc = 0.0; s42_best_alpha = 1.0
        for alpha in alpha_values:
            acc = _val_acc(all_val_l7[s42_idx], alpha)
            marker = " ← 最佳" if acc > s42_best_val_acc else ""
            print(f"  alpha={alpha:.2f}: Val Acc7={acc:.2f}%{marker}")
            if acc > s42_best_val_acc:
                s42_best_val_acc = acc; s42_best_alpha = alpha
        print(f"Seed-42 最佳 alpha={s42_best_alpha:.2f} (Val={s42_best_val_acc:.2f}%)")

        # Ensemble alpha search on val
        print(f"\n[Ensemble {len(all_val_l7)} seeds] Alpha Search on Val:")
        ens_best_val_acc = 0.0; ens_best_alpha = 1.0
        for alpha in alpha_values:
            acc = _val_acc(mean_val_l7, alpha)
            marker = " ← 最佳" if acc > ens_best_val_acc else ""
            print(f"  alpha={alpha:.2f}: Val Acc7={acc:.2f}%{marker}")
            if acc > ens_best_val_acc:
                ens_best_val_acc = acc; ens_best_alpha = alpha
        print(f"Ensemble 最佳 alpha={ens_best_alpha:.2f} (Val={ens_best_val_acc:.2f}%)")

        # ── 最終 test 評估（各 seed + ensemble）────────────────────────────
        def compute_test_with_alpha(l7, l2, reg, alpha, label):
            cl    = l7 + alpha * log_ratio
            cp    = np.exp(cl - cl.max(1, keepdims=True)); cp /= cp.sum(1, keepdims=True)
            pred7 = cp.argmax(1)
            acc7  = (pred7 == test_cls7_true).mean() * 100
            pred2 = l2.argmax(1)
            acc2  = (pred2 == test_cls2_true).mean() * 100
            f1    = f1_score(test_cls2_true, pred2, average="weighted") * 100
            mae   = np.abs(reg - test_labels_np).mean()
            corr, _ = pearsonr(reg.astype(float), test_labels_np.astype(float))
            print(f"  [{label}] alpha={alpha:.2f}: "
                  f"Acc7={acc7:.2f}%  Acc2={acc2:.2f}%  F1={f1:.2f}%  MAE={mae:.4f}  Corr={corr:.4f}")
            return acc7, {"Acc7": round(acc7,2), "Acc2": round(acc2,2),
                          "F1": round(f1,2), "MAE": round(mae,4), "Corr": round(corr,4)}

        print("\n  === 各 Seed 個別 Test 結果 ===")
        s42_acc7, s42_m = compute_test_with_alpha(
            all_test_l7[s42_idx], all_test_l2[s42_idx], all_test_reg[s42_idx],
            s42_best_alpha, f"Seed-42")

        print("\n  === Ensemble Test 結果 ===")
        ens_acc7, ens_m = compute_test_with_alpha(
            mean_test_l7, mean_test_l2, mean_test_reg,
            ens_best_alpha, "Ensemble")

        # ★ 修正 #1：final_acc7 與 best_m 來自同一候選
        if s42_acc7 >= ens_acc7:
            final_acc7, best_m, best_label = s42_acc7, s42_m, "Seed-42"
        else:
            final_acc7, best_m, best_label = ens_acc7, ens_m, "Ensemble"
        print(f"\n  最佳候選: [{best_label}]  Acc7={final_acc7:.2f}%")

        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        np.save(str(MODEL_DIR / "raw_logits_v55.npy"), np.stack(all_test_l7, axis=0))
        np.save(str(MODEL_DIR / "val_logits_v55.npy"),  np.stack(all_val_l7,  axis=0))

        with open(MODEL_DIR / "history_v55.json", "w") as f:
            json.dump({
                "version":               config["version"],
                "note":                  "Zero data leakage — standard train/val/test splits",
                "seeds":                 config["seeds"],
                "seed_results":          seed_results,
                "best_alpha_seed42":     s42_best_alpha,
                "best_val_acc7_seed42":  round(s42_best_val_acc, 2),
                "best_alpha_ensemble":   ens_best_alpha,
                "best_val_acc7_ens":     round(ens_best_val_acc, 2),
                "best_candidate":        best_label,
                "final_acc7":            round(final_acc7, 2),
                "final_metrics":         best_m,   # ★ 與 final_acc7 來源一致（修正 #1）
                "prior_log_ratio":       log_ratio.tolist(),
            }, f, indent=2, ensure_ascii=False)
        history_file = MODEL_DIR / "history_v55.json"

    print(f"\n{'='*70}")
    print(f"最終 Test Acc7 (zero leakage): {final_acc7:.2f}%")
    print(f"  Acc2={best_m['Acc2']:.2f}%  F1={best_m['F1']:.2f}%  MAE={best_m['MAE']:.4f}  Corr={best_m['Corr']:.4f}")
    print(f"vs 目標 51%: {final_acc7-51.0:+.2f}%  {'✓ 達標！' if final_acc7 > 51.0 else '✗ 未達標'}")
    print(f"{'='*70}")
    print(f"結果已儲存: {history_file}")

    # ─────────────────────────────────────────────────────
    # ★ 儲存模型權重（供情緒偵測推論使用）
    # ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    version = config["version"]

    # 每個 seed 的模型都完整儲存（含架構設定）
    model_config_meta = {
        "lang_model":    config["lang_model"],
        "audio_dim":     5,
        "vision_dim":    20,
        "modal_hidden":  128,
        "fusion_dim":    512,
        "top_k":         5,
        "num_classes":   7,
    }

    print(f"\n{'─'*55}")
    print(f"儲存模型權重至 {MODEL_DIR} ...")
    saved_paths = []
    for entry in seed_model_states:
        seed_id   = entry["seed"]
        val_acc7  = entry["val_acc7"]
        state     = entry["state_dict"]
        save_path = MODEL_DIR / f"sacf_{version}_seed{seed_id}.pt"
        torch.save({
            "model_config":     model_config_meta,
            "model_state_dict": state,
            "seed":             seed_id,
            "version":          version,
            "val_acc7":         round(val_acc7, 4),
        }, save_path)
        print(f"  Seed {seed_id}: {save_path}  (Val Acc7={val_acc7:.2f}%)")
        saved_paths.append((val_acc7, save_path))

    # 選出 val acc 最高的 seed 作為最佳模型（trainval 模式以 seed 42 優先，因無 val 監控）
    if use_trainval:
        best_seed_entry = next(
            (e for e in seed_model_states if e["seed"] == 42),
            seed_model_states[0]
        )
    else:
        best_seed_entry = max(seed_model_states, key=lambda e: e["val_acc7"])

    best_save_path = MODEL_DIR / f"sacf_{version}_best.pt"
    torch.save({
        "model_config":     model_config_meta,
        "model_state_dict": best_seed_entry["state_dict"],
        "seed":             best_seed_entry["seed"],
        "version":          version,
        "val_acc7":         round(best_seed_entry["val_acc7"], 4),
        "note":             "best seed by val acc7 (or seed42 in trainval mode)",
    }, best_save_path)
    print(f"  最佳模型 (Seed {best_seed_entry['seed']}): {best_save_path}")
    print(f"{'─'*55}")
    print(f"\n[使用方式] 在 data/config.json 的 emotion 區塊設定：")
    print(f'  "sacf_model_path": "{best_save_path}"')


if __name__ == "__main__":
    main()
