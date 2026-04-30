"""
SCAF Final — 多分支單一模型（Multi-Branch Single Model）
═══════════════════════════════════════════════════════════════════

本模型在「架構層面」就是一個單一模型，內部以 4 個並行分支提供多樣性，
端對端訓練、單一 forward pass、單一 sacf_final.pt 輸出。

【架構】
  輸入（文字 / 音訊 / 視覺）
        │
        ▼
  ┌────────────────────────────────────────┐
  │ DeBERTa-v3-large (24 層, 1024d)  [共享] │
  │ BiLSTM 音訊編碼器 (5 → 128)       [共享] │
  │ BiLSTM 視覺編碼器 (20 → 128)      [共享] │
  └────────────────────────────────────────┘
        │
        ├──→ Branch 1: PEA₁ + SACF₁ + Proj₁ + Heads₁
        ├──→ Branch 2: PEA₂ + SACF₂ + Proj₂ + Heads₂
        ├──→ Branch 3: PEA₃ + SACF₃ + Proj₃ + Heads₃
        └──→ Branch 4: PEA₄ + SACF₄ + Proj₄ + Heads₄
                          │
                  ┌───────┴────────┐
                  │  分支內部平均   │
                  │  (mean logits)  │
                  └────────────────┘
                          │
                  cls7 / cls2 / reg

【關鍵設計】
  ‧ 共享骨幹：DeBERTa（400M）只有一份，總參數量約 420M
  ‧ 4 個分支不同隨機初始化 → 不同收斂方向
  ‧ 每個分支獨立計算 cls7/cls2/reg 損失 → 強迫每個分支獨立勝任任務
  ‧ 同時，跨分支「mean logits」作為主要預測 → 集成增益
  ‧ 訓練完直接得到「一個模型」，無需後處理融合

【產出】
  emotion_system/models/sacf_final.pt   ← 單一檔案、單一 state_dict
"""

import os, math, json, pickle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl"
MODEL_DIR = PROJECT_ROOT / "emotion_system/models"
TASK_PROMPT = "Predict the sentiment intensity (-3 to 3, negative to positive) of the following text: "

NUM_BRANCHES = 4   # 內部分支數量；對應原 4 個訓練協議的概念


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class MOSIDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_text_len=80):
        self.tokenizer = tokenizer; self.max_text_len = max_text_len
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

    def __len__(self): return len(self.raw_text)
    def __getitem__(self, idx):
        enc = self.tokenizer(TASK_PROMPT + str(self.raw_text[idx]),
                             add_special_tokens=True, max_length=self.max_text_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        aud_len = min(int(self.audio_lengths[idx]), self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])
        aud_mask = torch.zeros(self.audio.shape[1]); aud_mask[:aud_len] = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len] = 1.0
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "audio": self.audio[idx], "audio_mask": aud_mask,
                "vision": self.vision[idx], "vision_mask": vis_mask,
                "cls7_label": self.cls7_labels[idx],
                "cls2_label": self.cls2_labels[idx],
                "reg_label": self.reg_labels[idx]}


# ─────────────────────────────────────────────────────────────
# Module components
# ─────────────────────────────────────────────────────────────
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
        self.audio_map = nn.Linear(modal_dim, lang_dim)
        self.vision_map = nn.Linear(modal_dim, lang_dim)
        self.token_attn = nn.Linear(lang_dim, 1)
        self.ffn = nn.Sequential(
            nn.Linear(lang_dim, lang_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lang_dim//2, lang_dim))
        self.gate = nn.Linear(lang_dim*2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(lang_dim)
    def forward(self, xl_hidden, xl_cls, gates, xa, xv):
        B, L, H = xl_hidden.shape
        topk_idx = gates.topk(min(self.top_k, L), dim=1).indices
        topk_h = xl_hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, H))
        w = F.softmax(self.token_attn(topk_h), dim=1)
        sa_q = (topk_h*w).sum(1)
        kv = torch.stack([self.audio_map(xa), self.vision_map(xv)], dim=1)
        attn = F.softmax(torch.bmm(sa_q.unsqueeze(1), kv.transpose(1, 2))/(H**0.5), dim=-1)
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
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[-2], h[-1]], dim=-1))


# ═══════════════════════════════════════════════════════════════
# 多分支單一模型 (Multi-Branch Single Model)
# ═══════════════════════════════════════════════════════════════
class SACFFinalModel(nn.Module):
    """
    SACF 最終模型 — 多分支單一模型架構。
    從外部看：一個 nn.Module、一個 forward、一個 state_dict。
    從內部看：共享骨幹 + 共享模態編碼器 + N 個並行融合/預測分支。
    """
    def __init__(self, lang_model="microsoft/deberta-v3-large",
                 audio_dim=5, vision_dim=20, modal_hidden=128,
                 fusion_dim=512, top_k=5, num_classes=7, dropout=0.15,
                 num_branches=NUM_BRANCHES):
        super().__init__()
        self.num_branches = num_branches

        # ── 共享 ──
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size
        self.audio_encoder = ModalityEncoder(audio_dim, modal_hidden, 2, dropout)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden, 2, dropout)

        # ── 每分支獨立 ──
        # 每個分支用稍微不同的 dropout 率（產生更多差異）
        per_branch_dropouts = [dropout, dropout * 1.3, dropout * 0.7, dropout * 1.5][:num_branches]
        self.pea = nn.ModuleList([
            PolarityEnhancedAttention(lang_dim, per_branch_dropouts[i])
            for i in range(num_branches)
        ])
        self.sacf = nn.ModuleList([
            SentimentAwareCrossModalAttention(lang_dim, modal_hidden, top_k, per_branch_dropouts[i])
            for i in range(num_branches)
        ])
        self.shared_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(lang_dim, fusion_dim), nn.LayerNorm(fusion_dim),
                          nn.GELU(), nn.Dropout(per_branch_dropouts[i]))
            for i in range(num_branches)
        ])
        self.cls7_heads = nn.ModuleList([nn.Linear(fusion_dim, num_classes) for _ in range(num_branches)])
        self.cls2_heads = nn.ModuleList([nn.Linear(fusion_dim, 2) for _ in range(num_branches)])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
                          nn.Linear(fusion_dim//2, 1), nn.Tanh())
            for _ in range(num_branches)
        ])

        # 不同分支的初始化偏置（小幅擾動）以加速分支差異化
        for i, head in enumerate(self.cls7_heads):
            with torch.no_grad():
                head.weight.add_(torch.randn_like(head.weight) * 0.001 * (i + 1))

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask,
                return_per_branch=False):
        # 共享編碼
        audio = F.normalize(torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        vision = F.normalize(torch.nan_to_num(vision, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        hidden = self.lang_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)

        # 4 分支並行
        l7_list, l2_list, reg_list = [], [], []
        for i in range(self.num_branches):
            xl_cls, gates = self.pea[i](hidden, attention_mask)
            fused = self.sacf[i](hidden, xl_cls, gates, xa, xv)
            feat = self.shared_proj[i](fused)
            l7_list.append(self.cls7_heads[i](feat))
            l2_list.append(self.cls2_heads[i](feat))
            reg_list.append(self.reg_heads[i](feat).squeeze(-1) * 3.0)

        # 內部 ensemble：mean logits
        l7_mean = torch.stack(l7_list).mean(0)
        l2_mean = torch.stack(l2_list).mean(0)
        reg_mean = torch.stack(reg_list).mean(0)

        if return_per_branch:
            return l7_mean, l2_mean, reg_mean, l7_list, l2_list, reg_list
        return l7_mean, l2_mean, reg_mean


# ─────────────────────────────────────────────────────────────
# Losses & EMA
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.label_smoothing = label_smoothing
    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight,
                              label_smoothing=self.label_smoothing, reduction='none')
        with torch.no_grad():
            log_pt = F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            focal = (1.0 - log_pt.exp()) ** self.gamma
        return (focal * ce).mean()


class OrdinalEMDLoss(nn.Module):
    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        cdf_pred = probs.cumsum(dim=-1)[:, :-1]
        cdf_true = F.one_hot(labels, 7).float().cumsum(dim=-1)[:, :-1]
        return (cdf_pred - cdf_true).abs().mean()


class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model; self.decay = decay
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


def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.where((c := np.bincount(cl, minlength=n).astype(float)) == 0, 1.0, c)
    return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))


# ─────────────────────────────────────────────────────────────
# Training: per-branch loss + diversity regularization
# ─────────────────────────────────────────────────────────────
def diversity_loss(l7_list):
    """
    輕度多樣性正則化：penalize 分支之間預測完全相同（避免分支退化為相同函數）。
    用「分支間 logits 相關性」的負值，越接近 0 越好。
    """
    n = len(l7_list)
    if n < 2: return torch.tensor(0.0, device=l7_list[0].device)
    # Cosine sim between branches (low sim = high diversity)
    flats = [F.normalize(l.flatten(1), dim=1) for l in l7_list]
    sim_sum = 0.0; count = 0
    for i in range(n):
        for j in range(i+1, n):
            sim_sum += (flats[i] * flats[j]).sum(1).mean()
            count += 1
    return sim_sum / max(count, 1)


def train_epoch(model, loader, cls7_crit, cls2_crit, reg_crit, optimizer, scheduler, device, scaler, ema,
                rdrop_alpha=0.05, emd_crit=None, emd_weight=0.25, diversity_weight=0.01):
    model.train(); total_loss = 0.0; nan_count = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device); amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device); vmask = batch["vision_mask"].to(device)
        cl7 = batch["cls7_label"].to(device); cl2 = batch["cls2_label"].to(device)
        rl = batch["reg_label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            l7_mean, l2_mean, reg_mean, l7_list, l2_list, reg_list = model(
                ids, mask, aud, amask, vis, vmask, return_per_branch=True)

            # 每分支獨立任務損失（強迫每個分支獨立勝任）
            per_branch_loss = 0.0
            for l7, l2, reg in zip(l7_list, l2_list, reg_list):
                bcls7 = cls7_crit(l7, cl7)
                if emd_crit is not None and emd_weight > 0:
                    bcls7 = (1 - emd_weight) * bcls7 + emd_weight * emd_crit(l7, cl7)
                per_branch_loss = per_branch_loss + bcls7 + 0.3 * cls2_crit(l2, cl2) + 0.4 * reg_crit(reg, rl)
            per_branch_loss = per_branch_loss / model.num_branches

            # 分支平均輸出的損失（主目標）
            mean_cls7 = cls7_crit(l7_mean, cl7)
            if emd_crit is not None and emd_weight > 0:
                mean_cls7 = (1 - emd_weight) * mean_cls7 + emd_weight * emd_crit(l7_mean, cl7)
            mean_loss = mean_cls7 + 0.3 * cls2_crit(l2_mean, cl2) + 0.4 * reg_crit(reg_mean, rl)

            # 多樣性正則化
            div_loss = diversity_loss(l7_list)

            loss = 0.5 * mean_loss + 0.5 * per_branch_loss + diversity_weight * div_loss

            # R-Drop
            if rdrop_alpha > 0:
                l7_mean_b, _, _ = model(ids, mask, aud, amask, vis, vmask)
                kl = (F.kl_div(F.log_softmax(l7_mean,-1), F.softmax(l7_mean_b,-1).detach(), reduction='batchmean') +
                      F.kl_div(F.log_softmax(l7_mean_b,-1), F.softmax(l7_mean,-1).detach(), reduction='batchmean'))/2
                loss = loss + rdrop_alpha * kl

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1; scheduler.step(); continue

        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        ema.update(); scheduler.step()
        if hasattr(scheduler, '_new_group_sched'):
            scheduler._new_group_sched.step()
        total_loss += loss.item()
    if nan_count > 0:
        print(f"  [warn] {nan_count} NaN batches", end="")
    return total_loss / max(len(loader) - nan_count, 1)


def get_test_outputs_tta(model, loader, device, n_tta=5):
    """TTA inference for the ENTIRE multi-branch model."""
    if n_tta <= 1:
        model.eval()
        l7s, l2s, regs = [], [], []
        with torch.no_grad():
            for b in loader:
                ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
                aud = b["audio"].to(device); amask = b["audio_mask"].to(device)
                vis = b["vision"].to(device); vmask = b["vision_mask"].to(device)
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
                l7s.append(l7.cpu().float().numpy()); l2s.append(l2.cpu().float().numpy()); regs.append(reg.cpu().float().numpy())
        return np.concatenate(l7s), np.concatenate(l2s), np.concatenate(regs)
    runs7, runs2, runsr = [], [], []
    model.train()
    for _ in range(n_tta):
        l7s, l2s, regs = [], [], []
        for b in loader:
            ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
            aud = b["audio"].to(device); amask = b["audio_mask"].to(device)
            vis = b["vision"].to(device); vmask = b["vision_mask"].to(device)
            with torch.no_grad():
                l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            l7s.append(l7.cpu().float().numpy()); l2s.append(l2.cpu().float().numpy()); regs.append(reg.cpu().float().numpy())
        runs7.append(np.concatenate(l7s)); runs2.append(np.concatenate(l2s)); runsr.append(np.concatenate(regs))
    model.eval()
    return np.mean(runs7, axis=0), np.mean(runs2, axis=0), np.mean(runsr, axis=0)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SCAF Final — 多分支單一模型訓練")
    print(f"  分支數：{NUM_BRANCHES}  |  共享：DeBERTa + BiLSTM  |  每分支：PEA + SACF + heads")
    print("=" * 70)

    config = {
        "lang_model": "microsoft/deberta-v3-large",
        "batch_size": 8, "num_epochs": 60,
        "lang_lr": 4e-6, "head_lr": 8e-5,
        "weight_decay": 0.01, "dropout": 0.15,
        "label_smoothing": 0.05, "rdrop_alpha": 0.05,
        "focal_gamma": 2.0,
        "swa_start": 42, "swa_step": 2,
        "emd_weight": 0.25, "diversity_weight": 0.01,
        "n_tta": 5, "seed": 42,
    }

    set_seed(config["seed"])

    print(f"\n載入資料：{DATA_PATH}")
    with open(DATA_PATH, "rb") as f: data = pickle.load(f)
    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])
    train_ds = MOSIDataset(data["train"], tokenizer)
    val_ds = MOSIDataset(data["valid"], tokenizer)
    test_ds = MOSIDataset(data["test"], tokenizer)
    print(f"  Train={len(train_ds)} Valid={len(val_ds)} Test={len(test_ds)}")

    bs = config["batch_size"]
    trainval_ds = ConcatDataset([train_ds, val_ds])
    train_loader = DataLoader(trainval_ds, bs, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=2, pin_memory=True)

    all_train_labels = np.concatenate([data["train"]["regression_labels"],
                                        data["valid"]["regression_labels"]])
    class_weights = compute_class_weights(all_train_labels)

    test_labels_np = np.array(data["test"]["regression_labels"])
    test_cls7_true = np.clip(np.round(test_labels_np).astype(int), -3, 3) + 3
    test_cls2_true = (test_labels_np >= 0).astype(int)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  設備：{device}")

    # 建立模型
    model = SACFFinalModel(dropout=config["dropout"], num_branches=NUM_BRANCHES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  總參數量：{n_params/1e6:.1f}M（其中 4 分支 PEA+SACF+heads 約 {(n_params - 400e6)/1e6:.1f}M）")

    # 凍結 DeBERTa 下層 6 層
    for i in range(6):
        for p in model.lang_backbone.encoder.layer[i].parameters():
            p.requires_grad = False

    backbone_p = [p for n, p in model.named_parameters() if p.requires_grad and "lang_backbone" in n]
    head_p = [p for n, p in model.named_parameters() if p.requires_grad and "lang_backbone" not in n]
    optimizer = optim.AdamW([
        {"params": backbone_p, "lr": config["lang_lr"]},
        {"params": head_p, "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if "cuda" in device else None
    ema = EMA(model, 0.9995)

    cw = class_weights.to(device)
    cls7_crit = FocalLoss(config["focal_gamma"], cw, config["label_smoothing"])
    cls2_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    reg_crit = nn.SmoothL1Loss()
    emd_crit = OrdinalEMDLoss()

    swa_states = []
    for epoch in range(config["num_epochs"]):
        if epoch == config["num_epochs"] // 3 and not getattr(model, '_unfroze', False):
            new_p = []
            for i in range(6):
                for p in model.lang_backbone.encoder.layer[i].parameters():
                    p.requires_grad = True; new_p.append(p)
            optimizer.add_param_group({
                "params": new_p, "lr": config["lang_lr"]/2,
                "weight_decay": config["weight_decay"]
            })
            current_step = epoch * len(train_loader)
            new_idx = len(optimizer.param_groups) - 1

            def _cosine_lambda(step, base=current_step, w=warmup_steps, t=total_steps):
                a = base + step
                if a < w: return float(a) / float(max(1, w))
                pp = float(a - w) / float(max(1, t - w))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * pp)))

            from torch.optim.lr_scheduler import LambdaLR
            new_sched = LambdaLR(optimizer, lr_lambda=[lambda s: 1.0]*new_idx + [_cosine_lambda])
            scheduler._new_group_sched = new_sched
            ema.add_new_params(); model._unfroze = True
            print(f"  [E{epoch+1}] 解凍 6 下層")

        loss = train_epoch(model, train_loader, cls7_crit, cls2_crit, reg_crit,
                           optimizer, scheduler, device, scaler, ema,
                           rdrop_alpha=config["rdrop_alpha"],
                           emd_crit=emd_crit, emd_weight=config["emd_weight"],
                           diversity_weight=config["diversity_weight"])

        ep1 = epoch + 1
        if ep1 >= config["swa_start"] and (ep1 - config["swa_start"]) % config["swa_step"] == 0:
            ema.apply_shadow()
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            ema.restore()
            print(f"  E{ep1:02d} | Loss={loss:.4f} [SWA #{len(swa_states)}]")
        elif ep1 % 5 == 0 or ep1 <= 5:
            print(f"  E{ep1:02d} | Loss={loss:.4f}")

    # SWA 平均（產生最終 single state_dict）
    print(f"\n[SWA] 平均 {len(swa_states)} 個快照...")
    swa_state = {}
    for k in swa_states[0]:
        if swa_states[0][k].dtype.is_floating_point:
            swa_state[k] = torch.stack([s[k].float() for s in swa_states]).mean(0).to(swa_states[0][k].dtype)
        else:
            swa_state[k] = swa_states[-1][k]
    model.load_state_dict(swa_state); model.to(device)

    # 測試集評估（含 TTA）
    print(f"\n[Inference] TTA×{config['n_tta']}...")
    test_l7, test_l2, test_reg = get_test_outputs_tta(model, test_loader, device, n_tta=config["n_tta"])

    pred7 = test_l7.argmax(1); pred2 = test_l2.argmax(1)
    acc7 = (pred7 == test_cls7_true).mean() * 100
    acc2 = (pred2 == test_cls2_true).mean() * 100
    f1 = f1_score(test_cls2_true, pred2, average='weighted') * 100
    mae = np.abs(test_reg - test_labels_np).mean()
    corr = pearsonr(test_reg.astype(float), test_labels_np.astype(float))[0]
    within1 = (np.abs(pred7 - test_cls7_true) <= 1).mean() * 100

    print(f"\n╭─────────────────────────────────────────────────────╮")
    print(f"│  SCAF 單一模型最終結果（多分支內部 ensemble）         │")
    print(f"├─────────────────────────────────────────────────────┤")
    print(f"│  Acc-7      ：{acc7:>6.2f} %                          │")
    print(f"│  Acc-2      ：{acc2:>6.2f} %                          │")
    print(f"│  F1         ：{f1:>6.2f} %                          │")
    print(f"│  MAE        ：{mae:>6.4f}                            │")
    print(f"│  Corr       ：{corr:>6.4f}                            │")
    print(f"│  Within-1   ：{within1:>6.2f} %                          │")
    print(f"╰─────────────────────────────────────────────────────╯")

    # 儲存單一最終權重
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    final_path = MODEL_DIR / "sacf_final.pt"
    torch.save({
        "model_class": "SACFFinalModel",
        "num_branches": NUM_BRANCHES,
        "model_config": {
            "lang_model": config["lang_model"],
            "audio_dim": 5, "vision_dim": 20, "modal_hidden": 128,
            "fusion_dim": 512, "top_k": 5, "num_classes": 7,
            "dropout": config["dropout"], "num_branches": NUM_BRANCHES,
        },
        "model_state_dict": swa_state,
        "config": config,
        "metrics": {"Acc-7": round(acc7, 2), "Acc-2": round(acc2, 2),
                    "F1": round(f1, 2), "MAE": round(mae, 4),
                    "Corr": round(corr, 4), "Within-1": round(within1, 2)},
    }, str(final_path))
    print(f"\n  ✓ 單一模型儲存：{final_path}")
    print(f"  ✓ 大小：{os.path.getsize(final_path)/1024**3:.2f} GB")
    print(f"  ✓ 推斷使用：emotion_system/sacf_final_loader.py 的 SACFFinal")

    # 寫摘要
    with open(MODEL_DIR / "sacf_final_summary.json", "w") as f:
        json.dump({
            "model": "SACFFinalModel (multi-branch single model)",
            "num_branches": NUM_BRANCHES,
            "config": config,
            "metrics": {"Acc-7": round(acc7, 2), "Acc-2": round(acc2, 2),
                        "F1": round(f1, 2), "MAE": round(mae, 4),
                        "Corr": round(corr, 4), "Within-1": round(within1, 2)},
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
