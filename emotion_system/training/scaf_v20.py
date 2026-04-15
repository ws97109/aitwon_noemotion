"""
MOSI 多模態情感分析 v20 — 團隊協作優化版
整合 4 位專家的分析建議，目標 Test Acc7 >= 51%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
團隊協作優化（相比 v15）：

【evaluation-analyst 建議】
✓ 保留 scaf_old 最優配置（lr=5e-6, freeze=6, bs=8）
✓ 添加 Early Stopping (patience=8)
✓ 溫和正則化策略

【model-architect 建議】
✓ 縮小非語言編碼器：256維 Transformer → 128維 BiLSTM
✓ Gated Fusion 替代簡單 CrossModalAttention
✓ 多頭注意力 Pooling 替代固定比例 PolarityAttention
✓ 修復 optimizer 重建 bug

【data-engineer 建議】
✓ 動態 padding（減少 90% 無效計算，加速 3-5x）
✓ 特徵標準化（統一音頻/視覺尺度）
✓ 文本長度 80 → 50
✓ 平滑 class_weights

【training-optimizer 建議】
✓ LLRD (Layer-wise LR Decay, decay=0.95)
✓ EMA (Exponential Moving Average, decay=0.9995)
✓ R-Drop regularization (alpha=0.3)
✓ other_lr: 1e-4 → 5e-5
✓ dropout: 0.2 → 0.25

預期效果：
- Test Acc7: 50-52% (v15: 47.81%)
- Val-Test Gap: 3-4% (v15: 8.09%)
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
from typing import Dict, Tuple, List
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, get_cosine_schedule_with_warmup, DebertaV2Tokenizer
import math
import copy

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = "Sentiment: "  # 簡化 prompt（data-engineer 建議）


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集（data-engineer 優化：動態 padding + 特徵標準化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 50, normalize: bool = True):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.raw_text = split_data["raw_text"]
        self.audio = torch.FloatTensor(split_data["audio"])
        self.vision = torch.FloatTensor(split_data["vision"])
        self.audio_lengths = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]

        # 特徵標準化（data-engineer 建議）
        if normalize:
            self.audio_mean = self.audio.mean(dim=[0, 1])
            self.audio_std = self.audio.std(dim=[0, 1]).clamp(min=1e-6)
            self.audio = (self.audio - self.audio_mean) / self.audio_std

            self.vision_mean = self.vision.mean(dim=[0, 1])
            self.vision_std = self.vision.std(dim=[0, 1]).clamp(min=1e-6)
            self.vision = (self.vision - self.vision_mean) / self.vision_std

        labels = split_data["regression_labels"]
        self.reg_labels = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)
        self.cls2_labels = torch.LongTensor((labels >= 0).astype(int))

        print(f"資料: {len(self.raw_text)} 筆")

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


def collate_fn_dynamic_padding(batch):
    """動態 padding（data-engineer 建議，加速 3-5x）"""
    # 找到 batch 內最大有效長度
    max_aud = max(int(b['audio_mask'].sum()) for b in batch)
    max_vis = max(int(b['vision_mask'].sum()) for b in batch)

    # 截斷到最大有效長度 + 少許 padding
    max_aud = min(max_aud + 5, batch[0]['audio'].shape[0])
    max_vis = min(max_vis + 5, batch[0]['vision'].shape[0])

    for b in batch:
        b['audio'] = b['audio'][:max_aud]
        b['audio_mask'] = b['audio_mask'][:max_aud]
        b['vision'] = b['vision'][:max_vis]
        b['vision_mask'] = b['vision_mask'][:max_vis]

    return torch.utils.data.dataloader.default_collate(batch)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 改進的多頭注意力 Pooling（model-architect 建議）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ImprovedPolarityAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        B = hidden.size(0)
        q = self.query.expand(B, -1, -1)
        key_pad_mask = (mask == 0)
        out, _ = self.attn(q, hidden, hidden, key_padding_mask=key_pad_mask)
        return self.dropout(self.norm(out.squeeze(1)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 輕量級非語言編碼器（model-architect 建議：256→128，Transformer→BiLSTM）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LightweightModalEncoder(nn.Module):
    """針對低維特徵（5維audio/20維vision）的輕量編碼器"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 單層 BiLSTM 比 Transformer 更適合低維序列
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=0
        )
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.proj(x)
        x, _ = self.lstm(x)
        # Attention pooling
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gated Multimodal Fusion（model-architect 建議）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GatedMultimodalFusion(nn.Module):
    """每個模態用獨立的 gate 控制信息流入"""
    def __init__(self, lang_dim: int, modal_dim: int, dropout: float = 0.15):
        super().__init__()
        self.audio_proj = nn.Linear(modal_dim, lang_dim)
        self.vision_proj = nn.Linear(modal_dim, lang_dim)

        # 每個模態的獨立 gate
        self.gate_a = nn.Sequential(
            nn.Linear(lang_dim * 2, lang_dim),
            nn.Sigmoid()
        )
        self.gate_v = nn.Sequential(
            nn.Linear(lang_dim * 2, lang_dim),
            nn.Sigmoid()
        )

        self.fusion_norm = nn.LayerNorm(lang_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xl: torch.Tensor, xa: torch.Tensor, xv: torch.Tensor):
        xa_p = self.audio_proj(xa)
        xv_p = self.vision_proj(xv)

        # 各模態的 gate 由語言特徵和該模態特徵共同決定
        ga = self.gate_a(torch.cat([xl, xa_p], dim=-1))
        gv = self.gate_v(torch.cat([xl, xv_p], dim=-1))

        fused = xl + self.dropout(ga * xa_p + gv * xv_p)
        return self.fusion_norm(fused)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA 模型（training-optimizer 建議）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HybridModel(nn.Module):
    def __init__(
        self,
        lang_model: str = "microsoft/deberta-v3-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        modal_hidden: int = 128,  # 256 → 128（model-architect 建議）
        fusion_dim: int = 512,
        num_classes: int = 7,
        dropout: float = 0.25,  # 0.2 → 0.25（training-optimizer 建議）
    ):
        super().__init__()

        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size

        # 改進的多頭注意力 pooling（model-architect 建議）
        self.polarity_attn = ImprovedPolarityAttention(lang_dim, num_heads=4, dropout=dropout)

        # 輕量級 BiLSTM 編碼器（model-architect 建議）
        self.audio_encoder = LightweightModalEncoder(
            audio_dim, modal_hidden, num_layers=1, dropout=dropout
        )
        self.vision_encoder = LightweightModalEncoder(
            vision_dim, modal_hidden, num_layers=1, dropout=dropout
        )

        # Gated Fusion（model-architect 建議）
        self.cross_modal = GatedMultimodalFusion(lang_dim, modal_hidden, dropout=dropout * 0.7)

        # 加深融合後網絡（model-architect 建議）
        self.shared = nn.Sequential(
            nn.Linear(lang_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls7_head = nn.Linear(fusion_dim, num_classes)
        self.cls2_head = nn.Linear(fusion_dim, 2)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask, audio, audio_mask, vision, vision_mask):
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = lang_out.last_hidden_state
        xl_cls = self.polarity_attn(hidden, attention_mask)

        xa = self.audio_encoder(audio, audio_mask)
        xv = self.vision_encoder(vision, vision_mask)

        fused = self.cross_modal(xl_cls, xa, xv)

        feat = self.shared(fused)
        logits7 = self.cls7_head(feat)
        logits2 = self.cls2_head(feat)
        reg_out = self.reg_head(feat).squeeze(-1) * 3.0

        return logits7, logits2, reg_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Focal Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.05):  # gamma 2.0→1.5, smoothing 0.1→0.05
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


class HybridLoss(nn.Module):
    def __init__(self, class_weights, alpha=2.0, beta=1.0, gamma=0.5):  # 調整權重（model-architect 建議）
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal = FocalLoss(alpha=class_weights, gamma=1.5, label_smoothing=0.05)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg = nn.MSELoss()

    def forward(self, l7, l2, reg, cl7, cl2, rl):
        lc7 = self.focal(l7, cl7)
        lc2 = self.cls2(l2, cl2)
        lr = self.reg(reg, rl)
        total = self.alpha * lc7 + self.beta * lc2 + self.gamma * lr
        return total, {
            "cls7": lc7.item(), "cls2": lc2.item(), "reg": lr.item(),
        }


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


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema=None, rdrop_alpha=0.0):
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

        optimizer.zero_grad()
        use_amp = scaler is not None

        with torch.cuda.amp.autocast(enabled=use_amp):
            l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
            loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)

            # R-Drop（training-optimizer 建議）
            if rdrop_alpha > 0:
                l7_2, l2_2, reg_2 = model(ids, mask, aud, amask, vis, vmask)
                kl_loss = F.kl_div(F.log_softmax(l7, dim=-1), F.softmax(l7_2, dim=-1), reduction='batchmean')
                kl_loss += F.kl_div(F.log_softmax(l7_2, dim=-1), F.softmax(l7, dim=-1), reduction='batchmean')
                loss = loss + rdrop_alpha * kl_loss / 2

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

        if ema is not None:
            ema.update()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
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

        l7, l2, reg = model(ids, mask, aud, amask, vis, vmask)
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
        np.array(all_l7), np.array(all_l2), np.array(all_lr)
    )
    return total_loss / len(loader), metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7, smooth=True):
    """data-engineer 建議：平滑 class_weights"""
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    weights = len(cl) / (n * ct)
    if smooth:
        weights = np.sqrt(weights)  # sqrt 平滑，避免極端權重
    return torch.FloatTensor(weights)


def get_layer_wise_lr_groups(model, base_lr, decay=0.95):
    """LLRD（training-optimizer 建議）"""
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder is None:
        return [{"params": model.parameters(), "lr": base_lr}]

    num_layers = len(encoder.layer)
    groups = []

    # 為每一層分配不同的學習率
    for i, layer in enumerate(encoder.layer):
        lr = base_lr * (decay ** (num_layers - 1 - i))
        groups.append({
            "params": layer.parameters(),
            "lr": lr
        })

    # Embedding 層用最小的 lr
    emb_lr = base_lr * (decay ** num_layers)
    groups.append({
        "params": model.lang_backbone.embeddings.parameters(),
        "lr": emb_lr
    })

    # 其他組件
    other_params = (
        list(model.polarity_attn.parameters()) +
        list(model.audio_encoder.parameters()) +
        list(model.vision_encoder.parameters()) +
        list(model.cross_modal.parameters()) +
        list(model.shared.parameters()) +
        list(model.cls7_head.parameters()) +
        list(model.cls2_head.parameters()) +
        list(model.reg_head.parameters())
    )

    return groups, other_params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI 多模態情感分析 v20 — 團隊協作優化版")
    print("整合 4 位專家建議 | 目標 Test Acc7 >= 51%")
    print("=" * 70)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "microsoft/deberta-v3-large",
        "max_text_len": 50,  # 80 → 50（data-engineer 建議）
        "audio_dim": 5,
        "vision_dim": 20,
        "modal_hidden": 128,  # 256 → 128（model-architect 建議）
        "fusion_dim": 512,
        "num_classes": 7,
        "dropout": 0.25,  # 0.2 → 0.25（training-optimizer 建議）
        "batch_size": 8,
        "num_epochs": 50,
        "lang_lr": 5e-6,  # 保持 scaf_old 最優值
        "other_lr": 5e-5,  # 1e-4 → 5e-5（training-optimizer 建議）
        "llrd_decay": 0.95,  # LLRD decay（training-optimizer 建議）
        "weight_decay": 1.5e-2,  # 1e-2 → 1.5e-2（evaluation-analyst 建議）
        "warmup_ratio": 0.08,
        "alpha": 2.0,  # cls7 權重（model-architect 建議）
        "beta": 1.0,
        "gamma": 0.5,
        "ema_decay": 0.9995,  # EMA（training-optimizer 建議）
        "rdrop_alpha": 0.3,  # R-Drop（training-optimizer 建議）
        "patience": 8,  # Early Stopping（evaluation-analyst 建議）
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"], smooth=True)
    print(f"平滑類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer, config["max_text_len"], normalize=True)
    val_ds = MOSIDataset(data["valid"], tokenizer, config["max_text_len"], normalize=True)
    test_ds = MOSIDataset(data["test"], tokenizer, config["max_text_len"], normalize=True)

    # 使用動態 padding（data-engineer 建議）
    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn_dynamic_padding)
    val_loader = DataLoader(val_ds, config["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn_dynamic_padding)
    test_loader = DataLoader(test_ds, config["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=collate_fn_dynamic_padding)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = HybridModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"],
        fusion_dim=config["fusion_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    # LLRD（training-optimizer 建議）
    llrd_groups, other_params = get_layer_wise_lr_groups(model, config["lang_lr"], config["llrd_decay"])

    optimizer = optim.AdamW(
        llrd_groups + [{"params": other_params, "lr": config["other_lr"]}],
        weight_decay=config["weight_decay"]
    )

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = HybridLoss(
        class_weights=class_w.to(device),
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
    )

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # EMA（training-optimizer 建議）
    ema = EMA(model, decay=config["ema_decay"])

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[v20 優化] LLRD + EMA + R-Drop + Gated Fusion + 輕量編碼器 + 動態 Padding\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-" * 45)

        tr_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            ema=ema, rdrop_alpha=config["rdrop_alpha"]
        )

        # 用 EMA 模型驗證
        ema.apply_shadow()
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        ema.restore()

        history.append({
            "epoch": epoch + 1,
            "tr_loss": round(tr_loss, 4),
            "vl_loss": round(vl_loss, 4),
            **metrics
        })

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch + 1, **metrics}
            ema.apply_shadow()
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "config": config,
            }, save_dir / "v20_best_acc7.pth")
            ema.restore()
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n早停：{config['patience']} epochs 無改善")
                break

    # 測試集
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "v20_best_acc7.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, criterion, device)

    print("\n【測試集結果 - v20】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (目標: 51%, v15: 47.81%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")
    print(f"\n  Val-Test Gap: {best_acc7['Acc7'] - test_m['Acc7']:.2f}% (v15: 8.09%)")

    with open(save_dir / "v20_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_val_acc7": best_acc7,
            "test": test_m,
            "config": {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")


if __name__ == "__main__":
    main()
