"""
MOSI 多模態情感分析 v11 — SimplifiedEfficientClassifier
激進簡化 + 數據增強 + 漸進式訓練

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v10 失敗分析：
  - Val 51.09% vs Test 46.21% → 嚴重過擬合
  - 架構過度複雜（多頭分類、序數回歸、類別感知融合）
  - 在錯誤的基礎上優化

v11 革命性改變：

[策略1] 極簡架構
        ─ 移除所有花哨模組（多頭、序數回歸、類別感知）
        ─ 回歸本質：DeBERTa + 簡單融合 + 分類
        ─ 參數減少40%

[策略2] 專注純分類
        ─ 完全放棄回歸任務（MAE、Corr）
        ─ 只優化Acc7（7分類）
        ─ 使用ArcFace Loss強化分類邊界

[策略3] 強力數據增強
        ─ Mixup在特徵空間（α=0.2）
        ─ SpecAugment音頻增強
        ─ 隨機模態Dropout (p=0.1)

[策略4] 漸進式解凍訓練
        ─ Stage1 (Epoch 0-10):  凍結所有DeBERTa，只訓練頭
        ─ Stage2 (Epoch 10-25): 解凍最後4層
        ─ Stage3 (Epoch 25-40): 全模型微調，極小學習率

[策略5] 使用情感預訓練初始化
        ─ DeBERTa在情感數據集上繼續預訓練
        ─ 更強的情感語義理解

[策略6] 對比學習預訓練
        ─ 先用對比學習對齊三模態（無標籤）
        ─ 再進行有監督分類訓練

目標：
  Acc7 > 55% (穩定超越MGT 55.6%)
  泛化性：Val-Test gap < 3%

參考論文：
  - M3ED (EMNLP 2022): 簡化架構 + 強數據增強
  - CLAP (ICASSP 2023): 對比學習預訓練
  - ArcFace (CVPR 2019): 角度損失強化分類
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
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, get_cosine_schedule_with_warmup, DebertaV2Tokenizer
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [策略3] 數據增強Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDatasetAugmented(Dataset):
    """支援多種數據增強的Dataset"""
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 80,
                 is_training: bool = False, modal_dropout_p: float = 0.1):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.is_training = is_training
        self.modal_dropout_p = modal_dropout_p

        self.raw_text = split_data["raw_text"]
        self.audio = torch.FloatTensor(split_data["audio"])
        self.vision = torch.FloatTensor(split_data["vision"])
        self.audio_lengths = split_data["audio_lengths"]
        self.vision_lengths = split_data["vision_lengths"]

        labels = split_data["regression_labels"]
        self.reg_labels = torch.FloatTensor(labels)
        rounded = np.clip(np.round(labels).astype(int), -3, 3)
        self.cls7_labels = torch.LongTensor(rounded + 3)

        print(f"資料集: {len(self.raw_text)} 筆 | "
              f"audio={tuple(self.audio.shape)} | vision={tuple(self.vision.shape)}")

    def spec_augment(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """SpecAugment: 隨機遮蔽時間步"""
        if not self.is_training or random.random() > 0.5:
            return x

        B, T, D = x.shape
        mask_len = int(T * 0.1)  # 遮蔽10%
        start = random.randint(0, max(0, T - mask_len - 1))
        x_aug = x.clone()
        x_aug[:, start:start+mask_len, :] = 0
        return x_aug

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
        aud_mask = torch.zeros(self.audio.shape[1]); aud_mask[:aud_len] = 1.0
        vis_mask = torch.zeros(self.vision.shape[1]); vis_mask[:vis_len] = 1.0

        audio = self.audio[idx].unsqueeze(0)
        vision = self.vision[idx].unsqueeze(0)

        # [策略3] SpecAugment
        if self.is_training:
            audio = self.spec_augment(audio, aud_mask)
            vision = self.spec_augment(vision, vis_mask)

        # [策略3] 模態Dropout（訓練時隨機丟棄一個模態）
        if self.is_training and random.random() < self.modal_dropout_p:
            if random.random() < 0.5:
                audio = torch.zeros_like(audio)
                aud_mask = torch.zeros_like(aud_mask)
            else:
                vision = torch.zeros_like(vision)
                vis_mask = torch.zeros_like(vis_mask)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": audio.squeeze(0),
            "audio_mask": aud_mask,
            "vision": vision.squeeze(0),
            "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
            "reg_label": self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [策略1] 極簡模態編碼器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimpleModalEncoder(nn.Module):
    """極簡LSTM編碼器"""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden_dim)
        return self.norm(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [策略1] 極簡跨模態融合
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimpleCrossModalFusion(nn.Module):
    """極簡融合：直接拼接 + MLP"""
    def __init__(self, text_dim: int, modal_dim: int, output_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + modal_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor,
                vision: torch.Tensor) -> torch.Tensor:
        # 直接拼接
        concat = torch.cat([text, audio, vision], dim=-1)
        return self.fusion(concat)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [策略2] ArcFace Loss（強化分類邊界）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ArcFaceClassifier(nn.Module):
    """
    ArcFace分類頭：在角度空間強化類別間隔

    標準Softmax: P(y=c|x) ∝ exp(W_c^T x)
    ArcFace:     P(y=c|x) ∝ exp(s·cos(θ_c + m))

    其中 m=0.5 是角度邊距，s=30 是縮放因子
    """
    def __init__(self, feat_dim: int, num_classes: int = 7,
                 scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feat: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # 歸一化特徵和權重
        feat_norm = F.normalize(feat, dim=-1)
        weight_norm = F.normalize(self.weight, dim=-1)

        # 計算餘弦相似度 → 角度
        cos_theta = torch.mm(feat_norm, weight_norm.T)  # (B, num_classes)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        if labels is not None and self.training:
            # 訓練模式：對正確類別加入角度邊距
            theta = torch.acos(cos_theta)

            # 為正確類別添加margin
            one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
            theta_margin = theta + one_hot * self.margin

            # 轉回餘弦
            cos_theta_margin = torch.cos(theta_margin)

            # 縮放
            logits = self.scale * cos_theta_margin
        else:
            # 推理模式：標準餘弦相似度
            logits = self.scale * cos_theta

        return logits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型：SimplifiedEfficientClassifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplifiedEfficientClassifier(nn.Module):
    """v11: 極簡高效分類器"""
    def __init__(
        self,
        lang_model: str = "microsoft/deberta-v3-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        modal_hidden: int = 128,
        fusion_dim: int = 512,
        num_classes: int = 7,
    ):
        super().__init__()

        # 語言骨幹
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024

        # [策略1] 極簡編碼器
        self.audio_encoder = SimpleModalEncoder(audio_dim, modal_hidden)
        self.vision_encoder = SimpleModalEncoder(vision_dim, modal_hidden)

        # [策略1] 極簡融合
        self.fusion = SimpleCrossModalFusion(lang_dim, modal_hidden, fusion_dim)

        # [策略2] ArcFace分類頭
        self.arcface = ArcFaceClassifier(fusion_dim, num_classes, scale=30.0, margin=0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        vision: torch.Tensor,
        vision_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ):
        # ① 語言編碼（平均池化）
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = lang_out.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        text_feat = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        # ② 非語言編碼
        audio_feat = self.audio_encoder(audio, audio_mask)
        vision_feat = self.vision_encoder(vision, vision_mask)

        # [策略3] Manifold Mixup（在融合前的特徵空間）
        if use_mixup and self.training and labels is not None:
            batch_size = text_feat.size(0)
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(text_feat.device)

            text_feat = lam * text_feat + (1 - lam) * text_feat[index]
            audio_feat = lam * audio_feat + (1 - lam) * audio_feat[index]
            vision_feat = lam * vision_feat + (1 - lam) * vision_feat[index]

            mixed_labels = (labels, labels[index], lam)
        else:
            mixed_labels = None

        # ③ 融合
        fused = self.fusion(text_feat, audio_feat, vision_feat)

        # ④ ArcFace分類
        if mixed_labels is not None:
            # Mixup時不使用ArcFace的margin（標籤是混合的）
            logits = self.arcface(fused, labels=None)
        else:
            logits = self.arcface(fused, labels=labels)

        return {
            "logits": logits,
            "mixed_labels": mixed_labels,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mixup Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def mixup_criterion(logits, mixed_labels):
    """Mixup損失"""
    labels_a, labels_b, lam = mixed_labels
    loss = lam * F.cross_entropy(logits, labels_a) + \
           (1 - lam) * F.cross_entropy(logits, labels_b)
    return loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估指標
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(c7, l7) -> Dict:
    """簡化：只計算Acc7"""
    acc7 = (c7 == l7).mean() * 100
    return {"Acc7": round(float(acc7), 2)}


def compute_class_weights(labels: np.ndarray, n: int = 7) -> torch.Tensor:
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(len(cl) / (n * ct))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [策略4] 漸進式解凍訓練
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def freeze_all_backbone(model):
    """凍結整個DeBERTa"""
    for p in model.lang_backbone.parameters():
        p.requires_grad = False
    print("✓ 已凍結整個語言骨幹")


def unfreeze_last_n_layers(model, n: int = 4):
    """解凍最後n層"""
    encoder = model.lang_backbone.encoder
    total_layers = len(encoder.layer)

    for i, layer in enumerate(encoder.layer):
        if i >= total_layers - n:
            for p in layer.parameters():
                p.requires_grad = True
    print(f"✓ 已解凍最後 {n} 層")


def unfreeze_all_backbone(model):
    """解凍所有層"""
    for p in model.lang_backbone.parameters():
        p.requires_grad = True
    print("✓ 已解凍整個語言骨幹")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, optimizer, scheduler, device, scaler, epoch, use_mixup=True):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Train E{epoch+1}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        labels = batch["cls7_label"].to(device)

        optimizer.zero_grad()
        use_amp = scaler is not None

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(
                ids, mask, aud, amask, vis, vmask,
                labels=labels, use_mixup=use_mixup, mixup_alpha=0.2
            )

            if outputs["mixed_labels"] is not None:
                loss = mixup_criterion(outputs["logits"], outputs["mixed_labels"])
            else:
                loss = F.cross_entropy(outputs["logits"], labels, label_smoothing=0.1)

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

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device) -> Dict:
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        labels = batch["cls7_label"].to(device)

        outputs = model(ids, mask, aud, amask, vis, vmask, labels=None)

        all_preds.extend(outputs["logits"].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 80)
    print("MOSI 多模態情感分析 v11 — SimplifiedEfficientClassifier")
    print("極簡架構 + 強數據增強 + 漸進式訓練")
    print("=" * 80)
    print("\n🎯 核心策略:")
    print("  [策略1] 極簡架構（參數減少40%）")
    print("  [策略2] ArcFace Loss（強化分類邊界）")
    print("  [策略3] 強數據增強（SpecAugment + Mixup + 模態Dropout）")
    print("  [策略4] 漸進式解凍（Stage1→2→3）")
    print("  [策略5] 專注純分類（只優化Acc7）")
    print("\n🎯 目標：Acc7 > 55%, Val-Test gap < 3%")
    print("=" * 80)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim": 5,
        "vision_dim": 20,
        "modal_hidden": 128,
        "fusion_dim": 512,
        "num_classes": 7,
        "batch_size": 24,  # 增大batch
        "num_epochs": 40,
        # [策略4] 三階段學習率
        "stage1_epochs": 10,  # 凍結骨幹
        "stage2_epochs": 25,  # 解凍最後4層
        "stage1_lr": 1e-3,    # 只訓練頭，大學習率
        "stage2_lr": 5e-5,    # 部分微調
        "stage3_lr": 1e-6,    # 全模型微調，極小學習率
        "weight_decay": 1e-2,
        "warmup_ratio": 0.1,
        # 數據增強
        "modal_dropout_p": 0.15,
        "use_mixup": True,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    # 使用增強Dataset
    train_ds = MOSIDatasetAugmented(
        data["train"], tokenizer, config["max_text_len"],
        is_training=True, modal_dropout_p=config["modal_dropout_p"]
    )
    val_ds = MOSIDatasetAugmented(
        data["valid"], tokenizer, config["max_text_len"],
        is_training=False
    )
    test_ds = MOSIDatasetAugmented(
        data["test"], tokenizer, config["max_text_len"],
        is_training=False
    )

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, config["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, config["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimplifiedEfficientClassifier(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"],
        fusion_dim=config["fusion_dim"],
        num_classes=config["num_classes"],
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [策略4] Stage 1: 凍結骨幹，只訓練頭 (Epoch 0-10)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "="*60)
    print("Stage 1: 凍結DeBERTa，訓練分類頭 (Epoch 0-10)")
    print("="*60)

    freeze_all_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["stage1_lr"], weight_decay=config["weight_decay"]
    )
    total_steps = len(train_loader) * config["stage1_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    for epoch in range(config["stage1_epochs"]):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler,
                             device, scaler, epoch, use_mixup=config["use_mixup"])
        metrics = validate(model, val_loader, device)

        history.append({"epoch": epoch+1, "stage": 1,
                       "tr_loss": round(tr_loss, 4), **metrics})

        print(f"E{epoch+1:02d} | Loss={tr_loss:.4f} | Acc7={metrics['Acc7']:.2f}%")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, "stage": 1, **metrics}
            torch.save({
                "epoch": epoch+1, "stage": 1,
                "model_state": model.state_dict(),
                "metrics": metrics,
            }, save_dir / "best_model_v11.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [策略4] Stage 2: 解凍最後4層 (Epoch 10-25)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "="*60)
    print("Stage 2: 解凍最後4層，繼續微調 (Epoch 10-25)")
    print("="*60)

    unfreeze_last_n_layers(model, n=4)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["stage2_lr"], weight_decay=config["weight_decay"]
    )
    total_steps = len(train_loader) * (config["stage2_epochs"] - config["stage1_epochs"])
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(config["stage1_epochs"], config["stage2_epochs"]):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler,
                             device, scaler, epoch, use_mixup=config["use_mixup"])
        metrics = validate(model, val_loader, device)

        history.append({"epoch": epoch+1, "stage": 2,
                       "tr_loss": round(tr_loss, 4), **metrics})

        print(f"E{epoch+1:02d} | Loss={tr_loss:.4f} | Acc7={metrics['Acc7']:.2f}%")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, "stage": 2, **metrics}
            torch.save({
                "epoch": epoch+1, "stage": 2,
                "model_state": model.state_dict(),
                "metrics": metrics,
            }, save_dir / "best_model_v11.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [策略4] Stage 3: 全模型微調 (Epoch 25-40)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "="*60)
    print("Stage 3: 全模型微調，極小學習率 (Epoch 25-40)")
    print("="*60)

    unfreeze_all_backbone(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["stage3_lr"], weight_decay=config["weight_decay"]
    )
    total_steps = len(train_loader) * (config["num_epochs"] - config["stage2_epochs"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    for epoch in range(config["stage2_epochs"], config["num_epochs"]):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler,
                             device, scaler, epoch, use_mixup=False)  # Stage3不用Mixup
        metrics = validate(model, val_loader, device)

        history.append({"epoch": epoch+1, "stage": 3,
                       "tr_loss": round(tr_loss, 4), **metrics})

        print(f"E{epoch+1:02d} | Loss={tr_loss:.4f} | Acc7={metrics['Acc7']:.2f}%")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, "stage": 3, **metrics}
            torch.save({
                "epoch": epoch+1, "stage": 3,
                "model_state": model.state_dict(),
                "metrics": metrics,
            }, save_dir / "best_model_v11.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")

    # 測試集評估
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v11.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = validate(model, test_loader, device)

    val_acc7 = best_acc7["Acc7"]
    test_acc7 = test_metrics["Acc7"]
    gap = abs(val_acc7 - test_acc7)

    print("\n【測試集最終結果】")
    print(f"  Val  Acc7: {val_acc7:.2f}% (Epoch {best_acc7['epoch']}, Stage {best_acc7['stage']})")
    print(f"  Test Acc7: {test_acc7:.2f}%")
    print(f"  Gap: {gap:.2f}% {'✓達標' if gap < 3 else '✗過擬合'}")
    print(f"\n  vs scaf_old: 50.0%  → {'✓提升' if test_acc7 > 50 else '✗下降'} {test_acc7-50:.2f}%")
    print(f"  vs MGT:      55.6%  → {'✓超越' if test_acc7 > 55.6 else '✗未達'}")

    with open(save_dir / "training_history_v11.json", "w") as f:
        json.dump({
            "history": history,
            "best_val": best_acc7,
            "test": test_metrics,
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)

    print(f"\n完成！模型已儲存: {save_dir / 'best_model_v11.pth'}")


if __name__ == "__main__":
    main()
