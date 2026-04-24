"""
MOSI 多模態情感分析 v31 — Simplified Fusion Architecture
基於 v30 的修復版本

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v30 問題診斷:
  ❌ 對比學習維度不匹配 (text:1024 vs audio/vision:80)
  ❌ Phase 1 準確率過低 (21-26%)
  ❌ Two-phase training 過於複雜
  ❌ 學習率太高 (1e-3)

v31 關鍵修復:
  ✅ 添加特徵投影層,統一維度到 256
  ✅ 簡化為單階段訓練
  ✅ 降低學習率到 5e-5 (DeBERTa 友好)
  ✅ 減小 batch size 到 32
  ✅ 保留核心 SOTA 技術:
      • MAG (Multimodal Adaptation Gate)
      • Directed Pairwise Attention (MulG)
      • Feature Cascading Fusion
      • InfoNCE Contrastive Learning (修復維度)

目標: Acc7 > 55% (超越 MGT baseline)
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
from typing import Dict, Tuple
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_text_len: int = 80):
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

        print(f"資料集: {len(self.raw_text)} 筆 | "
              f"audio={tuple(self.audio.shape)} | vision={tuple(self.vision.shape)}")

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

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx],
            "audio_mask": aud_mask,
            "vision": self.vision[idx],
            "vision_mask": vis_mask,
            "cls7_label": self.cls7_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRU序列編碼器 (from MulG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_dim * 2  # 128

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        seq_out, h = self.gru(packed)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(
            seq_out, batch_first=True, total_length=x.size(1)
        )

        h_forward = h[-2]
        h_backward = h[-1]
        pooled = torch.cat([h_forward, h_backward], dim=-1)

        return seq_out, pooled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multimodal Adaptation Gate (from MAG-BERT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultimodalAdaptationGate(nn.Module):
    def __init__(self, text_dim: int, nonverbal_dim: int):
        super().__init__()
        self.W_audio = nn.Linear(nonverbal_dim, text_dim)
        self.W_vision = nn.Linear(nonverbal_dim, text_dim)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, text_hidden: torch.Tensor,
                audio_feat: torch.Tensor, vision_feat: torch.Tensor):
        """
        Args:
            text_hidden: (B, L, text_dim)
            audio_feat: (B, nonverbal_dim)
            vision_feat: (B, nonverbal_dim)
        """
        audio_proj = self.W_audio(audio_feat)
        vision_proj = self.W_vision(vision_feat)
        shift = torch.tanh(audio_proj + vision_proj).unsqueeze(1)
        return text_hidden + self.alpha * shift


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Directed Pairwise Cross-Modal Attention (from MulG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DirectedPairwiseCrossModalAttention(nn.Module):
    def __init__(self, dim1: int, dim2: int, dropout: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim1, dim2))
        nn.init.xavier_uniform_(self.W)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1: torch.Tensor, seq2: torch.Tensor,
                mask1: torch.Tensor, mask2: torch.Tensor):
        B, T1, D1 = seq1.shape
        T2 = seq2.size(1)

        attn_scores = torch.bmm(
            torch.matmul(seq1, self.W),
            seq2.transpose(1, 2)
        )

        # M1 → M2
        mask2_expanded = mask2.unsqueeze(1).expand(B, T1, T2)
        attn_scores_12 = attn_scores.masked_fill(mask2_expanded == 0, float('-inf'))
        attn_weights_12 = F.softmax(attn_scores_12, dim=2)
        attn_weights_12 = self.dropout(attn_weights_12)
        attended_seq1 = torch.bmm(attn_weights_12, seq2)

        # M2 → M1
        mask1_expanded = mask1.unsqueeze(2).expand(B, T1, T2)
        attn_scores_21 = attn_scores.transpose(1, 2)
        attn_scores_21 = attn_scores_21.masked_fill(
            mask1_expanded.transpose(1, 2) == 0, float('-inf')
        )
        attn_weights_21 = F.softmax(attn_scores_21, dim=2)
        attn_weights_21 = self.dropout(attn_weights_21)
        attended_seq2 = torch.bmm(attn_weights_21, seq1)

        return attended_seq1, attended_seq2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# InfoNCE Contrastive Loss (修復版 - 統一維度)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor):
        """兩個特徵必須是相同維度 (B, D)"""
        B = feat1.size(0)
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        logits = torch.mm(feat1, feat2.T) / self.temperature
        labels = torch.arange(B, device=feat1.device)

        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)

        return (loss_12 + loss_21) / 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型: SimplifiedFusionModel (v31)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplifiedFusionModel(nn.Module):
    """
    v31: 簡化並修復 v30 的問題

    關鍵改進:
      1. 特徵投影層 → 統一維度到 proj_dim (256)
      2. 單階段訓練 → 移除複雜的 two-phase
      3. 保留核心技術: MAG + Directed Attention + Contrastive
    """
    def __init__(
        self,
        lang_model: str = "microsoft/deberta-v3-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        gru_hidden: int = 64,
        proj_dim: int = 256,
        num_classes: int = 7,
    ):
        super().__init__()

        # ① 語言骨幹
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024

        # ② GRU編碼器
        self.audio_encoder = GRUEncoder(audio_dim, gru_hidden, 2)
        self.vision_encoder = GRUEncoder(vision_dim, gru_hidden, 2)
        gru_out_dim = gru_hidden * 2  # 128

        # ③ MAG (單層,應用到最後一層)
        self.mag = MultimodalAdaptationGate(lang_dim, gru_out_dim)

        # ④ 特徵投影層 (關鍵修復: 統一維度到 proj_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(lang_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(gru_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(gru_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ⑤ Directed Pairwise Attention (在投影後的空間)
        self.cross_attn_ta = DirectedPairwiseCrossModalAttention(
            proj_dim, proj_dim, dropout=0.2
        )
        self.cross_attn_tv = DirectedPairwiseCrossModalAttention(
            proj_dim, proj_dim, dropout=0.2
        )
        self.cross_attn_av = DirectedPairwiseCrossModalAttention(
            proj_dim, proj_dim, dropout=0.2
        )

        # ⑥ Feature Cascading Fusion
        fusion_input_dim = proj_dim * 6  # 6個交叉特徵
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ⑦ 分類頭
        self.classifier = nn.Linear(proj_dim, num_classes)

        # ⑧ 對比學習
        self.contrastive_loss = InfoNCELoss(temperature=0.07)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        vision: torch.Tensor,
        vision_mask: torch.Tensor,
    ):
        # ① 編碼
        lang_out = self.lang_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        text_hidden = lang_out.last_hidden_state

        audio_seq, audio_pooled = self.audio_encoder(audio, audio_mask)
        vision_seq, vision_pooled = self.vision_encoder(vision, vision_mask)

        # ② MAG (應用到文本)
        text_hidden = self.mag(text_hidden, audio_pooled, vision_pooled)

        # 文本池化
        text_pooled = (text_hidden * attention_mask.unsqueeze(-1)).sum(1) / \
                     attention_mask.sum(1, keepdim=True).clamp(min=1)

        # ③ 投影到統一空間 (關鍵: 解決維度不匹配)
        text_feat = self.text_proj(text_pooled)     # (B, 256)
        audio_feat = self.audio_proj(audio_pooled)  # (B, 256)
        vision_feat = self.vision_proj(vision_pooled)  # (B, 256)

        # 為交叉注意力準備序列 (升維到 proj_dim)
        text_seq_proj = text_hidden @ self.text_proj[0].weight.T + self.text_proj[0].bias
        audio_seq_proj = audio_seq @ self.audio_proj[0].weight.T + self.audio_proj[0].bias
        vision_seq_proj = vision_seq @ self.vision_proj[0].weight.T + self.vision_proj[0].bias

        # ④ Directed Pairwise Attention (在統一維度空間)
        text_from_audio, audio_from_text = self.cross_attn_ta(
            text_seq_proj, audio_seq_proj, attention_mask, audio_mask
        )
        text_from_audio = (text_from_audio * attention_mask.unsqueeze(-1)).sum(1) / \
                         attention_mask.sum(1, keepdim=True).clamp(min=1)
        audio_from_text = (audio_from_text * audio_mask.unsqueeze(-1)).sum(1) / \
                         audio_mask.sum(1, keepdim=True).clamp(min=1)

        text_from_vision, vision_from_text = self.cross_attn_tv(
            text_seq_proj, vision_seq_proj, attention_mask, vision_mask
        )
        text_from_vision = (text_from_vision * attention_mask.unsqueeze(-1)).sum(1) / \
                          attention_mask.sum(1, keepdim=True).clamp(min=1)
        vision_from_text = (vision_from_text * vision_mask.unsqueeze(-1)).sum(1) / \
                          vision_mask.sum(1, keepdim=True).clamp(min=1)

        audio_from_vision, vision_from_audio = self.cross_attn_av(
            audio_seq_proj, vision_seq_proj, audio_mask, vision_mask
        )
        audio_from_vision = (audio_from_vision * audio_mask.unsqueeze(-1)).sum(1) / \
                           audio_mask.sum(1, keepdim=True).clamp(min=1)
        vision_from_audio = (vision_from_audio * vision_mask.unsqueeze(-1)).sum(1) / \
                           vision_mask.sum(1, keepdim=True).clamp(min=1)

        # ⑤ Feature Cascading Fusion
        all_features = torch.cat([
            text_from_audio, audio_from_text,
            text_from_vision, vision_from_text,
            audio_from_vision, vision_from_audio,
        ], dim=-1)
        fused = self.fusion(all_features)

        # ⑥ 分類
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "text_feat": text_feat,
            "audio_feat": audio_feat,
            "vision_feat": vision_feat,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, optimizer, scheduler, device, scaler, epoch):
    model.train()
    total_loss, total_cls_loss, total_con_loss = 0.0, 0.0, 0.0

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
            outputs = model(ids, mask, aud, amask, vis, vmask)

            # 分類損失
            cls_loss = F.cross_entropy(outputs["logits"], labels, label_smoothing=0.1)

            # 對比學習損失 (現在維度一致了!)
            con_loss = (
                model.contrastive_loss(outputs["text_feat"], outputs["audio_feat"]) +
                model.contrastive_loss(outputs["text_feat"], outputs["vision_feat"]) +
                model.contrastive_loss(outputs["audio_feat"], outputs["vision_feat"])
            ) / 3

            # 總損失
            loss = cls_loss + 0.1 * con_loss

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
        total_cls_loss += cls_loss.item()
        total_con_loss += con_loss.item()

    n = len(loader)
    return {
        "loss": round(total_loss / n, 4),
        "cls_loss": round(total_cls_loss / n, 4),
        "con_loss": round(total_con_loss / n, 4),
    }


@torch.no_grad()
def validate(model, loader, device):
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

        outputs = model(ids, mask, aud, amask, vis, vmask)

        all_preds.extend(outputs["logits"].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc7 = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    return {"Acc7": round(float(acc7), 2)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 80)
    print("MOSI 多模態情感分析 v31 — Simplified Fusion Architecture")
    print("=" * 80)
    print("\n🔧 v30 → v31 關鍵修復:")
    print("  ✅ 添加特徵投影層,統一維度到 256")
    print("  ✅ 修復對比學習維度不匹配問題")
    print("  ✅ 簡化為單階段訓練")
    print("  ✅ 降低學習率 1e-3 → 5e-5")
    print("  ✅ 減小 batch size 96 → 32")
    print("\n🎯 保留核心技術:")
    print("  • MAG (Multimodal Adaptation Gate)")
    print("  • Directed Pairwise Attention (MulG)")
    print("  • InfoNCE Contrastive Learning")
    print("\n🎯 目標: Acc7 > 55% (超越 MGT)")
    print("=" * 80)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim": 5,
        "vision_dim": 20,
        "gru_hidden": 64,
        "proj_dim": 256,
        "num_classes": 7,
        "batch_size": 32,
        "num_epochs": 60,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.1,
        "freeze_layers": 6,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds = MOSIDataset(data["test"], tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, config["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, config["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimplifiedFusionModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        gru_hidden=config["gru_hidden"],
        proj_dim=config["proj_dim"],
        num_classes=config["num_classes"],
    ).to(device)

    # 凍結DeBERTa前6層
    encoder = model.lang_backbone.encoder
    for i in range(config["freeze_layers"]):
        for p in encoder.layer[i].parameters():
            p.requires_grad = False
    print(f"✓ 已凍結DeBERTa前{config['freeze_layers']}層")

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    # 優化器和調度器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device} | Batch: {config['batch_size']} | LR: {config['learning_rate']}\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler,
                                    device, scaler, epoch)
        val_metrics = validate(model, val_loader, device)

        history.append({
            "epoch": epoch+1,
            **train_metrics,
            **val_metrics
        })

        print(f"E{epoch+1:03d} | Loss={train_metrics['loss']:.4f} "
              f"(cls={train_metrics['cls_loss']:.4f}, con={train_metrics['con_loss']:.4f}) | "
              f"Acc7={val_metrics['Acc7']:.2f}%")

        if val_metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **val_metrics}
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "metrics": val_metrics,
            }, save_dir / "best_model_v31.pth")
            print(f"  ✅ 新最佳 Acc7={val_metrics['Acc7']:.2f}%")

        # 早停
        if epoch - best_acc7["epoch"] > 20:
            print(f"\n早停: 20 epochs無提升")
            break

    # 測試集評估
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v31.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = validate(model, test_loader, device)

    val_acc7 = best_acc7["Acc7"]
    test_acc7 = test_metrics["Acc7"]
    gap = abs(val_acc7 - test_acc7)

    print("\n【測試集最終結果 - v31 Simplified Fusion】")
    print(f"  Val  Acc7: {val_acc7:.2f}% (Epoch {best_acc7['epoch']})")
    print(f"  Test Acc7: {test_acc7:.2f}%")
    print(f"  Gap: {gap:.2f}%")
    print(f"\n  vs scaf_old: 50.0%  → {'+' if test_acc7 > 50 else ''}{test_acc7-50:.2f}%")
    print(f"  vs MGT:      55.6%  → {'✓超越' if test_acc7 > 55.6 else '✗未達'} ({test_acc7-55.6:+.2f}%)")
    print(f"  vs MulG:     82.2%  → {test_acc7-82.2:+.2f}%")

    with open(save_dir / "training_history_v31.json", "w") as f:
        json.dump({
            "history": history,
            "best_val": best_acc7,
            "test": test_metrics,
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)

    print(f"\n完成！模型已儲存: {save_dir / 'best_model_v31.pth'}")


if __name__ == "__main__":
    main()
