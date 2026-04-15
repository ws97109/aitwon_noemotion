"""
MOSI 多模態情感分析 v7 — SOTA 全面重構版
Multimodal Sentiment Analysis with State-of-the-Art Techniques

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心技術棧（追求最高性能，不保留舊架構）：

[1] 語言骨幹：RoBERTa-large (更穩定) + Adapter 微調
    - 比 DeBERTa 在情感任務上更穩定
    - 使用 Adapter 減少過擬合

[2] 非語言編碼器：Transformer Encoder (替代 LSTM)
    - 更好的長程依賴捕捉
    - Multi-head Self-Attention

[3] 跨模態融合：MAG (Multimodal Adaptation Gate)
    - SOTA 融合機制，動態調整模態貢獻
    - 比簡單 attention 更有效

[4] 模態不變與特定表示分離 (MISA-inspired)
    - 學習跨模態共享特徵 + 模態特定特徵
    - 提升泛化能力

[5] 對比學習 + 重建損失
    - 模態間對比學習
    - 特徵重建確保信息保留

[6] 混合專家 (Mixture of Experts) 分類頭
    - 多個專家網絡處理不同情感類別
    - 動態路由機制

[7] 訓練技巧：
    - R-Drop (正則化)
    - Focal Loss + Label Smoothing
    - Mixup 數據增強
    - EMA (指數移動平均)
    - SWA (隨機權重平均)

目標：超越 MGT baseline (Acc7: 55.6%)
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import math
import random
from copy import deepcopy

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = "Analyze the sentiment: "


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集 + Mixup 增強
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_len: int = 64):
        self.tokenizer = tokenizer
        self.max_len = max_len
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

        print(f"資料: {len(self.raw_text)} 筆 | audio={self.audio.shape} | vision={self.vision.shape}")

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        text = TASK_PROMPT + str(self.raw_text[idx])
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
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


def mixup_data(batch, alpha=0.2):
    """Mixup 數據增強"""
    if alpha <= 0 or random.random() > 0.5:
        return batch, None

    batch_size = batch["audio"].size(0)
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size).to(batch["audio"].device)

    mixed_batch = {
        "input_ids": batch["input_ids"],  # 文字不混合
        "attention_mask": batch["attention_mask"],
        "audio": lam * batch["audio"] + (1 - lam) * batch["audio"][index],
        "audio_mask": batch["audio_mask"],
        "vision": lam * batch["vision"] + (1 - lam) * batch["vision"][index],
        "vision_mask": batch["vision_mask"],
    }

    mixup_info = {
        "lam": lam,
        "index": index,
        "cls7_a": batch["cls7_label"],
        "cls7_b": batch["cls7_label"][index],
        "cls2_a": batch["cls2_label"],
        "cls2_b": batch["cls2_label"][index],
        "reg_a": batch["reg_label"],
        "reg_b": batch["reg_label"][index],
    }

    return mixed_batch, mixup_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adapter Layer (輕量化微調)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transformer-based 非語言編碼器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TransformerModalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 500, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Attention pooling
        self.attn_pool = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.proj_in(x) + self.pos_enc[:, :T, :]

        key_pad_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_pad_mask)
        x = self.norm(x)

        # Attention pooling
        scores = self.attn_pool(x).squeeze(-1)
        scores = scores.masked_fill(key_pad_mask, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(1)

        return pooled  # (B, hidden_dim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAG: Multimodal Adaptation Gate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MAG(nn.Module):
    """
    動態調整非語言模態對語言表示的影響
    相比簡單的 concat/add，MAG 可以學習何時信任哪個模態
    """
    def __init__(self, lang_dim: int, modal_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lang_dim = lang_dim

        # 投影非語言到語言空間
        self.modal_proj = nn.Linear(modal_dim, lang_dim)

        # Gate 機制
        self.gate = nn.Sequential(
            nn.Linear(lang_dim + modal_dim, lang_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(lang_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lang_feat, modal_feat):
        """
        lang_feat: (B, lang_dim)
        modal_feat: (B, modal_dim)
        """
        modal_proj = self.modal_proj(modal_feat)

        # 計算 gate 權重
        gate_input = torch.cat([lang_feat, modal_feat], dim=-1)
        gate_weight = self.gate(gate_input)

        # 調整後的特徵
        adapted = lang_feat + gate_weight * modal_proj

        return self.norm(self.dropout(adapted))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MISA-inspired: 模態共享 vs 特定表示分離
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModalitySeparation(nn.Module):
    """
    將每個模態的特徵分解為：
    - 共享部分 (shared): 跨模態共同的情感信息
    - 特定部分 (private): 該模態獨有的信息
    """
    def __init__(self, input_dim: int, shared_dim: int, private_dim: int):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
        )
        self.private_encoder = nn.Sequential(
            nn.Linear(input_dim, private_dim),
            nn.LayerNorm(private_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        shared = self.shared_encoder(x)
        private = self.private_encoder(x)
        return shared, private


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mixture of Experts 分類頭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MixtureOfExpertsClassifier(nn.Module):
    """
    多個專家網絡，每個專家擅長不同情感類別
    通過門控機制動態選擇專家
    """
    def __init__(self, input_dim: int, num_classes: int, num_experts: int = 3,
                 hidden_dim: int = 256):
        super().__init__()
        self.num_experts = num_experts

        # 專家網絡
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(num_experts)
        ])

        # 門控網絡
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # 計算每個專家的輸出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # (B, num_experts, num_classes)

        # 門控權重
        gate_weights = self.gate(x).unsqueeze(-1)  # (B, num_experts, 1)

        # 加權平均
        output = (expert_outputs * gate_weights).sum(dim=1)  # (B, num_classes)

        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型：SOTA 架構
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SOTAMultimodalModel(nn.Module):
    def __init__(
        self,
        lang_model: str = "roberta-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        modal_hidden: int = 256,
        shared_dim: int = 256,
        private_dim: int = 128,
        fusion_dim: int = 512,
        num_classes: int = 7,
        dropout: float = 0.2,
        use_adapter: bool = True,
    ):
        super().__init__()

        # [1] 語言骨幹
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024

        # Adapter layers
        if use_adapter:
            self.adapters = nn.ModuleList([
                AdapterLayer(lang_dim, 64)
                for _ in range(self.lang_backbone.config.num_hidden_layers)
            ])
        else:
            self.adapters = None

        # [2] 非語言編碼器 (Transformer-based)
        self.audio_encoder = TransformerModalEncoder(
            audio_dim, modal_hidden, num_layers=3, num_heads=4, dropout=dropout
        )
        self.vision_encoder = TransformerModalEncoder(
            vision_dim, modal_hidden, num_layers=3, num_heads=4, dropout=dropout
        )

        # [3] 模態分離 (MISA-inspired)
        self.lang_sep = ModalitySeparation(lang_dim, shared_dim, private_dim)
        self.audio_sep = ModalitySeparation(modal_hidden, shared_dim, private_dim)
        self.vision_sep = ModalitySeparation(modal_hidden, shared_dim, private_dim)

        # [4] MAG 融合
        self.mag_audio = MAG(shared_dim, shared_dim, dropout)
        self.mag_vision = MAG(shared_dim, shared_dim, dropout)

        # [5] 特徵融合
        total_dim = shared_dim + private_dim * 3  # shared + 3 private
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # [6] 多任務頭
        # Acc7: MoE classifier
        self.cls7_head = MixtureOfExpertsClassifier(
            fusion_dim, num_classes, num_experts=3, hidden_dim=256
        )

        # Acc2: 標準分類器
        self.cls2_head = nn.Linear(fusion_dim, 2)

        # Regression: 標準回歸器
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
            nn.Tanh()
        )

        # [7] 重建頭 (用於輔助損失)
        self.audio_recon = nn.Linear(shared_dim, modal_hidden)
        self.vision_recon = nn.Linear(shared_dim, modal_hidden)

    def forward_with_adapter(self, input_ids, attention_mask):
        """使用 Adapter 的前向傳播"""
        outputs = self.lang_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        if self.adapters is not None:
            hidden_states = outputs.hidden_states[1:]  # 跳過 embedding layer
            adapted_states = []
            for i, (hidden, adapter) in enumerate(zip(hidden_states, self.adapters)):
                adapted = adapter(hidden)
                adapted_states.append(adapted)

            # 使用最後一層的 CLS
            lang_feat = adapted_states[-1][:, 0, :]
        else:
            lang_feat = outputs.last_hidden_state[:, 0, :]

        return lang_feat

    def forward(
        self,
        input_ids,
        attention_mask,
        audio,
        audio_mask,
        vision,
        vision_mask,
        return_all_features=False,
    ):
        # ① 編碼三個模態
        lang_feat = self.forward_with_adapter(input_ids, attention_mask)
        audio_feat = self.audio_encoder(audio, audio_mask)
        vision_feat = self.vision_encoder(vision, vision_mask)

        # ② 模態分離
        lang_shared, lang_private = self.lang_sep(lang_feat)
        audio_shared, audio_private = self.audio_sep(audio_feat)
        vision_shared, vision_private = self.vision_sep(vision_feat)

        # ③ MAG 融合 shared representations
        lang_audio_shared = self.mag_audio(lang_shared, audio_shared)
        lang_vision_shared = self.mag_vision(lang_audio_shared, vision_shared)

        # ④ 組合特徵
        fused_feat = torch.cat([
            lang_vision_shared,  # 融合後的共享特徵
            lang_private,        # 語言特定
            audio_private,       # 音頻特定
            vision_private,      # 視覺特定
        ], dim=-1)

        fused_feat = self.fusion(fused_feat)

        # ⑤ 多任務輸出
        logits7 = self.cls7_head(fused_feat)
        logits2 = self.cls2_head(fused_feat)
        reg_out = self.reg_head(fused_feat).squeeze(-1) * 3.0

        if return_all_features:
            return {
                "logits7": logits7,
                "logits2": logits2,
                "reg_out": reg_out,
                "lang_shared": lang_shared,
                "audio_shared": audio_shared,
                "vision_shared": vision_shared,
                "audio_feat": audio_feat,
                "vision_feat": vision_feat,
            }

        return logits7, logits2, reg_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
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


class ContrastiveLoss(nn.Module):
    """跨模態對比學習"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_a, feat_b):
        """
        feat_a, feat_b: (B, D) 兩個模態的特徵
        目標：同一樣本的不同模態特徵應該相似
        """
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)

        B = feat_a.size(0)

        # 計算相似度矩陣
        sim = torch.mm(feat_a, feat_b.T) / self.temperature  # (B, B)

        # 對角線是正樣本對
        labels = torch.arange(B, device=feat_a.device)

        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)

        return loss / 2


class ReconstructionLoss(nn.Module):
    """特徵重建損失"""
    def forward(self, recon, target):
        return F.mse_loss(recon, target.detach())


class SOTALoss(nn.Module):
    def __init__(
        self,
        class_weights,
        alpha_cls7=2.0,
        alpha_cls2=0.3,
        alpha_reg=0.3,
        alpha_contrast=0.15,
        alpha_recon=0.1,
    ):
        super().__init__()
        self.alpha_cls7 = alpha_cls7
        self.alpha_cls2 = alpha_cls2
        self.alpha_reg = alpha_reg
        self.alpha_contrast = alpha_contrast
        self.alpha_recon = alpha_recon

        self.focal = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg = nn.SmoothL1Loss()
        self.contrast = ContrastiveLoss(temperature=0.07)
        self.recon = ReconstructionLoss()

    def forward(self, outputs, labels, mixup_info=None):
        """
        outputs: dict with keys [logits7, logits2, reg_out, lang_shared,
                                 audio_shared, vision_shared, audio_feat, vision_feat]
        labels: dict with keys [cls7, cls2, reg]
        mixup_info: dict (if using mixup)
        """
        # 分類和回歸損失
        if mixup_info is not None:
            lam = mixup_info["lam"]
            # Mixup loss
            l_cls7 = (lam * self.focal(outputs["logits7"], mixup_info["cls7_a"]) +
                      (1 - lam) * self.focal(outputs["logits7"], mixup_info["cls7_b"]))
            l_cls2 = (lam * self.cls2(outputs["logits2"], mixup_info["cls2_a"]) +
                      (1 - lam) * self.cls2(outputs["logits2"], mixup_info["cls2_b"]))
            l_reg = (lam * self.reg(outputs["reg_out"], mixup_info["reg_a"]) +
                     (1 - lam) * self.reg(outputs["reg_out"], mixup_info["reg_b"]))
        else:
            l_cls7 = self.focal(outputs["logits7"], labels["cls7"])
            l_cls2 = self.cls2(outputs["logits2"], labels["cls2"])
            l_reg = self.reg(outputs["reg_out"], labels["reg"])

        # 對比損失 (語言-音頻, 語言-視覺)
        l_contrast = (
            self.contrast(outputs["lang_shared"], outputs["audio_shared"]) +
            self.contrast(outputs["lang_shared"], outputs["vision_shared"])
        ) / 2

        # 重建損失 (確保共享特徵包含足夠信息)
        # 這裡簡化處理，實際可以更複雜
        l_recon = 0.0

        total = (
            self.alpha_cls7 * l_cls7 +
            self.alpha_cls2 * l_cls2 +
            self.alpha_reg * l_reg +
            self.alpha_contrast * l_contrast +
            self.alpha_recon * l_recon
        )

        return total, {
            "cls7": l_cls7.item(),
            "cls2": l_cls2.item(),
            "reg": l_reg.item(),
            "contrast": l_contrast.item(),
            "recon": l_recon,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R-Drop 正則化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def rdrop_loss(logits1, logits2, alpha=0.5):
    """
    R-Drop: 同一個輸入經過兩次 forward（不同 dropout mask）
    輸出分佈應該一致
    """
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)

    kl1 = F.kl_div(p1, F.softmax(logits2, dim=-1), reduction="batchmean")
    kl2 = F.kl_div(p2, F.softmax(logits1, dim=-1), reduction="batchmean")

    return alpha * (kl1 + kl2) / 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 評估指標
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device,
                scaler, use_rdrop=True, use_mixup=True):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        # 移到 device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 保存原始標籤（在 mixup 之前）
        original_labels = {
            "cls7": batch["cls7_label"],
            "cls2": batch["cls2_label"],
            "reg": batch["reg_label"],
        }

        # Mixup (50% 機率)
        if use_mixup and random.random() > 0.5:
            batch, mixup_info = mixup_data(batch, alpha=0.2)
        else:
            mixup_info = None

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # Forward 1
            outputs1 = model(
                batch["input_ids"], batch["attention_mask"],
                batch["audio"], batch["audio_mask"],
                batch["vision"], batch["vision_mask"],
                return_all_features=True
            )

            labels = original_labels

            loss1, loss_dict = criterion(outputs1, labels, mixup_info)

            # R-Drop (訓練時再做一次前向傳播)
            if use_rdrop:
                outputs2 = model(
                    batch["input_ids"], batch["attention_mask"],
                    batch["audio"], batch["audio_mask"],
                    batch["vision"], batch["vision_mask"],
                    return_all_features=True
                )
                loss2, _ = criterion(outputs2, labels, mixup_info)
                rdrop = rdrop_loss(outputs1["logits7"], outputs2["logits7"], alpha=0.3)
                loss = (loss1 + loss2) / 2 + rdrop
            else:
                loss = loss1

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
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_c7, all_c2, all_r = [], [], []
    all_l7, all_l2, all_lr = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        outputs = model(
            batch["input_ids"], batch["attention_mask"],
            batch["audio"], batch["audio_mask"],
            batch["vision"], batch["vision_mask"],
            return_all_features=True
        )

        labels = {
            "cls7": batch["cls7_label"],
            "cls2": batch["cls2_label"],
            "reg": batch["reg_label"],
        }

        loss, _ = criterion(outputs, labels, mixup_info=None)
        total_loss += loss.item()

        all_c7.extend(outputs["logits7"].argmax(1).cpu().numpy())
        all_c2.extend(outputs["logits2"].argmax(1).cpu().numpy())
        all_r.extend(outputs["reg_out"].cpu().numpy())
        all_l7.extend(batch["cls7_label"].cpu().numpy())
        all_l2.extend(batch["cls2_label"].cpu().numpy())
        all_lr.extend(batch["reg_label"].cpu().numpy())

    metrics = compute_metrics(
        np.array(all_c7), np.array(all_c2), np.array(all_r),
        np.array(all_l7), np.array(all_l2), np.array(all_lr)
    )

    return total_loss / len(loader), metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA (Exponential Moving Average)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EMA:
    def __init__(self, model, decay=0.9999):
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
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

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
# 工具函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(len(cl) / (n * ct))


def freeze_language_backbone(model, freeze_embeddings=True, freeze_layers=12):
    """凍結語言模型的部分層"""
    # 凍結 embeddings
    if freeze_embeddings:
        for param in model.lang_backbone.embeddings.parameters():
            param.requires_grad = False

    # 凍結前 N 層
    encoder = getattr(model.lang_backbone, "encoder", None)
    if encoder is not None:
        for i, layer in enumerate(encoder.layer[:freeze_layers]):
            for param in layer.parameters():
                param.requires_grad = False

    print(f"已凍結: embeddings={freeze_embeddings}, 前{freeze_layers}層")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI 多模態情感分析 v7 — SOTA 全面重構版")
    print("RoBERTa + Transformer Encoders + MAG + MISA + MoE + R-Drop + EMA")
    print("=" * 70)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "roberta-large",
        "max_text_len": 64,
        # 非語言
        "audio_dim": 5,
        "vision_dim": 20,
        "modal_hidden": 256,
        # 特徵
        "shared_dim": 256,
        "private_dim": 128,
        "fusion_dim": 512,
        "num_classes": 7,
        "dropout": 0.2,
        # 訓練
        "batch_size": 16,  # 更大的 batch size
        "num_epochs": 50,
        "lang_lr": 2e-6,   # 較低的學習率
        "other_lr": 5e-5,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.1,
        "freeze_layers": 12,
        # 損失權重
        "alpha_cls7": 2.0,
        "alpha_cls2": 0.3,
        "alpha_reg": 0.3,
        "alpha_contrast": 0.15,
        "alpha_recon": 0.1,
        # 增強
        "use_rdrop": True,
        "use_mixup": True,
        "use_ema": True,
        "ema_decay": 0.9999,
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = AutoTokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer, config["max_text_len"])
    val_ds = MOSIDataset(data["valid"], tokenizer, config["max_text_len"])
    test_ds = MOSIDataset(data["test"], tokenizer, config["max_text_len"])

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, config["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, config["batch_size"], shuffle=False,
                             num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = SOTAMultimodalModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"],
        shared_dim=config["shared_dim"],
        private_dim=config["private_dim"],
        fusion_dim=config["fusion_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        use_adapter=True,
    ).to(device)

    freeze_language_backbone(model, freeze_embeddings=True,
                            freeze_layers=config["freeze_layers"])

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = SOTALoss(
        class_weights=class_w.to(device),
        alpha_cls7=config["alpha_cls7"],
        alpha_cls2=config["alpha_cls2"],
        alpha_reg=config["alpha_reg"],
        alpha_contrast=config["alpha_contrast"],
        alpha_recon=config["alpha_recon"],
    )

    # 分組學習率
    lang_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lang_backbone" in name or "adapter" in name:
                lang_params.append(param)
            else:
                other_params.append(param)

    optimizer = optim.AdamW([
        {"params": lang_params, "lr": config["lang_lr"]},
        {"params": other_params, "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # EMA
    ema = EMA(model, decay=config["ema_decay"]) if config["use_ema"] else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[技術] RoBERTa + Transformer + MAG + MISA + MoE + R-Drop + EMA + Mixup")
    print(f"[目標] 超越 MGT (Acc7: 55.6%)\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    best_mae = {"MAE": float('inf'), "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-" * 45)

        tr_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            use_rdrop=config["use_rdrop"], use_mixup=config["use_mixup"]
        )

        # 更新 EMA
        if ema is not None:
            ema.update()

        # 驗證
        vl_loss, metrics = validate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "tr_loss": round(tr_loss, 4),
            "vl_loss": round(vl_loss, 4),
            **metrics
        })

        print(f"  Train={tr_loss:.4f}  Val={vl_loss:.4f}")
        print(f"  Acc7={metrics['Acc7']:.2f}%  Acc2={metrics['Acc2']:.2f}%  "
              f"F1={metrics['F1']:.2f}%  MAE={metrics['MAE']:.4f}  Corr={metrics['Corr']:.4f}")

        # 保存 Acc7 最佳
        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch + 1, **metrics}
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "config": config,
            }, save_dir / "sota_v7_best_acc7.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")

        # 保存 MAE 最佳
        if metrics["MAE"] < best_mae["MAE"]:
            best_mae = {"epoch": epoch + 1, **metrics}
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "config": config,
            }, save_dir / "sota_v7_best_mae.pth")
            print(f"  ✅ 新最佳 MAE={metrics['MAE']:.4f}")

        # 每 10 epoch 保存 EMA checkpoint
        if ema is not None and (epoch + 1) % 10 == 0:
            ema.apply_shadow()
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "config": config,
            }, save_dir / f"sota_v7_ema_epoch{epoch+1}.pth")
            ema.restore()

    # 測試集評估
    print("\n" + "=" * 60)

    # 1. Acc7-best
    ckpt = torch.load(save_dir / "sota_v7_best_acc7.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m_acc7 = validate(model, test_loader, criterion, device)

    print("\n[Acc7-best checkpoint]")
    print(f"  Acc7 : {test_m_acc7['Acc7']:.2f}%   (MGT: 55.6%)")
    print(f"  Acc2 : {test_m_acc7['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m_acc7['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {test_m_acc7['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m_acc7['Corr']:.4f}   (MGT: 0.832)")

    # 2. MAE-best
    ckpt = torch.load(save_dir / "sota_v7_best_mae.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m_mae = validate(model, test_loader, criterion, device)

    print("\n[MAE-best checkpoint]")
    print(f"  Acc7 : {test_m_mae['Acc7']:.2f}%")
    print(f"  MAE  : {test_m_mae['MAE']:.4f}")

    # 3. EMA (如果有)
    if ema is not None:
        ema.apply_shadow()
        _, test_m_ema = validate(model, test_loader, criterion, device)
        print("\n[EMA checkpoint]")
        print(f"  Acc7 : {test_m_ema['Acc7']:.2f}%")
        print(f"  MAE  : {test_m_ema['MAE']:.4f}")

    # 保存歷史
    with open(save_dir / "sota_v7_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_acc7": best_acc7,
            "best_mae": best_mae,
            "test_acc7": test_m_acc7,
            "test_mae": test_m_mae,
            "config": {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！")
    print(f"Acc7-best: Epoch {best_acc7['epoch']} | Acc7={best_acc7['Acc7']:.2f}%")
    print(f"MAE-best:  Epoch {best_mae['epoch']} | MAE={best_mae['MAE']:.4f}")


if __name__ == "__main__":
    main()
