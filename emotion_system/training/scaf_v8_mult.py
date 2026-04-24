"""
MOSI 多模態情感分析 v8 — MulT 架構
基於 Multimodal Transformer (MulT) - SOTA 方法

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
參考論文：
Multimodal Transformer for Unaligned Multimodal Language Sequences
(ACL 2019)

核心思想：
1. 每個模態都有自己的 Transformer encoder
2. 使用 Crossmodal Transformer 進行模態間交互
   - Language → Audio (L→A)
   - Language → Vision (L→V)
   - Audio → Language (A→L)
   - Vision → Language (V→L)
   - Audio → Vision (A→V)
   - Vision → Audio (V→A)
3. 融合所有交互後的特徵
4. 多任務學習（分類 + 回歸）

為什麼 MulT 有效：
- 充分建模跨模態交互
- 不需要時間對齊
- 架構簡單穩定
- 在 MOSI/MOSEI 上都是 SOTA

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
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import math

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 資料集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MOSIDataset(Dataset):
    def __init__(self, split_data: dict, tokenizer, max_len: int = 50):
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

        print(f"數據: {len(self.raw_text)} 筆")

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        text = str(self.raw_text[idx])
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        # 音頻和視覺的實際長度
        aud_len = min(int(self.audio_lengths[idx]), self.audio.shape[1])
        vis_len = min(int(self.vision_lengths[idx]), self.vision.shape[1])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": self.audio[idx],
            "audio_len": aud_len,
            "vision": self.vision[idx],
            "vision_len": vis_len,
            "cls7_label": self.cls7_labels[idx],
            "cls2_label": self.cls2_labels[idx],
            "reg_label": self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Positional Encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Crossmodal Transformer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CrossmodalTransformer(nn.Module):
    """
    跨模態 Transformer
    用 modality A 作為 query，modality B 作為 key/value
    實現 A → B 的信息流動
    """
    def __init__(self, d_model: int, num_heads: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, query, key_value, query_mask=None, kv_mask=None):
        """
        query: (B, T_q, D) - 目標模態
        key_value: (B, T_kv, D) - 源模態
        query_mask: (B, T_q) - padding mask for query
        kv_mask: (B, T_kv) - padding mask for key/value

        Returns: (B, T_q, D) - 融合後的 query
        """
        # 創建 attention mask
        if query_mask is not None:
            query_mask = (query_mask == 0)  # True for padding
        if kv_mask is not None:
            kv_mask = (kv_mask == 0)  # True for padding

        output = self.transformer(
            tgt=query,
            memory=key_value,
            tgt_key_padding_mask=query_mask,
            memory_key_padding_mask=kv_mask,
        )
        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MulT 主模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MulTModel(nn.Module):
    """
    Multimodal Transformer

    架構：
    1. 單模態編碼
       - Language: BERT
       - Audio/Vision: 1D Conv + Position Encoding

    2. 跨模態 Transformer (6個方向)
       - L→A, L→V: 語言指導非語言
       - A→L, V→L: 非語言豐富語言
       - A→V, V→A: 非語言互相補充

    3. 特徵融合與分類
    """
    def __init__(
        self,
        lang_model: str = "bert-base-uncased",
        audio_dim: int = 5,
        vision_dim: int = 20,
        d_model: int = 128,  # 統一維度
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ① 單模態編碼器
        # Language
        self.bert = BertModel.from_pretrained(lang_model)
        bert_dim = self.bert.config.hidden_size  # 768
        self.lang_proj = nn.Linear(bert_dim, d_model)

        # Audio
        self.audio_conv = nn.Sequential(
            nn.Conv1d(audio_dim, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
        )
        self.audio_pos = PositionalEncoding(d_model)

        # Vision
        self.vision_conv = nn.Sequential(
            nn.Conv1d(vision_dim, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
        )
        self.vision_pos = PositionalEncoding(d_model)

        # ② 跨模態 Transformer (6個方向)
        # L→A, L→V
        self.trans_l_to_a = CrossmodalTransformer(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        self.trans_l_to_v = CrossmodalTransformer(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )

        # A→L, V→L
        self.trans_a_to_l = CrossmodalTransformer(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        self.trans_v_to_l = CrossmodalTransformer(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )

        # A→V, V→A
        self.trans_a_to_v = CrossmodalTransformer(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )
        self.trans_v_to_a = CrossmodalTransformer(
            d_model, num_heads, num_layers, dim_feedforward, dropout
        )

        # ③ 融合層
        # 每個模態都會被其他兩個模態增強，所以是 d_model * 3
        fusion_dim = d_model * 3

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, 256),  # 3個模態
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ④ 多任務頭
        self.cls7_head = nn.Linear(256, num_classes)
        self.cls2_head = nn.Linear(256, 2)
        self.reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask, audio, audio_len, vision, vision_len):
        B = input_ids.size(0)
        device = input_ids.device

        # ========================================
        # ① 單模態編碼
        # ========================================
        # Language: (B, L, 768) → (B, L, d_model)
        lang_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lang = self.lang_proj(lang_out.last_hidden_state)  # (B, L, d_model)
        lang_mask = attention_mask

        # Audio: (B, T_a, audio_dim) → (B, T_a, d_model)
        audio_t = audio.transpose(1, 2)  # (B, audio_dim, T_a)
        audio_feat = self.audio_conv(audio_t).transpose(1, 2)  # (B, T_a, d_model)
        audio_feat = self.audio_pos(audio_feat)

        # 創建 audio mask
        audio_mask = torch.zeros(B, audio.size(1), device=device)
        for i, length in enumerate(audio_len):
            audio_mask[i, :length] = 1

        # Vision: (B, T_v, vision_dim) → (B, T_v, d_model)
        vision_t = vision.transpose(1, 2)  # (B, vision_dim, T_v)
        vision_feat = self.vision_conv(vision_t).transpose(1, 2)  # (B, T_v, d_model)
        vision_feat = self.vision_pos(vision_feat)

        # 創建 vision mask
        vision_mask = torch.zeros(B, vision.size(1), device=device)
        for i, length in enumerate(vision_len):
            vision_mask[i, :length] = 1

        # ========================================
        # ② 跨模態 Transformer
        # ========================================
        # L→A, L→V: 語言指導非語言
        audio_from_lang = self.trans_l_to_a(audio_feat, lang, audio_mask, lang_mask)
        vision_from_lang = self.trans_l_to_v(vision_feat, lang, vision_mask, lang_mask)

        # A→L, V→L: 非語言豐富語言
        lang_from_audio = self.trans_a_to_l(lang, audio_feat, lang_mask, audio_mask)
        lang_from_vision = self.trans_v_to_l(lang, vision_feat, lang_mask, vision_mask)

        # A→V, V→A: 非語言互補
        vision_from_audio = self.trans_a_to_v(vision_feat, audio_feat, vision_mask, audio_mask)
        audio_from_vision = self.trans_v_to_a(audio_feat, vision_feat, audio_mask, vision_mask)

        # ========================================
        # ③ 池化 & 融合
        # ========================================
        # 每個模態都有3個版本：原始 + 從其他兩個模態來
        # 使用平均池化（考慮 mask）

        def masked_mean_pool(feat, mask):
            mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
            sum_feat = (feat * mask_expanded).sum(dim=1)  # (B, D)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            return sum_feat / sum_mask

        # Language 的3個版本
        lang_pooled = masked_mean_pool(lang, lang_mask)
        lang_a_pooled = masked_mean_pool(lang_from_audio, lang_mask)
        lang_v_pooled = masked_mean_pool(lang_from_vision, lang_mask)
        lang_fused = torch.cat([lang_pooled, lang_a_pooled, lang_v_pooled], dim=-1)

        # Audio 的3個版本
        audio_pooled = masked_mean_pool(audio_feat, audio_mask)
        audio_l_pooled = masked_mean_pool(audio_from_lang, audio_mask)
        audio_v_pooled = masked_mean_pool(audio_from_vision, audio_mask)
        audio_fused = torch.cat([audio_pooled, audio_l_pooled, audio_v_pooled], dim=-1)

        # Vision 的3個版本
        vision_pooled = masked_mean_pool(vision_feat, vision_mask)
        vision_l_pooled = masked_mean_pool(vision_from_lang, vision_mask)
        vision_a_pooled = masked_mean_pool(vision_from_audio, vision_mask)
        vision_fused = torch.cat([vision_pooled, vision_l_pooled, vision_a_pooled], dim=-1)

        # 組合所有特徵
        final_feat = torch.cat([lang_fused, audio_fused, vision_fused], dim=-1)
        final_feat = self.fusion(final_feat)  # (B, 256)

        # ========================================
        # ④ 多任務輸出
        # ========================================
        logits7 = self.cls7_head(final_feat)
        logits2 = self.cls2_head(final_feat)
        reg_out = self.reg_head(final_feat).squeeze(-1) * 3.0

        return logits7, logits2, reg_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 損失函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MulTLoss(nn.Module):
    def __init__(self, class_weights, alpha=1.5, beta=0.5, gamma=0.5):
        super().__init__()
        self.alpha = alpha  # cls7 權重
        self.beta = beta    # cls2 權重
        self.gamma = gamma  # reg 權重

        self.cls7 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.cls2 = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.reg = nn.SmoothL1Loss()

    def forward(self, logits7, logits2, reg_out, cls7_label, cls2_label, reg_label):
        l7 = self.cls7(logits7, cls7_label)
        l2 = self.cls2(logits2, cls2_label)
        lr = self.reg(reg_out, reg_label)

        total = self.alpha * l7 + self.beta * l2 + self.gamma * lr

        return total, {
            "cls7": l7.item(),
            "cls2": l2.item(),
            "reg": lr.item(),
        }


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
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        aud_len = batch["audio_len"]
        vis = batch["vision"].to(device)
        vis_len = batch["vision_len"]
        cl7 = batch["cls7_label"].to(device)
        cl2 = batch["cls2_label"].to(device)
        rl = batch["reg_label"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            l7, l2, reg = model(ids, mask, aud, aud_len, vis, vis_len)
            loss, _ = criterion(l7, l2, reg, cl7, cl2, rl)

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
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        aud_len = batch["audio_len"]
        vis = batch["vision"].to(device)
        vis_len = batch["vision_len"]
        cl7 = batch["cls7_label"].to(device)
        cl2 = batch["cls2_label"].to(device)
        rl = batch["reg_label"].to(device)

        l7, l2, reg = model(ids, mask, aud, aud_len, vis, vis_len)
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
def compute_class_weights(labels, n=7):
    cl = np.clip(np.round(labels).astype(int), -3, 3) + 3
    ct = np.bincount(cl, minlength=n).astype(float)
    ct = np.where(ct == 0, 1.0, ct)
    return torch.FloatTensor(len(cl) / (n * ct))


def freeze_bert_layers(model, freeze_embeddings=True, freeze_encoder_layers=6):
    """凍結 BERT 的部分層"""
    if freeze_embeddings:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

    # 凍結前 N 個 encoder layer
    for i in range(freeze_encoder_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False

    print(f"凍結: embeddings={freeze_embeddings}, 前{freeze_encoder_layers}層 encoder")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("MOSI 多模態情感分析 v8 — MulT (Multimodal Transformer)")
    print("基於 ACL 2019 論文的經典架構")
    print("=" * 70)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "bert-base-uncased",
        "max_text_len": 50,
        # 非語言
        "audio_dim": 5,
        "vision_dim": 20,
        # MulT 參數
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "num_classes": 7,
        "dropout": 0.2,
        # 訓練
        "batch_size": 24,
        "num_epochs": 40,
        "bert_lr": 2e-5,
        "other_lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.1,
        "freeze_layers": 6,
        # 損失權重
        "alpha": 1.5,  # cls7
        "beta": 0.5,   # cls2
        "gamma": 0.5,  # reg
    }

    print(f"\n載入資料: {config['data_path']}")
    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    class_w = compute_class_weights(data["train"]["regression_labels"])
    print(f"類別權重: {[f'{w:.3f}' for w in class_w.tolist()]}")

    tokenizer = BertTokenizer.from_pretrained(config["lang_model"])

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

    model = MulTModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    freeze_bert_layers(model, freeze_embeddings=True,
                      freeze_encoder_layers=config["freeze_layers"])

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數: {total_p/1e6:.1f}M | 可訓練: {trainable_p/1e6:.1f}M")

    criterion = MulTLoss(
        class_weights=class_w.to(device),
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
    )

    # 分組學習率：BERT 用小 lr，其他用大 lr
    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)

    optimizer = optim.AdamW([
        {"params": bert_params, "lr": config["bert_lr"]},
        {"params": other_params, "lr": config["other_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"[架構] MulT - 6方向跨模態 Transformer")
    print(f"[目標] 超越 MGT (Acc7: 55.6%)\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    best_mae = {"MAE": float('inf'), "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']} " + "-" * 45)

        tr_loss = train_epoch(model, train_loader, criterion,
                             optimizer, scheduler, device, scaler)
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
            }, save_dir / "mult_v8_best_acc7.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")

        # 保存 MAE 最佳
        if metrics["MAE"] < best_mae["MAE"]:
            best_mae = {"epoch": epoch + 1, **metrics}
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "metrics": metrics,
                "config": config,
            }, save_dir / "mult_v8_best_mae.pth")
            print(f"  ✅ 新最佳 MAE={metrics['MAE']:.4f}")

    # 測試集
    print("\n" + "=" * 60)

    # Acc7-best
    ckpt = torch.load(save_dir / "mult_v8_best_acc7.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_m = validate(model, test_loader, criterion, device)

    print("\n【測試集結果 - Acc7-best】")
    print(f"  Acc7 : {test_m['Acc7']:.2f}%   (MGT: 55.6%)")
    print(f"  Acc2 : {test_m['Acc2']:.2f}%   (MGT: 88.4%)")
    print(f"  F1   : {test_m['F1']:.2f}%   (MGT: 88.4%)")
    print(f"  MAE  : {test_m['MAE']:.4f}   (MGT: 0.654)")
    print(f"  Corr : {test_m['Corr']:.4f}   (MGT: 0.832)")

    # 保存歷史
    with open(save_dir / "mult_v8_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_acc7": best_acc7,
            "best_mae": best_mae,
            "test": test_m,
            "config": {k: str(v) for k, v in config.items()},
        }, f, indent=2)

    print(f"\n完成！")
    print(f"最佳 Val Acc7: {best_acc7['Acc7']:.2f}% (Epoch {best_acc7['epoch']})")
    print(f"最佳 Val MAE: {best_mae['MAE']:.4f} (Epoch {best_mae['epoch']})")


if __name__ == "__main__":
    main()
