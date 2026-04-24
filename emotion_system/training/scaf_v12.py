"""
MOSI 多模態情感分析 v12 — MulG-Enhanced Architecture
整合 MulG (82.2% Acc7) 的核心技術

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基於 Nature Scientific Reports 2025 的 MulG 模型

MulG 在 CMU-MOSI 達成:
  - Acc7: 82.2% ✓✓✓ (vs MGT 55.6%, vs scaf_old 50%)
  - 使用 GRU + Directed Pairwise Cross-Modal Attention
  - 計算高效，參數少

v12 策略: DeBERTa (文本) + MulG架構 (跨模態)

[核心技術1] Directed Pairwise Cross-Modal Attention
              ─ 計算模態間時間步的相關分數
              ─ A(t,t') = x^M1_t · W · x^M2_t'
              ─ 動態對齊異步特徵

[核心技術2] GRU序列編碼器
              ─ 替代LSTM，更高效
              ─ Hidden=40, Layers=2 (MulG最佳配置)
              ─ Reset Gate + Update Gate

[核心技術3] 特徵級聯融合
              ─ Step1: 所有模態對的跨模態處理
              ─ Step2: Concatenate融合向量
              ─ Step3: FC降維

[核心技術4] MulG訓練策略
              ─ Adam優化器, LR=1e-3
              ─ Batch=128, Epochs=100
              ─ Attention Dropout=0.1
              ─ Cross-Entropy Loss

[增強點] 保留DeBERTa-v3-large
         ─ MulG原版用GRU處理文本
         ─ v12用DeBERTa (更強的文本理解)

目標: Acc7 > 70% (接近MulG的82.2%)

參考文獻:
  - MulG (Nature SR 2025): https://www.nature.com/articles/s41598-025-93023-3
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
from transformers import AutoModel, DebertaV2Tokenizer

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
# [核心技術2] GRU序列編碼器 (MulG配置)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GRUEncoder(nn.Module):
    """
    MulG GRU編碼器
    - Hidden: 40 (MulG最佳配置)
    - Layers: 2
    - Bidirectional
    """
    def __init__(self, input_dim: int, hidden_dim: int = 40, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_dim * 2  # bidirectional

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x: (B, T, D)
            mask: (B, T)
        Returns:
            seq_out: (B, T, hidden_dim*2)  序列輸出（用於attention）
            pooled: (B, hidden_dim*2)       全局表示
        """
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        seq_out, h = self.gru(packed)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(
            seq_out, batch_first=True, total_length=x.size(1)
        )

        # 雙向GRU的最後隱藏狀態
        h_forward = h[-2]   # 前向最後層
        h_backward = h[-1]  # 後向最後層
        pooled = torch.cat([h_forward, h_backward], dim=-1)

        return seq_out, pooled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [核心技術1] Directed Pairwise Cross-Modal Attention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DirectedPairwiseCrossModalAttention(nn.Module):
    """
    MulG的核心: 定向成對跨模態注意力

    計算兩個模態序列間的時間步相關性:
    A(t,t') = x^M1_t · W · x^M2_t'

    動態對齊異步特徵，處理不同時間頻率的模態
    """
    def __init__(self, dim1: int, dim2: int, dropout: float = 0.1):
        super().__init__()
        # 注意力權重矩陣
        self.W = nn.Parameter(torch.randn(dim1, dim2))
        nn.init.xavier_uniform_(self.W)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1: torch.Tensor, seq2: torch.Tensor,
                mask1: torch.Tensor, mask2: torch.Tensor):
        """
        Args:
            seq1: (B, T1, D1) 模態1序列
            seq2: (B, T2, D2) 模態2序列
            mask1: (B, T1) 模態1遮罩
            mask2: (B, T2) 模態2遮罩

        Returns:
            attended_seq1: (B, T1, D2) M1的每個時間步attend到M2
            attended_seq2: (B, T2, D1) M2的每個時間步attend到M1
        """
        B, T1, D1 = seq1.shape
        B, T2, D2 = seq2.shape

        # ① 計算注意力分數矩陣
        # A(t,t') = x^M1_t · W · x^M2_t'
        # seq1 @ W @ seq2^T → (B, T1, T2)
        attn_scores = torch.bmm(
            torch.matmul(seq1, self.W),  # (B, T1, D2)
            seq2.transpose(1, 2)         # (B, D2, T2)
        )  # (B, T1, T2)

        # ② M1 → M2 方向的注意力
        # 對每個M1的時間步t，計算對M2所有時間步t'的注意力
        mask2_expanded = mask2.unsqueeze(1).expand(B, T1, T2)  # (B, T1, T2)
        attn_scores_12 = attn_scores.masked_fill(mask2_expanded == 0, float('-inf'))
        attn_weights_12 = F.softmax(attn_scores_12, dim=2)  # (B, T1, T2)
        attn_weights_12 = self.dropout(attn_weights_12)

        # M1的每個時間步t加權聚合M2的所有時間步
        attended_seq1 = torch.bmm(attn_weights_12, seq2)  # (B, T1, D2)

        # ③ M2 → M1 方向的注意力
        mask1_expanded = mask1.unsqueeze(2).expand(B, T1, T2)  # (B, T1, T2)
        attn_scores_21 = attn_scores.transpose(1, 2)  # (B, T2, T1)
        attn_scores_21 = attn_scores_21.masked_fill(
            mask1_expanded.transpose(1, 2) == 0, float('-inf')
        )
        attn_weights_21 = F.softmax(attn_scores_21, dim=2)  # (B, T2, T1)
        attn_weights_21 = self.dropout(attn_weights_21)

        # M2的每個時間步t'加權聚合M1的所有時間步
        attended_seq2 = torch.bmm(attn_weights_21, seq1)  # (B, T2, D1)

        return attended_seq1, attended_seq2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [核心技術3] 特徵級聯融合 (MulG策略)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FeatureCascadingFusion(nn.Module):
    """
    MulG的特徵融合策略:
    1. 所有模態對的跨模態處理
    2. Concatenate成融合向量
    3. FC降維
    """
    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int,
                 output_dim: int = 256):
        super().__init__()

        # 所有模態對的交互特徵維度
        # cross_attn 回傳的兩方向特徵維度是交換的：
        #   text_from_audio: dim=audio_dim, audio_from_text: dim=text_dim → 合計 text_dim+audio_dim
        #   text_from_vision: dim=vision_dim, vision_from_text: dim=text_dim → 合計 text_dim+vision_dim
        #   audio_from_vision: dim=vision_dim, vision_from_audio: dim=audio_dim → 合計 audio_dim+vision_dim
        total_dim = (text_dim + audio_dim) + \
                   (text_dim + vision_dim) + \
                   (audio_dim + vision_dim)

        # FC降維
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, text_audio_pair, text_vision_pair, audio_vision_pair):
        """
        Args:
            text_audio_pair: (text_attended, audio_attended) 各(B, D)
            text_vision_pair: (text_attended, vision_attended)
            audio_vision_pair: (audio_attended, vision_attended)

        Returns:
            fused: (B, output_dim)
        """
        # Concatenate所有交互特徵
        all_features = torch.cat([
            text_audio_pair[0], text_audio_pair[1],
            text_vision_pair[0], text_vision_pair[1],
            audio_vision_pair[0], audio_vision_pair[1],
        ], dim=-1)

        # FC降維
        fused = self.fusion(all_features)
        return fused


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型: MulG-Enhanced (v12)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MulGEnhancedModel(nn.Module):
    """
    整合MulG核心技術的模型

    架構:
      DeBERTa → 文本序列表示
      GRU × 2 → 音頻/視覺序列表示

      Directed Pairwise Cross-Modal Attention:
        Text ↔ Audio
        Text ↔ Vision
        Audio ↔ Vision

      Feature Cascading Fusion → 分類
    """
    def __init__(
        self,
        lang_model: str = "microsoft/deberta-v3-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        gru_hidden: int = 40,  # MulG最佳配置
        fusion_dim: int = 256,
        num_classes: int = 7,
    ):
        super().__init__()

        # ① 語言骨幹 (DeBERTa-v3-large)
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024

        # ② GRU編碼器 (MulG配置)
        self.audio_encoder = GRUEncoder(audio_dim, gru_hidden, num_layers=2)
        self.vision_encoder = GRUEncoder(vision_dim, gru_hidden, num_layers=2)

        gru_out_dim = gru_hidden * 2  # bidirectional: 80

        # ③ 定向成對跨模態注意力 (3對)
        self.cross_attn_text_audio = DirectedPairwiseCrossModalAttention(
            lang_dim, gru_out_dim, dropout=0.1
        )
        self.cross_attn_text_vision = DirectedPairwiseCrossModalAttention(
            lang_dim, gru_out_dim, dropout=0.1
        )
        self.cross_attn_audio_vision = DirectedPairwiseCrossModalAttention(
            gru_out_dim, gru_out_dim, dropout=0.1
        )

        # ④ 特徵級聯融合
        self.fusion = FeatureCascadingFusion(
            lang_dim, gru_out_dim, gru_out_dim, fusion_dim
        )

        # ⑤ 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        vision: torch.Tensor,
        vision_mask: torch.Tensor,
    ):
        # ① 編碼三個模態
        # 文本: DeBERTa序列輸出
        lang_out = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_seq = lang_out.last_hidden_state  # (B, L, 1024)

        # 音頻/視覺: GRU序列輸出
        audio_seq, audio_pooled = self.audio_encoder(audio, audio_mask)  # (B, T_a, 80)
        vision_seq, vision_pooled = self.vision_encoder(vision, vision_mask)  # (B, T_v, 80)

        # ② 定向成對跨模態注意力
        # Text ↔ Audio
        text_from_audio, audio_from_text = self.cross_attn_text_audio(
            text_seq, audio_seq, attention_mask, audio_mask
        )
        # 池化: 平均池化帶遮罩的序列
        text_from_audio_pooled = (text_from_audio * attention_mask.unsqueeze(-1)).sum(1) / \
                                attention_mask.sum(1, keepdim=True).clamp(min=1)
        audio_from_text_pooled = (audio_from_text * audio_mask.unsqueeze(-1)).sum(1) / \
                                audio_mask.sum(1, keepdim=True).clamp(min=1)

        # Text ↔ Vision
        text_from_vision, vision_from_text = self.cross_attn_text_vision(
            text_seq, vision_seq, attention_mask, vision_mask
        )
        text_from_vision_pooled = (text_from_vision * attention_mask.unsqueeze(-1)).sum(1) / \
                                 attention_mask.sum(1, keepdim=True).clamp(min=1)
        vision_from_text_pooled = (vision_from_text * vision_mask.unsqueeze(-1)).sum(1) / \
                                 vision_mask.sum(1, keepdim=True).clamp(min=1)

        # Audio ↔ Vision
        audio_from_vision, vision_from_audio = self.cross_attn_audio_vision(
            audio_seq, vision_seq, audio_mask, vision_mask
        )
        audio_from_vision_pooled = (audio_from_vision * audio_mask.unsqueeze(-1)).sum(1) / \
                                  audio_mask.sum(1, keepdim=True).clamp(min=1)
        vision_from_audio_pooled = (vision_from_audio * vision_mask.unsqueeze(-1)).sum(1) / \
                                  vision_mask.sum(1, keepdim=True).clamp(min=1)

        # ③ 特徵級聯融合
        fused = self.fusion(
            text_audio_pair=(text_from_audio_pooled, audio_from_text_pooled),
            text_vision_pair=(text_from_vision_pooled, vision_from_text_pooled),
            audio_vision_pair=(audio_from_vision_pooled, vision_from_audio_pooled),
        )

        # ④ 分類
        logits = self.classifier(fused)

        return logits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 訓練 / 驗證
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, epoch):
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
            logits = model(ids, mask, aud, amask, vis, vmask)
            loss = criterion(logits, labels)

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

        logits = model(ids, mask, aud, amask, vis, vmask)

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc7 = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    return {"Acc7": round(float(acc7), 2)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主訓練流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 80)
    print("MOSI 多模態情感分析 v12 — MulG-Enhanced Architecture")
    print("整合 MulG (82.2% Acc7) 核心技術")
    print("=" * 80)
    print("\n🎯 核心技術:")
    print("  [技術1] Directed Pairwise Cross-Modal Attention")
    print("  [技術2] GRU序列編碼器 (Hidden=40, Layers=2)")
    print("  [技術3] 特徵級聯融合")
    print("  [技術4] MulG訓練策略 (LR=1e-3, Batch=128)")
    print("\n🎯 目標: Acc7 > 70% (MulG達成82.2%)")
    print("\n參考: MulG (Nature SR 2025) - https://www.nature.com/articles/s41598-025-93023-3")
    print("=" * 80)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim": 5,
        "vision_dim": 20,
        "gru_hidden": 40,      # MulG最佳配置
        "fusion_dim": 256,
        "num_classes": 7,
        "batch_size": 128,     # MulG配置
        "num_epochs": 100,     # MulG配置
        "learning_rate": 1e-3, # MulG配置
        "weight_decay": 1e-4,
        "warmup_ratio": 0.1,
        "freeze_layers": 6,    # 凍結DeBERTa前6層
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

    model = MulGEnhancedModel(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        gru_hidden=config["gru_hidden"],
        fusion_dim=config["fusion_dim"],
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

    # [核心技術4] MulG訓練策略
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"MulG配置: LR={config['learning_rate']}, Batch={config['batch_size']}, Epochs={config['num_epochs']}\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer,
                             scheduler, device, scaler, epoch)
        metrics = validate(model, val_loader, device)

        history.append({
            "epoch": epoch+1,
            "tr_loss": round(tr_loss, 4),
            **metrics
        })

        print(f"E{epoch+1:03d} | Loss={tr_loss:.4f} | Acc7={metrics['Acc7']:.2f}%")

        if metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **metrics}
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "metrics": metrics,
            }, save_dir / "best_model_v12.pth")
            print(f"  ✅ 新最佳 Acc7={metrics['Acc7']:.2f}%")

        # 早停：如果50 epochs沒提升則停止
        if epoch - best_acc7["epoch"] > 50:
            print(f"\n早停: {50} epochs無提升")
            break

    # 測試集評估
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v12.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = validate(model, test_loader, device)

    val_acc7 = best_acc7["Acc7"]
    test_acc7 = test_metrics["Acc7"]
    gap = abs(val_acc7 - test_acc7)

    print("\n【測試集最終結果】")
    print(f"  Val  Acc7: {val_acc7:.2f}% (Epoch {best_acc7['epoch']})")
    print(f"  Test Acc7: {test_acc7:.2f}%")
    print(f"  Gap: {gap:.2f}%")
    print(f"\n  vs scaf_old: 50.0%  → {'+' if test_acc7 > 50 else ''}{test_acc7-50:.2f}%")
    print(f"  vs MGT:      55.6%  → {'✓超越' if test_acc7 > 55.6 else '✗未達'}")
    print(f"  vs MulG:     82.2%  → {'+' if test_acc7 > 82.2 else ''}{test_acc7-82.2:.2f}%")

    with open(save_dir / "training_history_v12.json", "w") as f:
        json.dump({
            "history": history,
            "best_val": best_acc7,
            "test": test_metrics,
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)

    print(f"\n完成！模型已儲存: {save_dir / 'best_model_v12.pth'}")


if __name__ == "__main__":
    main()
