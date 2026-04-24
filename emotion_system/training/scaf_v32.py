"""
MOSI 多模態情感分析 v32 — Enhanced SACF Fusion
基於 scaf_old 成功經驗 + v31 修復

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
借鑒 scaf_old (50% Acc7) 的成功要素:
  ✅ 分層學習率: lang=5e-6, modal=1e-4
  ✅ 極小 batch_size=8 (更好泛化)
  ✅ 情感感知對比學習 (sentiment-aware contrastive)
  ✅ modal_hidden=128

融合 v31 的改進:
  ✅ 統一特徵投影到 256 維
  ✅ Directed Pairwise Attention (MulG)
  ✅ MAG (Multimodal Adaptation Gate)

目標: Acc7 > 50.5% (超越 scaf_old)
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
from typing import Tuple
from tqdm import tqdm
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
            "reg_label": self.reg_labels[idx],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LSTM 序列編碼器 (from scaf_old)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.hidden_dim = hidden_dim * 2  # 256

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        # 取最後一層的前向和後向hidden state
        h_forward = h[-2]
        h_backward = h[-1]
        pooled = torch.cat([h_forward, h_backward], dim=-1)
        return pooled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Polarity-Enhanced Attention (from scaf_old)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PolarityEnhancedAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        """Returns: pooled (B, H), gates (B, L)"""
        g = self.gate(hidden)  # (B, L, 1)
        m = mask.unsqueeze(-1).float()
        enhanced = (0.75 * hidden + 0.25 * hidden * g) * m
        pooled = enhanced.sum(1) / m.sum(1).clamp(min=1e-9)
        gates = (g * m).squeeze(-1)
        return self.dropout(pooled), gates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Directed Pairwise Attention (from MulG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DirectedPairwiseAttention(nn.Module):
    def __init__(self, dim1: int, dim2: int, dropout: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim1, dim2))
        nn.init.xavier_uniform_(self.W)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1: torch.Tensor, seq2: torch.Tensor,
                mask1: torch.Tensor, mask2: torch.Tensor):
        """返回雙向注意力後的pooled特徵"""
        B, T1, D1 = seq1.shape
        T2 = seq2.size(1)

        attn_scores = torch.bmm(
            torch.matmul(seq1, self.W),
            seq2.transpose(1, 2)
        )

        # seq1 attend to seq2
        mask2_expanded = mask2.unsqueeze(1).expand(B, T1, T2)
        scores_12 = attn_scores.masked_fill(mask2_expanded == 0, float('-inf'))
        weights_12 = F.softmax(scores_12, dim=2)
        weights_12 = self.dropout(weights_12)
        attended_1 = torch.bmm(weights_12, seq2)  # (B, T1, D2)

        # 池化
        pooled_1 = (attended_1 * mask1.unsqueeze(-1)).sum(1) / \
                   mask1.sum(1, keepdim=True).clamp(min=1)

        # seq2 attend to seq1
        mask1_expanded = mask1.unsqueeze(2).expand(B, T1, T2)
        scores_21 = attn_scores.transpose(1, 2)
        scores_21 = scores_21.masked_fill(
            mask1_expanded.transpose(1, 2) == 0, float('-inf')
        )
        weights_21 = F.softmax(scores_21, dim=2)
        weights_21 = self.dropout(weights_21)
        attended_2 = torch.bmm(weights_21, seq1)  # (B, T2, D1)

        # 池化
        pooled_2 = (attended_2 * mask2.unsqueeze(-1)).sum(1) / \
                   mask2.sum(1, keepdim=True).clamp(min=1)

        return pooled_1, pooled_2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 情感感知對比損失 (from scaf_old, 修復梯度流)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SentimentContrastiveLoss(nn.Module):
    """基於情感標籤距離定義正負樣本對"""
    def __init__(self, delta_pos: float = 0.5, delta_neg: float = 1.5,
                 margin: float = 0.2):
        super().__init__()
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg
        self.margin = margin

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor,
                reg_labels: torch.Tensor):
        """
        Args:
            feat1, feat2: (B, D) 兩個模態特徵
            reg_labels: (B,) 情感回歸標籤
        """
        B = feat1.size(0)
        feat1_n = F.normalize(feat1, dim=-1)
        feat2_n = F.normalize(feat2, dim=-1)

        # 情感差值矩陣
        diff = (reg_labels.unsqueeze(0) - reg_labels.unsqueeze(1)).abs()

        # 正對: 情感差值 < delta_pos
        # 負對: 情感差值 > delta_neg
        pos_mask = (diff < self.delta_pos).float()
        # 排除對角線
        eye = torch.eye(B, device=feat1.device)
        pos_mask = pos_mask * (1 - eye)

        neg_mask = (diff > self.delta_neg).float()

        # 相似度矩陣
        sim = torch.mm(feat1_n, feat2_n.T)

        # 正對損失: 相似度應接近1
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)

        # 負對損失: 相似度應低於margin
        neg_loss = (neg_mask * F.relu(sim - self.margin) ** 2).sum() / \
                   (neg_mask.sum() + 1e-9)

        return pos_loss + neg_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主模型: EnhancedSACFFusion (v32)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EnhancedSACFFusion(nn.Module):
    """
    v32: 融合 scaf_old 成功要素 + v31 改進

    架構:
      1. DeBERTa + PolarityEnhancedAttention
      2. LSTM 編碼 audio/vision (hidden=128)
      3. 特徵投影到統一256維
      4. Directed Pairwise Attention (3對)
      5. 拼接融合 → 分類
      6. 情感感知對比學習
    """
    def __init__(
        self,
        lang_model: str = "microsoft/deberta-v3-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        modal_hidden: int = 128,
        proj_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ① 語言骨幹 + Polarity Attention
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size  # 1024
        self.polarity_attn = PolarityEnhancedAttention(lang_dim, dropout)

        # ② LSTM 編碼器
        self.audio_encoder = ModalityEncoder(audio_dim, modal_hidden)
        self.vision_encoder = ModalityEncoder(vision_dim, modal_hidden)
        lstm_out_dim = modal_hidden * 2  # 256

        # ③ 特徵投影層 (統一到 proj_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(lang_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ④ Directed Pairwise Attention (投影後空間)
        self.attn_ta = DirectedPairwiseAttention(proj_dim, proj_dim, dropout)
        self.attn_tv = DirectedPairwiseAttention(proj_dim, proj_dim, dropout)
        self.attn_av = DirectedPairwiseAttention(proj_dim, proj_dim, dropout)

        # ⑤ 融合 + 分類
        fusion_dim = proj_dim * 6  # 6個交叉特徵
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

        # ⑥ 情感感知對比損失
        self.contrastive = SentimentContrastiveLoss(
            delta_pos=0.5, delta_neg=1.5, margin=0.2
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
        # ① 語言編碼 + Polarity Attention
        lang_out = self.lang_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_hidden = lang_out.last_hidden_state
        text_pooled, _ = self.polarity_attn(text_hidden, attention_mask)

        # ② 非語言編碼
        audio_pooled = self.audio_encoder(audio, audio_mask)
        vision_pooled = self.vision_encoder(vision, vision_mask)

        # ③ 投影到統一空間
        text_feat = self.text_proj(text_pooled)     # (B, 256)
        audio_feat = self.audio_proj(audio_pooled)  # (B, 256)
        vision_feat = self.vision_proj(vision_pooled)  # (B, 256)

        # 準備序列 (用於注意力)
        text_seq = text_hidden @ self.text_proj[0].weight.T + self.text_proj[0].bias
        # audio/vision 已經池化,擴展為"序列"
        audio_seq = audio_feat.unsqueeze(1)  # (B, 1, 256)
        vision_seq = vision_feat.unsqueeze(1)  # (B, 1, 256)
        audio_seq_mask = torch.ones(audio.size(0), 1, device=audio.device)
        vision_seq_mask = torch.ones(vision.size(0), 1, device=vision.device)

        # ④ Directed Pairwise Attention
        t_from_a, a_from_t = self.attn_ta(
            text_seq, audio_seq, attention_mask, audio_seq_mask
        )
        t_from_v, v_from_t = self.attn_tv(
            text_seq, vision_seq, attention_mask, vision_seq_mask
        )
        a_from_v, v_from_a = self.attn_av(
            audio_seq, vision_seq, audio_seq_mask, vision_seq_mask
        )

        # ⑤ 融合
        fused = torch.cat([
            t_from_a, a_from_t,
            t_from_v, v_from_t,
            a_from_v, v_from_a,
        ], dim=-1)
        fused = self.fusion(fused)

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
    total_loss, total_cls, total_con = 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc=f"Train E{epoch+1}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        labels = batch["cls7_label"].to(device)
        reg_labels = batch["reg_label"].to(device)

        optimizer.zero_grad()
        use_amp = scaler is not None

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(ids, mask, aud, amask, vis, vmask)

            # 分類損失
            cls_loss = F.cross_entropy(outputs["logits"], labels, label_smoothing=0.1)

            # 情感感知對比損失 (3對)
            con_loss = (
                model.contrastive(outputs["text_feat"], outputs["audio_feat"], reg_labels) +
                model.contrastive(outputs["text_feat"], outputs["vision_feat"], reg_labels) +
                model.contrastive(outputs["audio_feat"], outputs["vision_feat"], reg_labels)
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
        total_cls += cls_loss.item()
        total_con += con_loss.item()

    n = len(loader)
    return {
        "loss": round(total_loss / n, 4),
        "cls": round(total_cls / n, 4),
        "con": round(total_con / n, 4),
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
    print("MOSI 多模態情感分析 v32 — Enhanced SACF Fusion")
    print("=" * 80)
    print("\n🎯 融合成功要素:")
    print("  • scaf_old: 分層學習率 + 情感對比學習 (50% Acc7)")
    print("  • v31: 統一特徵投影 + Directed Attention")
    print("\n🔧 關鍵配置:")
    print("  • Batch size: 8 (scaf_old成功配置)")
    print("  • Lang LR: 5e-6 | Modal LR: 1e-4 (分層學習率)")
    print("  • Modal hidden: 128 | Proj dim: 256")
    print("  • Sentiment-aware contrastive learning")
    print("\n🎯 目標: Acc7 > 50.5%")
    print("=" * 80)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "microsoft/deberta-v3-large",
        "max_text_len": 80,
        "audio_dim": 5,
        "vision_dim": 20,
        "modal_hidden": 128,
        "proj_dim": 256,
        "num_classes": 7,
        "dropout": 0.2,
        "batch_size": 8,
        "num_epochs": 40,
        "lang_lr": 5e-6,
        "modal_lr": 1e-4,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
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

    model = EnhancedSACFFusion(
        lang_model=config["lang_model"],
        audio_dim=config["audio_dim"],
        vision_dim=config["vision_dim"],
        modal_hidden=config["modal_hidden"],
        proj_dim=config["proj_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
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

    # 分層學習率優化器 (關鍵!)
    lang_params = []
    modal_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lang_backbone' in name or 'polarity_attn' in name:
                lang_params.append(param)
            else:
                modal_params.append(param)

    optimizer = optim.AdamW([
        {'params': lang_params, 'lr': config["lang_lr"]},
        {'params': modal_params, 'lr': config["modal_lr"]},
    ], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n開始訓練 | 設備: {device}")
    print(f"Lang params: {len(lang_params)} @ {config['lang_lr']}")
    print(f"Modal params: {len(modal_params)} @ {config['modal_lr']}\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc7 = {"Acc7": 0.0, "epoch": 0}
    history = []

    for epoch in range(config["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler,
                                    device, scaler, epoch)
        val_metrics = validate(model, val_loader, device)

        history.append({"epoch": epoch+1, **train_metrics, **val_metrics})

        print(f"E{epoch+1:03d} | Loss={train_metrics['loss']:.4f} "
              f"(cls={train_metrics['cls']:.4f}, con={train_metrics['con']:.4f}) | "
              f"Acc7={val_metrics['Acc7']:.2f}%")

        if val_metrics["Acc7"] > best_acc7["Acc7"]:
            best_acc7 = {"epoch": epoch+1, **val_metrics}
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "metrics": val_metrics,
            }, save_dir / "best_model_v32.pth")
            print(f"  ✅ 新最佳 Acc7={val_metrics['Acc7']:.2f}%")

        # 早停
        if epoch - best_acc7["epoch"] > 15:
            print(f"\n早停: 15 epochs無提升")
            break

    # 測試集評估
    print("\n" + "=" * 60)
    ckpt = torch.load(save_dir / "best_model_v32.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = validate(model, test_loader, device)

    val_acc7 = best_acc7["Acc7"]
    test_acc7 = test_metrics["Acc7"]
    gap = abs(val_acc7 - test_acc7)

    print("\n【測試集最終結果 - v32 Enhanced SACF】")
    print(f"  Val  Acc7: {val_acc7:.2f}% (Epoch {best_acc7['epoch']})")
    print(f"  Test Acc7: {test_acc7:.2f}%")
    print(f"  Gap: {gap:.2f}%")
    print(f"\n  vs scaf_old: 50.0%  → {test_acc7-50:+.2f}%")
    print(f"  vs 目標:     50.5%  → {'✓超越' if test_acc7 > 50.5 else '✗未達'} ({test_acc7-50.5:+.2f}%)")

    with open(save_dir / "training_history_v32.json", "w") as f:
        json.dump({
            "history": history,
            "best_val": best_acc7,
            "test": test_metrics,
            "config": {k: str(v) for k, v in config.items()}
        }, f, indent=2)

    print(f"\n完成！模型已儲存: {save_dir / 'best_model_v32.pth'}")


if __name__ == "__main__":
    main()
