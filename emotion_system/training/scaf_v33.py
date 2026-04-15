"""
MOSI 多模態情感分析 v33 — Fast Convergence Version
快速收斂版本,用於測試

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v33 策略: 更快收斂,快速驗證
  ✅ 統一學習率 1e-4 (更快收斂)
  ✅ Batch size 16 (加快訓練)
  ✅ 簡化架構 (移除複雜組件)
  ✅ 保留核心: Polarity Attention + Cross-modal Fusion
  ✅ 情感對比學習

目標: 快速達到 >50% Acc7
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
from tqdm import tqdm
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent

TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, "
    "negative to positive) of the following text: "
)


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

        print(f"資料集: {len(self.raw_text)} 筆")

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


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.hidden_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[-2], h[-1]], dim=-1)


class SimpleAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (hidden * weights * mask.unsqueeze(-1)).sum(1)
        return pooled


class SentimentContrastive(nn.Module):
    def __init__(self, delta_pos: float = 0.5, delta_neg: float = 1.5):
        super().__init__()
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, reg_labels: torch.Tensor):
        B = f1.size(0)
        f1_n = F.normalize(f1, dim=-1)
        f2_n = F.normalize(f2, dim=-1)

        diff = (reg_labels.unsqueeze(0) - reg_labels.unsqueeze(1)).abs()
        pos_mask = (diff < self.delta_pos).float() * (1 - torch.eye(B, device=f1.device))
        neg_mask = (diff > self.delta_neg).float()

        sim = torch.mm(f1_n, f2_n.T)
        pos_loss = (pos_mask * (1 - sim) ** 2).sum() / (pos_mask.sum() + 1e-9)
        neg_loss = (neg_mask * F.relu(sim - 0.2) ** 2).sum() / (neg_mask.sum() + 1e-9)

        return pos_loss + neg_loss


class FastFusionModel(nn.Module):
    def __init__(
        self,
        lang_model: str = "microsoft/deberta-v3-large",
        audio_dim: int = 5,
        vision_dim: int = 20,
        hidden_dim: int = 128,
        num_classes: int = 7,
    ):
        super().__init__()

        # 語言骨幹
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        lang_dim = self.lang_backbone.config.hidden_size
        self.text_pool = SimpleAttentionPool(lang_dim)

        # 非語言編碼
        self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim // 2)
        self.vision_encoder = LSTMEncoder(vision_dim, hidden_dim // 2)

        # 投影層
        self.text_proj = nn.Linear(lang_dim, hidden_dim)
        self.audio_proj = nn.Linear(hidden_dim, hidden_dim)
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 分類
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 對比學習
        self.contrastive = SentimentContrastive()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        vision: torch.Tensor,
        vision_mask: torch.Tensor,
    ):
        # 編碼
        lang_out = self.lang_backbone(input_ids, attention_mask)
        text_feat = self.text_pool(lang_out.last_hidden_state, attention_mask)
        audio_feat = self.audio_encoder(audio, audio_mask)
        vision_feat = self.vision_encoder(vision, vision_mask)

        # 投影
        text_feat = self.text_proj(text_feat)
        audio_feat = self.audio_proj(audio_feat)
        vision_feat = self.vision_proj(vision_feat)

        # 融合
        fused = torch.cat([text_feat, audio_feat, vision_feat], dim=-1)
        fused = self.fusion(fused)

        # 分類
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "text_feat": text_feat,
            "audio_feat": audio_feat,
            "vision_feat": vision_feat,
        }


def train_epoch(model, loader, optimizer, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"E{epoch+1}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        aud = batch["audio"].to(device)
        amask = batch["audio_mask"].to(device)
        vis = batch["vision"].to(device)
        vmask = batch["vision_mask"].to(device)
        labels = batch["cls7_label"].to(device)
        reg_labels = batch["reg_label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(ids, mask, aud, amask, vis, vmask)

            cls_loss = F.cross_entropy(outputs["logits"], labels, label_smoothing=0.1)
            con_loss = (
                model.contrastive(outputs["text_feat"], outputs["audio_feat"], reg_labels) +
                model.contrastive(outputs["text_feat"], outputs["vision_feat"], reg_labels)
            ) / 2

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

        outputs = model(ids, mask, aud, amask, vis, vmask)
        all_preds.extend(outputs["logits"].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc7 = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    return round(float(acc7), 2)


def main():
    print("=" * 70)
    print("MOSI v33 — Fast Convergence")
    print("=" * 70)
    print("🎯 統一LR=1e-4 | Batch=16 | 快速收斂測試")
    print("=" * 70)

    config = {
        "data_path": PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
        "model_dir": PROJECT_ROOT / "emotion_system/models",
        "lang_model": "microsoft/deberta-v3-large",
        "batch_size": 16,
        "num_epochs": 30,
        "learning_rate": 1e-4,
        "freeze_layers": 6,
    }

    with open(config["data_path"], "rb") as f:
        data = pickle.load(f)

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["lang_model"])

    train_ds = MOSIDataset(data["train"], tokenizer)
    val_ds = MOSIDataset(data["valid"], tokenizer)
    test_ds = MOSIDataset(data["test"], tokenizer)

    train_loader = DataLoader(train_ds, config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, config["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, config["batch_size"], shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FastFusionModel().to(device)

    # 凍結前6層
    for i in range(config["freeze_layers"]):
        for p in model.lang_backbone.encoder.layer[i].parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=1e-3
    )

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    print(f"\n訓練開始 | 設備: {device} | Batch: {config['batch_size']}\n")

    save_dir = Path(config["model_dir"])
    save_dir.mkdir(exist_ok=True)

    best_acc7 = 0.0
    best_epoch = 0

    for epoch in range(config["num_epochs"]):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, epoch)
        acc7 = validate(model, val_loader, device)

        print(f"E{epoch+1:02d} | Loss={loss:.4f} | Acc7={acc7:.2f}%", end="")

        if acc7 > best_acc7:
            best_acc7 = acc7
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_dir / "best_model_v33.pth")
            print(f" ✅ 新最佳")
        else:
            print()

        if epoch - best_epoch > 10:
            print(f"\n早停")
            break

    # 測試
    model.load_state_dict(torch.load(save_dir / "best_model_v33.pth", map_location=device, weights_only=False))
    test_acc7 = validate(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"【v33 最終結果】")
    print(f"  Val  Acc7: {best_acc7:.2f}% (E{best_epoch})")
    print(f"  Test Acc7: {test_acc7:.2f}%")
    print(f"  vs 目標 50.5%: {test_acc7-50.5:+.2f}% {'✓' if test_acc7 > 50.5 else '✗'}")


if __name__ == "__main__":
    main()
