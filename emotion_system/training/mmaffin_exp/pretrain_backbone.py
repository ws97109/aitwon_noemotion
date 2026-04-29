"""Pretrain DeBERTa-v3-large backbone on MMAFFIn (MMS+XED) 3-class sentiment.

Goal: produce a backbone state dict that later replaces the random init
inside SACFModel.lang_backbone in scaf_final_mmaffin.py.

Outputs two files under emotion_system/models/:
  - mmaffin_pretrain_backbone.pt   (state_dict of AutoModel only; small enough to load)
  - mmaffin_pretrain_log.json      (per-epoch train/val metrics)
"""
import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH      = Path(__file__).resolve().parent / "data" / "pretrain_corpus.pkl"
MODEL_DIR      = PROJECT_ROOT / "emotion_system" / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
BACKBONE_OUT   = MODEL_DIR / "mmaffin_pretrain_backbone.pt"
LOG_OUT        = MODEL_DIR / "mmaffin_pretrain_log.json"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class SentimentCorpus(Dataset):
    def __init__(self, rows, tokenizer, max_len=80):
        self.rows = rows
        self.tok  = tokenizer
        self.max  = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r   = self.rows[i]
        enc = self.tok(
            r["text"],
            add_special_tokens=True, max_length=self.max,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(r["label"], dtype=torch.long),
        }


class BackbonePlusHead(nn.Module):
    """DeBERTa backbone + tiny CLS head for 3-class sentiment pretraining.
    Only backbone weights matter downstream; head is discarded."""
    def __init__(self, model_name, n_classes=3, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h // 2, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        m      = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(pooled)


def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for b in loader:
            ids, mask, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
            logits = model(ids, mask)
            loss_sum += crit(logits, y).item() * y.size(0)
            correct  += (logits.argmax(-1) == y).sum().item()
            total    += y.size(0)
    return {"loss": loss_sum / total, "acc": correct / total * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="microsoft/deberta-v3-large")
    ap.add_argument("--max_len",    type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs",     type=int, default=3)
    ap.add_argument("--lr",         type=float, default=2e-5)
    ap.add_argument("--warmup",     type=float, default=0.06)
    ap.add_argument("--wd",         type=float, default=0.01)
    ap.add_argument("--dropout",    type=float, default=0.1)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--gpu",        default="0", help="CUDA device id")
    ap.add_argument("--eval_every", type=int, default=1, help="evaluate every N epochs")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print("=" * 70)
    print(f"MMAFFIn sentiment pretraining")
    print(f"  data    = {DATA_PATH}")
    print(f"  model   = {args.model}")
    print(f"  device  = {device} (CUDA_VISIBLE_DEVICES={args.gpu})")
    print(f"  out_pt  = {BACKBONE_OUT}")
    print("=" * 70)

    with open(DATA_PATH, "rb") as f:
        corp = pickle.load(f)
    print(f"  train rows: {len(corp['train'])} | val rows: {len(corp['val'])}")

    tok = DebertaV2Tokenizer.from_pretrained(args.model)
    train_ds = SentimentCorpus(corp["train"], tok, max_len=args.max_len)
    val_ds   = SentimentCorpus(corp["val"],   tok, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = BackbonePlusHead(args.model, n_classes=3, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    crit   = nn.CrossEntropyLoss()

    history = []
    best_val_acc = -1.0
    best_state   = None
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total   = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}", leave=False)
        for b in pbar:
            ids  = b["input_ids"].to(device, non_blocking=True)
            mask = b["attention_mask"].to(device, non_blocking=True)
            y    = b["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(ids, mask)
                loss   = crit(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            ep_loss += loss.item() * y.size(0)
            ep_correct += (logits.argmax(-1) == y).sum().item()
            ep_total   += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             acc=f"{ep_correct/ep_total*100:.1f}%")

        train_m = {"loss": ep_loss/ep_total, "acc": ep_correct/ep_total*100}
        if ep % args.eval_every == 0 or ep == args.epochs:
            val_m = evaluate(model, val_loader, device)
        else:
            val_m = {"loss": None, "acc": None}
        elapsed = (time.time() - t0) / 60.0
        print(f"  E{ep} | train loss={train_m['loss']:.4f} acc={train_m['acc']:.2f}% | "
              f"val loss={val_m['loss']} acc={val_m['acc']}  [{elapsed:.1f} min]")
        history.append({"epoch": ep, "train": train_m, "val": val_m,
                        "elapsed_min": round(elapsed, 2)})

        if val_m["acc"] is not None and val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            # detach + cpu clone so GPU memory is freed
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.backbone.state_dict().items()}
            print(f"  ✓ new best val acc={best_val_acc:.2f}%")

    if best_state is None:
        # fallback: final weights
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.backbone.state_dict().items()}

    torch.save({
        "backbone_state_dict": best_state,
        "model_name":          args.model,
        "best_val_acc":        round(float(best_val_acc), 4),
        "epochs":              args.epochs,
        "train_samples":       len(train_ds),
        "val_samples":         len(val_ds),
    }, BACKBONE_OUT)

    with open(LOG_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "history": history,
            "best_val_acc": best_val_acc,
            "backbone_path": str(BACKBONE_OUT),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved backbone  -> {BACKBONE_OUT}")
    print(f"   Saved log       -> {LOG_OUT}")
    print(f"   best val acc    = {best_val_acc:.2f}%")
    print(f"   total time      = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
