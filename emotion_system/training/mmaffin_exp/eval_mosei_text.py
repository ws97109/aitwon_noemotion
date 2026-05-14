"""CMU-MOSEI text-only evaluation of SACF — cross-dataset generalization.

We load the CMU-MOSEI text dataset from Hugging Face (vintp/CMU-Mosei-text),
build a TextOnlySACF regression model initialized from sacf_final.pt's language
branch + Polarity-Enhanced Attention + shared projection, then fine-tune the
regression head on MOSEI train and evaluate on MOSEI test. We report the same
five metrics used on CMU-MOSI: Acc-7, Acc-2, F1, MAE, Corr.

Run:  python3 emotion_system/training/mmaffin_exp/eval_mosei_text.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, DebertaV2Tokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "emotion_system" / "training"))
from scaf_final import PolarityEnhancedAttention

MODEL_DIR = PROJECT_ROOT / "emotion_system" / "models"
SACF_CKPT = MODEL_DIR / "sacf_final.pt"


class TextOnlySACFRegression(nn.Module):
    """Text-only SACF with a regression head (predicts sentiment in [-3, +3])."""

    def __init__(self, lang_model: str, fusion_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.lang_backbone = AutoModel.from_pretrained(lang_model)
        h = self.lang_backbone.config.hidden_size
        self.polarity_attn = PolarityEnhancedAttention(h, dropout)
        self.shared = nn.Sequential(
            nn.Linear(h, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # Regression head: outputs in [-1, +1] via tanh, then scaled to [-3, +3]
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 1), nn.Tanh(),
        )

    def encode(self, input_ids, attention_mask):
        hidden = self.lang_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled, _ = self.polarity_attn(hidden, attention_mask)
        return self.shared(pooled)

    def forward(self, input_ids, attention_mask):
        e = self.encode(input_ids, attention_mask)
        return 3.0 * self.reg_head(e).squeeze(-1)  # scale tanh to [-3, +3]


def load_sacf_backbone(model: TextOnlySACFRegression, ckpt_path: Path) -> str:
    if not ckpt_path.exists():
        return f"WARNING: {ckpt_path.name} not found, using HF init."
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    own = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in state.items():
        # Strip per-branch prefix if needed (e.g., "branches.0.polarity_attn.")
        if k.startswith("branches.0."):
            k_norm = k.replace("branches.0.", "")
        else:
            k_norm = k
        if k_norm in own and own[k_norm].shape == v.shape:
            own[k_norm] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(own)
    return f"  loaded {loaded} tensors from {ckpt_path.name} (skip {skipped})"


class MOSEIDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int = 128):
        self.rows = rows
        self.tok = tokenizer
        self.max = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"] if r["text"] else r.get("ASR", "")
        if not text:
            text = "."
        enc = self.tok(text, add_special_tokens=True, max_length=self.max,
                       padding="max_length", truncation=True, return_tensors="pt")
        y = torch.tensor(float(r["sentiment"]), dtype=torch.float32)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          y,
        }


def compute_mosi_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute Acc-7, Acc-2 (non-negative >= 0), F1, MAE, Corr — matching CMU-MOSI eval."""
    # Acc-7: clip + round to integer in [-3, 3]
    pred_int = np.clip(np.round(preds), -3, 3).astype(int)
    true_int = np.clip(np.round(labels), -3, 3).astype(int)
    acc7 = float((pred_int == true_int).mean() * 100)

    # Within-1: pred is within ±1 of true class
    w1 = float((np.abs(pred_int - true_int) <= 1).mean() * 100)

    # Acc-2 + F1: standard non-negative protocol (>= 0 = positive)
    pred_pos = (preds >= 0).astype(int)
    true_pos = (labels >= 0).astype(int)
    acc2 = float((pred_pos == true_pos).mean() * 100)
    f1 = float(f1_score(true_pos, pred_pos, average="weighted") * 100)

    # MAE & Corr
    mae = float(np.abs(preds - labels).mean())
    if preds.std() < 1e-8:
        corr = 0.0
    else:
        corr = float(pearsonr(preds, labels)[0])

    return {"Acc-7": acc7, "Acc-2": acc2, "F1": f1, "MAE": mae, "Corr": corr, "Within-1": w1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_model", default="microsoft/deberta-v3-large")
    parser.add_argument("--lang_lr", type=float, default=5e-6)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_train", type=int, default=None,
                        help="Cap on training samples for debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Skip SACF backbone init")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Eval without any fine-tuning")
    parser.add_argument("--output", default=str(HERE / "mosei_text_results.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(" CMU-MOSEI text-only evaluation of SACF (cross-dataset generalization)")
    print("=" * 72)
    print(f"  device      = {device}")
    print(f"  backbone    = {args.lang_model}")
    print(f"  epochs      = {args.epochs}  | bs = {args.batch_size}  | max_len = {args.max_len}")
    print(f"  lr          = lang {args.lang_lr:.1e} / head {args.head_lr:.1e}")
    print(f"  pretrained  = {'sacf_final.pt' if not args.no_pretrain else 'HF only'}")
    print(f"  mode        = {'zero-shot' if args.zero_shot else 'fine-tune + eval'}")

    print("\n  Loading vintp/CMU-Mosei-text from Hugging Face ...")
    ds = load_dataset("vintp/CMU-Mosei-text")
    train_rows = [r for r in ds["train"]]
    val_rows   = [r for r in ds["validation"]]
    test_rows  = [r for r in ds["test"]]
    print(f"  train n={len(train_rows)} | val n={len(val_rows)} | test n={len(test_rows)}")

    if args.max_train is not None and len(train_rows) > args.max_train:
        np.random.shuffle(train_rows)
        train_rows = train_rows[: args.max_train]
        print(f"  capped train to {len(train_rows)}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(args.lang_model)
    train_ds = MOSEIDataset(train_rows, tokenizer, max_len=args.max_len)
    val_ds   = MOSEIDataset(val_rows,   tokenizer, max_len=args.max_len)
    test_ds  = MOSEIDataset(test_rows,  tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = TextOnlySACFRegression(args.lang_model)
    if not args.no_pretrain:
        msg = load_sacf_backbone(model, SACF_CKPT)
        print(msg)
    model.to(device)

    if not args.zero_shot:
        # Fine-tune
        lang_params = [p for n, p in model.named_parameters() if "lang_backbone" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "lang_backbone" not in n and p.requires_grad]
        optim = torch.optim.AdamW(
            [
                {"params": lang_params, "lr": args.lang_lr, "weight_decay": args.weight_decay},
                {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
            ]
        )
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(0.06 * total_steps)
        sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
        scaler = torch.cuda.amp.GradScaler()

        for ep in range(args.epochs):
            model.train()
            losses = []
            pbar = tqdm(train_loader, desc=f"E{ep+1} train")
            for b in pbar:
                ids   = b["input_ids"].to(device)
                amask = b["attention_mask"].to(device)
                y     = b["label"].to(device)
                optim.zero_grad()
                with torch.cuda.amp.autocast():
                    pred = model(ids, amask)
                    loss = F.smooth_l1_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                sched.step()
                losses.append(loss.item())
                pbar.set_postfix({"loss": f"{np.mean(losses[-50:]):.4f}"})

            # quick val check
            model.eval()
            preds_v, labels_v = [], []
            with torch.no_grad():
                for b in val_loader:
                    ids   = b["input_ids"].to(device)
                    amask = b["attention_mask"].to(device)
                    pred = model(ids, amask).cpu().numpy()
                    preds_v.append(pred); labels_v.append(b["label"].numpy())
            preds_v  = np.concatenate(preds_v)
            labels_v = np.concatenate(labels_v)
            m = compute_mosi_metrics(preds_v, labels_v)
            print(f"  E{ep+1}: val Acc-7={m['Acc-7']:.2f} Acc-2={m['Acc-2']:.2f} "
                  f"F1={m['F1']:.2f} MAE={m['MAE']:.4f} Corr={m['Corr']:.4f}")

    # Final eval on test
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in tqdm(test_loader, desc="MOSEI test"):
            ids   = b["input_ids"].to(device)
            amask = b["attention_mask"].to(device)
            p = model(ids, amask).cpu().numpy()
            preds.append(p); labels.append(b["label"].numpy())
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = compute_mosi_metrics(preds, labels)

    print("\n" + "=" * 72)
    print(" CMU-MOSEI Test Results — SACF-Text (this study)")
    print("=" * 72)
    for k, v in metrics.items():
        if k in ("MAE", "Corr"):
            print(f"  {k:9s}  =  {v:.4f}")
        else:
            print(f"  {k:9s}  =  {v:.2f} %")

    # Save
    out = {
        "config": vars(args),
        "device": device,
        "metrics": metrics,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ok  saved {args.output}")


if __name__ == "__main__":
    main()
