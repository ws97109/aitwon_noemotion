"""Zero-shot evaluation of the MMAFFIn-pretrained backbone on MOSI text-only.

Rationale: the backbone was trained with a 3-class head (neg/neu/pos). If the
head's predictions on MOSI text align with MOSI's sign-of-label, it means the
pretraining transferred sentiment knowledge. We can then estimate an *upper
bound* for how much pretraining alone could help.

Does NOT interfere with GPU0 training — runs on GPU1 or CPU.
"""
import argparse
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, DebertaV2Tokenizer

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
MODEL_DIR    = PROJECT_ROOT / "emotion_system" / "models"
MOSI_PATH    = PROJECT_ROOT / "emotion_system" / "data" / "mosi" / "unaligned_50.pkl"
BACKBONE_PT  = MODEL_DIR / "mmaffin_pretrain_backbone.pt"

TASK_PROMPT = "Predict the sentiment intensity (-3 to 3, negative to positive) of the following text: "


class Model3Class(nn.Module):
    """Same architecture as BackbonePlusHead in pretrain_backbone.py, but head
    is RE-TRAINED from scratch since we discarded it during pretrain save."""
    def __init__(self, model_name, n_classes=3, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h//2, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(pooled)


class MOSITextDataset(Dataset):
    def __init__(self, texts, labels3, tokenizer, max_len=80, with_prompt=True):
        self.texts, self.labels = texts, labels3
        self.tok, self.max_len = tokenizer, max_len
        self.with_prompt = with_prompt

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        t = str(self.texts[i])
        if self.with_prompt:
            t = TASK_PROMPT + t
        enc = self.tok(t, add_special_tokens=True, max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


def to3class(reg):
    reg = np.asarray(reg)
    out = np.ones_like(reg, dtype=np.int64)  # neutral
    out[reg < -0.5] = 0   # negative
    out[reg >  0.5] = 2   # positive
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1", help="Use GPU 1 to not interfere with main training on GPU0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--probe_epochs", type=int, default=3,
                    help="0 = pure zero-shot eval (random head); >0 = train a linear probe on frozen backbone to measure representation quality")
    ap.add_argument("--compare_random_backbone", action="store_true",
                    help="Also run probe on random-init backbone for comparison")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={args.gpu})")

    print(f"Loading MOSI from {MOSI_PATH}")
    with open(MOSI_PATH, "rb") as f:
        data = pickle.load(f)

    # Train+Val for probe training; Test for reporting
    train_txt = list(data["train"]["raw_text"]) + list(data["valid"]["raw_text"])
    train_reg = np.concatenate([data["train"]["regression_labels"], data["valid"]["regression_labels"]])
    test_txt  = list(data["test"]["raw_text"])
    test_reg  = np.array(data["test"]["regression_labels"])

    train_y = to3class(train_reg)
    test_y  = to3class(test_reg)
    print(f"  train+val: n={len(train_txt)} class dist = {np.bincount(train_y, minlength=3).tolist()}")
    print(f"  test:      n={len(test_txt)}  class dist = {np.bincount(test_y, minlength=3).tolist()}")

    tok = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")

    train_ds = MOSITextDataset(train_txt, train_y, tok)
    test_ds  = MOSITextDataset(test_txt,  test_y,  tok)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Class-balance weights for the linear probe (MOSI is highly imbalanced)
    counts = np.bincount(train_y, minlength=3).astype(float)
    cw = torch.FloatTensor(len(train_y) / (3 * np.where(counts == 0, 1, counts))).to(device)

    def run_probe(backbone_src, tag):
        print(f"\n{'=' * 60}\n[{tag}] loading backbone = {backbone_src}\n{'=' * 60}")
        model = Model3Class("microsoft/deberta-v3-large").to(device)
        if backbone_src == "mmaffin":
            ckpt = torch.load(str(BACKBONE_PT), map_location="cpu")
            state = ckpt["backbone_state_dict"]
            miss, unex = model.backbone.load_state_dict(state, strict=False)
            print(f"  loaded. missing={len(miss)} unexpected={len(unex)}")
        elif backbone_src == "hf":
            print(f"  using HF pretrained weights as-is (no MMAFFIn)")
        else:
            raise ValueError(backbone_src)

        # Freeze backbone — we only train the head (linear probe)
        for p in model.backbone.parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M (head only)")

        opt  = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)
        crit = nn.CrossEntropyLoss(weight=cw)

        def eval_test():
            model.eval(); correct, tot = 0, 0
            preds, labels = [], []
            with torch.no_grad():
                for b in test_loader:
                    ids = b["input_ids"].to(device); mk = b["attention_mask"].to(device); y = b["label"].to(device)
                    logits = model(ids, mk)
                    preds.append(logits.argmax(-1).cpu().numpy()); labels.append(y.cpu().numpy())
                    correct += (logits.argmax(-1) == y).sum().item(); tot += y.size(0)
            preds = np.concatenate(preds); labels = np.concatenate(labels)
            acc = correct / tot * 100
            # per-class recall
            per = []
            for c in range(3):
                mask = labels == c
                r = (preds[mask] == c).mean() * 100 if mask.sum() > 0 else 0.0
                per.append(r)
            return acc, per

        # 1) Pure zero-shot (random head, frozen backbone)
        acc_zs, per_zs = eval_test()
        print(f"  [zero-shot, random head] Test Acc3 = {acc_zs:.2f}%  per-class (neg/neu/pos) = "
              f"{per_zs[0]:.1f}/{per_zs[1]:.1f}/{per_zs[2]:.1f}")

        # 2) Linear probe — train head only, keep backbone frozen
        if args.probe_epochs > 0:
            for ep in range(1, args.probe_epochs + 1):
                model.train(); tl = 0; tc = 0; tn = 0
                for b in train_loader:
                    ids = b["input_ids"].to(device); mk = b["attention_mask"].to(device); y = b["label"].to(device)
                    opt.zero_grad()
                    with torch.amp.autocast("cuda"):
                        logits = model(ids, mk)
                        loss   = crit(logits, y)
                    loss.backward(); opt.step()
                    tl += loss.item()*y.size(0); tc += (logits.argmax(-1)==y).sum().item(); tn += y.size(0)
                acc_train = tc/tn*100
                acc, per = eval_test()
                print(f"  [probe E{ep}] train loss={tl/tn:.4f} acc={acc_train:.2f}% | "
                      f"test Acc3={acc:.2f}%  per-class={per[0]:.1f}/{per[1]:.1f}/{per[2]:.1f}")
            return {"tag": tag, "zero_shot_acc3": acc_zs, "probe_acc3": acc,
                    "probe_per_class": per, "zero_shot_per_class": per_zs}
        return {"tag": tag, "zero_shot_acc3": acc_zs, "zero_shot_per_class": per_zs}

    results = []
    if args.compare_random_backbone:
        results.append(run_probe("hf", "HF baseline (no MMAFFIn)"))
    results.append(run_probe("mmaffin", "MMAFFIn-pretrained"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    # Random 3-class baseline = 33.3%. MOSI majority-class (pos) ~ 40%
    test_majority = np.bincount(test_y, minlength=3).max() / len(test_y) * 100
    print(f"  MOSI test 3-class majority baseline: {test_majority:.2f}%")
    print(f"  Random baseline: 33.33%")
    for r in results:
        print(f"  {r['tag']}:")
        print(f"    zero-shot Acc3 = {r['zero_shot_acc3']:.2f}% (per-class {r['zero_shot_per_class']})")
        if "probe_acc3" in r:
            print(f"    linear probe Acc3 = {r['probe_acc3']:.2f}% (per-class {r['probe_per_class']})")

    # Save
    import json
    out = HERE / "zero_shot_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "majority_baseline_pct":  float(test_majority),
            "random_baseline_pct":    33.33,
            "results":                results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
