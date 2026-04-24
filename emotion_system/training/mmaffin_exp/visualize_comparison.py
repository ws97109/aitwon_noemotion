"""Compare baseline vs MMAFFIn-pretrained MOSI runs.

Expects the following files under emotion_system/models/:
  history_<version>.json       (final metrics)
  raw_logits_<version>.npy     (shape [n_seeds, n_test, 7])
  epoch_log_<version>.json     (per-epoch train loss per seed)

Produces five PNGs under emotion_system/training/mmaffin_exp/figs/:
  1. final_metrics_bar.png       — Acc7/Acc2/F1/MAE/Corr side-by-side
  2. per_seed_test_acc7.png      — per-seed Acc7 dots + ensemble stars
  3. training_curves.png         — train loss per epoch, avg across seeds
  4. confusion_matrices.png      — 7×7 CM for baseline & pretrained (ensemble)
  5. per_class_acc.png           — recall per class comparison
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_DIR    = PROJECT_ROOT / "emotion_system" / "models"
OUT_DIR      = Path(__file__).resolve().parent / "figs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

_DATA_CANDIDATES = [
    PROJECT_ROOT / "aitwon_emotion/emotion_system/data/mosi/unaligned_50.pkl",
    PROJECT_ROOT / "emotion_system/data/mosi/unaligned_50.pkl",
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if p.exists()), None)

CLASS_LABELS_7 = ["-3", "-2", "-1", "0", "+1", "+2", "+3"]
METRIC_KEYS    = ["Acc7", "Acc2", "F1", "MAE", "Corr"]
METRIC_BETTER  = {"Acc7": "up", "Acc2": "up", "F1": "up", "MAE": "down", "Corr": "up"}

COLORS = {"baseline": "#1f77b4", "pretrained": "#d62728"}


def load_run(version):
    h_path = MODEL_DIR / f"history_{version}.json"
    l_path = MODEL_DIR / f"raw_logits_{version}.npy"
    e_path = MODEL_DIR / f"epoch_log_{version}.json"
    if not h_path.exists():
        raise FileNotFoundError(h_path)
    with open(h_path, "r", encoding="utf-8") as f:
        h = json.load(f)
    logits = np.load(l_path) if l_path.exists() else None
    epochs = None
    if e_path.exists():
        with open(e_path, "r", encoding="utf-8") as f:
            epochs = json.load(f)
    return {"history": h, "logits": logits, "epochs": epochs, "version": version}


def ensure_test_labels():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    lbl = np.array(data["test"]["regression_labels"])
    cls7 = np.clip(np.round(lbl).astype(int), -3, 3) + 3
    return cls7


# ─────────────────────────────────────────────────────────────────────
def plot_final_metrics(base, pre, out_path):
    bm = base["history"]["final_metrics"]
    pm = pre ["history"]["final_metrics"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(METRIC_KEYS))
    w = 0.35
    b_vals = [bm[k] for k in METRIC_KEYS]
    p_vals = [pm[k] for k in METRIC_KEYS]
    ax.bar(x - w/2, b_vals, w, label=f"baseline ({base['version']})", color=COLORS["baseline"])
    ax.bar(x + w/2, p_vals, w, label=f"pretrained ({pre['version']})", color=COLORS["pretrained"])
    ax.set_xticks(x); ax.set_xticklabels(METRIC_KEYS)
    ax.set_title("MOSI Test Metrics: Baseline vs MMAFFIn-Pretrained (3-Seed Ensemble)")
    # annotate delta
    for i, (bv, pv, k) in enumerate(zip(b_vals, p_vals, METRIC_KEYS)):
        ax.text(i - w/2, bv, f"{bv:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, pv, f"{pv:.2f}", ha="center", va="bottom", fontsize=9)
        diff = pv - bv
        arrow_good = "↑" if (METRIC_BETTER[k] == "up" and diff > 0) or (METRIC_BETTER[k] == "down" and diff < 0) else "↓"
        ax.text(i, max(bv, pv) * 1.08, f"Δ={diff:+.2f} {arrow_good}",
                ha="center", fontsize=9,
                color="green" if (arrow_good == "↑") else "firebrick")
    ax.legend(loc="upper right")
    ax.set_ylabel("value")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_per_seed(base, pre, test_cls7, out_path):
    b_logits, p_logits = base["logits"], pre["logits"]
    if b_logits is None or p_logits is None:
        print("  (skip per-seed: missing logits)")
        return
    def seed_accs(logits):
        seed_accs = [(logits[i].argmax(1) == test_cls7).mean() * 100 for i in range(logits.shape[0])]
        ens_acc   = (logits.mean(0).argmax(1) == test_cls7).mean() * 100
        return seed_accs, ens_acc
    b_sa, b_ens = seed_accs(b_logits)
    p_sa, p_ens = seed_accs(p_logits)
    n = max(len(b_sa), len(p_sa))
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(n)
    w = 0.35
    ax.bar(xs - w/2, b_sa, w, label="baseline seeds", color=COLORS["baseline"], alpha=0.75)
    ax.bar(xs + w/2, p_sa, w, label="pretrained seeds", color=COLORS["pretrained"], alpha=0.75)
    ax.axhline(b_ens, color=COLORS["baseline"], linestyle="--",
               label=f"baseline ensemble={b_ens:.2f}%")
    ax.axhline(p_ens, color=COLORS["pretrained"], linestyle="--",
               label=f"pretrained ensemble={p_ens:.2f}%")
    seeds = base["history"].get("seeds", list(range(n)))
    ax.set_xticks(xs); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Test Acc7 (%)")
    ax.set_title("Per-Seed Test Acc7 (bars) vs Ensemble (dashed)")
    for i, v in enumerate(b_sa):
        ax.text(i - w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(p_sa):
        ax.text(i + w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_training_curves(base, pre, out_path):
    if base["epochs"] is None or pre["epochs"] is None:
        print("  (skip training curves: missing epoch_log)")
        return
    def avg_loss_per_epoch(epochs_obj):
        rows = epochs_obj["epochs"]
        by_ep = {}
        for r in rows:
            by_ep.setdefault(r["epoch"], []).append(r["train_loss"])
        ep_sorted = sorted(by_ep.keys())
        return ep_sorted, [float(np.mean(by_ep[e])) for e in ep_sorted], \
               [float(np.std(by_ep[e])) for e in ep_sorted]
    b_ep, b_mean, b_std = avg_loss_per_epoch(base["epochs"])
    p_ep, p_mean, p_std = avg_loss_per_epoch(pre["epochs"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(b_ep, b_mean, color=COLORS["baseline"], label=f"baseline ({base['version']})", linewidth=2)
    ax.fill_between(b_ep, np.array(b_mean) - np.array(b_std), np.array(b_mean) + np.array(b_std),
                    color=COLORS["baseline"], alpha=0.2)
    ax.plot(p_ep, p_mean, color=COLORS["pretrained"], label=f"pretrained ({pre['version']})", linewidth=2)
    ax.fill_between(p_ep, np.array(p_mean) - np.array(p_std), np.array(p_mean) + np.array(p_std),
                    color=COLORS["pretrained"], alpha=0.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Loss (avg ± std across seeds)")
    ax.set_title("Training Loss Curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_confusion_matrices(base, pre, test_cls7, out_path):
    if base["logits"] is None or pre["logits"] is None:
        print("  (skip CM: missing logits)")
        return
    b_pred = base["logits"].mean(0).argmax(1)
    p_pred = pre ["logits"].mean(0).argmax(1)

    def cm(pred):
        m = np.zeros((7, 7), dtype=int)
        for t, p in zip(test_cls7, pred):
            m[t, p] += 1
        return m
    bm = cm(b_pred); pm = cm(p_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, M, tag in zip(axes, [bm, pm], [f"baseline ({base['version']})", f"pretrained ({pre['version']})"]):
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(7)); ax.set_xticklabels(CLASS_LABELS_7)
        ax.set_yticks(range(7)); ax.set_yticklabels(CLASS_LABELS_7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        acc = (M.diagonal().sum() / M.sum()) * 100
        ax.set_title(f"{tag}\nAcc7={acc:.2f}%")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > M.max() * 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_per_class(base, pre, test_cls7, out_path):
    if base["logits"] is None or pre["logits"] is None:
        print("  (skip per-class: missing logits)")
        return
    def recall(logits):
        pred = logits.mean(0).argmax(1)
        out = []
        for c in range(7):
            mask = test_cls7 == c
            if mask.sum() == 0:
                out.append(0.0)
            else:
                out.append((pred[mask] == c).mean() * 100)
        return out
    b_r = recall(base["logits"])
    p_r = recall(pre ["logits"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7); w = 0.35
    ax.bar(x - w/2, b_r, w, label="baseline", color=COLORS["baseline"])
    ax.bar(x + w/2, p_r, w, label="pretrained", color=COLORS["pretrained"])
    ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS_7)
    ax.set_xlabel("True class (sentiment intensity)")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-Class Recall Comparison")
    for i in range(7):
        ax.text(i - w/2, b_r[i], f"{b_r[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, p_r[i], f"{p_r[i]:.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  -> {out_path}")


def print_summary(base, pre):
    bm = base["history"]["final_metrics"]; pm = pre["history"]["final_metrics"]
    print("\n" + "=" * 72)
    print(f"Baseline   ({base['version']}): {bm}")
    print(f"Pretrained ({pre['version']}): {pm}")
    print("Δ (pretrained - baseline):")
    for k in METRIC_KEYS:
        d = pm[k] - bm[k]
        good = (METRIC_BETTER[k] == "up" and d > 0) or (METRIC_BETTER[k] == "down" and d < 0)
        tag  = "BETTER" if good else ("WORSE" if d != 0 else "SAME")
        print(f"  {k:5s}: {d:+.4f}  [{tag}]")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",   default="v60_baseline")
    ap.add_argument("--pretrained", default="v60_mmaffin")
    args = ap.parse_args()

    print(f"Loading baseline   = {args.baseline}")
    print(f"Loading pretrained = {args.pretrained}")
    base = load_run(args.baseline)
    pre  = load_run(args.pretrained)

    test_cls7 = ensure_test_labels()

    print("\nGenerating plots:")
    plot_final_metrics     (base, pre,             OUT_DIR / "1_final_metrics_bar.png")
    plot_per_seed          (base, pre, test_cls7,  OUT_DIR / "2_per_seed_test_acc7.png")
    plot_training_curves   (base, pre,             OUT_DIR / "3_training_curves.png")
    plot_confusion_matrices(base, pre, test_cls7,  OUT_DIR / "4_confusion_matrices.png")
    plot_per_class         (base, pre, test_cls7,  OUT_DIR / "5_per_class_recall.png")

    print_summary(base, pre)
    print(f"\nAll figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
