"""Generate the MMS-task macro-F1 bar chart from 表 4-2 of 論文＿李昇峰_v18.docx.

This mirrors the existing "Average macro-F1 — Five MMAFFBen Text-Only Tasks"
chart (regenerate_mmaffben_avg_chart.py) but plots only the **MMS** column
(三語情感強度任務) instead of the five-task average.

Visual conventions kept identical to the original:
  • one fixed colour per model (regardless of rank order)
  • value labels on top of every bar
  • y-axis starts at 0 (no truncation) with head-room for labels
  • x-axis labels fully shown (rotated, tight bounding box)

Output:
  docs/figures/mmaffben_mms_macroF1.svg
  docs/figures/mmaffben_mms_macroF1.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── MMS-task ma-F1 from 表 4-2 (MMS column) ─────────────────────────────────
MMS = {
    "SACF-Text (Ours)": 61.42,
    "GPT-4o-mini":      48.6,
    "EmoLlama-chat-7b": 48.6,
    "MMAFFLM-3b":       43.3,
    "MMAFFLM-7b":       43.0,
}

# colour per model (consistent with the average-score chart)
COLORS = {
    "MMAFFLM-3b":       "#F59E0B",   # orange
    "MMAFFLM-7b":       "#1D4ED8",   # blue
    "SACF-Text (Ours)": "#DC2626",   # red
    "GPT-4o-mini":      "#10B981",   # green
    "EmoLlama-chat-7b": "#8B5CF6",   # purple
}


def make_chart(items, outname):
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    names = [k for k, _ in items]
    vals  = [v for _, v in items]
    cols  = [COLORS[n] for n in names]

    bars = ax.bar(range(len(names)), vals, color=cols, width=0.6,
                  edgecolor="white", linewidth=0)
    for bar, v in zip(bars, vals):
        # 2 decimals if the value is not a clean 1-decimal number, else 1.
        label = f"{v:.2f}" if abs(v - round(v, 1)) > 1e-6 else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                label, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold")

    ax.set_title(
        "macro-F1 — MMAFFBen MMS Task (Multilingual Emotion)",
        fontsize=14, fontweight="bold", color="#1D4ED8", pad=14)
    ax.set_ylabel("macro-F1 on the MMS task (%)", fontsize=10.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10.5)
    # y starts at 0 (no truncation) with head-room so labels are not clipped
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E7EB", alpha=0.6, lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(OUTDIR / f"{outname}.svg"), bbox_inches="tight")
    fig.savefig(str(OUTDIR / f"{outname}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    items = sorted(MMS.items(), key=lambda x: -x[1])
    print("MMS-task ma-F1 (sorted descending):")
    for name, v in items:
        print(f"  {name:<20s}{v:>7.2f}")
    make_chart(items, "mmaffben_mms_macroF1")
    print("\n  ✓ Saved:  docs/figures/mmaffben_mms_macroF1.png")
    print("  ✓ Saved:  docs/figures/mmaffben_mms_macroF1.svg")
