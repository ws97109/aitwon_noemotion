"""Re-generate the MMAFFBen average ma-F1 bar chart with correct averages
from 表 4-2 of 論文＿李昇峰_v18.docx.

The previous chart had two wrong values:
  • MMAFFLM-3b shown as 58.22 → correct is 58.2
  • MMAFFLM-7b shown as 55.44 → correct is 57.1   (largest error)
  • GPT-4o-mini shown as 51.96 → correct is 51.8

This script recomputes the average from the per-task ma-F1 numbers in
the source table and draws a corrected bar chart sorted by average score
(descending).  Colours are kept consistent with the previous chart per
model identity (i.e., MMAFFLM-3b stays orange, SACF-Text stays red, etc.).

Output:
  docs/figures/mmaffben_avg_macroF1.svg
  docs/figures/mmaffben_avg_macroF1.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── Per-task ma-F1 from 表 4-2  (5 MMAFFBen text-only tasks) ────────────────
PER_TASK = {
    # name : [EWECT-usual, EWECT-virus, Onlineshopping, MMS, XED]
    "MMAFFLM-3b":         [66.9, 60.3, 93.9, 43.3, 26.5],
    "MMAFFLM-7b":         [67.6, 58.2, 93.7, 43.0, 22.8],
    "SACF-Text (Ours)":   [62.36, 50.70, 92.82, 61.42, 16.41],
    "GPT-4o-mini":        [69.5, 57.6, 61.9, 48.6, 21.2],
    "EmoLlama-chat-7b":   [45.6, 30.5, 44.0, 48.6, 20.3],
}

# colour per model (consistent regardless of rank order)
COLORS = {
    "MMAFFLM-3b":         "#F59E0B",   # orange
    "MMAFFLM-7b":         "#1D4ED8",   # blue
    "SACF-Text (Ours)":   "#DC2626",   # red
    "GPT-4o-mini":        "#10B981",   # green
    "EmoLlama-chat-7b":   "#8B5CF6",   # purple
}


# Averages as printed in 表 4-2 (kept to match the thesis exactly).
# These match the per-task means up to rounding (one decimal in the table,
# except SACF-Text which is reported to two decimals).
TABLE_AVG = {
    "MMAFFLM-3b":         58.2,
    "MMAFFLM-7b":         57.1,
    "SACF-Text (Ours)":   56.74,
    "GPT-4o-mini":        51.8,
    "EmoLlama-chat-7b":   37.8,
}


def compute_averages():
    """Return list of (model, table-reported avg) sorted descending."""
    avgs = list(TABLE_AVG.items())
    avgs.sort(key=lambda x: -x[1])
    return avgs


def make_chart(avgs, outname):
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    names = [a[0] for a in avgs]
    vals  = [a[1] for a in avgs]
    cols  = [COLORS[n] for n in names]

    bars = ax.bar(range(len(names)), vals, color=cols, width=0.6,
                  edgecolor="white", linewidth=0)
    for bar, v in zip(bars, vals):
        # Match the table's printed precision: 2 decimals for SACF-Text,
        # 1 decimal otherwise.
        label = f"{v:.2f}" if abs(v - round(v, 1)) > 1e-6 else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.0,
                label, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold")

    ax.set_title(
        "Average macro-F1 — Five MMAFFBen Text-Only Tasks",
        fontsize=14, fontweight="bold", color="#1D4ED8", pad=14)
    ax.set_ylabel("Average macro-F1 across the five tasks (%)", fontsize=10.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10.5)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E7EB", alpha=0.6, lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(OUTDIR / f"{outname}.svg"), bbox_inches="tight")
    fig.savefig(str(OUTDIR / f"{outname}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def print_diff_table():
    """Print recalculated vs old-chart values for sanity-check."""
    old_chart = {
        "MMAFFLM-3b":        58.22,
        "SACF-Text (Ours)":  56.74,
        "MMAFFLM-7b":        55.44,
        "GPT-4o-mini":       51.96,
        "EmoLlama-chat-7b":  37.80,
    }
    print(f"\n{'Model':<22s}{'Recalc avg':>12s}{'Old chart':>12s}{'Δ':>9s}")
    print("-" * 55)
    for name, scores in PER_TASK.items():
        avg = sum(scores) / len(scores)
        chart = old_chart.get(name)
        if chart is None:
            continue
        diff = avg - chart
        print(f"{name:<22s}{avg:>12.2f}{chart:>12.2f}{diff:>+9.2f}")


if __name__ == "__main__":
    print("Recomputing averages from 表 4-2 per-task scores ...")
    avgs = compute_averages()
    print("\nRecalculated averages (sorted descending):")
    for name, v in avgs:
        print(f"  {name:<22s}{v:>6.2f}")
    print_diff_table()
    make_chart(avgs, "mmaffben_avg_macroF1")
    print(f"\n  ✓ Saved:  docs/figures/mmaffben_avg_macroF1.png")
    print(f"  ✓ Saved:  docs/figures/mmaffben_avg_macroF1.svg")
