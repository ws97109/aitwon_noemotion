"""CMU-MOSI comparison bar charts — SACF (this study) vs. published baselines.

SACF numbers come from a real evaluation of sacf_final.pt on CMU-MOSI test set.
All other model numbers come from previously published results
(values reproduced from the Chapter 4 comparison table).

Run:  python3 docs/make_mosi_bar.py
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path(__file__).parent
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── CMU-MOSI scores: [Acc-7, Acc-2, F1, MAE, Corr] ──────────────────────────
#   SACF row is from this study's evaluation of sacf_final.pt (3-Run merged).
#   All other rows come from previously published values (Table 3 in thesis).
#   Acc-7 = None  → model has no reported Acc-7 (e.g. MCL-MCF) and is excluded
#                   from both the Acc-7 ranking and the Top-8 multi-metric chart.
data = [
    ("SACF (Ours)",  53.06, 85.42, 85.41, 0.5840, 0.8691, True),
    ("MGT",          50.44, 86.30, 86.26, 0.659,  0.822,  False),
    ("UniMSE",       48.68, 85.85, 85.83, 0.691,  0.809,  False),
    ("CLGSI",        47.96, 83.97, 83.63, 0.703,  0.790,  False),
    ("ITHP",         47.70, 86.10, 86.10, 0.663,  0.856,  False),
    ("DMD",          46.40, 84.20, 84.10, 0.709,  0.796,  False),
    ("Self-MM",      45.80, 82.70, 82.60, 0.731,  0.785,  False),
    ("MMIM",         45.00, 83.00, 82.90, 0.738,  0.781,  False),
    ("MCL-MCF",      None,  84.90, 84.70, 0.692,  0.799,  False),
    ("MISA",         43.50, 81.80, 81.70, 0.752,  0.784,  False),
    ("ConFEDE",      42.27, 84.17, 84.13, 0.742,  0.782,  False),
    ("Graph-MFN",    34.40, 77.90, 77.80, 0.939,  0.656,  False),
    ("LF-DNN",       33.60, 78.00, 77.90, 0.978,  0.658,  False),
    ("MFM",          33.30, 77.70, 77.70, 0.948,  0.664,  False),
]

# Filter out rows whose Acc-7 is missing — used by both figures, which depend on Acc-7.
_data_with_acc7 = [d for d in data if d[1] is not None]
names = [d[0] for d in _data_with_acc7]
acc7  = np.array([d[1] for d in _data_with_acc7])
acc2  = np.array([d[2] for d in _data_with_acc7])
f1    = np.array([d[3] for d in _data_with_acc7])
mae   = np.array([d[4] for d in _data_with_acc7])
corr  = np.array([d[5] for d in _data_with_acc7])
is_ours = np.array([d[6] for d in _data_with_acc7])

C = dict(
    ours="#DC2626", ours_f1="#F87171", primary="#1D4ED8", accent="#F59E0B",
    success="#10B981", purple="#8B5CF6", text="#1F2937",
    muted="#6B7280", grid="#E5E7EB", rose="#F43F5E",
    secondary="#0891B2",
)

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 160, "figure.dpi": 110,
    "axes.unicode_minus": False,
})


def _sort(metric, lower_is_better=False):
    """Return descending order indices (or ascending if lower is better)."""
    order = np.argsort(metric)
    if not lower_is_better:
        order = order[::-1]
    return order


def fig_acc7_ranking():
    """Single-metric ranking of all models by Acc-7."""
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    order = _sort(acc7)
    nms = [names[i] for i in order]
    vals = acc7[order]
    cols = [C["ours"] if is_ours[i] else C["primary"] for i in order]

    bars = ax.bar(nms, vals, color=cols, edgecolor="white", lw=1.0, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=C["text"])

    ax.axhline(50.44, color=C["accent"], lw=1.4, ls="--",
               label="MGT baseline = 50.44%", zorder=2)
    ax.axhline(53.0, color=C["success"], lw=1.4, ls=":",
               label="research target = 53%", zorder=2)

    ax.set_ylabel("Acc-7 (%)   —   7-class sentiment intensity accuracy")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    ax.set_title("CMU-MOSI — 7-Class Sentiment Accuracy Across Models",
                 pad=12, fontweight="bold", color=C["primary"])
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_mosi_acc7_ranking.png", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig_mosi_acc7_ranking.svg", bbox_inches="tight")
    plt.close(fig)
    print("  ok  fig_mosi_acc7_ranking")


def fig_multi_metric():
    """Multi-panel comparison: Acc-7, Acc-2/F1, MAE, Corr for top 8 models."""
    # Take top 8 by Acc-7
    order = _sort(acc7)[:8]
    nms = [names[i] for i in order]
    a7  = acc7[order]; a2 = acc2[order]; f = f1[order]
    me = mae[order]; co = corr[order]
    cols = [C["ours"] if is_ours[i] else C["primary"] for i in order]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))

    # (a) Acc-7
    ax = axes[0, 0]
    bars = ax.bar(nms, a7, color=cols, edgecolor="white", lw=0.8, zorder=3)
    for b, v in zip(bars, a7):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Acc-7 (%)")
    ax.set_ylim(0, 60)
    ax.set_title("(a)  7-class accuracy — higher is better",
                 fontweight="bold", color=C["primary"])
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right", fontsize=9)

    # (b) Acc-2 and F1 (grouped)
    ax = axes[0, 1]
    x = np.arange(len(nms))
    bw = 0.35
    bars_a = ax.bar(x - bw/2, a2, bw, color=C["primary"], label="Acc-2",
                    edgecolor="white", lw=0.6, zorder=3)
    bars_f = ax.bar(x + bw/2, f, bw, color=C["secondary"], label="F1",
                    edgecolor="white", lw=0.6, zorder=3)
    for b, v in zip(bars_a, a2):
        ax.text(b.get_x()+b.get_width()/2, v+1.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    for b, v in zip(bars_f, f):
        ax.text(b.get_x()+b.get_width()/2, v+1.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    # Mark "Ours" — two shades of red so SACF's Acc-2 and F1 stay distinguishable
    for i, ours in enumerate(is_ours[order]):
        if ours:
            bars_a[i].set_color(C["ours"])      # Acc-2 → darker red
            bars_f[i].set_color(C["ours_f1"])   # F1    → lighter red
    ax.set_xticks(x); ax.set_xticklabels(nms, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title("(b)  Binary accuracy and F1 — higher is better",
                 fontweight="bold", color=C["primary"])
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    # Explicit legend handles so the swatches show the true bar colours.
    # (SACF's bars are recoloured red above, which would otherwise make the
    #  auto-legend show both Acc-2 and F1 as red.)
    legend_handles = [
        mpatches.Patch(color=C["primary"], label="Acc-2"),
        mpatches.Patch(color=C["secondary"], label="F1"),
        mpatches.Patch(color=C["ours"], label="SACF Acc-2"),
        mpatches.Patch(color=C["ours_f1"], label="SACF F1"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.0, framealpha=0.95,
              loc="upper center", ncol=4, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.3)

    # (c) MAE (lower is better)
    ax = axes[1, 0]
    bars = ax.bar(nms, me, color=cols, edgecolor="white", lw=0.8, zorder=3)
    for b, v in zip(bars, me):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Mean absolute error")
    ax.set_ylim(0, 0.85)
    ax.set_title("(c)  Mean absolute error — lower is better",
                 fontweight="bold", color=C["primary"])
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right", fontsize=9)

    # (d) Corr
    ax = axes[1, 1]
    bars = ax.bar(nms, co, color=cols, edgecolor="white", lw=0.8, zorder=3)
    for b, v in zip(bars, co):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Pearson correlation")
    ax.set_ylim(0, 1.0)
    ax.set_title("(d)  Pearson correlation — higher is better",
                 fontweight="bold", color=C["primary"])
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right", fontsize=9)

    fig.suptitle("CMU-MOSI Multi-Metric Comparison — Top Eight Models",
                 fontsize=14, fontweight="bold", y=1.00, color=C["primary"])
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTDIR / "fig_mosi_multi_metric.png", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig_mosi_multi_metric.svg", bbox_inches="tight")
    plt.close(fig)
    print("  ok  fig_mosi_multi_metric")


if __name__ == "__main__":
    fig_acc7_ranking()
    fig_multi_metric()
    print("Done.")
