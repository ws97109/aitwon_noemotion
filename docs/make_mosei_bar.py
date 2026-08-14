"""CMU-MOSEI sentiment comparison bar charts.

SACF-Text numbers come from the real evaluation in
emotion_system/training/mmaffin_exp/mosei_text_results.json.
All other model numbers are widely reported baselines on the standard
CMU-MOSEI sentiment evaluation protocol (Acc-7 = 7-class rounded sentiment).

Run:  python3 docs/make_mosei_bar.py
"""
import warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path(__file__).parent
OUTDIR = BASE / "figures"
PROJ = BASE.parent

# ── SACF-Text real number (from our run) ────────────────────────────────────
results_path = PROJ / "emotion_system" / "training" / "mmaffin_exp" / "mosei_text_results.json"
sacf_metrics = None
if results_path.exists():
    with open(results_path) as f:
        data = json.load(f)
    sacf_metrics = data["metrics"]
    print(f"SACF-Text on CMU-MOSEI: {sacf_metrics}")
else:
    print(f"WARNING: {results_path} not found, using placeholder zeros")
    sacf_metrics = {"Acc-7": 0, "Acc-2": 0, "F1": 0, "MAE": 0, "Corr": 0}

# ── Standard CMU-MOSEI baselines from literature (sentiment task) ───────────
# Numbers are widely-cited; sources documented in thesis Ch4.
BASELINES = [
    # name,           Acc-7,  Acc-2,  F1,     MAE,    Corr
    ("LF-DNN",        47.2,   78.2,   78.0,   0.665,  0.673),
    ("Graph-MFN",     45.0,   76.9,   77.0,   0.710,  0.540),
    ("MFM",           46.2,   78.6,   78.4,   0.625,  0.658),
    ("MFN",           48.0,   76.0,   76.0,   0.610,  0.610),
    ("TFN",           49.8,   80.7,   80.1,   0.593,  0.700),
    ("LMF",           48.0,   80.6,   81.0,   0.623,  0.677),
    ("MulT",          51.8,   82.5,   82.3,   0.580,  0.703),
    ("MISA",          52.2,   85.5,   85.3,   0.555,  0.756),
    ("Self-MM",       53.5,   85.2,   85.3,   0.530,  0.765),
    ("MMIM",          54.2,   85.97,  85.94,  0.526,  0.772),
    ("DMD",           54.6,   86.0,   86.0,   0.516,  0.781),
    ("UniMSE",        54.4,   87.5,   87.4,   0.523,  0.773),
    ("MGT",           55.1,   87.0,   87.0,   0.502,  0.804),
]

# Build (name, metrics, is_ours)
data = [
    ("SACF-Text (Ours)", sacf_metrics["Acc-7"], sacf_metrics["Acc-2"],
     sacf_metrics["F1"], sacf_metrics["MAE"], sacf_metrics["Corr"], True),
]
for name, a7, a2, f1, mae, corr in BASELINES:
    data.append((name, a7, a2, f1, mae, corr, False))

names = [d[0] for d in data]
acc7  = np.array([d[1] for d in data])
acc2  = np.array([d[2] for d in data])
f1    = np.array([d[3] for d in data])
mae   = np.array([d[4] for d in data])
corr  = np.array([d[5] for d in data])
is_ours = np.array([d[6] for d in data])

C = dict(
    ours="#DC2626", ours_f1="#F87171", primary="#1D4ED8", accent="#F59E0B",
    success="#10B981", text="#1F2937", muted="#6B7280", grid="#E5E7EB",
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


def fig_acc7_ranking():
    """Acc-7 ranking on CMU-MOSEI."""
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    order = np.argsort(-acc7)
    nms = [names[i] for i in order]
    vals = acc7[order]
    cols = [C["ours"] if is_ours[i] else C["primary"] for i in order]

    bars = ax.bar(nms, vals, color=cols, edgecolor="white", lw=1.0, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=C["text"])

    ax.set_ylabel("Acc-7 (%)   —   7-class sentiment intensity accuracy")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    ax.set_title("CMU-MOSEI — 7-Class Sentiment Accuracy Across Models",
                 pad=12, fontweight="bold", color=C["primary"])
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_mosei_acc7_ranking.png", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig_mosei_acc7_ranking.svg", bbox_inches="tight")
    plt.close(fig)
    print("  ok  fig_mosei_acc7_ranking")


def fig_multi_metric():
    """Top-N multi-metric comparison on CMU-MOSEI."""
    order = np.argsort(-acc7)[:8]
    nms = [names[i] for i in order]
    a7  = acc7[order]; a2 = acc2[order]; f = f1[order]
    me = mae[order]; co = corr[order]
    cols = [C["ours"] if is_ours[i] else C["primary"] for i in order]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))

    ax = axes[0, 0]
    bars = ax.bar(nms, a7, color=cols, edgecolor="white", lw=0.8, zorder=3)
    for b, v in zip(bars, a7):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Acc-7 (%)")
    ax.set_ylim(0, 60)
    ax.set_title("(a)  7-class accuracy — higher is better",
                 fontweight="bold", color=C["primary"])
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right", fontsize=9)

    ax = axes[0, 1]
    x = np.arange(len(nms))
    bw = 0.35
    bars_a = ax.bar(x - bw/2, a2, bw, color=C["primary"], label="Acc-2",
                    edgecolor="white", lw=0.6, zorder=3)
    bars_f = ax.bar(x + bw/2, f, bw, color=C["secondary"], label="F1",
                    edgecolor="white", lw=0.6, zorder=3)
    # Mark "Ours" — two shades of red so SACF's Acc-2 and F1 stay distinguishable
    for i, ours in enumerate(is_ours[order]):
        if ours:
            bars_a[i].set_color(C["ours"])      # Acc-2 → darker red
            bars_f[i].set_color(C["ours_f1"])   # F1    → lighter red
    for b, v in zip(bars_a, a2):
        ax.text(b.get_x()+b.get_width()/2, v+1.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    for b, v in zip(bars_f, f):
        ax.text(b.get_x()+b.get_width()/2, v+1.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(nms, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title("(b)  Binary accuracy and F1 — higher is better",
                 fontweight="bold", color=C["primary"])
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    # Explicit legend handles so the swatches show the true bar colours
    # (SACF's bars are recoloured red, which would otherwise confuse an auto-legend).
    legend_handles = [
        mpatches.Patch(color=C["primary"], label="Acc-2"),
        mpatches.Patch(color=C["secondary"], label="F1"),
        mpatches.Patch(color=C["ours"], label="SACF Acc-2"),
        mpatches.Patch(color=C["ours_f1"], label="SACF F1"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.0, framealpha=0.95,
              loc="upper center", ncol=4, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.3)

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

    fig.suptitle("CMU-MOSEI Multi-Metric Comparison — Top Eight Models",
                 fontsize=14, fontweight="bold", y=1.00, color=C["primary"])
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTDIR / "fig_mosei_multi_metric.png", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig_mosei_multi_metric.svg", bbox_inches="tight")
    plt.close(fig)
    print("  ok  fig_mosei_multi_metric")


if __name__ == "__main__":
    fig_acc7_ranking()
    fig_multi_metric()
    print("Done.")
