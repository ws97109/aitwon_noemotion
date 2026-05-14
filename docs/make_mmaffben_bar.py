"""MMAFFBen 5-task bar chart — SACF-Text (this study) vs. baselines from Liu et al. 2025.

SACF-Text numbers come from a real evaluation of sacf_final.pt on the five
text-only MMAFFBen tasks (mmaffben_results.json).
All other model numbers come from Liu et al. 2025 (MMAFFBen paper).

Run:  python3 docs/make_mmaffben_bar.py
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)
PROJ = BASE.parent

# ── SACF-Text real numbers ───────────────────────────────────────────────────
results_path = PROJ / "emotion_system" / "training" / "mmaffin_exp" / "mmaffben_results.json"
with open(results_path) as f:
    data = json.load(f)
r = data["results"]
tasks = ["EWECT-usual", "EWECT-virus", "MMS", "XED", "Onlineshopping"]
sacf_text = [r[t]["test_metrics"]["ma-F1"] for t in tasks]
sacf_avg = float(np.mean(sacf_text))

# ── Baselines from Liu et al. 2025 (emollm follow-up paper / MMAFFBen) ──────
# Column order = EWECT-usual, EWECT-virus, MMS, XED, Onlineshopping
baselines = {
    "MMAFFLM-7b":       [67.6, 58.2, 79.3, 43.3, 28.8],
    "MMAFFLM-3b":       [66.9, 60.3, 93.9, 43.5, 26.5],
    "GPT-4o-mini":      [69.5, 57.6, 61.9, 48.6, 22.2],
    "EmoLlama-chat-7b": [45.6, 30.5, 44.0, 48.6, 20.3],
}

models = ["SACF-Text (Ours)"] + list(baselines.keys())
values = np.array([sacf_text] + [baselines[k] for k in baselines.keys()])
avgs = values.mean(axis=1)

# ── Colours ──────────────────────────────────────────────────────────────────
C = dict(
    sacf="#DC2626", primary="#1D4ED8", accent="#F59E0B",
    success="#10B981", purple="#8B5CF6", text="#1F2937",
    muted="#6B7280", grid="#E5E7EB",
)
model_colors = [C["sacf"], C["primary"], C["accent"], C["success"], C["purple"]]

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 160, "figure.dpi": 110,
    "axes.unicode_minus": False,
})


def make_bar(out_name):
    fig, ax = plt.subplots(figsize=(13.0, 6.4))

    n_tasks = len(tasks)
    n_models = len(models)
    x = np.arange(n_tasks)
    bar_w = 0.16

    for mi, (name, vals, col) in enumerate(zip(models, values, model_colors)):
        offset = (mi - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=col,
                      label=f"{name}  (avg {avgs[mi]:.2f})",
                      edgecolor="white", linewidth=0.6, zorder=3)
        # Value labels above each bar
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=C["text"], rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("macro-F1 (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)

    title = ("SACF-Text vs. Reference Models on the Five MMAFFBen Text-Only Tasks")
    ax.set_title(title, pad=18, fontweight="bold", color=C["primary"])
    ax.text(0.5, 1.025,
            "SACF-Text from our evaluation of the final checkpoint; "
            "all other model values from Liu et al. 2025 (MMAFFBen paper).",
            transform=ax.transAxes, ha="center", fontsize=9.5,
            color=C["muted"], style="italic")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=5, frameon=False, fontsize=9.5)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out_png = OUTDIR / f"{out_name}.png"
    out_svg = OUTDIR / f"{out_name}.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok  {out_png}")
    print(f"  ok  {out_svg}")


def make_avg_bar(out_name):
    """A second figure: average ma-F1 across the five tasks per model."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    order = np.argsort(-avgs)
    sorted_models = [models[i] for i in order]
    sorted_avgs = avgs[order]
    sorted_colors = [model_colors[i] for i in order]

    bars = ax.bar(sorted_models, sorted_avgs, color=sorted_colors,
                  edgecolor="white", linewidth=1.0, zorder=3)
    for b, v in zip(bars, sorted_avgs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10.5,
                color=C["text"], fontweight="bold")

    ax.set_ylabel("Average macro-F1 across the five tasks (%)")
    ax.set_ylim(0, max(sorted_avgs) + 8)
    ax.grid(axis="y", color=C["grid"], alpha=0.6, zorder=0)
    ax.set_title("Average macro-F1 — Five MMAFFBen Text-Only Tasks",
                 pad=12, fontweight="bold", color=C["primary"])

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    out_png = OUTDIR / f"{out_name}.png"
    out_svg = OUTDIR / f"{out_name}.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok  {out_png}")
    print(f"  ok  {out_svg}")


if __name__ == "__main__":
    print("SACF-Text real numbers:", dict(zip(tasks, [f"{v:.2f}" for v in sacf_text])),
          f"avg = {sacf_avg:.2f}")
    make_bar("fig_mmaffben_text_bar")
    make_avg_bar("fig_mmaffben_text_avg")
    print("Done.")
