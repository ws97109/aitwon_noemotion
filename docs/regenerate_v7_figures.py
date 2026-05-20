"""SACF v7 figures — aligned with sacf_final.pt (Hierarchical SACF + 2-stage + 3-Run).

Updates relative to v6:
  • Per-branch Acc-7 corrected to 49.56 / 50.00 / 50.29 / 49.71  (internal mean 50.00%).
  • Final Acc-7 fused to 53.06% (already correct in npz; carried through).
  • Training timeline redrawn as TWO STAGES — Stage 1 (60 ep base) + Stage 2
    (20 ep cross-modal contrastive fine-tune) — and three INDEPENDENT RUNS
    (Run 1 seed=42, Run 2 seed=42 with extended Stage 2, Run 3 seed=5678 + BAN).
  • Inference figure: the 3-Run ensemble is performed in PARAMETER SPACE *at
    training time*, so inference uses ONE forward only. Stages relabelled.
  • Architecture figure: bottom label says "3-Run Parameter-Space Average".
  • All boxes sized to prevent overflow; no underscore-as-subscript in displayed
    text; all labels in plain English natural language.

Run:  python3 docs/regenerate_v7_figures.py
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).parent
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

DATA = np.load(str(BASE / "paper_v2_data.npz"))
m = DATA["metrics"]
ACC7_FUSED, ACC7_RAW, ACC2, F1, MAE, CORR = [float(x) for x in m]
W1 = 91.55
TRAIN_DIST = DATA["train_dist"]; TEST_DIST = DATA["test_dist"]
TRAINVAL_DIST = DATA["trainval_dist"]

C = dict(
    primary="#1D4ED8", secondary="#0891B2", accent="#F59E0B",
    danger="#DC2626", success="#10B981", purple="#8B5CF6",
    text="#1F2937", muted="#6B7280", grid="#E5E7EB", bg="#F9FAFB",
    teal="#14B8A6", indigo="#6366F1", rose="#F43F5E", lime="#84CC16",
    violet="#7C3AED",
)
plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.alpha": 0.5,
    "savefig.dpi": 150, "figure.dpi": 110,
    "axes.unicode_minus": False,
})


def save(fig, name):
    fig.savefig(str(OUTDIR / f"{name}.svg"), bbox_inches="tight")
    fig.savefig(str(OUTDIR / f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def rbox(ax, cx, cy, w, h, fc, text, fs=9, tc="white", ec=None,
         bold=False, lw=1.0, alpha=1.0, rad=0.06):
    if ec is None: ec = fc
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle=f"round,pad=0.005,rounding_size={rad}",
                          fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=4)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, zorder=5)


def arr(ax, x1, y1, x2, y2, color=None, lw=1.4, hw=0.18, z=3):
    if color is None: color = C["text"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f"->,head_width={hw},head_length={hw*1.4}",
                                 color=color, lw=lw), zorder=z)


def bg(ax, x0, y0, w, h, fc, ec, lw=1.2, alpha=0.18):
    r = FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.01,rounding_size=0.18",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=1)
    ax.add_patch(r)


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Overall architecture (conceptual)
# ════════════════════════════════════════════════════════════════════════════
def fig_arch():
    fig, ax = plt.subplots(figsize=(14.0, 13.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 15.8); ax.axis("off")

    ax.text(7, 15.30, "Hierarchical SACF — Multi-Branch Single Model",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["primary"])
    ax.text(7, 14.85,
            f"Acc-7 = {ACC7_FUSED:.2f}%   |   Acc-2 = {ACC2:.2f}%   |   "
            f"F1 = {F1:.2f}%   |   MAE = {MAE:.4f}   |   Corr = {CORR:.4f}",
            ha="center", va="center", fontsize=10.5, color=C["muted"])

    # Row 1: Inputs
    bg(ax, 0.4, 13.30, 13.2, 1.2, "#FEF3C7", C["accent"])
    ax.text(0.7, 14.30, "(i)  Multimodal Inputs",
            ha="left", va="center", fontsize=11,
            color=C["accent"], fontweight="bold")
    rbox(ax, 2.5, 13.70, 3.6, 0.55, C["primary"],
         "Text  (transcript + prompt)", fs=9.5, bold=True)
    rbox(ax, 7.0, 13.70, 3.6, 0.55, C["accent"],
         "Audio  (acoustic features, 5-dim)", fs=9.5, bold=True)
    rbox(ax, 11.5, 13.70, 3.6, 0.55, C["success"],
         "Vision  (facial features, 20-dim)", fs=9.5, bold=True)

    # Row 2: Shared encoders
    bg(ax, 0.4, 11.40, 13.2, 1.25, "#DBEAFE", C["primary"])
    ax.text(0.7, 12.50, "(ii)  Shared Encoders",
            ha="left", va="center", fontsize=11,
            color=C["primary"], fontweight="bold")
    rbox(ax, 2.5, 11.90, 3.6, 0.8, C["primary"],
         "DeBERTa-v3-large\n(24-layer Transformer)", fs=9, bold=True)
    rbox(ax, 7.0, 11.90, 3.6, 0.8, C["accent"],
         "Bidirectional LSTM\nfor audio sequence", fs=9, bold=True)
    rbox(ax, 11.5, 11.90, 3.6, 0.8, C["success"],
         "Bidirectional LSTM\nfor vision sequence", fs=9, bold=True)
    for cx in (2.5, 7.0, 11.5):
        arr(ax, cx, 13.20, cx, 12.75)

    # Row 3: 4 branches (Hierarchical SACF)
    branch_top = 10.85
    branch_bottom = 6.85
    bg(ax, 0.4, branch_bottom, 13.2, branch_top - branch_bottom - 0.15, "#EDE9FE", C["purple"])
    ax.text(0.7, 10.50,
            "(iii)  Hierarchical SACF",
            ha="left", va="center", fontsize=10.5,
            color=C["purple"], fontweight="bold")

    for cx in (2.0, 5.0, 8.0, 11.0):
        arr(ax, 2.5, 11.30, cx, 10.85, color=C["primary"], lw=0.9, hw=0.14)
        arr(ax, 7.0, 11.30, cx, 10.85, color=C["accent"], lw=0.9, hw=0.14)
        arr(ax, 11.5, 11.30, cx, 10.85, color=C["success"], lw=0.9, hw=0.14)

    branch_colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    branch_dp = [0.10, 0.20, 0.30, 0.40]
    for i in range(4):
        cx = 2.0 + i * 3.0
        bg(ax, cx - 1.30, 7.00, 2.6, 3.10, "#FFFFFF", branch_colors[i], lw=1.6, alpha=0.6)
        ax.text(cx, 9.85, f"Branch {i+1}  (drop = {branch_dp[i]})",
                ha="center", fontsize=9.5, fontweight="bold", color=branch_colors[i])
        rbox(ax, cx, 9.25, 2.4, 0.5, branch_colors[i],
             "Polarity-Enhanced\nAttention", fs=8, bold=True)
        rbox(ax, cx, 8.60, 2.4, 0.5, branch_colors[i],
             "Hierarchical SACF\n(two-stage fusion)", fs=8, bold=True)
        rbox(ax, cx, 7.95, 2.4, 0.5, branch_colors[i],
             "Shared Projection", fs=8.5, bold=True)
        rbox(ax, cx, 7.30, 2.4, 0.55, "#FFFFFF",
             "Class-7 / Class-2 / Regression",
             fs=8.0, tc=branch_colors[i], ec=branch_colors[i], bold=True)

    # Row 4: Mean of branches
    bg(ax, 3.5, 5.50, 7.0, 0.95, "#FFE4E6", C["rose"])
    rbox(ax, 7.0, 5.95, 6.6, 0.65, C["rose"],
         "Internal Mean — average of the 4 branch outputs",
         fs=10, bold=True)
    for i in range(4):
        cx = 2.0 + i * 3.0
        arr(ax, cx, 6.55, 7.0, 6.48, color=branch_colors[i], lw=0.8, hw=0.10)

    # Row 5: Inference
    bg(ax, 0.4, 2.85, 13.2, 2.2, "#DCFCE7", C["success"])
    ax.text(0.7, 4.90,
            "(iv)  Zero-Leakage Inference",
            ha="left", va="center", fontsize=11,
            color=C["success"], fontweight="bold")
    rbox(ax, 3.0, 4.15, 3.0, 0.7, C["primary"],
         "Test-time augmentation\n(five stochastic passes)", fs=8.5, bold=True)
    rbox(ax, 7.0, 4.15, 3.0, 0.7, C["accent"],
         "Three-run weights\nalready merged in checkpoint", fs=8.5, bold=True)
    rbox(ax, 11.0, 4.15, 3.0, 0.7, C["success"],
         "Classification + Regression\nprobability fusion",
         fs=8.5, bold=True)
    arr(ax, 7.0, 5.30, 7.0, 4.80)
    arr(ax, 4.70, 4.15, 5.30, 4.15, lw=1.4, hw=0.10)
    arr(ax, 8.70, 4.15, 9.30, 4.15, lw=1.4, hw=0.10)
    arr(ax, 7.0, 3.55, 7.0, 2.95)

    rbox(ax, 7.0, 2.45, 5.6, 0.75, C["danger"],
         f"Final prediction              Acc-7 = {ACC7_FUSED:.2f}%",
         fs=11, bold=True)

    handles = [
        mpatches.Patch(color=C["primary"], label="Text"),
        mpatches.Patch(color=C["accent"], label="Audio"),
        mpatches.Patch(color=C["success"], label="Vision"),
        mpatches.Patch(color=C["rose"], label="Aggregation"),
        mpatches.Patch(color=C["purple"], label="Parallel Branches"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.07),
              ncol=5, fontsize=9, frameon=False)

    save(fig, "v7_fig_arch")


# ════════════════════════════════════════════════════════════════════════════
# FIG PEA
# ════════════════════════════════════════════════════════════════════════════
def fig_pea():
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.8); ax.axis("off")
    ax.text(6.75, 6.40, "Polarity-Enhanced Attention (PEA) Module",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(6.75, 5.95,
            "Learn how much each text token contributes to the sentiment, "
            "then pool token representations weighted by that contribution",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Step 1 — DeBERTa output
    bg(ax, 0.3, 3.0, 3.0, 2.6, "#DBEAFE", C["primary"])
    ax.text(1.8, 5.30, "DeBERTa Output", ha="center", fontsize=10,
            fontweight="bold", color=C["primary"])
    ax.text(1.8, 4.95, "(token representations)",
            ha="center", fontsize=8, color=C["muted"], style="italic")
    tokens = ["[CLS]", "good", "movie", "!", "[PAD]"]
    for ti, tok in enumerate(tokens):
        ty = 4.55 - ti * 0.36
        rbox(ax, 1.8, ty, 2.4, 0.30, C["primary"],
             f"Token {ti}   ({tok})", fs=8.5, bold=True)

    # Step 2 — Gate MLP
    bg(ax, 3.7, 3.2, 3.4, 2.2, "#FEF3C7", C["accent"])
    ax.text(5.4, 5.10, "Salience Scorer", ha="center", fontsize=10,
            fontweight="bold", color=C["accent"])
    rbox(ax, 5.4, 4.45, 3.0, 0.50, C["accent"],
         "First linear layer + tanh", fs=9, bold=True)
    rbox(ax, 5.4, 3.85, 3.0, 0.50, C["accent"],
         "Second linear layer + sigmoid", fs=9, bold=True)
    ax.text(5.4, 3.45, "(produces a score between 0 and 1 per token)",
            ha="center", fontsize=8, color=C["muted"], style="italic")
    arr(ax, 3.30, 4.30, 3.75, 4.30, lw=1.4)

    # Step 3 — Gate values
    bg(ax, 7.5, 3.0, 2.3, 2.6, "#FFE4E6", C["rose"])
    ax.text(8.65, 5.30, "Salience Scores",
            ha="center", fontsize=10, fontweight="bold", color=C["rose"])
    ax.text(8.65, 4.95, "(higher = more sentiment)",
            ha="center", fontsize=8, color=C["muted"], style="italic")
    pol_vals = [0.10, 0.92, 0.30, 0.50, 0.00]
    for ti, (tok, pol) in enumerate(zip(tokens, pol_vals)):
        ty = 4.55 - ti * 0.36
        rbox(ax, 8.65, ty, 2.0, 0.30, C["rose"],
             f"Token {ti}:  {pol:.2f}", fs=8.5, bold=True)
    arr(ax, 7.10, 4.30, 7.55, 4.30, lw=1.4)

    # Step 4 — Weighted pooling
    bg(ax, 10.1, 3.0, 3.1, 2.6, "#DCFCE7", C["success"])
    ax.text(11.65, 5.30, "Weighted Pooling",
            ha="center", fontsize=10, fontweight="bold", color=C["success"])
    ax.text(11.65, 4.65,
            "Average all token\nrepresentations weighted\nby their salience scores",
            ha="center", va="center", fontsize=9, fontweight="bold")
    rbox(ax, 11.65, 3.60, 2.6, 0.55, C["success"],
         "Sentiment-aware\nlanguage vector",
         fs=9, bold=True)
    arr(ax, 9.85, 4.30, 10.15, 4.30, lw=1.4)

    # Bottom: PEA outputs
    bg(ax, 0.3, 0.4, 12.9, 2.0, "#EDE9FE", C["purple"])
    ax.text(6.75, 2.10, "Where the PEA Outputs Go",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    rbox(ax, 3.4, 1.35, 6.0, 0.55, C["purple"],
         "Sentiment-aware vector  →  shared projection of each branch",
         fs=9, bold=True)
    rbox(ax, 10.0, 1.35, 5.6, 0.55, C["purple"],
         "Salience scores  →  Top-K tokens for SACF",
         fs=9, bold=True)
    arr(ax, 11.65, 3.30, 11.65, 1.65, color=C["purple"], hw=0.12)

    save(fig, "v7_fig_pea")


# ════════════════════════════════════════════════════════════════════════════
# FIG SACF steps — now Hierarchical SACF (two-stage)
# ════════════════════════════════════════════════════════════════════════════
def fig_sacf_steps():
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(7, 6.60, "Hierarchical Sentiment-Aware Cross-modal Fusion — Four Steps",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(7, 6.15,
            "Build a sentiment-focused query from the most salient tokens, "
            "then bring in audio and vision evidence",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    step_colors = [C["primary"], C["violet"], C["accent"], C["success"]]
    step_titles = [
        "Step 1\nSelect Salient Tokens",
        "Step 2\nBuild Sentiment Query",
        "Step 3\nAlign Audio and Vision",
        "Step 4\nGated Residual Fusion",
    ]
    step_bodies = [
        ("Pick the top-K tokens\nwith the highest PEA\nsalience scores.\n\n"
         "These are the words most\nlikely to carry sentiment\n(opinion adjectives,\nemotional verbs, etc.)."),
        ("Combine the K selected\ntoken representations\ninto a single vector\nvia attention-weighted\naveraging.\n\n"
         "Result: a sentiment-\nfocused query vector."),
        ("Project audio and vision\nfeatures into the\nlanguage embedding space\nas keys and values.\n\n"
         "Use scaled dot-product\nattention against the\nsentiment query to pull\nin cross-modal evidence."),
        ("Combine cross-modal\nevidence with the original\nsentence representation\nvia a two-layer\nfeed-forward network\nand a sigmoid gate.\n\n"
         "Add a residual link and\nlayer normalization for\ntraining stability."),
    ]
    step_outputs = [
        "Top-K token tensor",
        "Sentiment query vector",
        "Cross-modal context",
        "Fused vector",
    ]

    box_w = 3.20
    box_h = 4.80
    spacing = 0.20
    start_x = 0.30

    for i, (title, body, out, col) in enumerate(
            zip(step_titles, step_bodies, step_outputs, step_colors)):
        cx = start_x + i * (box_w + spacing) + box_w / 2
        bg(ax, cx - box_w/2, 0.4, box_w, box_h, "#FFFFFF", col, lw=1.6, alpha=0.85)
        rbox(ax, cx, 0.4 + box_h - 0.55, box_w - 0.25, 0.85, col,
             title, fs=10.5, tc="white", bold=True)
        ax.text(cx, 0.4 + box_h/2 - 0.20, body,
                ha="center", va="center", fontsize=9, color=C["text"])
        rbox(ax, cx, 0.4 + 0.40, box_w - 0.25, 0.55, col,
             out, fs=9.5, tc="white", bold=True)
        if i < 3:
            arr(ax, cx + box_w/2 - 0.08, 2.80,
                start_x + (i+1)*(box_w + spacing) + 0.08, 2.80,
                color=col, lw=2.0, hw=0.18)

    save(fig, "v7_fig_sacf_steps")


# ════════════════════════════════════════════════════════════════════════════
# FIG branches — corrected per-branch Acc-7 numbers
# ════════════════════════════════════════════════════════════════════════════
def fig_branches():
    fig, ax = plt.subplots(figsize=(14.0, 7.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.8); ax.axis("off")
    ax.text(7, 7.40, "Four Parallel Branches — Sources of Diversity",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(7, 6.95,
            "Three mechanisms keep the branches diverse so that their average "
            "behaves like a multi-model ensemble",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    bg(ax, 0.3, 4.4, 13.4, 2.0, "#FEF3C7", C["accent"])
    ax.text(7, 6.20, "Three Diversity Mechanisms",
            ha="center", fontsize=10.5, fontweight="bold", color=C["accent"])
    rbox(ax, 2.5, 5.55, 4.0, 0.50, C["accent"],
         "(a) Different dropout rates\nfor each branch",
         fs=9, bold=True)
    rbox(ax, 7.0, 5.55, 4.0, 0.50, C["accent"],
         "(b) Independent attention\nand projection weights",
         fs=9, bold=True)
    rbox(ax, 11.5, 5.55, 4.0, 0.50, C["accent"],
         "(c) Small Gaussian noise on\nthe 7-class head init",
         fs=9, bold=True)
    rbox(ax, 7.0, 4.75, 12.6, 0.55, C["accent"],
         "At inference the branches differ deterministically by weight (dropout is turned off)",
         fs=9.5, bold=True)

    branch_colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    branch_dp = [0.10, 0.20, 0.30, 0.40]
    per_branch_acc7 = [49.56, 50.00, 50.29, 49.71]
    bw = 2.9
    for i in range(4):
        cx = 1.05 + i * 3.0 + bw/2
        bg(ax, cx - bw/2, 1.30, bw, 2.85, "#FFFFFF", branch_colors[i], lw=1.6, alpha=0.65)
        ax.text(cx, 3.95, f"Branch {i+1}",
                ha="center", fontsize=10.5, fontweight="bold",
                color=branch_colors[i])
        ax.text(cx, 3.60, f"dropout = {branch_dp[i]}",
                ha="center", fontsize=8.5, color=C["text"])
        rbox(ax, cx, 3.05, bw - 0.4, 0.42, branch_colors[i],
             "Attention → Hier. SACF → Proj", fs=8.5, bold=True)
        rbox(ax, cx, 2.45, bw - 0.4, 0.42, "#FFFFFF",
             "Projected feature",
             fs=9, tc=branch_colors[i], ec=branch_colors[i], bold=True)
        rbox(ax, cx, 1.85, bw - 0.4, 0.42, branch_colors[i],
             "Class-7 / Class-2 / Regression",
             fs=8.5, bold=True)
        ax.text(cx, 1.45,
                f"single-branch Acc-7 ≈ {per_branch_acc7[i]:.2f}%",
                ha="center", fontsize=8.5,
                color=branch_colors[i], fontweight="bold")

    bg(ax, 1.0, 0.10, 12.0, 0.95, "#FFE4E6", C["rose"])
    rbox(ax, 7.0, 0.55, 11.5, 0.55, C["rose"],
         "Internal Mean — average of 4 branches ≈ 50.00%   |   "
         f"After 3-Run ensemble and Reg-Cls fusion: Acc-7 = {ACC7_FUSED:.2f}%",
         fs=9.0, bold=True)
    for i in range(4):
        cx = 1.05 + i * 3.0 + bw/2
        arr(ax, cx, 1.35, 7.0, 1.10, color=branch_colors[i], lw=0.8, hw=0.10)

    save(fig, "v7_fig_branches")


# ════════════════════════════════════════════════════════════════════════════
# FIG loss composition — emphasise CMC as Stage-2 contrastive term
# ════════════════════════════════════════════════════════════════════════════
def fig_loss_comp():
    fig, ax = plt.subplots(figsize=(14.0, 8.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.4); ax.axis("off")
    ax.text(7, 8.05, "Multi-Task Loss — Composition Structure",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(7, 7.65,
            "Three conceptual layers:  per-branch task losses  →  cross-branch "
            "aggregation and contrast  →  consistency regularization and total",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Layer 1
    bg(ax, 0.4, 5.0, 13.2, 2.2, "#DBEAFE", C["primary"])
    ax.text(7, 6.95, "Layer 1 — Per-branch Task Losses",
            ha="center", fontsize=11, fontweight="bold", color=C["primary"])
    rbox(ax, 2.0, 6.30, 3.2, 0.55, C["primary"],
         "Soft-Ordinal\nCross-Entropy", fs=9, bold=True)
    rbox(ax, 5.7, 6.30, 3.2, 0.55, C["secondary"],
         "Ordinal Earth-Mover's\nDistance", fs=9, bold=True)
    rbox(ax, 9.3, 6.30, 2.6, 0.55, C["accent"],
         "Binary Polarity\nCross-Entropy", fs=9, bold=True)
    rbox(ax, 12.3, 6.30, 2.6, 0.55, C["danger"],
         "Smooth-L1\nRegression", fs=9, bold=True)
    ax.text(2.0, 5.75, "(soft labels for class distance)",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")
    ax.text(5.7, 5.75, "(penalizes distant misclassifications)",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")
    ax.text(9.3, 5.75, "(positive vs. negative)",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")
    ax.text(12.3, 5.75, "(continuous strength)",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")
    rbox(ax, 7, 5.30, 13.0, 0.42, C["text"],
         "Per-branch loss  =  weighted combination of the four above",
         fs=10, bold=True)

    # Layer 2
    bg(ax, 0.4, 2.4, 13.2, 2.3, "#FFE4E6", C["rose"])
    ax.text(7, 4.45,
            "Layer 2 — Cross-Branch Aggregation, Diversity and Cross-modal Contrast (Stage 2 only)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["rose"])
    rbox(ax, 2.3, 3.70, 4.0, 0.55, C["rose"],
         "Mean-output loss\n(on average of 4 branches)",
         fs=9, bold=True)
    rbox(ax, 7.0, 3.70, 4.0, 0.55, C["rose"],
         "Per-branch loss\n(average over branches)",
         fs=9, bold=True)
    rbox(ax, 11.7, 3.70, 4.0, 0.55, C["rose"],
         "Branch diversity penalty\n(decorrelate features)",
         fs=9, bold=True)
    rbox(ax, 7.0, 2.85, 12.0, 0.50, C["danger"],
         "Cross-modal contrast — pull text, audio and vision embeddings together "
         "for the same sample, push apart for different samples (added in Stage 2)",
         fs=9.0, bold=True)

    # Layer 3
    bg(ax, 0.4, 0.3, 13.2, 1.85, "#DCFCE7", C["success"])
    ax.text(7, 1.95,
            "Layer 3 — Consistency Regularization and Final Total",
            ha="center", fontsize=11, fontweight="bold", color=C["success"])
    rbox(ax, 7.0, 1.30, 12.4, 0.55, C["success"],
         "R-Drop consistency — two stochastic forward passes "
         "should agree on the output distribution",
         fs=9.5, bold=True)
    rbox(ax, 7.0, 0.60, 12.8, 0.55, C["text"],
         "Total loss  =  weighted sum of all the above",
         fs=10.5, bold=True)

    save(fig, "v7_fig_loss_comp")


# ════════════════════════════════════════════════════════════════════════════
# FIG training timeline — TWO STAGES + THREE RUNS
# ════════════════════════════════════════════════════════════════════════════
def fig_train_timeline():
    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    ax.set_xlim(0, 80); ax.set_ylim(-2.6, 6.0); ax.axis("off")
    ax.text(40, 5.60,
            "Training Pipeline — Two-Stage Protocol with Three Independent Runs",
            ha="center", fontsize=12.5, fontweight="bold", color=C["primary"])

    # Single run -- Stage 1 (60 ep) + Stage 2 (20 ep)
    bg(ax, 0.5, 1.3, 78, 3.8, "#DBEAFE", C["primary"])
    ax.text(40, 4.80, "Single-run training  (Stage 1: 60 epochs   +   Stage 2: 20 epochs)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])

    # Stage 1 box on the left half
    bg(ax, 1, 1.9, 54, 2.3, "#FEF3C7", C["accent"], alpha=0.55)
    ax.text(28, 3.95, "Stage 1   —   Base training (no cross-modal contrast)",
            ha="center", fontsize=10, fontweight="bold", color=C["accent"])
    rbox(ax, 11, 3.20, 18, 0.65, C["accent"],
         "Phase 1  —  freeze DeBERTa\nbottom 6 layers (E1–20)",
         fs=8.5, bold=True, tc="white")
    rbox(ax, 28, 3.20, 14, 0.65, C["success"],
         "Phase 2  —  full-model\nfine-tune (E20–42)",
         fs=8.5, bold=True)
    rbox(ax, 46, 3.20, 16, 0.65, C["rose"],
         "Phase 3  —  weight averaging\n(E42–60, ten snapshots)",
         fs=8.5, bold=True)
    ax.text(28, 2.30,
            "Exponential moving average of model weights maintained throughout",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")

    # Stage 2 box on the right
    bg(ax, 56, 1.9, 22.5, 2.3, "#FEE2E2", C["danger"], alpha=0.55)
    ax.text(67.2, 3.95, "Stage 2   —   Contrastive fine-tune",
            ha="center", fontsize=10, fontweight="bold", color=C["danger"])
    rbox(ax, 67.2, 3.20, 20, 0.65, C["danger"],
         "Lower learning rate  +  cross-modal\ncontrastive loss added  (20 epochs)",
         fs=8.5, bold=True, tc="white")
    ax.text(67.2, 2.30,
            "Per-epoch snapshot weight averaging (16 snapshots)",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")

    # Three independent runs panel
    bg(ax, 0.5, -2.5, 78, 3.2, "#EDE9FE", C["purple"], alpha=0.30)
    ax.text(40, 0.40, "Three Independent Runs   →   Parameter-Space Average",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    runs = [
        ("Run 1",  "seed 42, standard protocol",        C["primary"], 13),
        ("Run 2",  "seed 42, Stage 2 extended",         C["accent"],  40),
        ("Run 3",  "seed 5678, two BAN rounds",         C["success"], 66),
    ]
    weights = ["weight 0.25", "weight 0.45", "weight 0.30"]
    for (lbl, sub, col, cx), w in zip(runs, weights):
        rbox(ax, cx, -0.30, 22, 0.65, col,
             lbl, fs=10.5, bold=True)
        ax.text(cx, -0.90, sub,
                ha="center", fontsize=8.5, color=C["text"])
        ax.text(cx, -1.30, w,
                ha="center", fontsize=9, color=col, fontweight="bold")
    arr(ax, 13, -1.55, 40, -2.00, color=C["primary"], lw=1.2, hw=0.12)
    arr(ax, 40, -1.55, 40, -2.00, color=C["accent"], lw=1.2, hw=0.12)
    arr(ax, 66, -1.55, 40, -2.00, color=C["success"], lw=1.2, hw=0.12)
    rbox(ax, 40, -2.25, 56, 0.55, C["danger"],
         "Merged checkpoint  =  0.25 × Run 1  +  0.45 × Run 2  +  0.30 × Run 3",
         fs=10.0, bold=True)

    save(fig, "v7_fig_train_timeline")


# ════════════════════════════════════════════════════════════════════════════
# FIG inference — only one forward at inference (ensemble done at training time)
# ════════════════════════════════════════════════════════════════════════════
def fig_inference():
    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(7, 6.85, "Zero-Leakage Inference Pipeline",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])
    ax.text(7, 6.40,
            "The three-run weights are pre-averaged into one checkpoint at "
            "training time, so inference uses only one forward pass per sample.",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Stage A — TTA
    bg(ax, 0.3, 4.2, 4.2, 1.95, "#DBEAFE", C["primary"])
    ax.text(2.4, 5.85, "Step A   Test-time augmentation",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    for i in range(5):
        rbox(ax, 1.05 + i * 0.65, 4.95, 0.5, 0.40, C["primary"],
             f"pass {i+1}", fs=7.5, bold=True)
    ax.text(2.4, 4.40,
            "Five stochastic forward passes\nfor each test sample,\nthen averaged",
            ha="center", fontsize=8.5, fontweight="bold",
            color=C["primary"])

    # Stage B — Three-run ensemble (already inside the weights)
    bg(ax, 5.0, 4.2, 4.0, 1.95, "#FEF3C7", C["accent"])
    ax.text(7.0, 5.85, "Step B   Three-run ensemble",
            ha="center", fontsize=10.5, fontweight="bold", color=C["accent"])
    ax.text(7.0, 5.05,
            "Already merged into one\nparameter-averaged checkpoint\nat training time",
            ha="center", fontsize=8.5, color=C["accent"], fontweight="bold")
    ax.text(7.0, 4.35,
            "No extra forward pass\nis needed at inference",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")

    # Stage C — Reg-Cls fusion
    bg(ax, 9.7, 4.2, 4.0, 1.95, "#DCFCE7", C["success"])
    ax.text(11.7, 5.85, "Step C   Probability fusion",
            ha="center", fontsize=10.5, fontweight="bold", color=C["success"])
    ax.text(11.7, 5.10,
            "Combine classification and\nregression probabilities\nin log-space",
            ha="center", fontsize=8.5, color=C["text"], fontweight="bold")
    ax.text(11.7, 4.45,
            "(fusion hyper-parameters set a priori)",
            ha="center", fontsize=8, color=C["muted"], style="italic")

    arr(ax, 4.5, 5.15, 5.0, 5.15, lw=1.6)
    arr(ax, 9.0, 5.15, 9.7, 5.15, lw=1.6)

    # Bottom flow
    bg(ax, 0.3, 0.4, 13.4, 3.0, "#EDE9FE", C["purple"])
    ax.text(7, 3.20, "Detailed inference flow",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    flow = [
        ("Test sample", C["text"], 1.2, 2.0),
        ("Load merged\ncheckpoint\n(one model)", C["accent"], 3.4, 2.0),
        ("Five stochastic\nforward passes\nwith dropout", C["primary"], 5.7, 2.0),
        ("Average\nthe five outputs", C["secondary"], 8.0, 2.0),
        ("Fuse classification\nand regression\nprobabilities", C["success"], 10.4, 2.0),
        ("Argmax\nfinal class", C["danger"], 12.8, 2.0),
    ]
    for label, col, cx, _ in flow:
        rbox(ax, cx, 1.95, 2.05, 1.10, col, label, fs=8.5, bold=True)
    for i in range(len(flow) - 1):
        cx1 = flow[i][2] + 1.05
        cx2 = flow[i+1][2] - 1.05
        arr(ax, cx1, 1.95, cx2, 1.95, lw=1.4, hw=0.15)

    ax.text(7, 0.85,
            "All fusion hyper-parameters are fixed before evaluation — "
            "no test-set statistics are used (zero data leakage).",
            ha="center", fontsize=9.5, color=C["text"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.40", fc="#F9FAFB",
                      ec=C["danger"], lw=1.0))

    save(fig, "v7_fig_inference")


# ════════════════════════════════════════════════════════════════════════════
# FIG distribution
# ════════════════════════════════════════════════════════════════════════════
def fig_distribution():
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    classes_label = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]
    x = np.arange(7); w = 0.36
    ax.bar(x - w/2, TRAINVAL_DIST, w, color=C["primary"],
           label="Train  (n = 1,513)", alpha=0.9)
    ax.bar(x + w/2, TEST_DIST, w, color=C["rose"],
           label="Test   (n = 686)", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Sentiment class label")
    ax.set_ylabel("Class proportion (%)")
    ax.set_title("CMU-MOSI 7-Class Sentiment Distribution  (Train and Test)")
    ax.legend(fontsize=10, loc="upper left")
    shift = abs(TEST_DIST[0] - TRAINVAL_DIST[0])
    ax.annotate(
        f"Class −3:   Test {TEST_DIST[0]:.1f}%  vs.  Train {TRAINVAL_DIST[0]:.1f}%\n"
        f"Distribution shift  =  +{shift:.1f}%",
        xy=(0, max(TRAINVAL_DIST[0], TEST_DIST[0])),
        xytext=(0.6, max(TRAINVAL_DIST) * 0.92),
        fontsize=8.5, color=C["danger"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["danger"], lw=1.2))
    plt.tight_layout()
    save(fig, "v7_fig_distribution")


# ════════════════════════════════════════════════════════════════════════════
# FIG regcls — example bar charts
# ════════════════════════════════════════════════════════════════════════════
def fig_regcls():
    L7 = DATA["L7"]; R = DATA["R"]; Y7 = DATA["y7"]
    z = L7[316]
    r_pred = float(R[316])
    y_true = int(Y7[316]) - 3

    K = 7
    classes_label = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]
    classes_idx = np.arange(K)
    p_cls = np.exp((z - z.max())); p_cls /= p_cls.sum()
    sigma = 0.65
    y_shift = r_pred + 3.0
    d2 = (classes_idx - y_shift) ** 2
    p_reg = np.exp(-d2 / (2 * sigma ** 2)); p_reg /= p_reg.sum()
    alpha = 0.65
    log_p = alpha * np.log(p_cls + 1e-9) + (1 - alpha) * np.log(p_reg + 1e-9)
    log_p -= log_p.max()
    p_final = np.exp(log_p); p_final /= p_final.sum()

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7))
    truth_idx = y_true + 3
    cls_argmax = int(p_cls.argmax()) - 3
    fused_argmax = int(p_final.argmax()) - 3

    ax = axes[0]
    ax.bar(classes_idx, p_cls, color=C["primary"], alpha=0.85)
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"Ground truth ({y_true:+d})")
    ax.axvline(int(p_cls.argmax()), color=C["danger"], lw=1.5, ls=":",
               label=f"argmax classification ({cls_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(a)  Classification head probabilities")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    ax = axes[1]
    ax.bar(classes_idx, p_reg, color=C["accent"], alpha=0.85)
    ax.axvline(y_shift, color=C["text"], lw=1.5,
               label=f"regression prediction = {r_pred:+.2f}")
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(b)  Regression mapped to class probabilities (Gaussian)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    ax = axes[2]
    ax.bar(classes_idx, p_final, color=C["success"], alpha=0.85)
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.axvline(int(p_final.argmax()), color=C["danger"], lw=1.5, ls=":",
               label=f"argmax fused ({fused_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(c)  Fused probabilities (geometric mean in log-space)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Classification and Regression Probability Fusion — Example",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "v7_fig_regcls")


if __name__ == "__main__":
    print("Regenerating v7 conceptual figures (Hierarchical SACF + two-stage + three-run) ...")
    fig_arch();             print("  ok  v7_fig_arch")
    fig_pea();              print("  ok  v7_fig_pea")
    fig_sacf_steps();       print("  ok  v7_fig_sacf_steps")
    fig_branches();         print("  ok  v7_fig_branches")
    fig_loss_comp();        print("  ok  v7_fig_loss_comp")
    fig_train_timeline();   print("  ok  v7_fig_train_timeline")
    fig_inference();        print("  ok  v7_fig_inference")
    fig_distribution();     print("  ok  v7_fig_distribution")
    fig_regcls();           print("  ok  v7_fig_regcls")
    print("Done.")
