"""SACF v5 figures — fully English labels + true math-mode subscripts.

Changes vs v4:
  • All Chinese text on figures replaced with English equivalents.
  • Subscripts now use LaTeX-style mathtext ($x_{\\text{cls}}$, $L_{\\text{softCE}}$)
    instead of underscores in the rendered text, so labels read as proper
    typographic subscripts.
  • Box dimensions and spacing widened where the previous draft was tight.
  • All formula text blocks use monospace + clear background to avoid clipping.

Run:  python3 docs/regenerate_v5_figures.py
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
    "mathtext.default": "regular",
    "mathtext.fontset": "dejavusans",
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


def arr(ax, x1, y1, x2, y2, color=None, lw=1.4, hw=0.18, z=3, ls="-"):
    if color is None: color = C["text"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f"->,head_width={hw},head_length={hw*1.4}",
                                 color=color, lw=lw, linestyle=ls), zorder=z)


def bg(ax, x0, y0, w, h, fc, ec, lw=1.2, alpha=0.18):
    r = FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.01,rounding_size=0.18",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=1)
    ax.add_patch(r)


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Overall architecture (English)
# ════════════════════════════════════════════════════════════════════════════
def fig_arch():
    fig, ax = plt.subplots(figsize=(14.2, 13.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 15.5); ax.axis("off")

    ax.text(7, 14.95, "SACFFinalModel — Multi-Branch Single Model Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["primary"])
    ax.text(7, 14.45,
            f"Acc-7 = {ACC7_FUSED:.2f}%  |  Acc-2 = {ACC2:.2f}%  |  F1 = {F1:.2f}%  "
            f"|  MAE = {MAE:.4f}  |  Corr = {CORR:.4f}",
            ha="center", va="center", fontsize=10.5, color=C["muted"])

    # Row 1: Inputs
    bg(ax, 0.4, 12.9, 13.2, 1.2, "#FEF3C7", C["accent"])
    ax.text(7, 13.85, "(i)  Multimodal Inputs",
            ha="center", fontsize=11, color=C["accent"], fontweight="bold")
    rbox(ax, 2.5, 13.30, 3.0, 0.55, C["primary"],
         "Text (raw_text + prompt)", fs=9.5, bold=True)
    rbox(ax, 7.0, 13.30, 3.0, 0.55, C["accent"],
         "Audio (COVAREP, 5-dim)", fs=9.5, bold=True)
    rbox(ax, 11.5, 13.30, 3.0, 0.55, C["success"],
         "Vision (FACET, 20-dim)", fs=9.5, bold=True)

    # Row 2: Shared encoders
    bg(ax, 0.4, 11.0, 13.2, 1.5, "#DBEAFE", C["primary"])
    ax.text(7, 12.35, "(ii)  Shared Encoders",
            ha="center", fontsize=11, color=C["primary"], fontweight="bold")
    rbox(ax, 2.5, 11.55, 3.0, 0.7, C["primary"],
         "DeBERTa-v3-large\n24L · 1024d → $t_{\\mathrm{emb}}$", fs=9, bold=True)
    rbox(ax, 7.0, 11.55, 3.0, 0.7, C["accent"],
         "BiLSTM-Audio\n2L · 5→128 → $a_{\\mathrm{emb}}$", fs=9, bold=True)
    rbox(ax, 11.5, 11.55, 3.0, 0.7, C["success"],
         "BiLSTM-Vision\n2L · 20→128 → $v_{\\mathrm{emb}}$", fs=9, bold=True)
    for cx in (2.5, 7.0, 11.5):
        arr(ax, cx, 13.02, cx, 11.94)

    # Row 3: 4 branches
    branch_top = 10.50
    branch_bottom = 6.85
    bg(ax, 0.4, branch_bottom, 13.2, branch_top - branch_bottom + 0.30, "#EDE9FE", C["purple"])
    ax.text(7, branch_top + 0.05,
            "(iii)  4 Parallel Branches (independent params; per-branch dropout = {0.10, 0.20, 0.30, 0.40})",
            ha="center", fontsize=10.5, color=C["purple"], fontweight="bold")

    for cx in (2.0, 5.0, 8.0, 11.0):
        arr(ax, 2.5, 11.20, cx, 10.30, color=C["primary"], lw=0.6, hw=0.10)
        arr(ax, 7.0, 11.20, cx, 10.30, color=C["accent"], lw=0.6, hw=0.10)
        arr(ax, 11.5, 11.20, cx, 10.30, color=C["success"], lw=0.6, hw=0.10)

    branch_colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    branch_dp = [0.10, 0.20, 0.30, 0.40]
    for i in range(4):
        cx = 2.0 + i * 3.0
        bg(ax, cx - 1.30, 7.00, 2.6, 3.10, "#FFFFFF", branch_colors[i], lw=1.6, alpha=0.6)
        ax.text(cx, 9.85, f"Branch {i+1}  (p = {branch_dp[i]})",
                ha="center", fontsize=9.5, fontweight="bold", color=branch_colors[i])
        rbox(ax, cx, 9.25, 2.4, 0.5, branch_colors[i], "PEA  (gate $\\sigma$)", fs=8.5)
        rbox(ax, cx, 8.60, 2.4, 0.5, branch_colors[i], "SACF  (top-K cross-modal)", fs=8.5)
        rbox(ax, cx, 7.95, 2.4, 0.5, branch_colors[i],
             f"Proj  (1024 → 512)  →  $e_{{{i+1}}}$", fs=8.5)
        rbox(ax, cx, 7.30, 2.4, 0.55, "#FFFFFF",
             f"$\\ell 7_{{{i+1}}}$ /  $\\ell 2_{{{i+1}}}$ /  $\\mathrm{{reg}}_{{{i+1}}}$",
             fs=8.5, tc=branch_colors[i], ec=branch_colors[i], bold=True)

    # Row 4: Mean aggregation
    bg(ax, 4.0, 5.50, 6.0, 0.95, "#FFE4E6", C["rose"])
    rbox(ax, 7.0, 5.95, 5.6, 0.65, C["rose"],
         "Mean-of-Branches  →  ($\\ell 7_{\\mathrm{mean}}$,  $\\ell 2_{\\mathrm{mean}}$,  $\\mathrm{reg}_{\\mathrm{mean}}$)",
         fs=10, bold=True)
    for i in range(4):
        cx = 2.0 + i * 3.0
        arr(ax, cx, 7.00, 7.0, 6.35, color=branch_colors[i], lw=0.8, hw=0.10)

    # Row 5: Inference fusion
    bg(ax, 0.4, 2.9, 13.2, 2.2, "#DCFCE7", C["success"])
    ax.text(7, 4.95,
            "(iv)  Zero-Leakage Inference:  TTA × 5  +  3-Seed Ensemble  +  Reg-Cls Probability Fusion",
            ha="center", fontsize=10.5, color=C["success"], fontweight="bold")
    rbox(ax, 3.0, 4.15, 3.4, 0.7, C["primary"],
         "$p_{\\mathrm{cls}}$ = softmax($\\ell 7_{\\mathrm{mean}}$ / $T_{\\mathrm{cls}}$)",
         fs=9.5, bold=True)
    rbox(ax, 7.0, 4.15, 3.4, 0.7, C["accent"],
         "$p_{\\mathrm{reg}}[k] \\propto \\exp(-(k-r)^2 / 2\\sigma^2)$",
         fs=9.5, bold=True)
    rbox(ax, 11.0, 4.15, 3.4, 0.7, C["success"],
         "$\\log p_{\\mathrm{final}} = \\alpha\\log p_{\\mathrm{cls}} + (1{-}\\alpha)\\log p_{\\mathrm{reg}}$",
         fs=9, bold=True)
    arr(ax, 7.0, 5.50, 7.0, 4.60)
    arr(ax, 3.0, 3.75, 5.5, 3.35, hw=0.12)
    arr(ax, 7.0, 3.75, 7.0, 3.35, hw=0.12)
    arr(ax, 11.0, 3.75, 8.5, 3.35, hw=0.12)

    rbox(ax, 7.0, 2.45, 5.6, 0.75, C["danger"],
         f"$\\hat{{y}}$ = argmax($p_{{\\mathrm{{final}}}}$)        Acc-7 = {ACC7_FUSED:.2f}%",
         fs=11, bold=True)

    handles = [
        mpatches.Patch(color=C["primary"], label="Text (DeBERTa)"),
        mpatches.Patch(color=C["accent"], label="Audio (BiLSTM)"),
        mpatches.Patch(color=C["success"], label="Vision (BiLSTM)"),
        mpatches.Patch(color=C["rose"], label="Aggregation"),
        mpatches.Patch(color=C["purple"], label="Parallel Branches"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.07),
              ncol=5, fontsize=9, frameon=False)

    save(fig, "v5_fig_arch")


# ════════════════════════════════════════════════════════════════════════════
# FIG PEA — English
# ════════════════════════════════════════════════════════════════════════════
def fig_pea():
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(6.75, 6.60, "Polarity-Enhanced Attention (PEA) Module",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(6.75, 6.18,
            "Learn token-level sentiment salience gate $g_i$ then weighted-pool the DeBERTa hidden states",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Input
    bg(ax, 0.3, 3.0, 2.6, 2.7, "#DBEAFE", C["primary"])
    ax.text(1.6, 5.42, "DeBERTa Output", ha="center", fontsize=10,
            fontweight="bold", color=C["primary"])
    tokens = ["[CLS]", "good", "movie", "!", "[PAD]"]
    pol_vals = [0.10, 0.92, 0.30, 0.50, 0.0]
    for ti, (tok, _) in enumerate(zip(tokens, pol_vals)):
        ty = 4.85 - ti * 0.38
        rbox(ax, 1.6, ty, 2.2, 0.30, C["primary"],
             f"$h_{ti}$  ({tok})", fs=8.5, bold=True)

    # Gate MLP
    bg(ax, 3.7, 3.2, 3.6, 2.3, "#FEF3C7", C["accent"])
    ax.text(5.5, 5.20, "Gate MLP", ha="center", fontsize=10,
            fontweight="bold", color=C["accent"])
    rbox(ax, 5.5, 4.55, 3.2, 0.50, C["accent"],
         "Linear ($d \\to d/4$)  +  tanh", fs=9, bold=True)
    rbox(ax, 5.5, 3.95, 3.2, 0.50, C["accent"],
         "Linear ($d/4 \\to 1$)  +  $\\sigma$ (sigmoid)", fs=9, bold=True)
    arr(ax, 2.9, 4.30, 3.85, 4.30, lw=1.3)

    # Gate values
    bg(ax, 7.8, 3.0, 2.0, 2.7, "#FFE4E6", C["rose"])
    ax.text(8.8, 5.42, "Gate  $g_i \\in [0,1]$",
            ha="center", fontsize=10, fontweight="bold", color=C["rose"])
    for ti, (tok, pol) in enumerate(zip(tokens, pol_vals)):
        ty = 4.85 - ti * 0.38
        rbox(ax, 8.8, ty, 1.7, 0.30, C["rose"],
             f"$g_{ti}$ = {pol:.2f}", fs=8.5, bold=True)
    arr(ax, 7.15, 4.30, 7.85, 4.30, lw=1.3)

    # Weighted pooling
    bg(ax, 10.4, 3.0, 3.0, 2.7, "#DCFCE7", C["success"])
    ax.text(11.9, 5.42, "Weighted Pooling",
            ha="center", fontsize=10, fontweight="bold", color=C["success"])
    ax.text(11.9, 4.75,
            "$x_l = \\frac{\\sum_i (0.75 h_i + 0.25 h_i g_i)\\, m_i}{\\sum_i m_i}$",
            ha="center", fontsize=10.5, fontweight="bold")
    rbox(ax, 11.9, 3.65, 2.6, 0.50, C["success"],
         "$x_l \\in \\mathbb{R}^{B \\times d_{\\mathrm{lang}}}$",
         fs=9.5, bold=True)
    arr(ax, 9.85, 4.30, 10.45, 4.30, lw=1.3)

    # Bottom: PEA outputs
    bg(ax, 0.3, 0.4, 12.9, 2.0, "#EDE9FE", C["purple"])
    ax.text(6.75, 2.10, "PEA Outputs",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    rbox(ax, 3.0, 1.35, 5.2, 0.55, C["purple"],
         "$x_l$  →  input to each branch's shared projection",
         fs=9.5, bold=True)
    rbox(ax, 9.8, 1.35, 6.4, 0.55, C["purple"],
         "$g$  →  Top-K (K=5) tokens  →  build SACF query",
         fs=9.5, bold=True)
    arr(ax, 11.9, 3.30, 11.9, 1.65, color=C["purple"], hw=0.12)

    save(fig, "v5_fig_pea")


# ════════════════════════════════════════════════════════════════════════════
# FIG SACF steps — English
# ════════════════════════════════════════════════════════════════════════════
def fig_sacf_steps():
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(7, 6.60, "Sentiment-Aware Cross-modal Fusion (SACF) — Step-by-Step",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(7, 6.18,
            "Build a sentiment-aware query from PEA's top-K tokens, then align with audio/vision keys & values",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    step_colors = [C["primary"], C["violet"], C["accent"], C["success"]]
    step_titles = [
        "Step 1: Top-K Tokens",
        "Step 2: Sentiment Query",
        "Step 3: Cross-Modal KV",
        "Step 4: Gated Fusion",
    ]
    step_formulas = [
        ("$I = \\mathrm{TopK}(g, K{=}5)$\n"
         "$H_{\\mathrm{topk}} = H[I]$\n"
         "$\\in \\mathbb{R}^{B \\times 5 \\times d}$"),
        ("$w = \\mathrm{softmax}(W_{\\mathrm{tok}} H_{\\mathrm{topk}})$\n"
         "$q_{\\mathrm{sa}} = \\sum_i w_i \\cdot H_{\\mathrm{topk}}[i]$"),
        ("$\\mathrm{KV} = \\mathrm{stack}(W_a a_{\\mathrm{emb}},\\; W_v v_{\\mathrm{emb}})$\n"
         "$x^{*} = \\mathrm{softmax}(q_{\\mathrm{sa}} \\mathrm{KV}^{T} / \\sqrt{d})\\, \\mathrm{KV}$"),
        ("$x = \\mathrm{FFN}(x_{\\mathrm{cls}} + x^{*})$\n"
         "$g_w = \\sigma(W_g \\, [x_{\\mathrm{cls}};\\, x])$\n"
         "$f = \\mathrm{LN}(x_{\\mathrm{cls}} + \\mathrm{Dropout}(x g_w))$"),
    ]

    box_w = 3.10
    box_h = 4.65
    spacing = 0.30
    start_x = 0.40

    for i, (title, form, col) in enumerate(zip(step_titles, step_formulas, step_colors)):
        cx = start_x + i * (box_w + spacing) + box_w / 2
        bg(ax, cx - box_w/2, 0.6, box_w, box_h, "#FFFFFF", col, lw=1.6, alpha=0.85)
        # Title bar uses slightly smaller font and longer box-width margin
        rbox(ax, cx, 0.6 + box_h - 0.45, box_w - 0.30, 0.55, col,
             title, fs=9.5, tc="white", bold=True)
        ax.text(cx, 0.6 + box_h - 2.00, form,
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.40", fc="#F9FAFB",
                          ec=col, lw=1.0))
        outs = ["$H_{\\mathrm{topk}} \\in \\mathbb{R}^{B \\times K \\times d}$",
                "$q_{\\mathrm{sa}} \\in \\mathbb{R}^{B \\times d}$",
                "$x^{*} \\in \\mathbb{R}^{B \\times d}$",
                "$f \\in \\mathbb{R}^{B \\times d_{\\mathrm{lang}}}$"]
        rbox(ax, cx, 0.6 + 0.40, box_w - 0.30, 0.45, col,
             outs[i], fs=9, tc="white", bold=True)
        if i < 3:
            arr(ax, cx + box_w/2 - 0.10, 2.95,
                start_x + (i+1)*(box_w + spacing) + 0.10, 2.95,
                color=col, lw=2.0, hw=0.18)

    save(fig, "v5_fig_sacf_steps")


# ════════════════════════════════════════════════════════════════════════════
# FIG branches — English
# ════════════════════════════════════════════════════════════════════════════
def fig_branches():
    fig, ax = plt.subplots(figsize=(14.0, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.text(7, 7.20, "Four Parallel Branches — Sources of Diversity & Internal Ensemble",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(7, 6.80,
            "Three diversity mechanisms:  (a) different dropout rates;  "
            "(b) independent PEA/SACF/Proj weights;  (c) small Gaussian "
            "perturbation on cls-7 head init",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    bg(ax, 0.3, 4.4, 13.4, 1.9, "#FEF3C7", C["accent"])
    ax.text(7, 6.10, "Diversity Mechanisms",
            ha="center", fontsize=10.5, fontweight="bold", color=C["accent"])
    rbox(ax, 2.8, 5.40, 4.4, 0.55, C["accent"],
         "(a) Branch dropout = [0.10, 0.20, 0.30, 0.40]", fs=9.5, bold=True)
    rbox(ax, 7.4, 5.40, 3.8, 0.55, C["accent"],
         "(b) Independent PEA / SACF / Proj weights", fs=9.5, bold=True)
    rbox(ax, 11.4, 5.40, 4.6, 0.55, C["accent"],
         "(c) Cls-7 init perturb. $0.005(i{+}1)\\cdot \\mathcal{N}(0,1)$",
         fs=9.5, bold=True)
    rbox(ax, 7.0, 4.65, 12.6, 0.55, C["accent"],
         "→ at inference dropout is off, so the 4 branches differ deterministically by weight",
         fs=9.5, bold=True)

    branch_colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    branch_dp = [0.10, 0.20, 0.30, 0.40]
    per_branch_acc7 = [52.62, 52.18, 51.89, 52.34]
    bw = 2.9
    for i in range(4):
        cx = 1.05 + i * 3.0 + bw/2
        bg(ax, cx - bw/2, 1.30, bw, 2.85, "#FFFFFF", branch_colors[i], lw=1.6, alpha=0.65)
        ax.text(cx, 3.95, f"Branch {i+1}",
                ha="center", fontsize=10.5, fontweight="bold", color=branch_colors[i])
        ax.text(cx, 3.60, f"dropout = {branch_dp[i]}",
                ha="center", fontsize=8.5, color=C["text"])
        rbox(ax, cx, 3.05, bw - 0.4, 0.42, branch_colors[i],
             "PEA  →  SACF  →  Proj", fs=8.5, bold=True)
        rbox(ax, cx, 2.45, bw - 0.4, 0.42, "#FFFFFF",
             f"$e_{{{i+1}}} \\in \\mathbb{{R}}^{{B \\times 512}}$",
             fs=9, tc=branch_colors[i], ec=branch_colors[i], bold=True)
        rbox(ax, cx, 1.85, bw - 0.4, 0.42, branch_colors[i],
             f"($\\ell 7_{{{i+1}}}$,  $\\ell 2_{{{i+1}}}$,  $\\mathrm{{reg}}_{{{i+1}}}$)",
             fs=8.5, bold=True)
        ax.text(cx, 1.45,
                f"Acc-7  ≈  {per_branch_acc7[i]:.2f}%",
                ha="center", fontsize=8.5,
                color=branch_colors[i], fontweight="bold")

    bg(ax, 3.5, 0.10, 7.0, 0.95, "#FFE4E6", C["rose"])
    rbox(ax, 7.0, 0.55, 6.4, 0.55, C["rose"],
         "Internal Mean  →  $\\ell 7_{\\mathrm{mean}} = \\frac{1}{4}\\sum_i \\ell 7_i$  "
         "    Acc-7 ≈ 53.21%",
         fs=10, bold=True)
    for i in range(4):
        cx = 1.05 + i * 3.0 + bw/2
        arr(ax, cx, 1.35, 7.0, 1.10, color=branch_colors[i], lw=0.8, hw=0.10)

    save(fig, "v5_fig_branches")


# ════════════════════════════════════════════════════════════════════════════
# FIG loss composition — English
# ════════════════════════════════════════════════════════════════════════════
def fig_loss_comp():
    fig, ax = plt.subplots(figsize=(14.0, 9.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.0); ax.axis("off")
    ax.text(7, 8.60, "Multi-Task Loss — Composition Structure",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(7, 8.20,
            "Layer 1 (per-branch task loss)  →  Layer 2 (aggregation + diversity + cross-modal contrast)  →  "
            "Layer 3 (R-Drop consistency + $L_{\\mathrm{total}}$)",
            ha="center", fontsize=10, color=C["muted"], style="italic")

    bg(ax, 0.4, 5.2, 13.2, 2.6, "#DBEAFE", C["primary"])
    ax.text(7, 7.55,
            "Layer 1 — Per-branch $i$ task loss  $L_{\\mathrm{branch},i}$",
            ha="center", fontsize=11, fontweight="bold", color=C["primary"])
    rbox(ax, 3.0, 6.75, 4.2, 0.65, C["primary"],
         "$L_{\\mathrm{softCE}} = - \\sum_k\\, \\mathrm{soft\\_target}_k \\log\\!\\mathrm{softmax}(\\ell 7)_k$",
         fs=9, bold=True)
    rbox(ax, 7.5, 6.75, 4.2, 0.65, C["secondary"],
         "$L_{\\mathrm{EMD}} = \\sum_k\\, |\\mathrm{CDF}_{\\mathrm{pred}} - \\mathrm{CDF}_{\\mathrm{true}}|$",
         fs=9, bold=True)
    rbox(ax, 12.0, 6.75, 1.7, 0.65, C["accent"],
         "$L_{\\mathrm{cls2}}$ (CE)", fs=9.5, bold=True)
    ax.text(3.0, 6.20, "(Gaussian soft-ordinal CE,  $\\sigma = 1.0$)",
            ha="center", fontsize=8.5, color=C["muted"])
    ax.text(7.5, 6.20, "(Earth-mover’s ordinal loss,  $w_{\\mathrm{EMD}} = 0.25$)",
            ha="center", fontsize=8.5, color=C["muted"])
    ax.text(12.0, 6.20, "(binary polarity)",
            ha="center", fontsize=8.5, color=C["muted"])
    rbox(ax, 7.0, 5.65, 9.8, 0.50, C["text"],
         "$L_{\\mathrm{branch},i} = (1 - w_{\\mathrm{EMD}})\\, L_{\\mathrm{softCE}} "
         "+ w_{\\mathrm{EMD}}\\, L_{\\mathrm{EMD}} "
         "+ 0.3\\, L_{\\mathrm{cls2}} + 0.4\\, L_{\\mathrm{SmoothL1}}$",
         fs=10, bold=True)

    bg(ax, 0.4, 2.6, 13.2, 2.2, "#FFE4E6", C["rose"])
    ax.text(7, 4.60,
            "Layer 2 — Branch aggregation, diversity, and cross-modal contrast",
            ha="center", fontsize=11, fontweight="bold", color=C["rose"])
    rbox(ax, 2.5, 3.80, 4.0, 0.65, C["rose"],
         "$L_{\\mathrm{mean}} = L_{\\mathrm{branch}}$  on  $\\ell 7_{\\mathrm{mean}}$",
         fs=9.5, bold=True)
    rbox(ax, 7.0, 3.80, 4.0, 0.65, C["rose"],
         "$L_{\\mathrm{per\\_branch}} = \\frac{1}{4} \\sum_i L_{\\mathrm{branch},i}$",
         fs=9.5, bold=True)
    rbox(ax, 11.5, 3.80, 2.2, 0.65, C["rose"],
         "$L_{\\mathrm{diversity}}$", fs=9.5, bold=True)
    ax.text(11.5, 3.30,
            "(cosine sim. penalty on  $e_i$)",
            ha="center", fontsize=8.0, color=C["muted"])
    rbox(ax, 7.0, 3.00, 9.4, 0.55, C["danger"],
         "$L_{\\mathrm{CMC}}$  =  InfoNCE cross-modal contrast  "
         "(pull $t_{\\mathrm{emb}}, a_{\\mathrm{emb}}, v_{\\mathrm{emb}}$ together,  push apart different samples)",
         fs=9.5, bold=True)

    bg(ax, 0.4, 0.6, 13.2, 1.9, "#DCFCE7", C["success"])
    ax.text(7, 2.30,
            "Layer 3 — R-Drop consistency  +  $L_{\\mathrm{total}}$",
            ha="center", fontsize=11, fontweight="bold", color=C["success"])
    rbox(ax, 7.0, 1.70, 12.4, 0.55, C["success"],
         "$L_{\\mathrm{R-Drop}} = \\frac{1}{2}\\,(\\mathrm{KL}(p_{f_1} \\,||\\, p_{f_2}) "
         "+ \\mathrm{KL}(p_{f_2} \\,||\\, p_{f_1}))$",
         fs=9.5, bold=True)
    rbox(ax, 7.0, 1.00, 13.0, 0.55, C["text"],
         "$L_{\\mathrm{total}} = w_{\\mathrm{mean}} L_{\\mathrm{mean}} "
         "+ w_{\\mathrm{per}} L_{\\mathrm{per\\_branch}} "
         "+ w_{\\mathrm{div}} L_{\\mathrm{diversity}} "
         "+ w_{\\mathrm{CMC}} L_{\\mathrm{CMC}} + 0.05\\, L_{\\mathrm{R-Drop}}$",
         fs=10, bold=True)

    save(fig, "v5_fig_loss_comp")


# ════════════════════════════════════════════════════════════════════════════
# FIG training timeline — English
# ════════════════════════════════════════════════════════════════════════════
def fig_train_timeline():
    fig, ax = plt.subplots(figsize=(14.0, 6.4))
    ax.set_xlim(0, 80); ax.set_ylim(-2.0, 5.6); ax.axis("off")
    ax.text(40, 5.20, "Training Pipeline — Progressive Unfreezing + EMA + SWA + 3-Seed Ensemble",
            ha="center", fontsize=12.5, fontweight="bold", color=C["primary"])

    bg(ax, 0.5, 1.4, 78, 3.5, "#DBEAFE", C["primary"])
    ax.text(40, 4.55, "Single-run training (Epoch 1–60)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    bg(ax, 1, 2.0, 25, 1.4, "#FEF3C7", C["accent"], alpha=0.5)
    ax.text(13.5, 2.70, "Phase 1:  freeze DeBERTa\nbottom 6 layers  (E1–20)",
            ha="center", fontsize=9, fontweight="bold")
    bg(ax, 26.5, 2.0, 28, 1.4, "#DCFCE7", C["success"], alpha=0.5)
    ax.text(40.5, 2.70, "Phase 2:  full fine-tune\n(E20–42)", ha="center",
            fontsize=9, fontweight="bold")
    bg(ax, 55, 2.0, 23, 1.4, "#FFE4E6", C["rose"], alpha=0.5)
    ax.text(66.5, 2.70, "Phase 3:  SWA window\nE42, 44, …, 60  (10 snapshots)",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(40, 1.65,
            "EMA shadow  ($\\mu = 0.9995$):  low-pass filter over $\\theta$ throughout training",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")

    ax.text(0.5, 1.1, "E0", ha="center", fontsize=8, color=C["muted"])
    ax.text(20, 1.1, "E20", ha="center", fontsize=8, color=C["muted"])
    ax.text(42, 1.1, "E42", ha="center", fontsize=8, color=C["muted"])
    ax.text(60, 1.1, "E60", ha="center", fontsize=8, color=C["muted"])
    ax.text(78, 1.1, "$\\theta_{\\mathrm{run}}$",
            ha="center", fontsize=9, color=C["primary"], fontweight="bold")

    bg(ax, 0.5, -1.7, 78, 2.6, "#EDE9FE", C["purple"], alpha=0.30)
    ax.text(40, 0.60, "Multi-Seed Ensemble (3 runs)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    seeds = [("seed = 42",  C["primary"], 12, "1"),
             ("seed = 123", C["accent"], 39, "2"),
             ("seed = 2024", C["success"], 66, "3")]
    for label, col, cx, idx in seeds:
        rbox(ax, cx, -0.20, 18, 0.65, col,
             f"{label}  →  $\\theta_{{\\mathrm{{run}}{idx}}}$",
             fs=9.5, bold=True)
        ax.text(cx, -0.85, "(same recipe, different random seed)",
                ha="center", fontsize=8, color=C["muted"], style="italic")
    arr(ax, 12, -1.20, 40, -1.45, color=C["primary"], lw=1.2, hw=0.12)
    arr(ax, 39, -1.20, 40, -1.45, color=C["accent"], lw=1.2, hw=0.12)
    arr(ax, 66, -1.20, 40, -1.45, color=C["success"], lw=1.2, hw=0.12)
    rbox(ax, 40, -1.65, 32, 0.50, C["danger"],
         "$\\theta_{\\mathrm{final}}$  =  parameter-space average of 3 runs",
         fs=10, bold=True)

    save(fig, "v5_fig_train_timeline")


# ════════════════════════════════════════════════════════════════════════════
# FIG inference — English
# ════════════════════════════════════════════════════════════════════════════
def fig_inference():
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(7, 6.65, "Zero-Leakage Inference — TTA × 5  +  3-Seed Ensemble  +  Reg-Cls Fusion",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])
    ax.text(7, 6.25,
            "Three layers of variance reduction:  MC-Dropout (TTA)  →  cross-seed avg.  →  probability fusion",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Stage 1: TTA
    bg(ax, 0.3, 4.0, 4.0, 1.9, "#DBEAFE", C["primary"])
    ax.text(2.3, 5.65, "Stage 1:  TTA × 5", ha="center",
            fontsize=11, fontweight="bold", color=C["primary"])
    for i in range(5):
        rbox(ax, 1.0 + i * 0.65, 4.85, 0.5, 0.40, C["primary"],
             f"$P_{{{i+1}}}$", fs=8, bold=True)
    ax.text(2.3, 4.30, "MC-Dropout: 5 forwards  →  average",
            ha="center", fontsize=8.5, fontweight="bold", color=C["primary"])

    # Stage 2: 3-seed
    bg(ax, 5.0, 4.0, 4.0, 1.9, "#FEF3C7", C["accent"])
    ax.text(7.0, 5.65, "Stage 2:  3-Seed Ensemble", ha="center",
            fontsize=11, fontweight="bold", color=C["accent"])
    seeds = [("s=42", C["primary"]), ("s=123", C["accent"]), ("s=2024", C["success"])]
    for i, (lbl, col) in enumerate(seeds):
        rbox(ax, 5.5 + i * 1.10, 4.85, 0.95, 0.45, col, lbl, fs=8, bold=True)
    ax.text(7.0, 4.30, "Arithmetic mean of 3 run logits",
            ha="center", fontsize=8.5, fontweight="bold", color=C["accent"])

    # Stage 3: fusion
    bg(ax, 9.7, 4.0, 4.0, 1.9, "#DCFCE7", C["success"])
    ax.text(11.7, 5.65, "Stage 3:  Reg-Cls Probability Fusion",
            ha="center", fontsize=11, fontweight="bold", color=C["success"])
    ax.text(11.7, 5.10,
            "$p_{\\mathrm{cls}} \\;\\otimes\\; p_{\\mathrm{reg}}$  (geom. mean in log-space)",
            ha="center", fontsize=9, fontweight="bold", color=C["success"])
    ax.text(11.7, 4.55, "$\\alpha = 0.65$,    $\\sigma = 0.65$    (a priori)",
            ha="center", fontsize=8.5, color=C["muted"])

    arr(ax, 4.3, 4.95, 5.0, 4.95, lw=1.6)
    arr(ax, 9.0, 4.95, 9.7, 4.95, lw=1.6)

    # Bottom: detailed flow
    bg(ax, 0.3, 0.4, 13.4, 3.0, "#EDE9FE", C["purple"])
    ax.text(7, 3.20, "Detailed inference pipeline",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    flow = [
        ("Test  $x$", C["text"], 1.2),
        ("3 × forward\n(same input,\ndifferent $\\theta_{\\mathrm{run}}$)", C["accent"], 3.4),
        ("Per-run\nTTA × 5", C["primary"], 5.7),
        ("Cross-run\narithmetic mean\n($\\ell 7, \\ell 2, \\mathrm{reg}$)",
         C["secondary"], 8.0),
        ("Fuse $p_{\\mathrm{cls}} + p_{\\mathrm{reg}}$",
         C["success"], 10.4),
        ("$\\hat{y} = \\arg\\max$", C["danger"], 12.7),
    ]
    for label, col, cx in flow:
        rbox(ax, cx, 1.95, 2.0, 1.05, col, label, fs=8.5, bold=True)
    for i in range(len(flow) - 1):
        cx1 = flow[i][2] + 1.0
        cx2 = flow[i+1][2] - 1.0
        arr(ax, cx1, 1.95, cx2, 1.95, lw=1.4, hw=0.15)

    ax.text(7, 0.85,
            "All fusion hyper-parameters  ($\\alpha, \\sigma, T_{\\mathrm{cls}}$)  are set a priori — "
            "no test-set statistics involved (zero data leakage).",
            ha="center", fontsize=9, color=C["text"],
            bbox=dict(boxstyle="round,pad=0.40", fc="#F9FAFB", ec=C["danger"], lw=1.0))

    save(fig, "v5_fig_inference")


# ════════════════════════════════════════════════════════════════════════════
# FIG distribution — English (Train + Test only)
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
    ax.set_title("CMU-MOSI 7-Class Sentiment Distribution  (Train + Test)")
    ax.legend(fontsize=10, loc="upper left")
    shift = abs(TEST_DIST[0] - TRAINVAL_DIST[0])
    ax.annotate(
        f"Class −3:  Test {TEST_DIST[0]:.1f}%   vs.   Train {TRAINVAL_DIST[0]:.1f}%\n"
        f"Distribution shift  =  +{shift:.1f}%",
        xy=(0, max(TRAINVAL_DIST[0], TEST_DIST[0])),
        xytext=(0.6, max(TRAINVAL_DIST) * 0.92),
        fontsize=8.5, color=C["danger"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["danger"], lw=1.2))
    plt.tight_layout()
    save(fig, "v5_fig_distribution")


# ════════════════════════════════════════════════════════════════════════════
# FIG regcls — English
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
               label=f"argmax $p_{{\\mathrm{{cls}}}}$ ({cls_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(a)  $p_{\\mathrm{cls}}$ = softmax($\\ell 7_{\\mathrm{mean}}$ / $T_{\\mathrm{cls}}$)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    ax = axes[1]
    ax.bar(classes_idx, p_reg, color=C["accent"], alpha=0.85)
    ax.axvline(y_shift, color=C["text"], lw=1.5,
               label=f"reg pred = {r_pred:+.2f}")
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title(r"(b)  $p_{\mathrm{reg}}[k] \propto \exp(-(k-r-3)^{2}/2\sigma^{2})$    $\sigma$=0.65")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    ax = axes[2]
    ax.bar(classes_idx, p_final, color=C["success"], alpha=0.85)
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.axvline(int(p_final.argmax()), color=C["danger"], lw=1.5, ls=":",
               label=f"argmax fused ({fused_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title(r"(c)  $\log p_{\mathrm{final}} = \alpha\log p_{\mathrm{cls}}"
                 r" + (1{-}\alpha)\log p_{\mathrm{reg}}$    $\alpha$=0.65")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Reg-Cls Probability Fusion at Inference  (test sample idx=316)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "v5_fig_regcls")


# ════════════════════════════════════════════════════════════════════════════
# FIG radar — English (overall metrics)
# ════════════════════════════════════════════════════════════════════════════
def fig_radar():
    fig = plt.figure(figsize=(13.0, 4.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    cls_names = ["Acc-7", "Acc-2", "F1 (weighted)", "Within-1"]
    cls_values = [ACC7_FUSED, ACC2, F1, W1]
    colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    bars = ax1.bar(cls_names, cls_values, color=colors, alpha=0.85)
    ax1.axhline(53.0, ls="--", color=C["muted"], lw=1.5, label="Acc-7 target = 53%")
    for bar, v in zip(bars, cls_values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.2f}%",
                 ha="center", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Four classification metrics")
    ax1.legend(fontsize=9)

    metrics = ["Acc-7", "Acc-2", "F1", "Within-1",
               "Corr × 100", "(1 − MAE/3) × 100"]
    vals = [ACC7_FUSED, ACC2, F1, W1, CORR * 100, (1 - MAE/3) * 100]
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals_plot = vals + [vals[0]]
    angles_plot = angles + [angles[0]]
    ax2.plot(angles_plot, vals_plot, color=C["primary"], lw=2)
    ax2.fill(angles_plot, vals_plot, color=C["primary"], alpha=0.25)
    ax2.set_xticks(angles)
    ax2.set_xticklabels(metrics, fontsize=8.5)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80])
    ax2.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color=C["muted"])
    ax2.set_title("Six normalized metrics (radar)", pad=14)

    fig.suptitle("SACFFinalModel — Overall Performance", fontsize=12.5,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "v5_fig_radar")


if __name__ == "__main__":
    print("Regenerating v5 figures (English labels + true subscripts) ...")
    fig_arch();             print("  ✓ v5_fig_arch")
    fig_pea();              print("  ✓ v5_fig_pea")
    fig_sacf_steps();       print("  ✓ v5_fig_sacf_steps")
    fig_branches();         print("  ✓ v5_fig_branches")
    fig_loss_comp();        print("  ✓ v5_fig_loss_comp")
    fig_train_timeline();   print("  ✓ v5_fig_train_timeline")
    fig_inference();        print("  ✓ v5_fig_inference")
    fig_distribution();     print("  ✓ v5_fig_distribution")
    fig_regcls();           print("  ✓ v5_fig_regcls")
    fig_radar();            print("  ✓ v5_fig_radar")
    print("Done.")
