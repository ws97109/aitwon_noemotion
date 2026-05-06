"""SACF-v2 (DKD+DIST+SORD+RegCls fusion + 2-stage SWA) — Chapter 3 generator.
Run: python3 docs/generate_paper_v2.py
Outputs:
  docs/figures/v2_*.svg / .png  (12 figures)
  docs/SACF_Methodology_Chapter3_v2.docx
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

BASE = Path(__file__).parent
ROOT = BASE.parent
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)
MODEL_DIR = ROOT / "emotion_system" / "models"

# ── load extracted data ─────────────────────────────────────────────────────
D = np.load(str(BASE / "paper_v2_data.npz"))
metrics = D["metrics"]   # [acc7_fused, acc7_raw, acc2, f1, mae, corr, within1]
ACC7_FUSED, ACC7_RAW, ACC2, F1, MAE, CORR, W1 = [float(x) for x in metrics]
PER_CLASS_ACC = D["per_class_acc"]
CLASS_SUPPORT = D["class_support"]
CM = D["confusion_matrix"]
TRAIN_DIST = D["train_dist"]; VAL_DIST = D["val_dist"]
TEST_DIST = D["test_dist"]; TRAINVAL_DIST = D["trainval_dist"]
ITER1_LOSS = D["iter1_loss"]; ITER4_LOSS = D["iter4_loss"]
L7 = D["L7"]; L2 = D["L2"]; R = D["R"]; Y7 = D["y7"]

# ── palette ─────────────────────────────────────────────────────────────────
C = dict(
    primary="#1D4ED8", secondary="#0891B2", accent="#F59E0B",
    danger="#DC2626", success="#10B981", purple="#8B5CF6",
    text="#1F2937", muted="#6B7280", grid="#E5E7EB", bg="#F9FAFB",
    teal="#14B8A6", indigo="#6366F1", rose="#F43F5E", lime="#84CC16",
)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.alpha": 0.5,
    "savefig.dpi": 150, "figure.dpi": 110,
})


def save(fig, name):
    p_svg = OUTDIR / f"{name}.svg"
    p_png = OUTDIR / f"{name}.png"
    fig.savefig(str(p_svg), bbox_inches="tight")
    fig.savefig(str(p_png), bbox_inches="tight")
    plt.close(fig)
    return {"svg": p_svg, "png": p_png}


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
# FIG 1 — Overall architecture
# ════════════════════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 13.5); ax.axis("off")

    # Title bar
    ax.text(7, 13.0,
            "SACF-v2: Multi-Branch Single Model with KD + SORD + Reg-Cls Fusion",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=C["primary"])
    ax.text(7, 12.55, f"Acc-7 = {ACC7_FUSED:.2f}%   |   Acc-2 = {ACC2:.2f}%   |   F1 = {F1:.2f}%   |   MAE = {MAE:.4f}   |   Corr = {CORR:.4f}",
            ha="center", va="center", fontsize=10.5, color=C["muted"])

    # ── Row 1: Inputs ───────────────────────────────────────────────────────
    bg(ax, 0.4, 11.0, 13.2, 1.2, "#FEF3C7", C["accent"])
    ax.text(7, 12.0, "(i)  Multimodal Inputs", ha="center", fontsize=10.5,
            color=C["accent"], fontweight="bold")
    rbox(ax, 2.5, 11.4, 3.0, 0.55, C["primary"],
         "Text  (raw_text + task prompt)", fs=9.5, bold=True)
    rbox(ax, 7.0, 11.4, 3.0, 0.55, C["accent"],
         "Audio  (COVAREP, 5-d, ≤375 frames)", fs=9.5, bold=True)
    rbox(ax, 11.5, 11.4, 3.0, 0.55, C["success"],
         "Vision  (FACET, 20-d, ≤500 frames)", fs=9.5, bold=True)

    # ── Row 2: Shared encoders ──────────────────────────────────────────────
    bg(ax, 0.4, 9.2, 13.2, 1.5, "#DBEAFE", C["primary"])
    ax.text(7, 10.55, "(ii)  Shared Encoders", ha="center", fontsize=10.5,
            color=C["primary"], fontweight="bold")
    rbox(ax, 2.5, 9.85, 3.0, 0.7, C["primary"],
         "DeBERTa-v3-large\n24 layers · d=1024 · ~400M", fs=9, bold=True)
    rbox(ax, 7.0, 9.85, 3.0, 0.7, C["accent"],
         "BiLSTM-Audio\n2 layers · 5→128", fs=9, bold=True)
    rbox(ax, 11.5, 9.85, 3.0, 0.7, C["success"],
         "BiLSTM-Vision\n2 layers · 20→128", fs=9, bold=True)

    # arrows row1 → row2
    for cx in (2.5, 7.0, 11.5):
        arr(ax, cx, 11.13, cx, 10.20)

    # ── Row 3: 4 parallel branches ──────────────────────────────────────────
    bg(ax, 0.4, 5.4, 13.2, 3.5, "#EDE9FE", C["purple"])
    ax.text(7, 8.75, "(iii)  4 Parallel Branches  (per-branch dropout = [0.10, 0.20, 0.30, 0.40])",
            ha="center", fontsize=10.5, color=C["purple"], fontweight="bold")

    branch_colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    branch_dropouts = [0.10, 0.20, 0.30, 0.40]
    for i in range(4):
        cx = 2.0 + i * 3.0
        # branch container
        bg(ax, cx - 1.30, 5.6, 2.6, 2.95, "#FFFFFF", branch_colors[i], lw=1.6, alpha=0.6)
        ax.text(cx, 8.30, f"Branch {i+1}  (p_drop={branch_dropouts[i]})",
                ha="center", fontsize=9.5, fontweight="bold", color=branch_colors[i])
        rbox(ax, cx, 7.85, 2.4, 0.5, branch_colors[i], "PEA  (gate σ)", fs=8.5)
        rbox(ax, cx, 7.20, 2.4, 0.5, branch_colors[i], "SACF  (top-K + cross-modal)", fs=8.5)
        rbox(ax, cx, 6.55, 2.4, 0.5, branch_colors[i], "Proj  (1024 → 512)", fs=8.5)
        rbox(ax, cx, 5.90, 2.4, 0.55, "#FFFFFF", "cls7 / cls2 / reg",
             fs=8.5, tc=branch_colors[i], ec=branch_colors[i], bold=True)

    # arrows from shared encoders to each branch (just 3 representative)
    for cx in (2.0, 5.0, 8.0, 11.0):
        arr(ax, 2.5, 9.50, cx, 8.50, color=C["primary"], lw=0.8)
        arr(ax, 7.0, 9.50, cx, 8.50, color=C["accent"], lw=0.8)
        arr(ax, 11.5, 9.50, cx, 8.50, color=C["success"], lw=0.8)

    # ── Row 4: Mean aggregation ─────────────────────────────────────────────
    bg(ax, 4.5, 4.1, 5.0, 0.9, "#FFE4E6", C["rose"])
    rbox(ax, 7.0, 4.55, 4.6, 0.65, C["rose"],
         "Mean-of-Branches  →  (cls7_mean, cls2_mean, reg_mean)",
         fs=9.5, bold=True)
    for i in range(4):
        cx = 2.0 + i * 3.0
        arr(ax, cx, 5.60, 7.0, 4.95, color=branch_colors[i], lw=0.8)

    # ── Row 5: Inference fusion ─────────────────────────────────────────────
    bg(ax, 0.4, 1.8, 13.2, 2.0, "#DCFCE7", C["success"])
    ax.text(7, 3.65, "(iv)  Inference: Reg-Cls Probability Fusion (α=0.65, σ=0.65 — chosen a priori)",
            ha="center", fontsize=10.5, color=C["success"], fontweight="bold")
    rbox(ax, 3.0, 2.85, 3.4, 0.7, C["primary"],
         "p_cls = softmax(cls7_logits / T)", fs=9, bold=True)
    rbox(ax, 7.0, 2.85, 3.4, 0.7, C["accent"],
         "p_reg[k] ∝ exp(−(k−r)² / 2σ²)", fs=9, bold=True)
    rbox(ax, 11.0, 2.85, 3.4, 0.7, C["success"],
         "log p ← α·log p_cls + (1−α)·log p_reg", fs=9, bold=True)
    arr(ax, 7.0, 4.20, 7.0, 3.30, lw=1.5)
    arr(ax, 3.0, 2.45, 5.5, 2.10)
    arr(ax, 7.0, 2.45, 7.0, 2.10)
    arr(ax, 11.0, 2.45, 8.5, 2.10)

    # ── Final prediction ────────────────────────────────────────────────────
    rbox(ax, 7.0, 1.30, 5.4, 0.75, C["danger"],
         f"ŷ = argmax(p_final)        Acc-7 = {ACC7_FUSED:.2f}%",
         fs=11, bold=True)

    # ── Legend ──────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=C["primary"], label="Text (DeBERTa)"),
        mpatches.Patch(color=C["accent"], label="Audio (BiLSTM)"),
        mpatches.Patch(color=C["success"], label="Vision (BiLSTM)"),
        mpatches.Patch(color=C["rose"], label="Aggregation"),
        mpatches.Patch(color=C["purple"], label="Parallel Branches"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              ncol=5, fontsize=8.5, frameon=False)

    fig.tight_layout()
    return save(fig, "v2_fig1_architecture")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Knowledge Distillation Pipeline
# ════════════════════════════════════════════════════════════════════════════
def fig2_kd_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.text(7, 7.1, "Knowledge Distillation:  12-Model Ensemble Teacher  →  Single Student",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])

    # Teacher zone
    bg(ax, 0.3, 3.5, 5.5, 3.2, "#FEE2E2", C["danger"])
    ax.text(3.05, 6.30, "Teacher  (12-model logit ensemble)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["danger"])
    teachers = ["v59-s42", "v59-s123", "v59-s2024",
                "v60_baseline-s42", "v60_baseline-s123", "v60_baseline-s2024",
                "v60_mmaffin-s42", "v60_mmaffin-s123", "v60_mmaffin-s2024",
                "v63-s101", "v63-s202", "v63-s303"]
    for i, t in enumerate(teachers):
        rx, ry = 0.7 + (i % 4) * 1.30, 5.55 - (i // 4) * 0.55
        rbox(ax, rx + 0.55, ry, 1.20, 0.42, C["danger"], t, fs=7.5)
    rbox(ax, 3.05, 3.85, 4.6, 0.5, C["danger"],
         "logits_teacher ∈ ℝ^{1513×7}  (only train+val)",
         fs=9, bold=True)

    # Arrow teacher → KD losses
    arr(ax, 3.05, 3.50, 7.0, 2.85, lw=2.0, color=C["danger"])

    # KD loss zone
    bg(ax, 5.8, 1.2, 4.1, 2.6, "#DBEAFE", C["primary"])
    ax.text(7.85, 3.45, "Distillation Losses",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    rbox(ax, 7.85, 2.85, 3.6, 0.55, C["primary"],
         "DKD  = α·TCKD + β·NCKD  (β=8)", fs=9, bold=True)
    rbox(ax, 7.85, 2.20, 3.6, 0.55, C["secondary"],
         "DIST  =  1 − corr(p_S, p_T)_inter\n         + 1 − corr(p_S, p_T)_intra",
         fs=8.5, bold=True)
    rbox(ax, 7.85, 1.50, 3.6, 0.55, C["text"],
         "T = 4   (softmax temperature)", fs=9)

    # Student
    bg(ax, 10.4, 3.5, 3.4, 3.2, "#DBEAFE", C["primary"])
    ax.text(12.10, 6.30, "Student (this work)", ha="center",
            fontsize=10.5, fontweight="bold", color=C["primary"])
    rbox(ax, 12.10, 5.55, 2.8, 0.5, C["primary"],
         "DeBERTa-v3-large", fs=9, bold=True)
    rbox(ax, 12.10, 4.95, 2.8, 0.5, C["accent"], "+ BiLSTM × 2", fs=9)
    rbox(ax, 12.10, 4.35, 2.8, 0.5, C["purple"], "+ 4 Branches", fs=9)
    rbox(ax, 12.10, 3.75, 2.8, 0.5, C["success"], "+ Reg-Cls Fusion", fs=9)
    arr(ax, 9.90, 2.85, 12.10, 3.45, lw=1.6, color=C["primary"])

    # Bottom: combined objective
    bg(ax, 0.3, 0.1, 13.4, 0.95, "#F3F4F6", C["text"], alpha=0.45)
    ax.text(7.0, 0.55,
            "Total Loss  =  0.5 · L_mean(SORD)  +  0.5 · L_per_branch(SORD)  "
            "+  1.0 · L_DKD  +  1.5 · L_DIST  +  0.02 · L_diversity",
            ha="center", fontsize=10.5, color=C["text"], fontweight="bold")

    fig.tight_layout()
    return save(fig, "v2_fig2_kd_pipeline")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — DKD decomposition (TCKD + NCKD)
# ════════════════════════════════════════════════════════════════════════════
def fig3_dkd_decomp():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # Synthetic example
    K = 7
    classes = list(range(-3, 4))
    target = 4  # class index "+1"
    teacher_logits = np.array([-1.5, -0.5, 1.0, 2.0, 3.5, 2.0, -0.5])
    student_logits = np.array([-1.0, -0.3, 0.7, 1.2, 2.0, 0.8, -0.5])
    T = 4.0
    pT = np.exp(teacher_logits / T); pT /= pT.sum()
    pS = np.exp(student_logits / T); pS /= pS.sum()

    # Panel 1: full distributions
    ax = axes[0]
    x = np.arange(K)
    w = 0.36
    ax.bar(x - w/2, pT, w, color=C["danger"], label="Teacher (T=4)", alpha=0.85)
    ax.bar(x + w/2, pS, w, color=C["primary"], label="Student (T=4)", alpha=0.85)
    ax.axvspan(target - 0.5, target + 0.5, color=C["success"], alpha=0.18, label="Target class")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_xlabel("Class label"); ax.set_ylabel("Softmax probability")
    ax.set_title("Full softmax distribution")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: TCKD — binary {target, non-target}
    ax = axes[1]
    bin_T = np.array([pT[target], 1 - pT[target]])
    bin_S = np.array([pS[target], 1 - pS[target]])
    xs = ["target", "non-target"]
    ax.bar(np.arange(2) - w/2, bin_T, w, color=C["danger"], alpha=0.85, label="Teacher")
    ax.bar(np.arange(2) + w/2, bin_S, w, color=C["primary"], alpha=0.85, label="Student")
    ax.set_xticks(np.arange(2)); ax.set_xticklabels(xs)
    ax.set_ylabel("Probability mass")
    ax.set_title("TCKD  =  KL(target / non-target)")
    ax.legend(fontsize=8)

    # Panel 3: NCKD — non-target only renormalized
    ax = axes[2]
    pT_nc = pT.copy(); pT_nc[target] = 0; pT_nc /= pT_nc.sum()
    pS_nc = pS.copy(); pS_nc[target] = 0; pS_nc /= pS_nc.sum()
    pT_nc_show = pT_nc.copy(); pS_nc_show = pS_nc.copy()
    ax.bar(x - w/2, pT_nc_show, w, color=C["danger"], alpha=0.85, label="Teacher (non-target only)")
    ax.bar(x + w/2, pS_nc_show, w, color=C["primary"], alpha=0.85, label="Student (non-target only)")
    ax.axvspan(target - 0.5, target + 0.5, color="#9CA3AF", alpha=0.18, label="(masked)")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_xlabel("Class label"); ax.set_ylabel("Renormalized probability")
    ax.set_title("NCKD  =  KL on non-target classes  (β=8)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Decoupled KD (DKD):  L_DKD = α·TCKD + β·NCKD",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, "v2_fig3_dkd_decomp")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — SORD soft labels
# ════════════════════════════════════════════════════════════════════════════
def fig4_sord():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    K = 7
    classes_label = list(range(-3, 4))
    # Left: Gaussian soft target visualization for different y
    ax = axes[0]
    sigma = 1.0
    centers = np.arange(K)
    for y, color in zip([0, 3, 6],
                        [C["danger"], C["accent"], C["success"]]):
        d2 = (centers - y) ** 2
        # SORD: softmax(-d²/σ²)
        soft = np.exp(-d2 / sigma**2); soft /= soft.sum()
        ax.plot(classes_label, soft, marker="o", lw=2, color=color,
                label=f"y = {classes_label[y]}  (class index {y})")
    ax.set_xlabel("Class label  (-3..+3)")
    ax.set_ylabel("SORD soft target probability")
    ax.set_title(f"SORD soft labels  (σ = {sigma})")
    ax.set_xticks(classes_label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.5)

    # Right: comparison vs one-hot
    ax = axes[1]
    y = 4
    # one-hot
    oh = np.zeros(K); oh[y] = 1.0
    # SORD
    d2 = (np.arange(K) - y) ** 2
    soft = np.exp(-d2 / sigma**2); soft /= soft.sum()
    # label smoothing (typical eps=0.05)
    ls = np.full(K, 0.05/(K-1)); ls[y] = 1 - 0.05
    x_pos = np.arange(K)
    w = 0.27
    ax.bar(x_pos - w, oh, w, color=C["muted"], alpha=0.7, label="One-hot")
    ax.bar(x_pos, ls, w, color=C["primary"], alpha=0.85,
           label="Label smoothing (ε=0.05)")
    ax.bar(x_pos + w, soft, w, color=C["success"], alpha=0.85,
           label=f"SORD soft (σ={sigma})")
    ax.set_xticks(x_pos); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Target probability")
    ax.set_title(f"Target encoding for y = {classes_label[y]} (+1)")
    ax.legend(fontsize=8)

    fig.suptitle("Soft Ordinal Regression Distribution (SORD)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, "v2_fig4_sord")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Reg-Cls fusion at inference
# ════════════════════════════════════════════════════════════════════════════
def fig5_regcls_fusion():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    K = 7
    classes_label = list(range(-3, 4))
    # Pick a real test sample where fusion improves (cls is close, reg helps)
    # Find a sample where pred_raw differs from pred_fused
    pred_raw = L7.argmax(1)
    import scipy.special
    p_cls_all = scipy.special.softmax(L7, axis=1)
    centers = np.arange(K).astype(np.float32)
    sigma_inf = 0.65
    p_reg_all = np.exp(-((centers[None, :] - (R[:, None] + 3))**2) / (2 * sigma_inf**2))
    p_reg_all = p_reg_all / p_reg_all.sum(1, keepdims=True)
    log_p = 0.65 * np.log(p_cls_all + 1e-9) + 0.35 * np.log(p_reg_all + 1e-9)
    log_p -= log_p.max(1, keepdims=True)
    p_fused_all = np.exp(log_p); p_fused_all /= p_fused_all.sum(1, keepdims=True)
    pred_fused = p_fused_all.argmax(1)
    # find sample where fused matches truth but raw doesn't
    idx = None
    for i in range(len(Y7)):
        if pred_fused[i] == Y7[i] and pred_raw[i] != Y7[i] and Y7[i] in (1,2,3,4,5):
            idx = i; break
    if idx is None: idx = 0
    p_cls = p_cls_all[idx]; p_reg = p_reg_all[idx]; p_fused = p_fused_all[idx]
    truth = Y7[idx]
    r_val = R[idx]

    ax = axes[0]
    ax.bar(classes_label, p_cls, color=C["primary"], alpha=0.85)
    ax.axvline(classes_label[truth], color=C["success"], lw=2, ls="--", label=f"Ground truth ({classes_label[truth]})")
    ax.axvline(classes_label[p_cls.argmax()], color=C["danger"], lw=1.5, ls=":", label=f"argmax cls ({classes_label[p_cls.argmax()]})")
    ax.set_xticks(classes_label); ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(a)  p_cls = softmax(cls7_logits)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(classes_label, p_reg, color=C["accent"], alpha=0.85)
    ax.axvline(r_val, color=C["text"], lw=2, ls="-", label=f"reg pred = {r_val:.2f}")
    ax.axvline(classes_label[truth], color=C["success"], lw=2, ls="--", label=f"truth ({classes_label[truth]})")
    ax.set_xticks(classes_label); ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title(f"(b)  p_reg ∝ exp(−(k−r)²/2σ²)   σ=0.65")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(classes_label, p_fused, color=C["success"], alpha=0.9)
    ax.axvline(classes_label[truth], color=C["success"], lw=2, ls="--", label=f"truth ({classes_label[truth]})")
    ax.axvline(classes_label[p_fused.argmax()], color=C["danger"], lw=1.5, ls=":",
               label=f"argmax fused ({classes_label[p_fused.argmax()]})")
    ax.set_xticks(classes_label); ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(c)  log p_final = α·log p_cls + (1−α)·log p_reg   α=0.65")
    ax.legend(fontsize=8)

    fig.suptitle(f"Reg-Cls Probability Fusion at Inference  (test sample idx={idx})",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, "v2_fig5_regcls_fusion")


# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — Two-stage training timeline
# ════════════════════════════════════════════════════════════════════════════
def fig6_training_timeline():
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.set_xlim(0, 80); ax.set_ylim(0, 5); ax.axis("off")

    # Header
    ax.text(40, 4.7, "Two-Stage Training Pipeline (single .pt artifact)",
            ha="center", fontsize=12.5, fontweight="bold", color=C["primary"])

    # Stage 1 — iter1 (60 ep)
    bg(ax, 0.5, 0.5, 60, 3.6, "#DBEAFE", C["primary"])
    ax.text(30, 3.85, "Stage 1: iter1 (Epoch 1–60)  —  full training from scratch",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    # phases inside iter1
    bg(ax, 1, 1.2, 19, 1.4, "#FEF3C7", C["accent"], alpha=0.5)
    ax.text(10.5, 1.9, "Phase 1: freeze DeBERTa\nlayers 0–5 (E1–20)",
            ha="center", fontsize=9, fontweight="bold")
    bg(ax, 20.5, 1.2, 21.5, 1.4, "#DCFCE7", C["success"], alpha=0.5)
    ax.text(31.25, 1.9, "Phase 2: full fine-tune\n(E20–60)", ha="center",
            fontsize=9, fontweight="bold")
    bg(ax, 42.5, 1.2, 18, 1.4, "#FFE4E6", C["rose"], alpha=0.5)
    ax.text(51.5, 1.9, "Phase 3: SWA window\nE42, 44, 46, …, 60  (10 snapshots)",
            ha="center", fontsize=9, fontweight="bold")

    # Stage 2 — iter4 (14 ep continuation)
    bg(ax, 61, 0.5, 18, 3.6, "#FFE4E6", C["rose"])
    ax.text(70, 3.85, "Stage 2: iter4 (Epoch 61–74)  —  load iter1, polish",
            ha="center", fontsize=10.5, fontweight="bold", color=C["rose"])
    bg(ax, 61.5, 1.2, 16.5, 1.4, "#FCE7F3", C["rose"], alpha=0.5)
    ax.text(69.75, 1.9, "low LR (×¼) + heavy SWA\nseed 777, 12 SWA snapshots",
            ha="center", fontsize=9, fontweight="bold", color=C["rose"])

    # ticks and final
    ax.text(0.5, 0.05, "E0", ha="center", fontsize=8, color=C["muted"])
    ax.text(60, 0.05, "E60", ha="center", fontsize=8, color=C["muted"])
    ax.text(78.5, 0.05, "final", ha="center", fontsize=8, color=C["muted"])

    # final outcome
    ax.text(40, -0.1, f"→  sacf_final.pt   (1.65 GB)   |   Acc-7 = {ACC7_FUSED:.2f}%",
            ha="center", fontsize=11, fontweight="bold", color=C["danger"])
    fig.tight_layout()
    return save(fig, "v2_fig6_training_timeline")


# ════════════════════════════════════════════════════════════════════════════
# FIG 7 — Loss curves
# ════════════════════════════════════════════════════════════════════════════
def fig7_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

    # Iter1
    ax = axes[0]
    epochs = np.arange(1, len(ITER1_LOSS) + 1)
    valid = ~np.isnan(ITER1_LOSS)
    ax.plot(epochs[valid], ITER1_LOSS[valid], marker="o", color=C["primary"],
            lw=2, ms=5, label="iter1 train loss")
    ax.axvspan(1, 20, color=C["accent"], alpha=0.15, label="Phase 1: layer freeze")
    ax.axvspan(20, 42, color=C["success"], alpha=0.10, label="Phase 2: full FT")
    ax.axvspan(42, 60, color=C["rose"], alpha=0.18, label="Phase 3: SWA window")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
    ax.set_title(f"iter1 training  (60 epochs, AMP, batch=8, seed=42)")
    ax.legend(fontsize=8.5, loc="upper right"); ax.set_xlim(0, 61)

    # Iter4
    ax = axes[1]
    epochs2 = np.arange(1, len(ITER4_LOSS) + 1)
    ax.plot(epochs2, ITER4_LOSS, marker="s", color=C["rose"], lw=2, ms=6,
            label="iter4 train loss")
    ax.axhline(np.nanmean(ITER4_LOSS), ls="--", color=C["muted"],
               label=f"iter4 mean = {np.nanmean(ITER4_LOSS):.3f}")
    ax.set_xlabel("Epoch  (within stage 2)")
    ax.set_ylabel("Total loss")
    ax.set_title("iter4  (14 epochs continuation, low LR, 12 SWA snapshots)")
    ax.legend(fontsize=8.5)
    ax.set_xticks(epochs2)

    fig.suptitle("Training Loss Trajectories",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, "v2_fig7_loss_curves")


# ════════════════════════════════════════════════════════════════════════════
# FIG 8 — Class distributions
# ════════════════════════════════════════════════════════════════════════════
def fig8_class_distribution():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    classes_label = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]
    x = np.arange(7); w = 0.22
    ax.bar(x - 1.5*w, TRAIN_DIST, w, color=C["primary"], label=f"Train (n=1,284)", alpha=0.9)
    ax.bar(x - 0.5*w, VAL_DIST, w, color=C["accent"], label=f"Valid (n=229)", alpha=0.9)
    ax.bar(x + 0.5*w, TRAINVAL_DIST, w, color=C["success"], label=f"Train+Val (n=1,513)", alpha=0.9)
    ax.bar(x + 1.5*w, TEST_DIST, w, color=C["rose"], label=f"Test (n=686)", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Sentiment class label")
    ax.set_ylabel("Class proportion (%)")
    ax.set_title("CMU-MOSI 7-Class Sentiment Distribution Across Splits")
    ax.legend(fontsize=9, loc="upper left")
    # Highlight the distribution shift
    ax.text(0.5, max(TEST_DIST), f"shift\nTest:{TEST_DIST[0]:.1f}%\nTrain:{TRAIN_DIST[0]:.1f}%",
            fontsize=8, ha="center", color=C["danger"], fontweight="bold")
    fig.tight_layout()
    return save(fig, "v2_fig8_class_distribution")


# ════════════════════════════════════════════════════════════════════════════
# FIG 9 — Confusion matrix
# ════════════════════════════════════════════════════════════════════════════
def fig9_confusion():
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    classes_label = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]
    cm = CM.astype(float)
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(classes_label); ax.set_yticklabels(classes_label)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Test Confusion Matrix  (Acc-7 fused = {ACC7_FUSED:.2f}%)")
    for i in range(7):
        for j in range(7):
            v = cm_n[i, j]
            color = "white" if v > 0.45 else C["text"]
            ax.text(j, i, f"{int(cm[i,j])}\n({v*100:.0f}%)", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
    fig.tight_layout()
    return save(fig, "v2_fig9_confusion")


# ════════════════════════════════════════════════════════════════════════════
# FIG 10 — Per-class accuracy bars
# ════════════════════════════════════════════════════════════════════════════
def fig10_per_class_acc():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    classes_label = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]
    x = np.arange(7)
    bars = ax.bar(x, PER_CLASS_ACC, color=[C["danger"], C["rose"], C["accent"],
                                            C["muted"], C["lime"], C["success"], C["secondary"]])
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{PER_CLASS_ACC[i]:.1f}%\n(n={CLASS_SUPPORT[i]})",
                ha="center", fontsize=9, color=C["text"])
    ax.axhline(ACC7_FUSED, color=C["primary"], lw=2, ls="--",
               label=f"Overall Acc-7 = {ACC7_FUSED:.2f}%")
    ax.axhline(100/7, color=C["muted"], lw=1.2, ls=":",
               label=f"Random baseline = {100/7:.1f}%")
    ax.set_xticks(x); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Sentiment class")
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_title("Per-Class Accuracy on CMU-MOSI Test (n=686)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, max(PER_CLASS_ACC) + 18)
    fig.tight_layout()
    return save(fig, "v2_fig10_per_class_acc")


# ════════════════════════════════════════════════════════════════════════════
# FIG 11 — Final metrics radar
# ════════════════════════════════════════════════════════════════════════════
def fig11_metrics_radar():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Left: bar
    ax = axes[0]
    labels = ["Acc-7", "Acc-2", "F1", "Within-1"]
    vals = [ACC7_FUSED, ACC2, F1, W1]
    colors = [C["primary"], C["success"], C["accent"], C["secondary"]]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.2f}%",
                ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Classification Metrics  (higher better)")
    ax.axhline(53.0, color=C["danger"], ls="--", lw=1.4, alpha=0.7, label="53% target")
    ax.legend(fontsize=8)

    # Right: regression metrics + radar
    ax = axes[1]
    # We use a polar plot for normalized metrics
    cats = ["Acc-7\n(/100)", "Acc-2\n(/100)", "F1\n(/100)", "Within-1\n(/100)",
            "Corr\n(×100)", "1−MAE/3\n(×100)"]
    vals2 = [ACC7_FUSED, ACC2, F1, W1, CORR * 100, (1 - MAE / 3) * 100]
    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
    vals2_c = vals2 + [vals2[0]]; angles_c = angles + [angles[0]]
    ax.remove()
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.plot(angles_c, vals2_c, color=C["primary"], lw=2, marker="o")
    ax2.fill(angles_c, vals2_c, color=C["primary"], alpha=0.25)
    ax2.set_xticks(angles); ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_ylim(0, 100)
    ax2.set_title("Holistic Performance Radar", pad=18)
    ax2.grid(True, alpha=0.5)

    fig.tight_layout()
    return save(fig, "v2_fig11_metrics")


# ════════════════════════════════════════════════════════════════════════════
# FIG 12 — Per-branch breakdown (real numbers from L7_b1..b4)
# ════════════════════════════════════════════════════════════════════════════
def fig12_per_branch():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    branches = []
    for i in range(1, 5):
        Lb = D[f"L7_b{i}"]
        acc = (Lb.argmax(1) == Y7).mean() * 100
        branches.append(acc)
    mean_acc = (L7.argmax(1) == Y7).mean() * 100
    fused_acc = ACC7_FUSED

    labels = [f"Branch {i+1}\n(p_drop={d})" for i, d in enumerate([0.10, 0.20, 0.30, 0.40])]
    labels.extend(["Mean of\nbranches", "+ Reg-Cls\nfusion (final)"])
    vals = branches + [mean_acc, fused_acc]
    colors = [C["primary"], C["accent"], C["success"], C["rose"], C["purple"], C["danger"]]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f}%",
                ha="center", fontweight="bold", fontsize=10)
    ax.axhline(53.0, color=C["danger"], ls="--", lw=1.4, label="53% target")
    ax.set_ylabel("Acc-7  (test set, n=686)")
    ax.set_title("Per-Branch  vs.  Internal Ensemble  vs.  Final Fused Prediction")
    ax.legend(fontsize=9)
    ax.set_ylim(min(vals) - 1.5, max(vals) + 2)
    fig.tight_layout()
    return save(fig, "v2_fig12_per_branch")


# ════════════════════════════════════════════════════════════════════════════
# Run all figures
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Generating figures ===")
PATHS = {}
PATHS["fig1"] = fig1_architecture()
PATHS["fig2"] = fig2_kd_pipeline()
PATHS["fig3"] = fig3_dkd_decomp()
PATHS["fig4"] = fig4_sord()
PATHS["fig5"] = fig5_regcls_fusion()
PATHS["fig6"] = fig6_training_timeline()
PATHS["fig7"] = fig7_loss_curves()
PATHS["fig8"] = fig8_class_distribution()
PATHS["fig9"] = fig9_confusion()
PATHS["fig10"] = fig10_per_class_acc()
PATHS["fig11"] = fig11_metrics_radar()
PATHS["fig12"] = fig12_per_branch()
for k, v in PATHS.items():
    print(f"  {k}  →  {v['svg'].name}, {v['png'].name}")


# ════════════════════════════════════════════════════════════════════════════
# WORD DOCUMENT — Chapter 3
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Generating docx ===")
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()
for sec in doc.sections:
    sec.top_margin=Cm(2.54); sec.bottom_margin=Cm(2.54)
    sec.left_margin=Cm(3.18); sec.right_margin=Cm(3.18)
doc.styles["Normal"].font.name = "Times New Roman"
doc.styles["Normal"].font.size = Pt(12)


def heading(d, text, level=1):
    p = d.add_paragraph()
    r = p.add_run(text); r.bold = True
    r.font.name = "Times New Roman"
    sizes = {1: 14, 2: 12, 3: 11}
    colors = {1: (0x1E, 0x40, 0xAF), 2: (0x1D, 0x4E, 0xD8), 3: (0x0F, 0x76, 0x6E)}
    r.font.size = Pt(sizes[level])
    r.font.color.rgb = RGBColor(*colors[level])
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def body(d, text):
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.75)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text); r.font.name = "Times New Roman"; r.font.size = Pt(12)


def caption(d, num, title, desc):
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(f"圖 {num}.  "); r1.bold = True; r1.font.size = Pt(10)
    r2 = p.add_run(title); r2.bold = True; r2.font.size = Pt(10)
    r3 = p.add_run(f"\n{desc}"); r3.font.size = Pt(9.5)
    p.paragraph_format.space_after = Pt(14)


def fig_block(d, img_path, num, title, desc, width=Inches(5.9)):
    d.add_paragraph()
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(img_path), width=width)
    caption(d, num, title, desc)


def add_table(d, headers, rows):
    t = d.add_table(rows=1+len(rows), cols=len(headers))
    t.style = "Table Grid"
    hr = t.rows[0]
    for i, h in enumerate(headers):
        c = hr.cells[i]; c.text = h
        r = c.paragraphs[0].runs[0]; r.bold = True; r.font.size = Pt(9.5)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        tc = c._tc; tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "1D4ED8")
        tcPr.append(shd); r.font.color.rgb = RGBColor(255, 255, 255)
    for ri, row_data in enumerate(rows):
        row = t.rows[ri + 1]
        for ci, ct in enumerate(row_data):
            c = row.cells[ci]; c.text = ct
            if c.paragraphs[0].runs:
                c.paragraphs[0].runs[0].font.size = Pt(9)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if ri % 2 == 0:
                tc = c._tc; tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "DBEAFE")
                tcPr.append(shd)
    d.add_paragraph()


# ─────────────────────────────────────────────────────────────────────────────
heading(doc, "第三章   研究方法", 1)

# ── 3.1 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.1   研究框架概覽", 2)
body(doc,
    "本章提出 SACFFinalModel — 一個專為多模態情感分析（Multimodal Sentiment Analysis, MSA）"
    "任務所設計之「多分支單一模型」（Multi-Branch Single Model）。"
    "本模型在架構層面就是單一 nn.Module、單一 forward、單一 .pt 權重檔；"
    "其內部以 4 條並行分支提供集成多樣性，相較於傳統「先訓練多個獨立模型再事後融合」之集成方法，"
    "本架構同時兼具：（1）部署簡單—僅需單一 1.65 GB 之權重檔；"
    "（2）推斷效率—DeBERTa 共享計算僅執行一次；（3）真正的「單一模型」可解釋與可重現性。")
body(doc,
    "為將「多模型集成」之效益完整壓縮進此單一模型，本研究進一步引入下列五項關鍵技術："
    "（1）解耦知識蒸餾（Decoupled Knowledge Distillation, DKD）— 以 12 個獨立模型之 logit 平均作為教師，"
    "將其暗知識（dark knowledge）注入學生；"
    "（2）DIST 相關係數蒸餾 — 以皮爾森相關係數為損失，補強學生與強教師之間的關係結構差距；"
    "（3）軟序數標籤（Soft Ordinal Regression Distribution, SORD）— 利用 −3..+3 七類別之序數性質，"
    "將 one-hot 取代為高斯型軟標籤；"
    "（4）回歸—分類融合推斷（Reg-Cls Probability Fusion）— 將回歸頭的標量預測轉為機率分布，"
    "與分類頭以幾何平均融合；"
    "（5）兩階段訓練（iter1 + iter4）— 先進行 60 epoch 之全模型訓練，再以低學習率與密集 SWA 快照"
    "進行 14 epoch 之精修，達成穩定且可重現之最終模型。")
body(doc,
    "本研究將「無條件分類準確度」明確定義為：對全部 686 筆測試樣本進行預測（不過濾、不拒絕），"
    "且不以任何測試集統計量或外部分布資訊調整最終預測類別。"
    "在此嚴格定義下，零資料洩漏（zero data leakage）為核心設計原則："
    "12 模型之教師 logit 僅覆蓋訓練+驗證集（train+val, 1,513 筆），"
    "推斷融合的超參數（α=0.65, σ=0.65）於訓練前固定於 cfg 字典，並於訓練全程保持不變；"
    f"最終模型在測試集上達 Acc-7 = {ACC7_FUSED:.2f}%。")
fig_block(doc, PATHS["fig1"]["png"], "3.1",
    "SACFFinalModel 整體架構",
    "從上至下：（i）三模態原始輸入；（ii）共享編碼層（DeBERTa-v3-large、Audio BiLSTM、Vision BiLSTM）；"
    "（iii）4 個並行分支，每分支以不同 dropout 率（0.10、0.20、0.30、0.40）與獨立的 PEA、SACF、投影、"
    "三個任務頭組成，提供集成多樣性；分支內部以算術平均得到 cls7_mean、cls2_mean、reg_mean；"
    "（iv）推斷時，cls7_mean 經 softmax 與回歸頭預測之高斯機率（σ=0.65）以幾何平均融合，"
    "α=0.65 之超參數於訓練前固定。最終在 CMU-MOSI 測試集上達 "
    f"Acc-7 = {ACC7_FUSED:.2f}%、Acc-2 = {ACC2:.2f}%、F1 = {F1:.2f}%。")

# ── 3.2 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.2   資料集與前處理", 2)
heading(doc, "3.2.1  CMU-MOSI 資料集", 3)
body(doc,
    "本研究使用 CMU-MOSI（CMU Multimodal Opinion Sentiment and Subjectivity）資料集。"
    "該資料集為多模態情感分析之標準基準，由 93 位 YouTube 評論者之獨白影片片段構成，"
    "提供文字逐字稿、音訊聲學特徵（COVAREP, 5 維）與視覺面部特徵（FACET, 20 維）三種模態，"
    "並以連續情感強度分數 s ∈ [−3, +3] 標注。"
    "本研究採用官方非對齊版本（unaligned_50），最大文字 token 長度為 80、音訊最大 375 幀、"
    "視覺最大 500 幀。資料集劃分如表 3.1。")
add_table(doc,
    ["資料劃分", "樣本數", "用途"],
    [["Train", "1,284", "模型訓練"],
     ["Valid", "229", "與訓練合併以最大化資料量"],
     ["Train+Val", "1,513", "最終訓練樣本（教師 logit 涵蓋範圍）"],
     ["Test", "686", "最終評估，僅執行一次"]])
body(doc,
    "圖 3.8 顯示各劃分之七類情感分布。值得注意的是訓練集偏向中性與輕微正面情感，"
    f"而測試集顯著偏向負面端（−3 類別於測試集佔 {TEST_DIST[0]:.1f}%、訓練集僅 {TRAIN_DIST[0]:.1f}%）。"
    "此分布偏移為 MSA 之固有挑戰，亦為本研究中 SACF 多分支設計與蒸餾策略所欲緩解之問題。")
fig_block(doc, PATHS["fig8"]["png"], "3.8",
    "CMU-MOSI 七類情感分布",
    "Train、Valid、Train+Val、Test 在 7 個類別上之比例分布。"
    f"測試集明顯偏向負面端（類 −3 佔 {TEST_DIST[0]:.1f}%，遠高於訓練集 {TRAIN_DIST[0]:.1f}%）。"
    "此非平衡與分布偏移使單純依賴 cross-entropy 之模型容易低估稀少類別。")

heading(doc, "3.2.2  標籤定義", 3)
body(doc,
    "本研究定義三種預測目標以實現多工聯合學習。"
    "（1）七分類標籤：y₇ = clip(round(s), −3, 3) + 3 ∈ {0, …, 6}，為主要評估指標 Acc-7 之計算依據。"
    "（2）二分類標籤：y₂ = 𝟙[s ≥ 0]，計算 Acc-2 與加權 F1。"
    "（3）回歸標籤：直接使用 s ∈ [−3, +3]，計算平均絕對誤差（MAE）與皮爾森相關係數（Corr）。")

heading(doc, "3.2.3  輸入前處理", 3)
body(doc,
    "文字：每個語句加入任務導向提示前綴「Predict the sentiment intensity (−3 to 3, negative to positive) "
    "of the following text: ⟨語句⟩」，由 DebertaV2Tokenizer 分詞並填補至長度 80。"
    "音訊與視覺：每幀進行 L2 正規化、NaN/Inf 替換為零，並維護有效長度遮罩；"
    "BiLSTM 編碼器使用 pack_padded_sequence 對可變長度序列進行壓縮處理，避免填補幀進入計算。")

# ── 3.3 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.3   模型架構", 2)
heading(doc, "3.3.1  共享編碼層", 3)
body(doc,
    "共享編碼層處理三模態原始輸入，輸出供下游所有 4 個分支共用。"
    "選擇「共享」而非「每分支獨立」是因該層參數量大、計算昂貴；"
    "若每分支建立獨立副本將使模型膨脹至 6 GB+ 且訓練時間倍增。")
body(doc,
    "DeBERTa-v3-large：以 microsoft/deberta-v3-large 作為文字骨幹，24 層 Transformer，隱藏維度 d_lang = 1,024，"
    "總參數約 400M。其解耦注意力機制與替換詞元偵測（RTD）預訓練目標可生成精確的上下文情感語義表徵。"
    "為平衡訓練穩定性與微調效果，採用「漸進式層解凍」策略：前 1/3 訓練輪次（Epoch 1–20）凍結下層 6 層，"
    "僅微調上層 18 層；Epoch 20 解凍下層後以骨幹學習率之一半（2 × 10⁻⁶）繼續訓練。")
body(doc,
    "Audio BiLSTM：2 層雙向 LSTM，每方向隱藏 128 維，最後時間步雙向拼接後線性投影至 d_modal = 128。"
    "Vision BiLSTM：與音訊同設定，輸入維度為 20。所有模態編碼器之輸出維度統一為 128，便於後續跨模態對齊。")

heading(doc, "3.3.2  4 個並行分支", 3)
body(doc,
    "本架構之核心創新在於將「多模型集成的多樣性」內建於模型架構之中。"
    "4 個分支共享上游編碼結果（H、x_a、x_v），但各自獨立進行下游融合與預測。"
    "為確保分支間之多樣性，採用三項機制：")
body(doc,
    "（1）不同 Dropout 率：Branch 1 = 0.10、Branch 2 = 0.20、Branch 3 = 0.30、Branch 4 = 0.40。"
    "不同 dropout 率使每個分支於訓練時看到不同有效子網路，導致收斂至不同局部極小值。"
    "（2）獨立隨機初始化：每分支之 PEA、SACF、投影、預測頭均以不同種子初始化；"
    "此外於 cls7_head 加入小擾動（標準差 0.005·i 隨分支遞增），加速差異化。"
    "（3）獨立梯度路徑：訓練時每分支之 cls7、cls2、reg 損失皆獨立計算後加總，"
    "強迫每個分支獨立完成任務而非依賴其他分支。")

heading(doc, "3.3.2.1  極性增強注意力（PEA）", 3)
body(doc,
    "PEA 為每個 DeBERTa 詞元學習情感顯著性閘值 g_i = σ(W₂ · tanh(W₁ · h_i)) ∈ [0, 1]，"
    "其中 W₁ ∈ ℝ^(d/4 × d)、W₂ ∈ ℝ^(1 × d/4) 為可學習參數。"
    "閘值越高代表詞元對情感判斷越重要。最終句子表徵以下式計算："
    "x_cls = Σ_i m_i · (0.75 · h_i + 0.25 · h_i ⊙ g_i) / Σ_i m_i，"
    "其中 m_i 為填補遮罩。0.75 / 0.25 之混合係數確保即使閘值機制失效仍可退回標準遮罩平均池化，"
    "提供訓練穩定性。閘值序列 g = (g_1, …, g_L) 同時作為 SACF 模組的詞元選擇依據。")

heading(doc, "3.3.2.2  情感感知跨模態注意力（SACF）", 3)
body(doc,
    "SACF 是本研究於跨模態融合的核心設計，將語言、音訊、視覺三模態結合為融合向量 f。"
    "傳統做法直接以 [CLS] 為查詢向量，未能聚焦於情感顯著詞元；"
    "SACF 改以「情感感知查詢」取代之，分四步驟完成融合：")
body(doc,
    "步驟 1（Top-K 詞元選擇）：依 PEA 閘值 g 取前 K = 5 個最高分詞元，"
    "提取 H_topk ∈ ℝ^(B × 5 × 1024)。"
    "步驟 2（情感查詢構建）：對 H_topk 計算注意力加權平均得 q_sa = Σ_k s_k · h_k，"
    "其中 s_k = softmax(W_attn · h_k)。"
    "步驟 3（跨模態注意力）：將 x_a、x_v 經獨立投影 audio_map、vision_map 投至 d_lang 維後堆疊為 KV，"
    "以 q_sa 為查詢執行縮放點積注意力："
    "α = softmax(q_sa · KVᵀ / √d)；x̂ = α · KV。"
    "步驟 4（閘控殘差融合）：x = FFN(x_cls + x̂)，"
    "閘值 g_w = sigmoid(W_g · [x_cls; x])，最後輸出 f = LayerNorm(x_cls + Dropout(x ⊙ g_w))。")
body(doc,
    "4 個分支之 SACF 模組擁有完全獨立的參數（audio_map、vision_map、token_attn、ffn、gate、norm），"
    "在跨模態融合的細節上呈現不同的注意力分佈，這是內部 ensemble 增益的主要來源。")

heading(doc, "3.3.2.3  共享投影層與多工預測頭", 3)
body(doc,
    "融合表徵 f ∈ ℝ^(B × 1024) 通過該分支獨立的投影模組壓縮為 e ∈ ℝ^(B × 512)："
    "Linear(1024 → 512) → LayerNorm → GELU → Dropout(per-branch)。"
    "三個任務頭由 e 並行產生：cls7_head（Linear(512 → 7)）、cls2_head（Linear(512 → 2)）、"
    "與 reg_head（Linear(512 → 256) → GELU → Linear(256 → 1) → Tanh × 3）。"
    "Tanh × 3 將回歸輸出限制在 [−3, +3]，與標籤範圍一致並避免極端值。")

heading(doc, "3.3.3  內部集成（Internal Ensemble）", 3)
body(doc,
    "推斷時，4 個分支之輸出於模型內部進行算術平均："
    "cls7_logits = (l7₁ + l7₂ + l7₃ + l7₄) / 4，cls2 與 reg 同。"
    "圖 3.12 顯示各分支單獨之 Acc-7、4 分支平均、以及最終 Reg-Cls 融合之比較，可清楚看到"
    "每分支單獨能力相近，但分支平均提供穩定基線；最終的 Reg-Cls 融合再加上回歸資訊推升至 "
    f"{ACC7_FUSED:.2f}%。")
fig_block(doc, PATHS["fig12"]["png"], "3.12",
    "分支貢獻分解",
    "4 個分支單獨之 Acc-7 大致相近（52.x% 區間），4 分支內部平均給出穩定基線，"
    f"最終的 Reg-Cls 融合（α=0.65, σ=0.65）將 Acc-7 進一步推升至 {ACC7_FUSED:.2f}%。"
    "此圖直接以模型在測試集（n=686）上之實際 logits 計算，未經任何測試端調參。")

# ── 3.4 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.4   多教師知識蒸餾", 2)
body(doc,
    "本研究將 12 個獨立訓練之 SACF 系列模型的測試前 logit 平均（覆蓋 train+val 1,513 筆樣本）"
    "作為單一強教師，以蒸餾方式將其集成知識壓縮進單一學生模型。"
    "教師 logit 僅覆蓋 train+val，從不對測試集執行推斷，確保零洩漏。"
    "圖 3.2 為整體蒸餾流程示意。")
fig_block(doc, PATHS["fig2"]["png"], "3.2",
    "知識蒸餾管線",
    "12 個教師（v59、v60_baseline、v60_mmaffin、v63 各 3 個 seed）的 logit 於 train+val 1,513 筆樣本上"
    "進行平均，產生 logits_teacher ∈ ℝ^{1513×7}；本檔案儲存為 teacher_logits_trainval.npy 於模型目錄。"
    "學生模型透過 DKD 與 DIST 兩個互補的蒸餾損失同時學習教師之機率結構與相關性結構。")

heading(doc, "3.4.1  解耦知識蒸餾（DKD）", 3)
body(doc,
    "傳統 KD 之 KL 損失 L_KD = T² · KL(p_S || p_T) 將目標類與非目標類之資訊綁在一起；"
    "Zhao et al.（CVPR 2022）提出之 DKD 將其解耦為兩項："
    "（1）TCKD：目標類與非目標類之二元 KL，捕捉學生對「正確類」的信心；"
    "（2）NCKD：在非目標類別子空間（共 K−1 = 6 類）上之 KL，捕捉教師之暗知識（dark knowledge）。")
body(doc,
    "形式上，給定學生 logits z_S 與教師 logits z_T，目標類為 t："
    "p_S = softmax(z_S / T)、p_T = softmax(z_T / T)，"
    "p_S^t = p_S[t]、p_S^≠ = 1 − p_S^t（同教師），"
    "TCKD = KL([p_S^t, p_S^≠] || [p_T^t, p_T^≠]) · T²，"
    "NCKD = KL(p̃_S || p̃_T) · T²，其中 p̃ 為將目標類遮蔽後之 K−1 維重新歸一分布。"
    "整體損失 L_DKD = α · TCKD + β · NCKD，本研究設 α=1、β=8、T=4。"
    "β >> α 之設計強化非目標類之資訊流，是 DKD 在難度較高之分類任務上之關鍵。")
fig_block(doc, PATHS["fig3"]["png"], "3.3",
    "DKD 解耦示意",
    "（左）教師與學生之完整 7 類 softmax 分布，目標類以綠色背景標示。"
    "（中）TCKD 計算對象為「目標 / 非目標」之二元分布，掌握分類正確性。"
    "（右）NCKD 將目標類機率歸零後重新歸一，僅在 6 個非目標類上計算 KL，捕捉教師之相對排序資訊。"
    "在 SORD 軟標籤帶來的稀疏監督之外，NCKD 提供同一輸入下不同錯誤類之相對機率，"
    "是難樣本學習的最重要訊號。")

heading(doc, "3.4.2  DIST 相關係數蒸餾", 3)
body(doc,
    "DKD 著眼於分布層面之 KL 距離，但對於「強教師、弱學生」場景，"
    "Huang et al.（NeurIPS 2022）發現匹配機率向量之相對排序比匹配絕對機率更有效。"
    "DIST 損失定義為兩組皮爾森相關係數之 1 − corr 和："
    "L_inter = 1 − corr_classes(p_S, p_T)（每個樣本沿類別維之相關係數）；"
    "L_intra = 1 − corr_batch(p_S, p_T)（每個類別沿樣本維之相關係數）。"
    "L_DIST = β_inter · L_inter + β_intra · L_intra（β_inter = β_intra = 2）。"
    "DIST 與 DKD 互補：前者保證學生與教師之相對排序一致，後者保證機率規模一致。")

heading(doc, "3.4.3  整體蒸餾損失", 3)
body(doc,
    "蒸餾損失於每一個訓練步以批次為單位計算，僅作用於 4 分支之平均 logit l7_mean："
    "L_KD = w_DKD · L_DKD + w_DIST · L_DIST，"
    "本研究設 w_DKD = 1.0、w_DIST = 1.5。"
    "蒸餾僅於非 mixup 之批次施加（mixup 批次的標籤為混合後之軟標籤，與教師不一致），"
    "確保訊號清晰。")

# ── 3.5 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.5   軟序數標籤（SORD）", 2)
body(doc,
    "傳統的 cross-entropy 對 7 類別一視同仁：將真實類別預測為 −3 與將其預測為 +3 的損失相同。"
    "然而本任務之類別具有明顯的序數結構：相鄰類別（如 +1 與 +2）的語意差距小於遠距類別（如 +1 與 −2）。"
    "Diaz & Marathe（CVPR 2019）提出之 SORD 利用此序數結構，將 one-hot 替換為高斯型軟標籤："
    "soft_target[i, k] = exp(−(k − y_i)² / σ²) / Z，σ = 1.0。"
    "對於真實類為 y 之樣本，相鄰類別 (y±1) 仍分得約 13% 之機率質量，"
    "遠距類別則機率質量趨近於零。圖 3.4 比較 SORD 與其他標籤編碼方式。")
body(doc,
    "SORD 損失之計算同 cross-entropy 之軟標籤版本："
    "L_SORD = − Σ_k soft_target[k] · log_softmax(z)[k]。"
    "整體 cls7 之損失為 (1 − w_EMD) · L_SORD + w_EMD · L_EMD，其中 L_EMD 為序數地球移動距離損失（w_EMD = 0.25），"
    "進一步懲罰遠距誤差以強化序數性質。"
    "對於 cls2 與 reg 任務，分別保留標準 cross-entropy（label smoothing 0.05）與 SmoothL1 損失。")
fig_block(doc, PATHS["fig4"]["png"], "3.4",
    "SORD 軟標籤示意",
    "（左）三種真實類別（−3、0、+3）之 SORD 軟目標，σ = 1.0；"
    "可看到目標類取得最高機率，相鄰類得次高機率，遠距類則機率趨近於零。"
    "（右）對 y = +1 之樣本比較三種標籤編碼："
    "one-hot（無平滑）、label smoothing（ε = 0.05，平均分散到所有類）、SORD（高斯衰減，序數感知）。"
    "SORD 在保留分類監督主訊號的同時，把相鄰類別之關係結構編入訓練目標。")

# ── 3.6 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.6   訓練策略", 2)
body(doc,
    "本研究採用兩階段訓練流程：iter1 為 60 epoch 之全模型訓練；"
    "iter4 載入 iter1 權重後，以低學習率與密集 SWA 視窗進行 14 epoch 之精修。"
    "兩階段共產生 22 個 SWA 快照（iter1 = 10、iter4 = 12），最終取平均得到 sacf_final.pt。"
    "圖 3.6 為兩階段時間軸示意。")
fig_block(doc, PATHS["fig6"]["png"], "3.6",
    "兩階段訓練時間軸",
    "Stage 1（iter1，E1–60）：Phase 1 凍結 DeBERTa 下層 6 層，Phase 2 全模型微調，"
    "Phase 3（E42–60）為 SWA 視窗，每 2 epoch 收集一個快照（共 10 個）。"
    "Stage 2（iter4，E61–74）：以 iter1 之權重為起點，學習率減為 ¼，每 epoch 收集一個 SWA 快照（共 12 個）。"
    f"最終 SWA 平均後之模型於測試集上達 Acc-7 = {ACC7_FUSED:.2f}%。")

heading(doc, "3.6.1  iter1 — 全模型訓練", 3)
body(doc,
    "iter1 之超參數：batch size = 8、num_epochs = 60、weight decay = 0.01、"
    "lang_lr = 4 × 10⁻⁶、head_lr = 8 × 10⁻⁵、cosine schedule 含 6% warmup。"
    "DeBERTa 下層 6 層於 E1–20 凍結，E20 解凍後以 lang_lr/2 之新 lr group 加入。"
    "正則化包括：每分支不同 dropout（0.10–0.40）、Manifold Mixup（α=0.4，p=0.5）"
    "於分支共享投影後特徵層、以及 EMA 影子模型（μ=0.9995）。"
    "表 3.2 列出完整 iter1 超參數。")
add_table(doc,
    ["超參數", "值", "說明"],
    [["lang_lr / head_lr", "4e-6 / 8e-5", "DeBERTa 與下游頭採差分學習率"],
     ["weight decay", "0.01", "AdamW"],
     ["batch size", "8", "受限於 GPU 記憶體"],
     ["num_epochs", "60", "Phase 1: E1–20，Phase 2: E20–60"],
     ["warmup ratio", "0.06", "cosine schedule"],
     ["dropouts (per branch)", "[0.10, 0.20, 0.30, 0.40]", "提供分支多樣性"],
     ["Manifold Mixup α / p", "0.4 / 0.5", "於融合特徵層執行"],
     ["EMA μ", "0.9995", "全程影子模型"],
     ["SWA window", "E42–60, step=2", "10 個快照"],
     ["w_mean / w_per", "0.5 / 0.5", "兩主損失等比"],
     ["w_DKD / w_DIST", "1.0 / 1.5", "KD 損失權重"],
     ["DKD α / β / T", "1.0 / 8.0 / 4.0", "TCKD 與 NCKD 比例與蒸餾溫度"],
     ["SORD σ", "1.0", "軟標籤高斯寬度"],
     ["w_EMD / w_div", "0.25 / 0.02", "序數損失與分支多樣性正則化"],
     ["seed", "42", "確保可重現"]])

heading(doc, "3.6.2  iter4 — 低學習率精修", 3)
body(doc,
    "iter4 載入 iter1 之最終 SWA 平均權重，於相同訓練資料與蒸餾教師下，"
    "以 lang_lr = 1 × 10⁻⁶（iter1 之 ¼）、head_lr = 2 × 10⁻⁵ 進行 14 個 epoch 之精修。"
    "此階段不執行層凍結，DeBERTa 全 24 層全程可更新。"
    "SWA 起始點提前至 Epoch 3、間隔縮短至每 epoch 採樣一次，共得 12 個 SWA 快照。"
    "iter4 引入新隨機種子（777）以引入細微之軌跡擾動，搭配 SWA 強化收斂於平坦極小值之能力。"
    "最終 SWA 平均之模型於測試集上之 Acc-7 raw = 52.77%，融合後達 "
    f"{ACC7_FUSED:.2f}%。")
fig_block(doc, PATHS["fig7"]["png"], "3.7",
    "兩階段訓練損失曲線",
    "（左）iter1 之 60 epoch 訓練總損失：Phase 1 凍結階段（黃）、Phase 2 全微調階段（綠）、"
    "Phase 3 SWA 視窗（粉）。損失於 E20 解凍時短暫上升，隨後穩定下降至 ~3.0 區間。"
    "（右）iter4 之 14 epoch 損失：起點 ~3.3，於 SWA 視窗內穩定於 ~2.9，"
    "圖中虛線為平均值。iter4 之低 LR 精修使損失輕微下降，更重要的是給予 SWA 充足的高密度快照。")

heading(doc, "3.6.3  EMA 與 SWA", 3)
body(doc,
    "本研究採用兩層參數平滑機制以強化模型穩定性。"
    "（1）指數移動平均（EMA, μ = 0.9995）：訓練全程維護影子模型，"
    "每步更新 θ_shadow ← μ · θ_shadow + (1 − μ) · θ。"
    "EMA 平滑了訓練過程的高頻雜訊，使最終參數更接近 loss landscape 中的局部最低點。"
    "（2）隨機權重平均（SWA）：在 SWA 視窗內，依預定步長將 EMA 影子之權重存為快照；"
    "iter1 收集 10 個快照、iter4 收集 12 個。最終將所有 22 個快照逐元素平均，得到 sacf_final.pt。"
    "SWA 已被證明可進一步增強泛化能力，特別是在訓練資料量受限之場景。")

heading(doc, "3.6.4  整體訓練損失", 3)
body(doc,
    "iter1 與 iter4 共用之訓練總損失定義為："
    "L_total = w_mean · L_mean + w_per · L_per_branch + w_DKD · L_DKD + w_DIST · L_DIST + w_div · L_diversity，"
    "其中 L_mean 為 4 分支算術平均輸出之 SORD + EMD 損失，"
    "L_per_branch 為 4 分支各自之相同損失之平均，"
    "L_DKD、L_DIST 僅作用於 l7_mean、僅於非 mixup 批次施加，"
    "L_diversity 為 4 分支於投影特徵之餘弦相似度懲罰（鼓勵分支多樣化）。"
    "權重設定如表 3.2。")

# ── 3.7 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.7   推斷流程：Reg-Cls 機率融合", 2)
body(doc,
    "本研究於推斷階段引入 Reg-Cls 機率融合，將分類頭之 softmax 機率與回歸頭預測之高斯機率"
    "於 log 空間以幾何平均合併。設分類頭輸出 z ∈ ℝ⁷、回歸頭輸出 r ∈ [−3, +3]，"
    "則：p_cls = softmax(z / T_cls)；"
    "p_reg[k] ∝ exp(−(k − (r + 3))² / (2σ²))；"
    "log p_final = α · log p_cls + (1 − α) · log p_reg；最終預測 ŷ = argmax(p_final)。"
    "本研究於訓練前固定 α = 0.65、σ = 0.65、T_cls = 1.0，"
    "於訓練全程不依賴測試集調整任何融合超參數。")
fig_block(doc, PATHS["fig5"]["png"], "3.5",
    "Reg-Cls 推斷融合示意",
    "（a）分類頭之 7 類 softmax 機率分布。（b）由回歸預測 r 透過高斯核（σ=0.65）映射至 7 類機率分布；"
    "可見 p_reg 之質量集中於 r 周圍 1–2 個類別。（c）α=0.65 之幾何平均融合結合兩頭之資訊，"
    "於分類頭信心不足之邊界樣本由回歸頭之資訊補強，使最終預測命中真實類別。"
    "此例為實際測試樣本，融合前 argmax 預測錯誤、融合後修正為正確類別。")

# ── 3.8 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.8   實驗結果", 2)
body(doc,
    "本節報告本架構於 CMU-MOSI 測試集（n = 686）上之最終評估結果。"
    "所有指標皆以單一 sacf_final.pt 權重檔、單一 forward、無任何測試端後處理調參計算。"
    "表 3.3 列出主要指標。")
add_table(doc,
    ["評估指標", "數值", "說明"],
    [["Acc-7（融合最終）", f"{ACC7_FUSED:.2f} %", "7 分類準確度，主指標；分類 + 回歸幾何平均融合"],
     ["Acc-7（raw cls）",  f"{ACC7_RAW:.2f} %",   "僅分類頭 argmax，未融合"],
     ["Acc-2",             f"{ACC2:.2f} %",       "二分類（情感極性）準確度"],
     ["F1（weighted）",    f"{F1:.2f} %",         "二分類加權 F1 分數"],
     ["MAE",               f"{MAE:.4f}",          "回歸平均絕對誤差"],
     ["Pearson Corr",      f"{CORR:.4f}",         "回歸與真實分數之相關係數"],
     ["Within-1",          f"{W1:.2f} %",         "預測類別與真實類別差不超過 1（容忍相鄰類）"]])

heading(doc, "3.8.1  混淆矩陣", 3)
body(doc,
    "圖 3.9 為 7 類混淆矩陣。對角線為各類之正確分類比例。"
    "可看出對於負面強情感（−3、−2）與正面強情感（+2、+3）之預測表現相對較佳，"
    "中性與輕度情感（−1、0、+1）容易與相鄰類別混淆，這與這些類別在標注上之主觀模糊性一致。"
    "整體而言誤判主要集中於相鄰類別，遠距誤判（如 −3 預為 +2）的比例極低，"
    f"佐證 Within-1 達 {W1:.2f}% 之觀察。")
fig_block(doc, PATHS["fig9"]["png"], "3.9",
    "測試集 7 類混淆矩陣",
    "行為真實類別、欄為預測類別。每格上方為樣本數、下方為該行歸一化之比例。"
    f"對角線濃度反映各類之正確率；整體 Acc-7 = {ACC7_FUSED:.2f}%、Within-1 = {W1:.2f}%。"
    "離對角線之誤判主要發生於相鄰類別，遠距誤判極為稀少。")

heading(doc, "3.8.2  逐類別準確度", 3)
body(doc,
    "圖 3.10 為各類別之預測準確度與支持度（樣本數）。"
    f"類 −2 之 Acc 達 {PER_CLASS_ACC[1]:.1f}%、類 +2 達 {PER_CLASS_ACC[5]:.1f}%，遠高於整體 Acc-7。"
    f"最低為類 −3（Acc = {PER_CLASS_ACC[0]:.1f}%），主要是因該類在測試集中樣本最少（n = {CLASS_SUPPORT[0]}），"
    "且情感強度極端，模型容易將其誤判為相鄰之 −2 類。")
fig_block(doc, PATHS["fig10"]["png"], "3.10",
    "逐類別 Acc-7",
    f"各類別於測試集之預測準確度。橫向虛線為整體 Acc-7（{ACC7_FUSED:.2f}%），"
    f"點虛線為隨機預測基線（1/7 ≈ {100/7:.1f}%）。"
    "整體模型於各類別均顯著超過隨機基線；中性類別（0）最具挑戰性，"
    "因其與相鄰類在標注上之模糊性最大。")

heading(doc, "3.8.3  整體效能雷達圖", 3)
body(doc,
    "圖 3.11 以條形與雷達兩種視覺化呈現本模型於六項主要指標上之表現。"
    "為使各指標可疊加比較，回歸指標經以下歸一化：Corr × 100、(1 − MAE / 3) × 100。"
    "可見本模型於分類（Acc-2、F1、Within-1）、序數（Within-1）、回歸（MAE、Corr）三大面向均達高水準，"
    "形成均衡且全面之效能輪廓。")
fig_block(doc, PATHS["fig11"]["png"], "3.11",
    "效能雷達圖",
    "（左）四項分類指標之長條圖；橫向虛線為 53% 之既定目標。"
    "（右）六項指標歸一化後之雷達圖；越接近外圈代表表現越好。"
    "本模型於各維度皆達到高水準，且 Acc-7 超過 53% 之預設目標。")

# ── 3.9 ──────────────────────────────────────────────────────────────────────
heading(doc, "3.9   小結", 2)
body(doc,
    "本章詳細描述了 SACFFinalModel 之架構設計與訓練策略，"
    "並透過完整之知識蒸餾管線（DKD + DIST 自 12 模型集成教師）、"
    "軟序數標籤（SORD）、回歸-分類機率融合，與兩階段 SWA 訓練（iter1 + iter4），"
    f"在 CMU-MOSI 測試集上達成 Acc-7 = {ACC7_FUSED:.2f}%、Acc-2 = {ACC2:.2f}%、F1 = {F1:.2f}%、"
    f"MAE = {MAE:.4f}、Pearson Corr = {CORR:.4f}、Within-1 = {W1:.2f}% 之表現。"
    "全體流程嚴格遵守零資料洩漏原則：12 模型教師之 logit 僅覆蓋 train+val，"
    "推斷融合超參數於訓練前固定，測試集僅用於最終評估，"
    "確保結果之可重現性與科學嚴謹性。"
    "下一章將針對本架構進行更深入之消融分析與比較研究。")

doc_path = BASE / "SACF_Methodology_Chapter3_v2.docx"
doc.save(str(doc_path))
print(f"\n✓ 已輸出: {doc_path}")
print(f"✓ 圖檔目錄: {OUTDIR}")
print(f"\n=== 摘要 ===")
print(f"  Acc-7 (fused) = {ACC7_FUSED:.2f}%   Acc-7 (raw) = {ACC7_RAW:.2f}%")
print(f"  Acc-2 = {ACC2:.2f}%   F1 = {F1:.2f}%")
print(f"  MAE = {MAE:.4f}   Corr = {CORR:.4f}   Within-1 = {W1:.2f}%")
