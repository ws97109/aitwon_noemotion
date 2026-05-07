"""SACF v3 figure regeneration — fixes overlapping/blocking issues from v2 set.

Changes from v2:
  - Fig 1 (architecture): redo branch-zone vertical spacing to remove arrow overlap
    with the "(iii) 4 Parallel Branches" header; legend moved outside axis.
  - Fig 2 (KD pipeline): widen + replace inline arrows with mid-line connectors so
    Distillation Losses box no longer obscures the teacher→student flow; multi-line
    DIST formula box increased in height.
  - Fig 3 (DKD decomp): removed conflicting "(masked)" axvspan from Panel 3 (NCKD);
    target column is hatched on the bar itself instead of a translucent overlay so
    bars are no longer obscured.
  - Fig 5 (RegCls fusion): fixed legend-bar overlap by moving legends outside.
  - Fig 6 (timeline): pulls the "→ sacf_final.pt" line into the axis area so it is
    not clipped at PNG export.

Run:  python3 docs/regenerate_v3_figures.py
Outputs:
  docs/figures/v3_fig1_architecture.{svg,png}
  docs/figures/v3_fig2_kd_pipeline.{svg,png}
  docs/figures/v3_fig3_dkd_decomp.{svg,png}
  docs/figures/v3_fig5_regcls_fusion.{svg,png}
  docs/figures/v3_fig6_training_timeline.{svg,png}
"""
import os, sys, warnings
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

# Reuse data from v2 (same metrics)
DATA = np.load(str(BASE / "paper_v2_data.npz"))
m = DATA["metrics"]
ACC7_FUSED, ACC7_RAW, ACC2, F1, MAE, CORR, W1 = [float(x) for x in m]

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
# FIG 1 — Overall architecture (v3: cleaner spacing, no arrow/header overlap)
# ════════════════════════════════════════════════════════════════════════════
def fig1_architecture_v3():
    fig, ax = plt.subplots(figsize=(14.0, 13.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 15.0); ax.axis("off")

    # Title
    ax.text(7, 14.55,
            "SACF-v2: Multi-Branch Single Model with KD + SORD + Reg-Cls Fusion",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=C["primary"])
    ax.text(7, 14.10,
            f"Acc-7 = {ACC7_FUSED:.2f}%   |   Acc-2 = {ACC2:.2f}%   |   "
            f"F1 = {F1:.2f}%   |   MAE = {MAE:.4f}   |   Corr = {CORR:.4f}",
            ha="center", va="center", fontsize=10.5, color=C["muted"])

    # ── Row 1: Inputs ───────────────────────────────────────────────────────
    bg(ax, 0.4, 12.5, 13.2, 1.2, "#FEF3C7", C["accent"])
    ax.text(7, 13.5, "(i)  Multimodal Inputs", ha="center", fontsize=10.5,
            color=C["accent"], fontweight="bold")
    rbox(ax, 2.5, 12.9, 3.0, 0.55, C["primary"],
         "Text  (raw_text + task prompt)", fs=9.5, bold=True)
    rbox(ax, 7.0, 12.9, 3.0, 0.55, C["accent"],
         "Audio  (COVAREP, 5-d, ≤375 frames)", fs=9.5, bold=True)
    rbox(ax, 11.5, 12.9, 3.0, 0.55, C["success"],
         "Vision  (FACET, 20-d, ≤500 frames)", fs=9.5, bold=True)

    # ── Row 2: Shared encoders ──────────────────────────────────────────────
    bg(ax, 0.4, 10.6, 13.2, 1.5, "#DBEAFE", C["primary"])
    ax.text(7, 11.95, "(ii)  Shared Encoders", ha="center", fontsize=10.5,
            color=C["primary"], fontweight="bold")
    rbox(ax, 2.5, 11.20, 3.0, 0.7, C["primary"],
         "DeBERTa-v3-large\n24 layers · d=1024 · ~400M", fs=9, bold=True)
    rbox(ax, 7.0, 11.20, 3.0, 0.7, C["accent"],
         "BiLSTM-Audio\n2 layers · 5→128", fs=9, bold=True)
    rbox(ax, 11.5, 11.20, 3.0, 0.7, C["success"],
         "BiLSTM-Vision\n2 layers · 20→128", fs=9, bold=True)

    # arrows row1 → row2
    for cx in (2.5, 7.0, 11.5):
        arr(ax, cx, 12.62, cx, 11.55)

    # Spacer to add gap between header text and arrows
    branch_top = 10.10  # branch container top y
    branch_bottom = 6.5
    bg(ax, 0.4, branch_bottom, 13.2, branch_top - branch_bottom + 0.30, "#EDE9FE", C["purple"])
    ax.text(7, branch_top + 0.05,
            "(iii)  4 Parallel Branches  (per-branch dropout = [0.10, 0.20, 0.30, 0.40])",
            ha="center", fontsize=10.5, color=C["purple"], fontweight="bold")

    # arrows from shared encoders to branch zone (terminate ABOVE header text)
    for cx in (2.0, 5.0, 8.0, 11.0):
        arr(ax, 2.5, 10.85, cx, 10.32, color=C["primary"], lw=0.7, hw=0.12)
        arr(ax, 7.0, 10.85, cx, 10.32, color=C["accent"], lw=0.7, hw=0.12)
        arr(ax, 11.5, 10.85, cx, 10.32, color=C["success"], lw=0.7, hw=0.12)

    branch_colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    branch_dropouts = [0.10, 0.20, 0.30, 0.40]
    for i in range(4):
        cx = 2.0 + i * 3.0
        bg(ax, cx - 1.30, 6.7, 2.6, 3.10, "#FFFFFF", branch_colors[i], lw=1.6, alpha=0.6)
        ax.text(cx, 9.55, f"Branch {i+1}  (p_drop={branch_dropouts[i]})",
                ha="center", fontsize=9.5, fontweight="bold", color=branch_colors[i])
        rbox(ax, cx, 8.95, 2.4, 0.5, branch_colors[i], "PEA  (gate σ)", fs=8.5)
        rbox(ax, cx, 8.30, 2.4, 0.5, branch_colors[i], "SACF  (top-K + cross-modal)", fs=8.5)
        rbox(ax, cx, 7.65, 2.4, 0.5, branch_colors[i], "Proj  (1024 → 512)", fs=8.5)
        rbox(ax, cx, 7.00, 2.4, 0.55, "#FFFFFF", "cls7 / cls2 / reg",
             fs=8.5, tc=branch_colors[i], ec=branch_colors[i], bold=True)

    # ── Row 4: Mean aggregation ─────────────────────────────────────────────
    bg(ax, 4.5, 5.20, 5.0, 0.95, "#FFE4E6", C["rose"])
    rbox(ax, 7.0, 5.65, 4.6, 0.65, C["rose"],
         "Mean-of-Branches  →  (cls7_mean, cls2_mean, reg_mean)",
         fs=9.5, bold=True)
    for i in range(4):
        cx = 2.0 + i * 3.0
        arr(ax, cx, 6.70, 7.0, 6.05, color=branch_colors[i], lw=0.8, hw=0.12)

    # ── Row 5: Inference fusion ─────────────────────────────────────────────
    bg(ax, 0.4, 2.6, 13.2, 2.2, "#DCFCE7", C["success"])
    ax.text(7, 4.65, "(iv)  Inference: Reg-Cls Probability Fusion (α=0.65, σ=0.65 — chosen a priori)",
            ha="center", fontsize=10.5, color=C["success"], fontweight="bold")
    rbox(ax, 3.0, 3.85, 3.4, 0.7, C["primary"],
         "p_cls = softmax(cls7_logits / T)", fs=9, bold=True)
    rbox(ax, 7.0, 3.85, 3.4, 0.7, C["accent"],
         "p_reg[k] ∝ exp(−(k−r)² / 2σ²)", fs=9, bold=True)
    rbox(ax, 11.0, 3.85, 3.4, 0.7, C["success"],
         "log p ← α·log p_cls + (1−α)·log p_reg", fs=9, bold=True)
    arr(ax, 7.0, 5.20, 7.0, 4.30)
    arr(ax, 3.0, 3.45, 5.5, 3.05, hw=0.13)
    arr(ax, 7.0, 3.45, 7.0, 3.05, hw=0.13)
    arr(ax, 11.0, 3.45, 8.5, 3.05, hw=0.13)

    rbox(ax, 7.0, 2.20, 5.4, 0.75, C["danger"],
         f"ŷ = argmax(p_final)        Acc-7 = {ACC7_FUSED:.2f}%",
         fs=11, bold=True)

    # ── Legend (outside data area, no overlap) ──────────────────────────────
    handles = [
        mpatches.Patch(color=C["primary"], label="Text (DeBERTa)"),
        mpatches.Patch(color=C["accent"], label="Audio (BiLSTM)"),
        mpatches.Patch(color=C["success"], label="Vision (BiLSTM)"),
        mpatches.Patch(color=C["rose"], label="Aggregation"),
        mpatches.Patch(color=C["purple"], label="Parallel Branches"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.07),
              ncol=5, fontsize=9, frameon=False)

    fig.tight_layout()
    save(fig, "v3_fig1_architecture")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — KD pipeline (v3: clearer connectors, taller DIST box)
# ════════════════════════════════════════════════════════════════════════════
def fig2_kd_pipeline_v3():
    fig, ax = plt.subplots(figsize=(14.0, 7.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.8); ax.axis("off")
    ax.text(7, 7.40, "Knowledge Distillation:  12-Model Ensemble Teacher  →  Single Student",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])

    # Teacher zone
    bg(ax, 0.3, 3.5, 5.5, 3.4, "#FEE2E2", C["danger"])
    ax.text(3.05, 6.55, "Teacher  (12-model logit ensemble)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["danger"])
    teachers = ["v59-s42", "v59-s123", "v59-s2024",
                "v60_baseline-s42", "v60_baseline-s123", "v60_baseline-s2024",
                "v60_mmaffin-s42", "v60_mmaffin-s123", "v60_mmaffin-s2024",
                "v63-s101", "v63-s202", "v63-s303"]
    for i, t in enumerate(teachers):
        rx, ry = 0.7 + (i % 4) * 1.30, 5.85 - (i // 4) * 0.55
        rbox(ax, rx + 0.55, ry, 1.20, 0.42, C["danger"], t, fs=7.5)
    rbox(ax, 3.05, 3.90, 4.6, 0.5, C["danger"],
         "logits_teacher ∈ ℝ^{1513×7}  (only train+val)",
         fs=9, bold=True)

    # KD loss zone — taller and shifted to allow DIST multi-line
    bg(ax, 5.8, 0.9, 4.1, 3.2, "#DBEAFE", C["primary"])
    ax.text(7.85, 3.75, "Distillation Losses",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    rbox(ax, 7.85, 3.10, 3.6, 0.55, C["primary"],
         "DKD  = α·TCKD + β·NCKD  (β=8)", fs=9, bold=True)
    rbox(ax, 7.85, 2.30, 3.6, 0.85, C["secondary"],
         "DIST  =  (1 − corr_inter(p_S, p_T))\n        +  (1 − corr_intra(p_S, p_T))",
         fs=8.5, bold=True)
    rbox(ax, 7.85, 1.40, 3.6, 0.55, C["text"],
         "T = 4   (softmax temperature)", fs=9)

    # Student zone
    bg(ax, 10.4, 3.5, 3.4, 3.4, "#DBEAFE", C["primary"])
    ax.text(12.10, 6.55, "Student (this work)", ha="center",
            fontsize=10.5, fontweight="bold", color=C["primary"])
    rbox(ax, 12.10, 5.80, 2.8, 0.5, C["primary"],
         "DeBERTa-v3-large", fs=9, bold=True)
    rbox(ax, 12.10, 5.20, 2.8, 0.5, C["accent"], "+ BiLSTM × 2", fs=9)
    rbox(ax, 12.10, 4.60, 2.8, 0.5, C["purple"], "+ 4 Branches", fs=9)
    rbox(ax, 12.10, 4.00, 2.8, 0.5, C["success"], "+ Reg-Cls Fusion", fs=9)

    # Arrows: teacher → KD losses → student (mid-line connectors so they don't overlap)
    arr(ax, 3.05, 3.50, 5.80, 2.85, lw=2.0, color=C["danger"], hw=0.18)
    arr(ax, 9.90, 2.85, 12.10, 3.50, lw=2.0, color=C["primary"], hw=0.18)

    # Bottom: combined objective
    bg(ax, 0.3, 0.05, 13.4, 0.70, "#F3F4F6", C["text"], alpha=0.45)
    ax.text(7.0, 0.40,
            "Total Loss  =  0.5 · L_mean(SORD)  +  0.5 · L_per_branch(SORD)  "
            "+  1.0 · L_DKD  +  1.5 · L_DIST  +  0.02 · L_diversity",
            ha="center", fontsize=10.5, color=C["text"], fontweight="bold")

    fig.tight_layout()
    save(fig, "v3_fig2_kd_pipeline")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — DKD decomposition (v3: target column hatched on bars, no overlay overlap)
# ════════════════════════════════════════════════════════════════════════════
def fig3_dkd_decomp_v3():
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))

    K = 7
    classes = list(range(-3, 4))
    target = 4  # +1
    teacher_logits = np.array([-1.5, -0.5, 1.0, 2.0, 3.5, 2.0, -0.5])
    student_logits = np.array([-1.0, -0.3, 0.7, 1.2, 2.0, 0.8, -0.5])
    T = 4.0
    pT = np.exp(teacher_logits / T); pT /= pT.sum()
    pS = np.exp(student_logits / T); pS /= pS.sum()

    # Panel 1
    ax = axes[0]; x = np.arange(K); w = 0.36
    bars_T = ax.bar(x - w/2, pT, w, color=C["danger"], label="Teacher (T=4)", alpha=0.85)
    bars_S = ax.bar(x + w/2, pS, w, color=C["primary"], label="Student (T=4)", alpha=0.85)
    # Mark target bar with thick green outline (instead of overlay axvspan)
    bars_T[target].set_edgecolor(C["success"]); bars_T[target].set_linewidth(2.5)
    bars_S[target].set_edgecolor(C["success"]); bars_S[target].set_linewidth(2.5)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_xlabel("Class label"); ax.set_ylabel("Softmax probability")
    ax.set_title("Full softmax distribution")
    target_handle = mpatches.Patch(facecolor="white", edgecolor=C["success"],
                                    lw=2.5, label="Target class (highlighted)")
    ax.legend(handles=[bars_T, bars_S, target_handle], loc="upper left", fontsize=8)

    # Panel 2 — TCKD (binary)
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

    # Panel 3 — NCKD (target column shown with hatch on the actual bars; no overlay)
    ax = axes[2]
    pT_nc = pT.copy(); pT_nc[target] = 0; pT_nc /= pT_nc.sum()
    pS_nc = pS.copy(); pS_nc[target] = 0; pS_nc /= pS_nc.sum()
    bars_T_nc = ax.bar(x - w/2, pT_nc, w, color=C["danger"], alpha=0.85, label="Teacher (non-target)")
    bars_S_nc = ax.bar(x + w/2, pS_nc, w, color=C["primary"], alpha=0.85, label="Student (non-target)")
    # Target bars are 0; mark with hatched ghost rectangles at the base
    ymax = max(pT_nc.max(), pS_nc.max()) * 1.05
    ghost_t = mpatches.Rectangle((target - w, 0), w, ymax * 0.05,
                                  facecolor="#9CA3AF", edgecolor="#6B7280",
                                  hatch="///", alpha=0.6, label="Target (masked → 0)")
    ax.add_patch(ghost_t)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_xlabel("Class label"); ax.set_ylabel("Renormalized probability")
    ax.set_title("NCKD  =  KL on non-target classes  (β=8)")
    ax.legend(handles=[bars_T_nc, bars_S_nc, ghost_t], loc="upper right", fontsize=7.5)

    fig.suptitle("Decoupled KD (DKD):  L_DKD = α·TCKD + β·NCKD",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "v3_fig3_dkd_decomp")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — Reg-Cls fusion (v3: legend outside, no overlap with truth lines)
# ════════════════════════════════════════════════════════════════════════════
def fig5_regcls_fusion_v3():
    L7 = DATA["L7"]; R = DATA["R"]; Y7 = DATA["y7"]
    z = L7[316]                # cls7 logits for one sample
    r_pred = float(R[316])     # regression pred
    y_true = int(Y7[316]) - 3  # ground truth label in [-3, 3]

    K = 7
    classes_label = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]
    classes_idx = np.arange(K)

    # p_cls
    p_cls = np.exp((z - z.max())); p_cls /= p_cls.sum()
    # p_reg from Gaussian centered on r_pred
    sigma = 0.65
    y_shift = r_pred + 3.0
    d2 = (classes_idx - y_shift) ** 2
    p_reg = np.exp(-d2 / (2 * sigma ** 2)); p_reg /= p_reg.sum()
    # fused
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
               label=f"argmax cls ({cls_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(a)  p_cls = softmax(cls7_logits)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    ax = axes[1]
    ax.bar(classes_idx, p_reg, color=C["accent"], alpha=0.85)
    ax.axvline(y_shift, color=C["text"], lw=1.5,
               label=f"reg pred = {r_pred:+.2f}")
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title(r"(b)  $p_{reg} \propto \exp(-(k-r)^2/2\sigma^2)$    σ=0.65")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    ax = axes[2]
    ax.bar(classes_idx, p_final, color=C["success"], alpha=0.85)
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.axvline(int(p_final.argmax()), color=C["danger"], lw=1.5, ls=":",
               label=f"argmax fused ({fused_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title(r"(c)  $\log p_{final} = \alpha\log p_{cls} + (1{-}\alpha)\log p_{reg}$    α=0.65")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    fig.suptitle(f"Reg-Cls Probability Fusion at Inference  (test sample idx=316)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "v3_fig5_regcls_fusion")


# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — Two-stage timeline (v3: extra bottom margin, label inside axis)
# ════════════════════════════════════════════════════════════════════════════
def fig6_training_timeline_v3():
    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    ax.set_xlim(0, 80); ax.set_ylim(-1.6, 5.2); ax.axis("off")

    # Header
    ax.text(40, 4.85, "Two-Stage Training Pipeline (single .pt artifact)",
            ha="center", fontsize=12.5, fontweight="bold", color=C["primary"])

    # Stage 1 — iter1 (60 ep)
    bg(ax, 0.5, 0.6, 60, 3.6, "#DBEAFE", C["primary"])
    ax.text(30, 3.95, "Stage 1: iter1 (Epoch 1–60)  —  full training from scratch",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    bg(ax, 1, 1.3, 19, 1.4, "#FEF3C7", C["accent"], alpha=0.5)
    ax.text(10.5, 2.0, "Phase 1: freeze DeBERTa\nlayers 0–5 (E1–20)",
            ha="center", fontsize=9, fontweight="bold")
    bg(ax, 20.5, 1.3, 21.5, 1.4, "#DCFCE7", C["success"], alpha=0.5)
    ax.text(31.25, 2.0, "Phase 2: full fine-tune\n(E20–60)", ha="center",
            fontsize=9, fontweight="bold")
    bg(ax, 42.5, 1.3, 18, 1.4, "#FFE4E6", C["rose"], alpha=0.5)
    ax.text(51.5, 2.0, "Phase 3: SWA window\nE42, 44, 46, …, 60  (10 snapshots)",
            ha="center", fontsize=9, fontweight="bold")

    bg(ax, 61, 0.6, 18, 3.6, "#FFE4E6", C["rose"])
    ax.text(70, 3.95, "Stage 2: iter4 (Epoch 61–74)  —  load iter1, polish",
            ha="center", fontsize=10.5, fontweight="bold", color=C["rose"])
    bg(ax, 61.5, 1.3, 16.5, 1.4, "#FCE7F3", C["rose"], alpha=0.5)
    ax.text(69.75, 2.0, "low LR (×¼) + heavy SWA\nseed 777, 12 SWA snapshots",
            ha="center", fontsize=9, fontweight="bold", color=C["rose"])

    ax.text(0.5, 0.1, "E0", ha="center", fontsize=8, color=C["muted"])
    ax.text(60, 0.1, "E60", ha="center", fontsize=8, color=C["muted"])
    ax.text(78.5, 0.1, "final", ha="center", fontsize=8, color=C["muted"])

    # Final outcome — placed inside axis (y=-0.7) so it is not clipped
    rbox(ax, 40, -0.85, 28, 0.85, C["danger"],
         f"→  sacf_final.pt   (1.65 GB)   |   Acc-7 = {ACC7_FUSED:.2f}%",
         fs=11, bold=True)

    fig.tight_layout()
    save(fig, "v3_fig6_training_timeline")


if __name__ == "__main__":
    print("Regenerating v3 figures (fixing v2 overlap issues)...")
    fig1_architecture_v3();          print("  ✓ v3_fig1_architecture")
    fig2_kd_pipeline_v3();           print("  ✓ v3_fig2_kd_pipeline")
    fig3_dkd_decomp_v3();            print("  ✓ v3_fig3_dkd_decomp")
    fig5_regcls_fusion_v3();         print("  ✓ v3_fig5_regcls_fusion")
    fig6_training_timeline_v3();     print("  ✓ v3_fig6_training_timeline")
    print("Done.")
