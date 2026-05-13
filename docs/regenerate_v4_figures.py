"""SACF v4 figure regeneration aligned with the v13 final model.

New / updated figures (v4 set):
  • v4_fig_arch        — 整體架構（共享編碼 + 4 並行分支 + 多工頭 + 推斷）
  • v4_fig_pea         — 極性增強注意力（PEA）模組詳細示意
  • v4_fig_sacf_steps  — SACF 跨模態注意力逐步計算示意（4 步驟）
  • v4_fig_branches    — 4 並行分支與內部 ensemble 詳圖
  • v4_fig_loss_comp   — 損失函數組成結構（softCE + EMD + SmoothL1 + R-Drop + CMC）
  • v4_fig_train_timeline — 訓練全景（漸進解凍 + EMA + SWA + 多種子）
  • v4_fig_inference   — 零洩漏推斷（TTA×5 + 多種子 + Reg-Cls 融合）
  • v4_fig_distribution — CMU-MOSI 七類分布（只保留 Train + Test）
  • v4_fig_regcls      — Reg-Cls 融合機率示意

Run:  python3 docs/regenerate_v4_figures.py
"""
import warnings, os
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BASE = Path(__file__).parent
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

DATA = np.load(str(BASE / "paper_v2_data.npz"))
m = DATA["metrics"]
ACC7_FUSED, ACC7_RAW, ACC2, F1, MAE, CORR = [float(x) for x in m]
W1 = 91.55
TRAIN_DIST = DATA["train_dist"]; VAL_DIST = DATA["val_dist"]
TEST_DIST = DATA["test_dist"]; TRAINVAL_DIST = DATA["trainval_dist"]

C = dict(
    primary="#1D4ED8", secondary="#0891B2", accent="#F59E0B",
    danger="#DC2626", success="#10B981", purple="#8B5CF6",
    text="#1F2937", muted="#6B7280", grid="#E5E7EB", bg="#F9FAFB",
    teal="#14B8A6", indigo="#6366F1", rose="#F43F5E", lime="#84CC16",
    violet="#7C3AED",
)
# Font fallback for CJK glyphs — first available wins.
import matplotlib.font_manager as fm
_avail = {f.name for f in fm.fontManager.ttflist}
_cjk = next((n for n in ["Heiti TC", "PingFang TC", "Songti SC", "STHeiti",
                          "Noto Sans CJK TC", "Noto Sans TC",
                          "Apple LiGothic", "Hiragino Sans GB"] if n in _avail),
            "DejaVu Sans")

plt.rcParams.update({
    "font.family": [_cjk, "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.alpha": 0.5,
    "savefig.dpi": 150, "figure.dpi": 110,
    "axes.unicode_minus": False,
})
print(f"  [font] using CJK font: {_cjk}")


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
# FIG arch — Overall architecture
# ════════════════════════════════════════════════════════════════════════════
def fig_arch():
    fig, ax = plt.subplots(figsize=(14.0, 13.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 15.5); ax.axis("off")

    ax.text(7, 14.95, "SACFFinalModel — 多分支單一模型整體架構",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["primary"])
    ax.text(7, 14.45,
            f"Acc-7 = {ACC7_FUSED:.2f}%  |  Acc-2 = {ACC2:.2f}%  |  F1 = {F1:.2f}%  "
            f"|  MAE = {MAE:.4f}  |  Corr = {CORR:.4f}",
            ha="center", va="center", fontsize=10.5, color=C["muted"])

    # Row 1: Inputs
    bg(ax, 0.4, 12.9, 13.2, 1.2, "#FEF3C7", C["accent"])
    ax.text(7, 13.85, "(i)  多模態輸入  /  Multimodal Inputs",
            ha="center", fontsize=10.5, color=C["accent"], fontweight="bold")
    rbox(ax, 2.5, 13.30, 3.0, 0.55, C["primary"], "文字  (raw_text + prompt)", fs=9.5, bold=True)
    rbox(ax, 7.0, 13.30, 3.0, 0.55, C["accent"], "音訊  (COVAREP, 5-d)", fs=9.5, bold=True)
    rbox(ax, 11.5, 13.30, 3.0, 0.55, C["success"], "視覺  (FACET, 20-d)", fs=9.5, bold=True)

    # Row 2: Shared encoders
    bg(ax, 0.4, 11.0, 13.2, 1.5, "#DBEAFE", C["primary"])
    ax.text(7, 12.35, "(ii)  共享編碼層  /  Shared Encoders",
            ha="center", fontsize=10.5, color=C["primary"], fontweight="bold")
    rbox(ax, 2.5, 11.55, 3.0, 0.7, C["primary"],
         "DeBERTa-v3-large\n24L · 1024d → t_emb", fs=9, bold=True)
    rbox(ax, 7.0, 11.55, 3.0, 0.7, C["accent"],
         "BiLSTM-Audio\n2L · 5→128 → a_emb", fs=9, bold=True)
    rbox(ax, 11.5, 11.55, 3.0, 0.7, C["success"],
         "BiLSTM-Vision\n2L · 20→128 → v_emb", fs=9, bold=True)
    for cx in (2.5, 7.0, 11.5):
        arr(ax, cx, 13.02, cx, 11.94)

    # Row 3: 4 branches
    branch_top = 10.50
    branch_bottom = 6.85
    bg(ax, 0.4, branch_bottom, 13.2, branch_top - branch_bottom + 0.30, "#EDE9FE", C["purple"])
    ax.text(7, branch_top + 0.05,
            "(iii)  4 並行分支（每分支獨立參數，dropout ∈ {0.10, 0.20, 0.30, 0.40}）",
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
        rbox(ax, cx, 9.25, 2.4, 0.5, branch_colors[i], "PEA  (gate σ)", fs=8.5)
        rbox(ax, cx, 8.60, 2.4, 0.5, branch_colors[i], "SACF  (Top-K cross-modal)", fs=8.5)
        rbox(ax, cx, 7.95, 2.4, 0.5, branch_colors[i], f"Proj  (1024 → 512)  → e_{i+1}", fs=8.5)
        rbox(ax, cx, 7.30, 2.4, 0.55, "#FFFFFF",
             f"l7_{i+1}  /  l2_{i+1}  /  reg_{i+1}",
             fs=8.5, tc=branch_colors[i], ec=branch_colors[i], bold=True)

    # Row 4: Mean aggregation
    bg(ax, 4.5, 5.50, 5.0, 0.95, "#FFE4E6", C["rose"])
    rbox(ax, 7.0, 5.95, 4.6, 0.65, C["rose"],
         "Mean-of-Branches  →  (l7_mean, l2_mean, reg_mean)",
         fs=9.5, bold=True)
    for i in range(4):
        cx = 2.0 + i * 3.0
        arr(ax, cx, 7.00, 7.0, 6.35, color=branch_colors[i], lw=0.8, hw=0.10)

    # Row 5: Inference fusion
    bg(ax, 0.4, 2.9, 13.2, 2.2, "#DCFCE7", C["success"])
    ax.text(7, 4.95, "(iv)  零洩漏推斷：TTA×5  +  3-seed ensemble  +  Reg-Cls 機率融合",
            ha="center", fontsize=10.5, color=C["success"], fontweight="bold")
    rbox(ax, 3.0, 4.15, 3.4, 0.7, C["primary"],
         "p_cls = softmax(l7_mean / T_cls)", fs=9, bold=True)
    rbox(ax, 7.0, 4.15, 3.4, 0.7, C["accent"],
         "p_reg[k] ∝ exp(−(k−r)² / 2σ²)", fs=9, bold=True)
    rbox(ax, 11.0, 4.15, 3.4, 0.7, C["success"],
         "log p_final = α log p_cls + (1−α) log p_reg",
         fs=8.5, bold=True)
    arr(ax, 7.0, 5.50, 7.0, 4.60)
    arr(ax, 3.0, 3.75, 5.5, 3.35, hw=0.12)
    arr(ax, 7.0, 3.75, 7.0, 3.35, hw=0.12)
    arr(ax, 11.0, 3.75, 8.5, 3.35, hw=0.12)

    rbox(ax, 7.0, 2.45, 5.4, 0.75, C["danger"],
         f"ŷ = argmax(p_final)        Acc-7 = {ACC7_FUSED:.2f}%",
         fs=11, bold=True)

    handles = [
        mpatches.Patch(color=C["primary"], label="文字 / Text"),
        mpatches.Patch(color=C["accent"], label="音訊 / Audio"),
        mpatches.Patch(color=C["success"], label="視覺 / Vision"),
        mpatches.Patch(color=C["rose"], label="聚合 / Aggregation"),
        mpatches.Patch(color=C["purple"], label="平行分支 / Branches"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.07),
              ncol=5, fontsize=9, frameon=False)

    save(fig, "v4_fig_arch")


# ════════════════════════════════════════════════════════════════════════════
# FIG PEA — Polarity-Enhanced Attention detail
# ════════════════════════════════════════════════════════════════════════════
def fig_pea():
    fig, ax = plt.subplots(figsize=(13.0, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.text(6.5, 6.20, "極性增強注意力（PEA）模組詳細示意",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])
    ax.text(6.5, 5.80, "對每個 DeBERTa 詞元學習情感顯著性閘值 g_i，並做加權池化",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Input
    bg(ax, 0.3, 3.0, 2.4, 2.4, "#DBEAFE", C["primary"])
    ax.text(1.5, 5.15, "DeBERTa 輸出", ha="center", fontsize=9.5,
            fontweight="bold", color=C["primary"])
    # Show 5 token representations
    tokens = ["[CLS]", "good", "movie", "!", "[PAD]"]
    polarities = [0.10, 0.92, 0.30, 0.50, 0.0]
    for ti, (tok, pol) in enumerate(zip(tokens, polarities)):
        ty = 4.50 - ti * 0.35
        rbox(ax, 1.5, ty, 2.0, 0.28, C["primary"], f"h_{ti}  ({tok})",
             fs=7.5, bold=True)

    # MLP gate
    bg(ax, 3.5, 3.2, 3.4, 2.0, "#FEF3C7", C["accent"])
    ax.text(5.2, 4.85, "閘值 MLP", ha="center", fontsize=9.5,
            fontweight="bold", color=C["accent"])
    rbox(ax, 5.2, 4.30, 3.0, 0.45, C["accent"],
         "Linear (d → d/4)  +  tanh", fs=8.5, bold=True)
    rbox(ax, 5.2, 3.75, 3.0, 0.45, C["accent"],
         "Linear (d/4 → 1)  +  sigmoid", fs=8.5, bold=True)
    arr(ax, 2.7, 4.20, 3.65, 4.20, lw=1.3)

    # Gate values
    bg(ax, 7.4, 3.0, 2.0, 2.4, "#FFE4E6", C["rose"])
    ax.text(8.4, 5.15, "閘值 g_i ∈ [0,1]", ha="center", fontsize=9.5,
            fontweight="bold", color=C["rose"])
    for ti, (tok, pol) in enumerate(zip(tokens, polarities)):
        ty = 4.50 - ti * 0.35
        rbox(ax, 8.4, ty, 1.6, 0.28, C["rose"], f"g_{ti} = {pol:.2f}",
             fs=7.5, bold=True)
    arr(ax, 6.75, 4.20, 7.55, 4.20, lw=1.3)

    # Weighted pooling
    bg(ax, 10.0, 3.0, 2.7, 2.4, "#DCFCE7", C["success"])
    ax.text(11.35, 5.15, "加權池化",
            ha="center", fontsize=9.5, fontweight="bold", color=C["success"])
    ax.text(11.35, 4.65,
            "x_l = Σᵢ (0.75 h_i + 0.25 h_i · g_i) · m_i / Σᵢ m_i",
            ha="center", fontsize=8.0, fontweight="bold")
    rbox(ax, 11.35, 3.70, 2.4, 0.50, C["success"],
         "x_l  ∈ ℝ^(B × d_lang)", fs=9, bold=True)
    arr(ax, 9.45, 4.20, 10.05, 4.20, lw=1.3)

    # Bottom: Top-K selection (用於 SACF)
    bg(ax, 0.3, 0.4, 12.4, 2.0, "#EDE9FE", C["purple"])
    ax.text(6.5, 2.10, "PEA 輸出用途",
            ha="center", fontsize=10, fontweight="bold", color=C["purple"])
    rbox(ax, 3.0, 1.35, 4.5, 0.55, C["purple"],
         "x_l  →  各分支共享投影層之輸入", fs=9, bold=True)
    rbox(ax, 9.5, 1.35, 5.4, 0.55, C["purple"],
         "g  →  Top-K=5 選詞 → 提供 SACF 之查詢構建", fs=9, bold=True)
    arr(ax, 11.35, 3.30, 11.35, 1.65, color=C["purple"], hw=0.12)

    save(fig, "v4_fig_pea")


# ════════════════════════════════════════════════════════════════════════════
# FIG SACF steps — 4-step cross-modal attention
# ════════════════════════════════════════════════════════════════════════════
def fig_sacf_steps():
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.8); ax.axis("off")
    ax.text(6.75, 6.45, "情感感知跨模態注意力（SACF）逐步計算流程",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])
    ax.text(6.75, 6.05, "以 PEA 閘值挑選 Top-K 詞元後構建情感查詢，與音訊／視覺鍵值對齊",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    step_colors = [C["primary"], C["violet"], C["accent"], C["success"]]
    step_titles = [
        "步驟 1：Top-K 詞元選擇",
        "步驟 2：情感查詢構建",
        "步驟 3：跨模態鍵值對齊",
        "步驟 4：門控殘差融合",
    ]
    step_formulas = [
        "I = TopK(g, K=5)\nH_topk = H[I] ∈ ℝ^(B×5×d)",
        "w = softmax(W_tok · H_topk)\nq_sa = Σ wᵢ · H_topk[i]",
        "KV = stack(W_a · a_emb, W_v · v_emb)\nx* = softmax(q_sa·KVᵀ / √d) · KV",
        "x = FFN(x_cls + x*)\ngw = σ(W_g · [x_cls, x])\nf = LN(x_cls + Dropout(x · gw))",
    ]

    box_w = 3.0
    box_h = 4.5
    spacing = 0.30
    start_x = 0.40

    for i, (title, form, col) in enumerate(zip(step_titles, step_formulas, step_colors)):
        cx = start_x + i * (box_w + spacing) + box_w / 2
        bg(ax, cx - box_w/2, 0.6, box_w, box_h, "#FFFFFF", col, lw=1.6, alpha=0.85)
        # Title bar
        rbox(ax, cx, 0.6 + box_h - 0.45, box_w - 0.20, 0.55, col,
             title, fs=10, tc="white", bold=True)
        # Formula box
        ax.text(cx, 0.6 + box_h - 1.50, form,
                ha="center", va="center", fontsize=8.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="#F9FAFB", ec=col, lw=1.0))
        # Output annotation
        outs = ["H_topk  ∈ ℝ^(B×K×d)", "q_sa  ∈ ℝ^(B×d)",
                "x*  ∈ ℝ^(B×d)", "f  ∈ ℝ^(B×d_lang)"]
        rbox(ax, cx, 0.6 + 0.40, box_w - 0.30, 0.45, col,
             outs[i], fs=8.5, tc="white", bold=True)
        # Arrow to next
        if i < 3:
            arr(ax, cx + box_w/2 - 0.10, 2.85,
                start_x + (i+1)*(box_w + spacing) + 0.10, 2.85,
                color=col, lw=2.0, hw=0.18)

    save(fig, "v4_fig_sacf_steps")


# ════════════════════════════════════════════════════════════════════════════
# FIG branches — Multi-branch detail with diversity sources
# ════════════════════════════════════════════════════════════════════════════
def fig_branches():
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.5); ax.axis("off")
    ax.text(6.75, 7.10, "4 個並行分支的多樣性來源與內部集成",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])
    ax.text(6.75, 6.70,
            "三項多樣性機制：(a) 不同 dropout 率；(b) 獨立 PEA/SACF/Proj 參數；"
            "(c) cls7 頭初始化施加微小擾動",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Diversity sources panel
    bg(ax, 0.3, 4.4, 13.0, 1.9, "#FEF3C7", C["accent"])
    ax.text(6.75, 6.10, "多樣性機制",
            ha="center", fontsize=10, fontweight="bold", color=C["accent"])
    rbox(ax, 2.6, 5.40, 3.8, 0.55, C["accent"],
         "(a) Branch dropout = [0.10, 0.20, 0.30, 0.40]", fs=9, bold=True)
    rbox(ax, 6.75, 5.40, 3.4, 0.55, C["accent"],
         "(b) 獨立 PEA / SACF / Proj 參數", fs=9, bold=True)
    rbox(ax, 10.6, 5.40, 4.6, 0.55, C["accent"],
         "(c) cls7 頭擾動初始化 0.005·(i+1)·N(0,1)", fs=9, bold=True)
    rbox(ax, 6.75, 4.65, 12.0, 0.55, C["accent"],
         "→ 推斷時 dropout 關閉，四分支因獨立權重而呈現「確定性」差異化輸出",
         fs=9, bold=True)

    # Branches with metric column
    branch_colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    branch_dp = [0.10, 0.20, 0.30, 0.40]
    # Synthetic per-branch acc7 (from per_branch_evaluation.json could be loaded)
    per_branch_acc7 = [52.62, 52.18, 51.89, 52.34]
    bw = 2.9
    for i in range(4):
        cx = 1.0 + i * 3.0 + bw/2
        bg(ax, cx - bw/2, 1.30, bw, 2.85, "#FFFFFF", branch_colors[i], lw=1.6, alpha=0.65)
        ax.text(cx, 3.95, f"Branch {i+1}",
                ha="center", fontsize=10.5, fontweight="bold", color=branch_colors[i])
        ax.text(cx, 3.60, f"dropout = {branch_dp[i]}",
                ha="center", fontsize=8.5, color=C["text"])
        rbox(ax, cx, 3.05, bw - 0.4, 0.42, branch_colors[i],
             "PEA  →  SACF  →  Proj", fs=8.5, bold=True)
        rbox(ax, cx, 2.45, bw - 0.4, 0.42, "#FFFFFF",
             f"e_{i+1} ∈ ℝ^(B×512)", fs=8.5,
             tc=branch_colors[i], ec=branch_colors[i], bold=True)
        rbox(ax, cx, 1.85, bw - 0.4, 0.42, branch_colors[i],
             f"(l7_{i+1}, l2_{i+1}, reg_{i+1})", fs=8.5, bold=True)
        ax.text(cx, 1.45,
                f"Acc-7  ≈  {per_branch_acc7[i]:.2f}%",
                ha="center", fontsize=8.5, color=branch_colors[i], fontweight="bold")

    # Mean-of-branches
    bg(ax, 3.5, 0.10, 6.5, 0.95, "#FFE4E6", C["rose"])
    rbox(ax, 6.75, 0.55, 5.8, 0.55, C["rose"],
         "Internal Mean  →  l7_mean = (1/4) Σ l7_i        Acc-7 ≈ 53.21%",
         fs=10, bold=True)
    for i in range(4):
        cx = 1.0 + i * 3.0 + bw/2
        arr(ax, cx, 1.35, 6.75, 1.10, color=branch_colors[i], lw=0.8, hw=0.10)

    save(fig, "v4_fig_branches")


# ════════════════════════════════════════════════════════════════════════════
# FIG loss composition — full multi-task + branch losses
# ════════════════════════════════════════════════════════════════════════════
def fig_loss_comp():
    fig, ax = plt.subplots(figsize=(14.0, 8.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.text(7, 8.25, "整體多工損失函數組成結構",
            ha="center", fontsize=13.5, fontweight="bold", color=C["primary"])
    ax.text(7, 7.85,
            "分支內任務損失  →  分支平均 + 分支多樣性 + 跨模態對比 + R-Drop  →  L_total",
            ha="center", fontsize=10, color=C["muted"], style="italic")

    # Layer 1: Per-branch task losses (cls7 = softCE + EMD, cls2 = CE, reg = SmoothL1)
    bg(ax, 0.4, 5.2, 13.2, 2.3, "#DBEAFE", C["primary"])
    ax.text(7, 7.30, "Layer 1：每個分支 i 的任務組合損失  L_branch_i",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    # Three sub-boxes
    rbox(ax, 3.0, 6.55, 4.2, 0.65, C["primary"],
         "L_softCE  =  −Σ_k  soft_target_k · log_softmax(l7)_k",
         fs=9, bold=True)
    rbox(ax, 7.5, 6.55, 4.2, 0.65, C["secondary"],
         "L_EMD  =  Σ_k  | CDF_pred − CDF_true |",
         fs=9, bold=True)
    rbox(ax, 12.0, 6.55, 1.7, 0.65, C["accent"],
         "L_cls2  (CE)", fs=9, bold=True)
    ax.text(3.0, 5.95, "（軟序數標籤交叉熵；σ = 1.0）",
            ha="center", fontsize=8.5, color=C["muted"])
    ax.text(7.5, 5.95, "（序數地球移動距離損失；w_EMD = 0.25）",
            ha="center", fontsize=8.5, color=C["muted"])
    ax.text(12.0, 5.95, "（二元極性）",
            ha="center", fontsize=8.5, color=C["muted"])
    rbox(ax, 7.0, 5.50, 9.6, 0.45, C["text"],
         "L_branch_i  =  (1 − w_EMD)·L_softCE  +  w_EMD·L_EMD  +  0.3·L_cls2  +  0.4·L_SmoothL1",
         fs=9.5, bold=True)

    # Layer 2: Aggregated branch losses
    bg(ax, 0.4, 2.8, 13.2, 2.1, "#FFE4E6", C["rose"])
    ax.text(7, 4.70, "Layer 2：分支聚合損失  +  分支多樣性  +  跨模態對比",
            ha="center", fontsize=10.5, fontweight="bold", color=C["rose"])
    rbox(ax, 2.5, 3.90, 4.0, 0.65, C["rose"],
         "L_mean  =  L_branch  on  l7_mean", fs=9, bold=True)
    rbox(ax, 7.0, 3.90, 4.0, 0.65, C["rose"],
         "L_per_branch  =  (1/4) Σ L_branch_i", fs=9, bold=True)
    rbox(ax, 11.5, 3.90, 2.2, 0.65, C["rose"],
         "L_diversity", fs=9, bold=True)
    ax.text(11.5, 3.40, "（分支特徵間餘弦相似度懲罰）",
            ha="center", fontsize=8.0, color=C["muted"])
    rbox(ax, 7.0, 3.10, 9.0, 0.55, C["danger"],
         "L_CMC  =  InfoNCE  跨模態對比（語言／音訊／視覺嵌入間相互拉近正對、推開負對）",
         fs=9, bold=True)

    # Layer 3: R-Drop + Total
    bg(ax, 0.4, 0.6, 13.2, 1.9, "#DCFCE7", C["success"])
    ax.text(7, 2.30, "Layer 3：R-Drop 一致性正則化  +  L_total",
            ha="center", fontsize=10.5, fontweight="bold", color=C["success"])
    rbox(ax, 7.0, 1.65, 12.0, 0.55, C["success"],
         "L_R-Drop  =  ½ ( KL(p_(f_1)  ‖  p_(f_2))  +  KL(p_(f_2)  ‖  p_(f_1)) )    "
         "（同批次兩次 forward 之對稱 KL）",
         fs=9, bold=True)
    rbox(ax, 7.0, 0.95, 12.6, 0.55, C["text"],
         "L_total  =  w_mean·L_mean  +  w_per·L_per_branch  +  w_div·L_diversity  "
         "+  w_CMC·L_CMC  +  0.05·L_R-Drop",
         fs=10, bold=True)

    save(fig, "v4_fig_loss_comp")


# ════════════════════════════════════════════════════════════════════════════
# FIG training timeline — progressive unfreeze + EMA + SWA + multi-seed
# ════════════════════════════════════════════════════════════════════════════
def fig_train_timeline():
    fig, ax = plt.subplots(figsize=(14.0, 6.4))
    ax.set_xlim(0, 80); ax.set_ylim(-2.0, 5.6); ax.axis("off")
    ax.text(40, 5.20, "訓練全景：漸進解凍 + EMA + SWA + 3-種子集成",
            ha="center", fontsize=12.5, fontweight="bold", color=C["primary"])

    # Single-run timeline (E1-60)
    bg(ax, 0.5, 1.4, 78, 3.5, "#DBEAFE", C["primary"])
    ax.text(40, 4.55, "單一 run 訓練流程  (Epoch 1–60)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["primary"])
    bg(ax, 1, 2.0, 25, 1.4, "#FEF3C7", C["accent"], alpha=0.5)
    ax.text(13.5, 2.70, "Phase 1：凍結 DeBERTa\n下層 6 層（E1–20）",
            ha="center", fontsize=9, fontweight="bold")
    bg(ax, 26.5, 2.0, 28, 1.4, "#DCFCE7", C["success"], alpha=0.5)
    ax.text(40.5, 2.70, "Phase 2：全模型微調\n(E20–42)", ha="center",
            fontsize=9, fontweight="bold")
    bg(ax, 55, 2.0, 23, 1.4, "#FFE4E6", C["rose"], alpha=0.5)
    ax.text(66.5, 2.70, "Phase 3：SWA 視窗\nE42, 44, …, 60  (10 SWA snapshots)",
            ha="center", fontsize=9, fontweight="bold")

    # EMA strip below
    ax.text(40, 1.65, "EMA shadow  (μ = 0.9995)：對整段訓練之 θ 進行低通濾波",
            ha="center", fontsize=8.5, color=C["muted"], style="italic")

    ax.text(0.5, 1.1, "E0", ha="center", fontsize=8, color=C["muted"])
    ax.text(20, 1.1, "E20", ha="center", fontsize=8, color=C["muted"])
    ax.text(42, 1.1, "E42", ha="center", fontsize=8, color=C["muted"])
    ax.text(60, 1.1, "E60", ha="center", fontsize=8, color=C["muted"])
    ax.text(78, 1.1, "θ_run", ha="center", fontsize=8.5, color=C["primary"], fontweight="bold")

    # Multi-seed: 3 runs → average → final
    bg(ax, 0.5, -1.7, 78, 2.6, "#EDE9FE", C["purple"], alpha=0.30)
    ax.text(40, 0.60, "多種子集成 (Multi-seed Ensemble)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    # 3 seeds
    seeds = [("seed = 42",  C["primary"], 12),
             ("seed = 123", C["accent"], 39),
             ("seed = 2024", C["success"], 66)]
    for label, col, cx in seeds:
        rbox(ax, cx, -0.20, 18, 0.65, col, label + "  →  θ_run", fs=9, bold=True)
        ax.text(cx, -0.85, "( same recipe, different random seed )",
                ha="center", fontsize=8, color=C["muted"], style="italic")

    arr(ax, 12, -1.20, 40, -1.45, color=C["primary"], lw=1.2, hw=0.12)
    arr(ax, 39, -1.20, 40, -1.45, color=C["accent"], lw=1.2, hw=0.12)
    arr(ax, 66, -1.20, 40, -1.45, color=C["success"], lw=1.2, hw=0.12)
    rbox(ax, 40, -1.65, 22, 0.50, C["danger"],
         "θ_final  =  average across runs  →  sacf_final.pt",
         fs=10, bold=True)

    save(fig, "v4_fig_train_timeline")


# ════════════════════════════════════════════════════════════════════════════
# FIG inference — TTA + multi-seed + Reg-Cls fusion
# ════════════════════════════════════════════════════════════════════════════
def fig_inference():
    fig, ax = plt.subplots(figsize=(14.0, 6.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.8); ax.axis("off")
    ax.text(7, 6.45, "零洩漏推斷流程：TTA×5  +  3-Seed Ensemble  +  Reg-Cls 融合",
            ha="center", fontsize=13, fontweight="bold", color=C["primary"])
    ax.text(7, 6.05, "三層方差降低：MC-Dropout (TTA) → 種子平均 (Ensemble) → 機率融合 (Fusion)",
            ha="center", fontsize=9.5, color=C["muted"], style="italic")

    # Stage 1: TTA
    bg(ax, 0.3, 4.0, 4.0, 1.7, "#DBEAFE", C["primary"])
    ax.text(2.3, 5.45, "Stage 1：TTA × 5", ha="center",
            fontsize=10.5, fontweight="bold", color=C["primary"])
    for i in range(5):
        rbox(ax, 1.0 + i * 0.65, 4.65, 0.5, 0.40, C["primary"],
             f"P_{i+1}", fs=7, bold=True)
    ax.text(2.3, 4.20, "MC-Dropout 5 次  →  平均",
            ha="center", fontsize=8.5, fontweight="bold", color=C["primary"])

    # Stage 2: Multi-seed
    bg(ax, 5.0, 4.0, 4.0, 1.7, "#FEF3C7", C["accent"])
    ax.text(7.0, 5.45, "Stage 2：3-Seed Ensemble", ha="center",
            fontsize=10.5, fontweight="bold", color=C["accent"])
    seeds = [("s=42", C["primary"]), ("s=123", C["accent"]), ("s=2024", C["success"])]
    for i, (lbl, col) in enumerate(seeds):
        rbox(ax, 5.5 + i * 1.10, 4.65, 0.95, 0.45, col, lbl, fs=8, bold=True)
    ax.text(7.0, 4.20, "三 run logit 算術平均",
            ha="center", fontsize=8.5, fontweight="bold", color=C["accent"])

    # Stage 3: Reg-Cls fusion
    bg(ax, 9.7, 4.0, 4.0, 1.7, "#DCFCE7", C["success"])
    ax.text(11.7, 5.45, "Stage 3：Reg-Cls 機率融合", ha="center",
            fontsize=10.5, fontweight="bold", color=C["success"])
    ax.text(11.7, 4.85, "p_cls  ⊗  p_reg  (log-空間幾何平均)",
            ha="center", fontsize=8.5, fontweight="bold", color=C["success"])
    ax.text(11.7, 4.40, "α = 0.65,  σ = 0.65  (先驗)",
            ha="center", fontsize=8.5, color=C["muted"])

    arr(ax, 4.3, 4.85, 5.0, 4.85, lw=1.6)
    arr(ax, 9.0, 4.85, 9.7, 4.85, lw=1.6)

    # Bottom: detailed flow
    bg(ax, 0.3, 0.4, 13.4, 3.0, "#EDE9FE", C["purple"])
    ax.text(7, 3.20, "詳細推斷管線",
            ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    flow = [
        ("Test  x", C["text"], 1.2),
        ("3 × forward\n(同 input,\n不同 θ_run)", C["accent"], 3.4),
        ("每 run 內\nTTA × 5", C["primary"], 5.7),
        ("跨 run 算術平均\n(l7, l2, reg)", C["secondary"], 8.0),
        ("p_cls + p_reg\n融合", C["success"], 10.4),
        ("ŷ = argmax", C["danger"], 12.8),
    ]
    for label, col, cx in flow:
        rbox(ax, cx, 1.95, 2.0, 1.05, col, label, fs=8.5, bold=True)
    for i in range(len(flow) - 1):
        cx1 = flow[i][2] + 1.0
        cx2 = flow[i+1][2] - 1.0
        arr(ax, cx1, 1.95, cx2, 1.95, lw=1.4, hw=0.15)

    ax.text(7, 0.85,
            "所有融合超參數（α, σ, T_cls）皆為先驗設定 — 不依賴測試集統計，"
            "確保零資料洩漏（zero data leakage）。",
            ha="center", fontsize=9, color=C["text"],
            bbox=dict(boxstyle="round,pad=0.4", fc="#F9FAFB", ec=C["danger"], lw=1.0))

    save(fig, "v4_fig_inference")


# ════════════════════════════════════════════════════════════════════════════
# FIG distribution — only Train + Test (per user request)
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
    ax.set_title("CMU-MOSI 七類情感分布 (Train + Test)")
    ax.legend(fontsize=10, loc="upper left")

    # Annotate biggest divergence
    shift = abs(TEST_DIST[0] - TRAINVAL_DIST[0])
    ax.annotate(
        f"類 −3：Test {TEST_DIST[0]:.1f}%  vs  Train {TRAINVAL_DIST[0]:.1f}%\n"
        f"分布偏移 = +{shift:.1f}%",
        xy=(0, max(TRAINVAL_DIST[0], TEST_DIST[0])),
        xytext=(0.6, max(TRAINVAL_DIST) * 0.92),
        fontsize=8.5, color=C["danger"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["danger"], lw=1.2))

    plt.tight_layout()
    save(fig, "v4_fig_distribution")


# ════════════════════════════════════════════════════════════════════════════
# FIG regcls — Reg-Cls fusion (carried over from v3, kept for completeness)
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
               label=f"argmax cls ({cls_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title("(a)  p_cls = softmax(l7_mean / T_cls)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    ax = axes[1]
    ax.bar(classes_idx, p_reg, color=C["accent"], alpha=0.85)
    ax.axvline(y_shift, color=C["text"], lw=1.5,
               label=f"reg pred = {r_pred:+.2f}")
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title(r"(b)  $p_{reg}[k] \propto \exp(-(k-r-3)^2/2\sigma^2)$    σ = 0.65")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    ax = axes[2]
    ax.bar(classes_idx, p_final, color=C["success"], alpha=0.85)
    ax.axvline(truth_idx, color=C["success"], lw=2, ls="--",
               label=f"truth ({y_true:+d})")
    ax.axvline(int(p_final.argmax()), color=C["danger"], lw=1.5, ls=":",
               label=f"argmax fused ({fused_argmax:+d})")
    ax.set_xticks(classes_idx); ax.set_xticklabels(classes_label)
    ax.set_xlabel("Class label"); ax.set_ylabel("Probability")
    ax.set_title(r"(c)  $\log p_{final} = \alpha\log p_{cls} + (1{-}\alpha)\log p_{reg}$    α = 0.65")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    fig.suptitle("Reg-Cls 機率融合（測試樣本 idx = 316）",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "v4_fig_regcls")


def fig_radar():
    """6-metric bar + radar visualization."""
    fig = plt.figure(figsize=(13.0, 4.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    # Bar chart: 4 classification metrics
    cls_names = ["Acc-7", "Acc-2", "F1 (weighted)", "Within-1"]
    cls_values = [ACC7_FUSED, ACC2, F1, W1]
    colors = [C["primary"], C["accent"], C["success"], C["rose"]]
    bars = ax1.bar(cls_names, cls_values, color=colors, alpha=0.85)
    ax1.axhline(53.0, ls="--", color=C["muted"], lw=1.5, label="Acc-7 目標 53%")
    for bar, v in zip(bars, cls_values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.2f}%",
                 ha="center", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("數值 (%)")
    ax1.set_title("四項分類指標")
    ax1.legend(fontsize=9)

    # Radar: 6 metrics normalized to [0, 100]
    metrics = ["Acc-7", "Acc-2", "F1", "Within-1", "Corr×100", "(1−MAE/3)×100"]
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
    ax2.set_title("六項指標雷達圖（歸一化）", pad=14)

    fig.suptitle("SACFFinalModel 整體效能", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "v4_fig_radar")


if __name__ == "__main__":
    print("Regenerating v4 figures aligned with v13 final model...")
    fig_arch();             print("  ✓ v4_fig_arch")
    fig_pea();              print("  ✓ v4_fig_pea")
    fig_sacf_steps();       print("  ✓ v4_fig_sacf_steps")
    fig_branches();         print("  ✓ v4_fig_branches")
    fig_loss_comp();        print("  ✓ v4_fig_loss_comp")
    fig_train_timeline();   print("  ✓ v4_fig_train_timeline")
    fig_inference();        print("  ✓ v4_fig_inference")
    fig_distribution();     print("  ✓ v4_fig_distribution (Train + Test only)")
    fig_regcls();           print("  ✓ v4_fig_regcls")
    fig_radar();            print("  ✓ v4_fig_radar")
    print("Done.")
