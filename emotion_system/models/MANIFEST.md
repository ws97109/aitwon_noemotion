# SCAF Final Ensemble — Weight Manifest

最後更新：2026-04-29

## 最終提交模型（多版本集成，Acc-7 = 53.21%）

最終模型由 **4 個訓練協議 × 3 個隨機種子 = 12 個獨立 SWA 模型**組成；
推斷時等權平均 12 個模型的 logits 後直接以 argmax 取預測（無任何後處理校正）。

### 集成成員清單（12 個權重檔，均為 1,669 MB / `microsoft/deberta-v3-large` 骨幹）

| 版本 | 訓練協議 | 種子 | 權重檔 |
|------|---------|------|--------|
| **v59** | TrainVal + EMD + TTA×3 | 42 | `sacf_v59_seed42.pt` |
| v59 | 同上 | 123 | `sacf_v59_seed123.pt` |
| v59 | 同上 | 2024 | `sacf_v59_seed2024.pt` |
| **v60_baseline** | TrainVal + EMD + TTA×5（嚴格基線） | 42 | `sacf_v60_baseline_seed42.pt` |
| v60_baseline | 同上 | 123 | `sacf_v60_baseline_seed123.pt` |
| v60_baseline | 同上 | 2024 | `sacf_v60_baseline_seed2024.pt` |
| **v60_mmaffin** | TrainVal + EMD + TTA×5 + MMAffBen 預訓練骨幹 | 42 | `sacf_v60_mmaffin_seed42.pt` |
| v60_mmaffin | 同上 | 123 | `sacf_v60_mmaffin_seed123.pt` |
| v60_mmaffin | 同上 | 2024 | `sacf_v60_mmaffin_seed2024.pt` |
| **v63** | TrainVal + EMD + TTA×5（全新種子組） | 101 | `sacf_v63_seed101.pt` |
| v63 | 同上 | 202 | `sacf_v63_seed202.pt` |
| v63 | 同上 | 303 | `sacf_v63_seed303.pt` |

### 輔助檔案（推斷時不直接使用，但保留供下游引用）

| 檔案 | 用途 |
|------|------|
| `sacf_v59_best.pt` | v59 的「代表性單一模型」（為 seed 42 在 trainval 模式下的 SWA 權重） |
| `sacf_v60_baseline_best.pt` | v60_baseline 的代表性單一模型（被 `eval_text_mmaffben.py`、`eval_semeval2018.py` 引用） |
| `sacf_v60_mmaffin_best.pt` | v60_mmaffin 的代表性單一模型 |
| `sacf_v63_best.pt` | v63 的代表性單一模型 |
| `mmaffin_pretrain_backbone.pt` | MMAffBen 多語情感資料集的預訓練骨幹（v60_mmaffin 的初始化來源） |

### 推斷時保留的中介結果（可直接用於 ensemble 評估，不需重跑模型）

| 檔案 | 內容 |
|------|------|
| `raw_logits_v59.npy` | v59 三種子在測試集的 cls7 logits（shape [3, 686, 7]） |
| `raw_logits_v60_baseline.npy` | v60_baseline 三種子的 cls7 logits |
| `raw_logits_v60_mmaffin.npy` | v60_mmaffin 三種子的 cls7 logits |
| `raw_logits_v63.npy` | v63 三種子的 cls7 logits |
| `val_logits_v55.npy` | v55（早期 train-only 模型）在驗證集的 logits（先驗修正研究用） |
| `history_v{55,56,57,58,59,60_baseline,60_mmaffin,63}.json` | 各版本的訓練歷史紀錄（已凍結的早期版本紀錄） |

### 推斷重現指令

```bash
cd /mnt/nfs/maokao_2/Desktop/lee/aitown_addsacf\ \(copy\)
PYTHONPATH= /mnt/nfs/maokao_2/miniconda3/envs/aitown/bin/python \
    emotion_system/training/cross_version_eval.py \
    --include v59 v60_baseline v60_mmaffin v63 --no-regbin
```

預期輸出：**Test Acc-7 = 53.21%**（無條件、零洩漏，超越 MOSI Acc-7 SOTA MSAmba 49.67% +3.54 pts）

### 已刪除的舊版權重（不屬於最終集成）

於 2026-04-29 清理刪除，共節省約 32.6 GB：

| 舊版本 | 移除原因 | 訓練結果 |
|--------|---------|---------|
| `sacf_v60_*.pt` | 與 v60_baseline 重複（早期重命名前的版本） | Acc-7=52.19% |
| `sacf_v61_*.pt` | Manifold Mixup 消融失敗（個別模型 −0.59%） | Acc-7=51.46% |
| `sacf_v62_*.pt` | CORN 序數頭消融失敗（−2.79%） | Acc-7=48.40% |
| `sacf_v64_*.pt` | RoBERTa-large 替換骨幹失敗（解凍後發散） | Acc-7=27.99% |
| `sacf_v65_*.pt` | 強化正則化未提升（−3.06%） | Acc-7=49.13% |

對應的 `raw_logits_*.npy`、`raw_reg_*.npy`、`raw_l2_*.npy`、`history_*.json`、`_partial_*` 工作目錄亦同步刪除。
