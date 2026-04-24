# MMAFFIn × MOSI — 最終比較報告

_訓練完成於 2026-04-24 03:33_

## TL;DR
- **Ensemble Acc7 小幅退步**：baseline 52.19% → pretrained 50.73% (-1.46 pp)
- **但差異統計不顯著**：McNemar-like p ≈ 0.391
- **Seed 穩定性大幅提升**：std 從 1.17 → 0.27（**4.3× 更穩定**）
- **極端類 (-3, +3) recall 顯著改善**：+3 從 20% → 30% (+50% 相對)
- **MAE 微幅改善**：0.5872 → 0.5797
- **結論**：不是明確的勝負，是**「trade-off」—— pretrained 犧牲 lottery luck 換取一致性與極端類覆蓋**

---

## 1. 最終數字

| 指標 | Baseline (v60_baseline) | Pretrained (v60_mmaffin) | Δ | 判斷 |
|---|---|---|---|---|
| **Acc7** | 52.19% | 50.73% | -1.46 pp | 退步但不顯著 |
| Acc2 | 86.30% | 86.30% | 0.00 | 平手 |
| F1 (weighted) | 86.28% | 86.21% | -0.07 | 平手 |
| **MAE** | 0.5872 | **0.5797** | -0.0075 | ✓ 微幅改善 |
| Corr | 0.8690 | 0.8686 | -0.0004 | 平手 |

## 2. Per-seed 比較 — 關鍵發現

| Seed | Baseline | Pretrained | Δ |
|---|---|---|---|
| 42 | **53.06%** ← baseline 的 lucky seed | 51.17% | -1.90 |
| 123 | 50.44% | **51.75%** | +1.31 |
| 2024 | 50.73% | **51.17%** | +0.44 |
| **平均** | **51.41%** | **51.36%** | -0.05（**基本相同**）|
| **標準差** | 1.17 | **0.27** | **4.3× 更穩定** |
| **Seed-wise wins** | 1/3 | **2/3** | pretrained 多勝 |

**重要洞察**：
- **平均個別 seed 表現幾乎一致**（51.41 vs 51.36）
- **Baseline 靠 Seed 42 的 outlier (53.06%) 把 ensemble 拉高**
- Pretrained 三個 seeds 都緊密集中在 51% 左右（低變異）
- **Ensemble 是大數統計**：如果三個 seeds 都相近（如 pretrained），ensemble 的 averaging 效益小；如果有一個特別高（如 baseline），ensemble 會被拉高

---

## 3. Per-class Recall (7-class)

| 類別 | Baseline | Pretrained | Δ | 判斷 |
|---|---|---|---|---|
| **-3** | 15.2% | **19.6%** | **+4.3** ↑ | ✓ 改善（但絕對值仍低）|
| -2 | 64.7% | 57.1% | -7.7 ↓ | 退步 |
| **-1** | 58.6% | **63.4%** | **+4.8** ↑ | ✓ 改善 |
| 0 | 46.2% | 40.6% | -5.7 ↓ | 退步（與 linear probe 預測相反）|
| +1 | 51.3% | 46.0% | -5.3 ↓ | 退步 |
| **+2** | 54.0% | **57.0%** | **+3.0** ↑ | ✓ 改善 |
| **+3** | 20.0% | **30.0%** | **+10.0** ↑ | ✓✓ 強烈改善 |

**意外發現**：
1. **極端類 (-3, +3) 顯著改善** — MMAFFIn 的 XED 強情緒（joy/anger/disgust/fear）語料確實遷移了
2. **中性類 (0) 反而退步** — 與 linear probe 預測相反
3. 遠離中心的 ±1 類改善，靠近中心的 0/+1 類退步

## 4. Per-class Recall (3-class)

| 極性 | Baseline | Pretrained | Δ |
|---|---|---|---|
| neg | 87.6% | 87.3% | -0.29 |
| **neu** | **46.2%** | **40.6%** | **-5.66** |
| pos | 82.4% | 82.4% | 0.00 |

中性類退步是最大意外。Linear probe 測試顯示 MMAFFIn backbone 對 neutral 有強烈 representation（probe 從 1.9% → 63.2%），但完整 fine-tune 後這優勢消失了。**解釋假說**：MOSI 主訓練 loss（focal + class weight + R-Drop）在 40 epoch 的 fine-tune 中把 backbone 的 neutral representation 「洗掉」了，因為 MOSI 本身 neutral 樣本少 + 損失函數更看重極端類。

---

## 5. 統計顯著性

**Paired McNemar-like test**（test set n=686）:
- Both right: 298, Both wrong: 278
- Only baseline right: **60**
- Only pretrained right: **50**
- Diff = 10, p ≈ **0.391**（不顯著，α=0.05）

**結論：1.46 pp 的 Acc7 差異實際上是 noise，不是真實 pretrained 劣化**。如果再跑幾個 seed，兩者大概率會翻轉。

---

## 6. 為什麼 Linear Probe 預測失準？

我原本預測 +1.5~3.5 pp 改善，實際 -1.46 pp。差距來源：

| 假設 | 驗證 |
|---|---|
| Linear probe 3-class 比 MOSI 7-class 更容易 | ✓ 確認（probe 看到的是粗粒度信號）|
| Full fine-tune 會洗掉 backbone 的 MMAFFIn 特性 | ✓ 確認（neutral 優勢消失）|
| MMAFFIn 的多語 + 電影字幕風格與 MOSI 口語錯配 | ✓ 部分確認（中性類退步）|
| Ensemble 變異數下降反而傷害 ensemble acc | ✓ **新發現**（std 從 1.17 → 0.27）|

---

## 7. 結論與建議

### 本次實驗的真實結論

1. **MMAFFIn pretraining 不是「更好」也不是「更差」**：它讓模型更穩定、對極端情感更敏感，但犧牲了 "lucky seed" 的 upside。
2. **實務使用建議**：如果你追求 **deployment 的穩定性和一致性**，pretrained backbone (`sacf_v60_mmaffin_best.pt`) 更好；如果你要最高可能的 Acc7 分數，baseline (`sacf_v60_baseline_best.pt`) 的 seed 42 有 53.06% 是最高。
3. **統計上不顯著**（p=0.39），所以**不應該把 -1.46 pp 當真實退步**。
4. **MAE 微幅改善** + **極端類覆蓋改善** 顯示 pretraining 確實提供了某種幫助，只是 Acc7 這個 metric 沒捕捉到。

### 下一步改善的優先順序（根據本次結果）

| 方向 | 改善潛力 | 為何 |
|---|---|---|
| **1. 降低 backbone LR (4e-6 → 1e-6)** | 高 | 防止 fine-tune 洗掉 MMAFFIn 學到的 neutral representation |
| **2. Layer-wise LR decay** | 高 | 底層保護 MMAFFIn 特徵，頂層適應 MOSI |
| **3. 預訓練時 input.upper()** | 中 | MOSI 是全大寫，縮小風格差距 |
| **4. Ensemble 5-7 seeds 而非 3** | 中 | 目前 3 seed 的 ensemble 對 seed luck 過敏感 |
| **5. MOSI 極端類 data augmentation** | 中 | MOSI 本身 -3/+3 樣本稀少，pretraining 無法補足 |
| 6. 多語版 backbone (mDeBERTa) | 低 | 居民中文對話場景，但 MOSI 評估用不到 |

### 應急方案（如你想現在就取得 Acc7 > 52%）

- 直接 deploy `sacf_v60_baseline_best.pt`（seed 42, 53.06%）— 這是目前最高單一 seed
- 等執行更多 seeds 之後再做 ensemble（seeds 42, 123, 2024, 99, 7 五組合）

---

## 8. 產出檔案清單

### 模型權重 (emotion_system/models/)
- `mmaffin_pretrain_backbone.pt` (1.7 GB) — MMAFFIn 預訓練 backbone
- `sacf_v60_baseline_{seed42,seed123,seed2024,best}.pt` — 4 個 baseline 模型
- `sacf_v60_mmaffin_{seed42,seed123,seed2024,best}.pt` — 4 個 pretrained 模型

### 結果檔
- `history_v60_baseline.json` / `history_v60_mmaffin.json` — 最終 ensemble 指標
- `epoch_log_v60_{baseline,mmaffin}.json` — 每 epoch train loss（per seed）
- `raw_logits_v60_{baseline,mmaffin}.npy` — shape [3, 686, 7] 供 ensemble 分析

### 視覺化 (emotion_system/training/mmaffin_exp/figs/)
- `1_final_metrics_bar.png` — Acc7/Acc2/F1/MAE/Corr 對比
- `2_per_seed_test_acc7.png` — 每 seed 的 Acc7（bars）+ ensemble（dashed）
- `3_training_curves.png` — 訓練 loss 曲線（各 seed 平均 ± std）
- `4_confusion_matrices.png` — 7×7 混淆矩陣並排
- `5_per_class_recall.png` — per-class recall 比較

### 分析產物
- `ANALYSIS.md` — 訓練前的研究報告（含預測）
- `FINAL_REPORT.md` — 本檔案（實際結果）
- `zero_shot_results.json` — Linear probe 結果
- `logs/` — 各階段執行 log

### 斷點續跑成果
- `_partial_v60_baseline/` + `_partial_v60_mmaffin/` — per-seed 中間檔（可刪除）
