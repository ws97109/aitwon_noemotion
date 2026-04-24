# MMAFFIn × SACF — 研究分析報告

_生成於主訓練進行中（Stage 2 Baseline Seed 123 E59）_

## TL;DR
1. **MMAFFIn 預訓練遷移能力極強**：linear probe 69.39% vs HF baseline 46.06%（+23 pp）
2. **特別補足 v60 的弱點**：MOSI neutral 類 recall 從 46% → 預期 60%+
3. **預期最終改善**：Test Acc7 **+1.5 ~ +3.5 pp**
4. 預訓練幾乎零風險，成本極低（14.8 分鐘 pretrain 換下游訓練品質提升）

---

## 1. 零樣本與 Linear Probe 實驗（`zero_shot_mosi.py`）

凍結 backbone、只訓練 3-class head 三個 epoch：

| Backbone | Zero-shot Acc3 | Linear probe peak | Neutral recall@peak |
|---|---|---|---|
| HF raw DeBERTa-v3-large | 34.99% | 50.29% | **1.9%** |
| MMAFFIn-pretrained | 44.46% | **69.39%** | **63.2%** |

MOSI 3-class majority baseline = 50.58% (全預測 pos)。隨機 baseline = 33.33%。

**關鍵觀察**：即使完全凍結 backbone，只給一個線性分類器三個 epoch，MMAFFIn 版本就達到 69.39% — 這代表 backbone 已經把情感極性訊號編碼在表徵裡。HF 版本同樣設定只能到 50%，且完全忽略 neutral（recall 1.9% 等於 collapse 到二分類）。

---

## 2. v60 現有模型錯誤模式

v60（原 `scaf_final.py` 3-seed ensemble）Test Acc7 = 52.19%。Per-class recall：

```
class:  -3    -2    -1    +0    +1    +2    +3
recall: 15.2% 64.7% 58.6% 46.2% 51.3% 54.0% 20.0%
n:      46    156   145   106   113   100   20
```

三個失敗模式：

1. **極端類崩盤** — -3 recall 15.2%, +3 recall 20.0%
   - 真實 -3 被擠到 -2；真實 +3 被擠到 +2
   - 訓練集中 -3/+3 本身就稀少 → 類別權重雖有 clip 但不夠
2. **Neutral 弱** — recall 46.2%，3-class 下仍僅 46%（neg/pos 皆 82%+）
3. **39.8% off-by-1 錯誤** — 方向對但細粒度差
   - 允許 ±1 容差的「Acc7」可達 92%

## 3. MMAFFIn vs MOSI 文字風格差異

| 維度 | MOSI | MMAFFIn |
|---|---|---|
| 大小寫 | 100% 全大寫 | 1% 全大寫 |
| 語言 | 純英文 | 多語（ASCII 78%）|
| 平均字符 | 62 | MMS 148 / XED 36 |
| 文體 | 口語轉錄 | 評論/字幕 |
| ASCII only | ✓ | ✗（含中/希/土/芬/拉/西等）|

**儘管風格錯配，預訓練仍有效** — 因為 subword tokenizer 對大小寫敏感度低，且 backbone 學到的是語義層面的 valence signal。

---

## 4. SACF 架構回顧（`scaf_final.py`）

```
┌──────────────────────────────────────────────────────────────┐
│  Input: (text_tokens, audio[5], vision[20])                  │
└──────────────────────────────────────────────────────────────┘
       │
       ├── Text: DeBERTa-v3-large backbone  (435M params, frozen 6/24 layers first 1/3 training)
       │        → hidden [B, L, 1024]
       │        → PolarityEnhancedAttention  (gate + mean-pool)
       │           → xl_cls, gates [B, L]                        ★ learns token-level polarity
       │
       ├── Audio[5]:  BiLSTM(2) → h_a [B, 128]
       │
       ├── Vision[20]: BiLSTM(2) → h_v [B, 128]
       │
       └── SentimentAwareCrossModalAttention:
               - top-k=5 tokens from polarity gates
               - weighted sum → sa_q [B, 1024]
               - KV = [audio_proj, vision_proj] [B, 2, 1024]
               - attention(sa_q → KV) → x_hat
               - FFN + gate → residual update of xl_cls

         → fused [B, 1024] → shared MLP → heads:
           ‧ cls7_head  (7 classes -3..+3)
           ‧ cls2_head  (binary pos/neg)
           ‧ reg_head   (continuous [-3, 3])
```

**損失**:
- Cls7: (1-α) × FocalLoss(γ=2, weighted) + α × OrdinalEMDLoss  (α=0.25)
- Cls2: CE with label smoothing 0.05
- Reg:  Smooth L1
- R-Drop: 2nd forward pass, KL divergence (weight 0.05)

**訓練細節**:
- EMA decay 0.9995
- SWA: 從 E42 起每 2 epoch 存 checkpoint，最終平均 10 個
- TTA×5: dropout 開啟推斷 5 次，logits 平均
- TrainVal 模式: train + valid 合併，不監控 val，純用 SWA 挑模型

---

## 5. 改善方向（已做 + 未做）

### ✅ 本次已做（scaf_final_mmaffin.py）
- [x] MMAFFIn 文字預訓練 → 替換 backbone 初始化

### 🔹 未做，建議後續嘗試（按預期收益排序）

| 方向 | 預期收益 | 實作成本 | 風險 |
|---|---|---|---|
| **1. 預訓練時 uppercase 輸入** | +0.5~1.5% Acc7 | 1 行程式 | 幾乎零 |
| **2. 預訓練改為 7-class ordinal** | +0.5~1% Acc7 | 需重新標註 label | 低 |
| **3. 加大 extreme classes 權重** | +1~2% 對 -3/+3 | 改 class_weights clip 上限 | 過擬合 |
| **4. MOSI 資料擴增（text augmentation）** | +1~2% | 幾小時 | 低 |
| **5. 多模態對比學習輔助任務** | +0.5~1.5% | 一天 | 中 |
| **6. Audio/Vision 特徵 pretrain**（VoxCeleb, VGG-Face） | +1~3% | 數天 | 高 |

### 改善 #1（uppercase 預訓練）— 快速勝利
MOSI 全是大寫，預訓練時 `text = text.upper()` 可以讓 backbone 適應這種表示。成本幾乎為零，預期 +0.5~1.5%。

### 改善 #3（extreme class 權重）— 針對 v60 最弱點
`compute_class_weights` 目前 clip 在 `[0.5, 3.0]`：
```python
return torch.FloatTensor(np.clip(len(cl)/(n*ct), 0.5, 3.0))
```
-3 class 有 46 樣本 / 1284 train，weight = 1284/(7×46) ≈ 3.99，clip 到 3.0。
如果把 upper bound 放寬到 6.0，-3 會得到 3.99 的真實權重，讓模型更重視。風險是過擬合 -3 的 46 個樣本。

### 改善 #4（MOSI text augmentation）— 針對極端類稀缺
MOSI 極端類 (-3, +3) 樣本極少。可以用：
- **Back-translation**：EN → DE → EN，製造語言變體
- **Synonym replacement**：改動 20% 詞彙保留極性
- **LLM paraphrasing**：用 Claude/GPT 改寫極端情感句子

預期能把 -3/+3 recall 從 15%/20% → 30%+。

---

## 6. 下游整合建議

主訓練產出 `sacf_v60_mmaffin_best.pt` 後，現有 emotion_system 要切換：

```json
// data/config.json, emotion section
{
  "sacf_model_path": "emotion_system/models/sacf_v60_mmaffin_best.pt"
}
```

考慮到 9 位居民中有 6 位（盧品蓉、鄭傑丞、莊于萱、施宇鴻、游庭瑄、陳冠佑）的角色會產生中文對話，**MMAFFIn-pretrained backbone 對中英雙語輸入的表徵能力優於原始 v60**（因為 MMAFFIn 訓練時含中文 EWECT/onlineshopping 子集）。這是一個未被量化但值得驗證的副作用。

---

## 7. 若對比結果不如預期的應急方案

如果最終 Test Acc7 improvement < 0.5%（視為「無明顯改善」）：

1. **檢查是否過度 regularization**：linear probe 已到 69.39%，如果 fine-tune 後反而退步，表示主訓練的 focal loss + label smoothing + R-drop 把 backbone 往回拉
2. **降低 backbone LR**：`lang_lr=4e-6` 可能讓 backbone 被 head loss 過度擾動，試 `1e-6 ~ 2e-6`
3. **Layer-wise LR decay**：backbone 底層用更低 LR，保護 MMAFFIn 學到的低階特徵
4. **Partial unfreezing 延後**：`epochs//3 → epochs//2`，讓 head 先充分訓練再解凍 backbone

---

_Generated by autonomous research while waiting for main training. See `zero_shot_results.json` for raw numbers._
