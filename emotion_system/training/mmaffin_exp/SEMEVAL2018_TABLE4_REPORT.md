# SemEval-2018 純文字評估報告（對應論文 Table 4）

依論文 *MMAFFBen* (Liu et al. 2025) Table 4 的設定：

- **訓練集**：SemEval-2018 Task 1 的 train + dev（每任務每語言）
- **測試集**：SemEval-2018 Task 1 的 **test-gold**
- **基底模型**：`scaf_final.SACFModel` 純文字分支（DeBERTa-v3-large）
- **backbone init**：`sacf_v60_best.pt`
- **訓練超參**：每 (lang, task) 獨立訓練 3 epoch，bs=16, lang_lr=5e-06, head_lr=5e-05

**指標**（與論文一致）
- EI (Emotion Intensity): pcc，欄位 *EI ave* = anger/fear/joy/sadness 4 個 PCC 平均
- SP (Sentiment Polarity, V-oc): pcc on 整數類別 (-3..+3)
- SI (Sentiment Intensity, V-reg): pcc on 連續分數 [0,1]
- EC (E-c, multi-label, 11 emotions): jac (Jaccard, samples avg)、mi-F1、ma-F1
- Overall = pcc(EI) + pcc(SP) + pcc(SI) + ma-F1(EC) 之平均

## 主結果（對應 Table 4 LM-T 區塊）

| Models | EN EI ave | SP val | SI val | Ec jac | Ec mi-F1 | Ec ma-F1 | AR EI ave | SP val | SI val | Ec jac | Ec mi-F1 | Ec ma-F1 | ES EI ave | SP val | SI val | Ec jac | Ec mi-F1 | Ec ma-F1 | Overall |
|--------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|--------:|
| **SACF-Text (ours, sacf_v60 init)** | **72.57** | **83.44** | **82.25** | **58.64** | **70.61** | **53.91** | **42.93** | **73.58** | **74.63** | **45.92** | **59.62** | **43.43** | **68.39** | **82.43** | **80.96** | **49.75** | **61.24** | **45.37** | **66.99** |
| EmoLlama-chat-7b | 79.90 | 87.30 | 85.60 | 60.90 | 72.40 | 59.20 | 43.00 | 66.10 | 74.50 | 36.20 | 52.60 | 38.30 | 73.70 | 74.40 | 81.70 | 40.10 | 56.30 | 45.70 | 67.45 |
| Llama3.2-1b-instruct | 14.40 | 27.10 | 34.80 | 36.80 | 51.00 | 39.20 | 6.30 | 18.70 | 22.00 | 15.00 | 25.50 | 20.90 | 13.00 | 13.10 | 32.60 | 14.60 | 22.60 | 14.60 | 21.39 |
| Llama3.2-3b-instruct | 54.70 | 65.20 | 71.10 | 32.70 | 45.80 | 35.60 | 32.70 | 33.80 | 36.50 | 17.30 | 26.70 | 20.60 | 53.70 | 64.40 | 29.10 | 42.70 | 35.10 | 51.10 | 45.71 |
| Mistral-7b-instruct | 58.40 | 77.70 | 70.80 | 30.70 | 43.00 | 35.60 | 35.90 | 64.10 | 30.60 | 42.10 | 57.60 | 73.40 | 56.70 | 29.80 | 40.70 | 26.70 | 34.20 | 55.80 | 52.46 |
| GPT-4o-mini | 71.40 | 82.40 | 82.00 | 42.10 | 56.40 | 44.60 | 63.90 | 85.40 | 86.40 | 56.50 | 64.10 | 47.10 | 73.00 | 81.40 | 80.60 | 42.10 | 60.00 | 70.20 | 72.37 |
| MMAFFLM-3b | 64.60 | 76.80 | 82.50 | 46.60 | 61.60 | 47.50 | 62.10 | 78.40 | 81.80 | 45.10 | 63.50 | 73.60 | 79.10 | 38.40 | 44.00 | 43.70 | 53.10 | 56.10 | 65.41 |
| MMAFFLM-7b | 70.30 | 77.80 | 83.20 | 43.20 | 59.30 | 42.30 | 62.10 | 78.40 | 81.80 | 39.70 | 57.40 | 40.50 | 69.70 | 78.00 | 73.00 | 35.70 | 52.30 | 38.40 | 66.29 |

**Our overall (across 3 langs) = 66.99**

> Baseline 數值取自論文 Table 4 「LM-T」區塊（純文字模型）。
> EI 為 anger/fear/joy/sadness 4 emotion 的 pcc 平均；Overall = 4 task 主要指標的算術平均。

## 各語言詳細（含每 emotion 的 EI pcc）

### EN (Overall = 73.04)

- EI ave (pcc) = **72.57** ± 2.86 (anger=74.13, fear=69.21, joy=76.42, sadness=70.54)
- SP valence (pcc on V-oc) = **83.44**
- SI valence (pcc on V-reg) = **82.25**
- EC: jac=58.64, mi-F1=70.61, ma-F1=53.91 (best dev threshold = 0.35)

### AR (Overall = 58.64)

- EI ave (pcc) = **42.93** ± 9.95 (anger=29.53, fear=37.20, joy=51.84, sadness=53.13)
- SP valence (pcc on V-oc) = **73.58**
- SI valence (pcc on V-reg) = **74.63**
- EC: jac=45.92, mi-F1=59.62, ma-F1=43.43 (best dev threshold = 0.30)

### ES (Overall = 69.28)

- EI ave (pcc) = **68.39** ± 5.69 (anger=65.68, fear=60.86, joy=76.06, sadness=70.95)
- SP valence (pcc on V-oc) = **82.43**
- SI valence (pcc on V-reg) = **80.96**
- EC: jac=49.75, mi-F1=61.24, ma-F1=45.37 (best dev threshold = 0.30)

_耗時：1129s，設備：cuda_