# MMAFFBen 純文字評估報告（Table 4 風格）

依論文 *MMAFFBen: A Multilingual and Multimodal Affective Analysis Benchmark* (Liu et al. 2025) 的訓練 / 測試協議：

- **訓練資料**：MMAFFIn `texts/*_train.json`（instruction-tuning 格式）
- **驗證資料**：MMAFFIn `texts/*_val.json`（過程監看，不用於早停）
- **測試資料**：**MMAFFBen** `texts/*_test.json`（最終指標來源）
- **基底模型**：`scaf_final.SACFModel` 的純文字分支 （DeBERTa-v3-large + PolarityEnhancedAttention + shared MLP + per-task head）
- **backbone 初始化**：`sacf_final.pt`（使用者 MOSI 訓練成果）
- **訓練超參數**：每任務獨立訓練 1 epoch，batch=16, lang_lr=5e-06, head_lr=5e-05（論文用 lr=5e-6）
- **指標**：SP / EC 任務皆報 macro-F1（ma-F1）；XED 多標籤額外報 Jaccard

## 主結果（macro-F1，越高越好）

| Models | EWECT-usual | EWECT-virus | MMS | XED | Onlineshopping | Average |
|--------|------:|------:|------:|------:|------:|--------:|
| **SACF-Text (ours, sacf_v60 backbone)** | **62.36** | **50.70** | **61.42** | **16.41** | **92.82** | **56.74** |
| EmoLlama-chat-7b | 45.60 | 36.50 | 44.00 | 48.60 | 20.30 | 39.00 |
| Llama3.2-1b-instruct | 24.20 | 18.00 | 49.10 | 30.90 | 28.70 | 30.18 |
| Llama3.2-3b-instruct | 51.90 | 39.90 | 52.20 | 33.10 | 11.90 | 37.80 |
| Mistral-7b-instruct | 45.10 | 36.50 | 55.40 | 33.10 | 23.30 | 38.68 |
| Llama3.2-11b-instruct | 42.30 | 35.80 | 57.40 | 37.50 | 20.00 | 38.60 |
| Qwen2.5-VL-7b | 46.30 | 38.10 | 56.20 | 31.30 | 12.80 | 36.94 |
| InternVL2.5-8B-MPO | 51.10 | 35.70 | 56.20 | 31.40 | 12.40 | 37.36 |
| GPT-4o-mini | 69.50 | 57.60 | 61.90 | 48.60 | 12.50 | 50.02 |
| MMAFFLM-3b | 66.90 | 60.30 | 93.90 | 43.50 | 26.50 | 58.22 |
| MMAFFLM-7b | 67.60 | 58.20 | 93.90 | 46.30 | 28.80 | 58.96 |

> 上表中「Average」= 列出任務之 ma-F1 平均。論文數值取自 Table 5 Text 區塊；MMAFFLM-3b/7b 為論文作者用 MMAFFIn 微調 Qwen2.5-VL 的成果，是同設定下的上界參照。

## 詳細指標（test set）

| Task | Tag | Lang | #cls | n_train | n_val | n_test | ma-F1 | 其他 |
|------|-----|------|-----:|--------:|------:|-------:|------:|------|
| Onlineshopping | SP | zh | 2 | 8000 | 1500 | 2500 | **92.82** | mi-F1=92.84, acc=92.84 |
| MMS | SP | multi | 3 | 27000 | 13500 | 13500 | **61.42** | mi-F1=63.27, acc=63.27 |
| EWECT-usual | EC | zh | 6 | 10000 | 2000 | 5000 | **62.36** | mi-F1=71.50, acc=71.50 |
| EWECT-virus | EC | zh | 6 | 8606 | 2000 | 3000 | **50.70** | mi-F1=74.53, acc=74.53 |
| XED | EC-multi | multi | 8 | 25000 | 12500 | 12500 | **16.41** | jac=13.21, subset-acc=9.68 |

## Val (MMAFFIn) vs Test (MMAFFBen) 對照

| Task | val ma-F1 | test ma-F1 | Δ |
|------|----------:|-----------:|----:|
| Onlineshopping | 93.73 | 92.82 | -0.91 |
| MMS | 61.14 | 61.42 | +0.28 |
| EWECT-usual | 62.96 | 62.36 | -0.60 |
| EWECT-virus | 48.93 | 50.70 | +1.77 |
| XED | 16.68 | 16.41 | -0.27 |

_耗時：1081s，設備：cuda_