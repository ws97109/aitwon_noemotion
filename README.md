# 生成式AI代理社區模擬系統 + SACF 多模態情感模型 — 使用手冊

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20本地推論-black.svg)](https://ollama.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)

本專案整合兩大子系統：

1. **生成式AI代理社區模擬**：9 位具有獨特性格的 AI 居民在 2D 虛擬村莊中生活、對話與互動，並內建完整的問卷調查系統（AI 居民可根據模擬記憶自動填寫問卷）。
2. **SACFFinalModel（Sentiment-Aware Cross-modal Fusion）**：多模態情感分析模型（DeBERTa-v3-large + 音訊/視覺 BiLSTM），可作為模擬中居民情緒偵測的參考訊號。

全系統使用 **Ollama 本地模型推論，不需要任何雲端 API 金鑰**。

---

## 目錄

1. [系統需求](#系統需求)
2. [前置作業（第一次使用必讀）](#前置作業第一次使用必讀)
3. [快速體驗：直接重播現成模擬](#快速體驗直接重播現成模擬)
4. [完整運行流程：模擬 → 壓縮 → 重播](#完整運行流程模擬--壓縮--重播)
5. [Web 介面功能總覽](#web-介面功能總覽)
6. [問卷調查系統操作](#問卷調查系統操作)
7. [SACF 情感模型（進階）](#sacf-情感模型進階)
8. [LoRA 對話微調（選用）](#lora-對話微調選用)
9. [連接埠與資料路徑總覽](#連接埠與資料路徑總覽)
10. [常見問題排解](#常見問題排解)
11. [AI 居民角色介紹](#ai-居民角色介紹)
12. [專案結構](#專案結構)

---

## 系統需求

| 項目 | 需求 |
|------|------|
| 作業系統 | macOS / Linux（Windows 未測試） |
| Python | 3.11 或 3.12（實際開發測試版本） |
| [Ollama](https://ollama.com) | 需在本機安裝並啟動 |
| 磁碟空間 | 模型約 6 GB 起（qwen2.5:7b 4.7 GB + bge-m3 1.2 GB） |
| GPU | **不需要**（模擬與問卷系統走 Ollama）；僅訓練 SACF 模型時需要 CUDA GPU |

---

## 前置作業（第一次使用必讀）

> **重要**：所有指令都必須在**專案根目錄**下執行。專案內所有路徑（`data/config.json`、`results/`、`survey_system/data` 等）都是相對於目前工作目錄的相對路徑，在其他目錄執行會找不到檔案或把資料寫到錯的地方。

### 步驟 1：建立 Python 環境並安裝依賴

```bash
cd <專案根目錄>

# 建立虛擬環境（擇一：venv 或 conda）
python3 -m venv .venv && source .venv/bin/activate
# 或： conda create -n aitown_tw python=3.12 && conda activate aitown_tw

# 安裝依賴
pip install -r requirements.txt
```

關於 `requirements.txt` 的兩點說明：

- **`uuid>=1.30` 這行請勿安裝／可直接刪除**：`uuid` 是 Python 內建標準函式庫，PyPI 上的同名套件是 Python 2 時代的過時套件，安裝後反而可能遮蔽內建模組。
- 最下方的 `torch / transformers / peft / accelerate / datasets` 區塊**只有訓練 SACF 模型或 LoRA 微調時才需要**，執行模擬與問卷系統可以跳過。註解中的 `--index-url .../cu121` 是 Linux + NVIDIA 專用；macOS 上直接 `pip install torch` 即可。

### 步驟 2：安裝並啟動 Ollama、下載模型

```bash
# 啟動 Ollama 服務（預設監聽 127.0.0.1:11434）
ollama serve

# 下載必要模型
ollama pull qwen2.5:7b      # 對話/生成模型（4.7 GB）
ollama pull bge-m3:latest   # 嵌入模型（1.2 GB）

# 驗證服務與模型
curl http://127.0.0.1:11434/api/tags
ollama list
```

若你想使用品質更好的 `qwen2.5:32b`（約 20 GB），改成 `ollama pull qwen2.5:32b`，並確保下面兩個設定檔的模型名稱一致。

### 步驟 3：設定 `.env`（問卷系統使用）

```bash
cp .env.example .env
```

`.env` 中**實際會被程式讀取的只有兩個變數**（其餘為預留設定，目前程式未使用）：

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama 服務位址（問卷系統用） |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 問卷 AI 填寫使用的模型，**必須已 `ollama pull`** |

> ⚠️ 若 `OLLAMA_MODEL` 指定的模型沒有安裝，問卷填寫**不會報錯**，而是所有回答靜默變成罐頭答案（如「無法回答」、評分一律 5 分）。請務必確認 `ollama list` 中有該模型。

### 步驟 4：檢查 `data/config.json`（模擬系統使用，最常見的卡關點）

模擬系統（`start.py`）的 LLM 設定**不讀 `.env`**，而是讀 `data/config.json`。出廠設定指向另一台機器的環境，第一次執行前必須確認以下三處：

1. **LLM 連接埠**：`agent.think.llm.base_url` 預設為 `http://127.0.0.1:11436/v1`，但 Ollama 預設跑在 **11434**。二擇一：
   - **改設定檔（建議）**：把 `agent.think.llm.base_url` 改成 `http://127.0.0.1:11434/v1`，並把 `agent.associate.embedding.base_url` 改成 `http://127.0.0.1:11434`；或
   - **多開一個 Ollama**：`OLLAMA_HOST=127.0.0.1:11436 ollama serve`。
2. **模型名稱**：`agent.think.llm.model` 預設為 `qwen2.5:32b`。若你只安裝了 `qwen2.5:7b`，請改成 `qwen2.5:7b`。
3. **情感模型路徑**：`agent.think.emotion.sacf_model_path` 預設是另一台 Linux 機器的 NFS 路徑，本機通常不存在。**不影響執行**——找不到 `.pt` 權重檔時，系統會自動退回用 LLM 判斷居民情緒。若已自行訓練出權重檔，改成 `emotion_system/models/sacf_final.pt` 即可；不想使用情感模組也可把 `emotion.enabled` 設為 `false`。

> 補充：`--resume` 續跑時會以「當下的」`data/config.json` 覆蓋檢查點中的 LLM／嵌入設定，所以中途修改此檔會直接改變後續模擬使用的模型。

---

## 快速體驗：直接重播現成模擬

`results/compressed/` 內已附多個壓縮完成的模擬結果（如 `final`、`test2` 等），**不必先跑模擬**就能體驗系統：

```bash
python replay.py
```

打開瀏覽器進入 **http://127.0.0.1:5001/** — 未指定模擬名稱時會顯示「模擬選擇頁」，列出所有可重播的模擬，點選即可觀看 2D 重播；頁面上也有問卷系統（`/surveys`）的入口。

---

## 完整運行流程：模擬 → 壓縮 → 重播

整條管線固定為三步，**順序不可省略**：

```
python start.py（跑模擬） → python compress.py（產生重播資料，必要） → python replay.py（Web 觀看）
```

### 第 1 步：執行模擬 `start.py`

```bash
# 新模擬：10 步、每步推進 10 分鐘模擬時間
python start.py --name my_sim --start "20250501-09:00" --step 10 --stride 10

# 續跑既有模擬（從最後一個檢查點接續）
python start.py --name my_sim --resume --step 100 --stride 10
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--name` | （空） | 模擬名稱。未提供會互動式詢問；新模擬撞名會要求換名；`--resume` 找不到名稱會要求重輸入 |
| `--start` | `20240213-09:30` | 模擬起始時間，格式 `YYYYMMDD-HH:MM` |
| `--resume` | 關閉 | 從最後檢查點續跑（沿用居民記憶、行程與位置） |
| `--step` | `10` | 模擬步數 |
| `--stride` | `10` | 每步推進的「模擬世界分鐘數」（stride=10 時村莊時鐘走 9:00 → 9:10 → 9:20…） |
| `--verbose` | `debug` | 日誌等級 |
| `--log` | （空） | 指定後日誌寫入檢查點資料夾內的檔案，而非顯示在終端機 |

**輸出**：每步寫入 `results/checkpoints/<name>/simulate-YYYYMMDD-HHMM.json`（完整居民狀態）與 `conversation.json`（所有對話）。

> 執行時間視模型與硬體而定；每步每位居民都會呼叫 LLM 數次，用 7b 模型跑 10 步約需數分鐘。

### 第 2 步：壓縮 `compress.py`（重播前必要）

```bash
python compress.py --name my_sim
```

讀取 `results/checkpoints/<name>/`，輸出到 `results/compressed/<name>/`：

- `movement.json` — 2D 重播的逐幀移動資料（**重播頁沒有它就無法載入**）
- `simulation.md` — 各居民狀態與對話的時間軸報告（問卷 AI 填寫時也會引用它作為活動記憶）
- `interactions.json` — 互動統計資料

### 第 3 步：啟動 Web 服務 `replay.py`

```bash
python replay.py                  # 預設 http://127.0.0.1:5001
python replay.py --name my_sim    # 指定預設模擬（問卷 AI 填寫會綁定此模擬的記憶）
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--name` | （空） | 預設模擬名稱；未指定時首頁顯示模擬選擇頁 |
| `--port` | `5001` | HTTP 連接埠（注意不是 Flask 慣用的 5000） |
| `--host` | `127.0.0.1` | 綁定位址 |
| `--no-debug` | 關閉 | 關閉 Flask debug 模式（對外開放時務必加上） |

重播頁網址參數（範例：`http://127.0.0.1:5001/?name=my_sim&step=0&speed=2&zoom=0.6`）：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `name` | — | 模擬名稱（必填，或由 `--name` 提供） |
| `step` | `0` | 起始影格 |
| `speed` | `2` | 播放速度 0–5（實際倍速為 2^speed） |
| `zoom` | `0.8` | 畫面縮放比例 |

---

## Web 介面功能總覽

| 網址 | 功能 |
|------|------|
| `/` | 2D 模擬重播（帶 `?name=`）；未指定名稱時為模擬選擇頁 |
| `/interaction-graph?name=<模擬名>` | 居民社交關係力導向圖（D3.js） |
| `/object-interaction?name=<模擬名>` | 居民與環境物品/地點的互動分析 |
| `/interaction-matrix` | 互動矩陣頁 |
| `/surveys` | 問卷列表 |
| `/surveys/create` | 建立/匯入問卷 |
| `/surveys/<id>` | 問卷詳情與回應 |
| `/surveys/<id>/analytics` | 問卷統計視覺化 |

> ⚠️ `/interaction-graph` 與 `/object-interaction` 在找不到該模擬的檢查點資料時，會**靜默改用內建示範資料**——若不同模擬的圖表長得一模一樣，代表讀到的不是真實資料，請確認 `?name=` 正確且 `results/checkpoints/<name>/` 存在。

API 端點（供程式化操作）：

| 方法與路徑 | 功能 |
|------|------|
| `POST /surveys/<id>/fill` | 讓所有 AI 居民填寫問卷（body 可帶 `simulation_name` 綁定模擬記憶） |
| `GET /surveys/simulations` | 取得可用模擬清單 |
| `GET /surveys/<id>/export?format=csv\|json\|excel` | 下載匯出檔 |
| `POST /surveys/<id>/clear-responses` | 清除回應（`{"mode":"all"}` 全刪；`{"mode":"fallback"}` 只刪罐頭回應佔比 ≥70% 的回應） |

---

## 問卷調查系統操作

### 完整流程

1. **啟動**：確認 Ollama 執行中（步驟 2）→ `python replay.py` → 打開 `http://127.0.0.1:5001/surveys`。
2. **建立問卷**（`/surveys/create`），三種方式：
   - **手動建立**：填寫標題、描述，逐題新增。
   - **JSON 匯入**：貼上 JSON（支援別名自動轉換：`choice/radio→single_choice`、`checkbox→multiple_choice`、`scale→rating`、`short_answer/paragraph→text`）。
   - **Google Forms URL**：⚠️ 功能有限——只能抓到表單標題，題目會以內建範例題代替，正式使用請改用手動或 JSON 匯入。
3. **AI 自動填寫**：在問卷詳情頁按下填寫按鈕（對應 `POST /surveys/<id>/fill`）。9 位居民會依序對每一題呼叫 Ollama 生成答案。
   - 若啟動時有 `python replay.py --name <模擬名>`（或 POST body 帶 `simulation_name`），居民會參考該模擬的**活動記錄（simulation.md）、情緒狀態（檢查點）與向量記憶**作答；否則僅依靜態人設作答。
4. **檢視與匯出**：詳情頁看逐題回應，`/surveys/<id>/analytics` 看統計圖表，匯出支援：
   - **CSV**（UTF-8 with BOM，Excel 可直接開啟；多選題每個選項一個布林欄位）
   - **JSON**（含問卷結構 + 全部回應，適合程式處理）
   - **Excel**（含「問卷回應」與「統計摘要」兩個工作表，需安裝 `openpyxl`）

### 支援的題型

| 類型 | 說明 |
|------|------|
| `single_choice` | 單選題（AI 答案會經過精確/模糊/數值區間比對，保證落在選項內） |
| `multiple_choice` | 多選題 |
| `rating` | 評分題（從題目文字如「1-5分」自動解析範圍，答案自動夾限） |
| `text` | 開放文字題（回答一律為繁體中文） |

### 其他要點

- 問卷與回應以 JSON 檔存放於 `survey_system/data/{surveys,responses,exports}/`。
- 重複填寫時，每位居民只保留最新一份回應（舊回應自動刪除）。
- AI 提示詞模板位於 `data/prompts/survey_*.txt`，可直接編輯調整答題風格。
- 系統測試：`python survey_system/test_survey.py`（涵蓋建立、管理、AI 填寫、匯出、匯入、統計；未啟動 Ollama 也能跑完，但 AI 填寫部分會是罐頭答案）。

---

## SACF 情感模型（進階）

SACFFinalModel 是針對 CMU-MOSI 的多模態情感分析模型：DeBERTa-v3-large 文字骨幹 + BiLSTM 音訊/視覺編碼器、4 個並行分支（極性增強注意力 PEA + 階層式跨模態融合 + 三種預測頭），詳細方法見 `emotion_system/SCAF_FLOWCHART.md`。

**在模擬中的角色**：居民對話時以文字單模態推論情緒（迴歸分數映射為 興奮/快樂/平靜/焦慮/悲傷），作為 LLM 情緒判斷的交叉參考。**倉庫不含 `.pt` 權重檔**（受 `.gitignore` 排除），找不到權重時模擬會自動退回純 LLM 情緒判斷，功能不受影響。

### 訓練

```bash
python emotion_system/training/scaf_final.py
```

- **無命令列參數**，所有超參數在檔內 `cfg` 字典中修改（兩階段訓練：Stage 1 基底 60 epochs → Stage 2 跨模態 InfoNCE 精修 20 epochs）。
- **需要資料集**：`emotion_system/data/mosi/unaligned_50.pkl`（CMU-MOSI 未對齊特徵，需自行取得放置）。
- **需要網路**：首次執行會從 HuggingFace 下載 `microsoft/deberta-v3-large`。
- **實務上需要 CUDA GPU**（骨幹約 4 億參數；無 GPU 時會退回 CPU 但速度不可行）。
- **輸出**：`emotion_system/models/sacf_final.pt`（約 1.65 GB）與 `sacf_final_summary.json`（若新結果劣於既有權重，改存為 `sacf_final_thisrun.pt`）。

### 評估 / 程式化載入

```bash
# 在 MOSI 測試集上評估（含 MC-Dropout TTA）
python emotion_system/sacf_final_loader.py --ckpt emotion_system/models/sacf_final.pt \
    --data emotion_system/data/mosi/unaligned_50.pkl --device cuda:0 --n_tta 5
```

```python
from emotion_system.sacf_final_loader import load_sacf_final
model = load_sacf_final("emotion_system/models/sacf_final.pt", device="cuda:0")
cls7_logits, cls2_logits, reg = model(input_ids, attention_mask, audio, audio_mask, vision, vision_mask)
```

**成績**：單次兩階段訓練（`sacf_final_summary.json`）Acc-7 51.46 / Acc-2 87.03 / F1 87.03 / MAE 0.5821 / Corr 0.8704；`SCAF_FLOWCHART.md` 記載的 **Acc-7 53.06** 為三次獨立訓練於參數層加權平均（0.25/0.45/0.30）後的成績，該合併步驟需另行執行，不含在訓練腳本內。

---

## LoRA 對話微調（選用）

`train.py`（專案根目錄）與 SACF 無關：它使用 HuggingFace Transformers + PEFT，從模擬產出的對話（`results/checkpoints/**/conversation.json`）微調因果語言模型：

```bash
python train.py --model_path Qwen/Qwen2-7B --data_path results/checkpoints --output_dir results/finetuned_model
```

主要參數：`--epochs 3`、`--batch_size 2`、`--grad_accum 8`、`--lr 2e-4`、`--lora_r 16`、`--lora_alpha 32`。需先跑過模擬（要有 conversation.json）並安裝 `peft` 等訓練依賴。

---

## 連接埠與資料路徑總覽

| 連接埠 | 用途 | 設定來源 |
|--------|------|----------|
| `11434` | Ollama 預設服務（問卷系統、情感模組 base_url） | `.env` 的 `OLLAMA_BASE_URL` |
| `11436` | 模擬 LLM 與嵌入（**出廠設定，預設沒有服務在跑**，見前置作業步驟 4） | `data/config.json` |
| `5001` | Flask Web 介面（重播 + 問卷） | `replay.py --port` |

| 路徑 | 內容 |
|------|------|
| `results/checkpoints/<name>/` | 模擬檢查點（`simulate-*.json`、`conversation.json`、居民向量記憶） |
| `results/compressed/<name>/` | 重播資料（`movement.json`、`simulation.md`、`interactions.json`） |
| `survey_system/data/` | 問卷（`surveys/`）、回應（`responses/`）、匯出檔（`exports/`） |
| `frontend/static/assets/village/agents/<姓名>/agent.json` | 各居民人設與初始位置（居民名單由此目錄動態掃描） |
| `data/config.json` | 模擬系統核心設定（LLM、嵌入、情感模組） |
| `data/prompts/` | 問卷 AI 填寫的提示詞模板 |

---

## 常見問題排解

| 症狀 | 原因與解法 |
|------|------------|
| `start.py` 一開始就連線錯誤或卡住 | `data/config.json` 指向 11436 但沒有服務在跑，或模型名稱未安裝 → 見[前置作業步驟 4](#步驟-4檢查-dataconfigjson模擬系統使用最常見的卡關點) |
| 問卷回應全是「無法回答」、評分全是 5 | Ollama 未啟動或 `.env` 的 `OLLAMA_MODEL` 未安裝（**不會報錯，靜默降級**）→ `curl http://127.0.0.1:11434/api/tags` 確認；可用 `/surveys/<id>/clear-responses` 的 `fallback` 模式清除罐頭回應後重填 |
| 重播頁顯示 `The data file doesn't exist` | 該模擬還沒跑 `python compress.py --name <名稱>` |
| 互動圖表每個模擬看起來都一樣 | 找不到 `results/checkpoints/<name>/` 的真實資料，頁面靜默改用示範資料 → 確認 `?name=` 參數與檢查點是否存在 |
| Excel 匯出報 `ImportError` | `pip install openpyxl` |
| 檔案找不到 / 資料寫到奇怪的位置 | 沒有在專案根目錄執行指令（所有路徑都是相對路徑） |
| 情感模組警告找不到 `.pt` | 正常現象——倉庫不含權重檔，系統自動退回 LLM 情緒判斷；可自行訓練或將 `emotion.enabled` 設為 `false` |
| 問卷答案沒有反映模擬經歷 | 啟動時未帶 `--name`（或 fill API 未帶 `simulation_name`），居民僅依靜態人設作答 |
| 舊文件（`運行.txt` 等）與本手冊不一致 | 以本手冊為準：重播埠是 **5001**（非 5000）；問卷填寫器已無 `use_ollama` 參數；不需要 OpenAI API 金鑰 |

---

## AI 居民角色介紹

| 姓名 | 年齡 | 職業/身份 | 性格特點 |
|------|------|-----------|----------|
| 盧品蓉 | 20歲 | 學生 | 友善、樂於助人、社交活躍 |
| 鄭傑丞 | 21歲 | 學生 | 認真負責、注重細節、理性思考 |
| 莊于萱 | 19歲 | 學生 | 創意豐富、藝術天賦、自由奔放 |
| 施宇鴻 | 22歲 | 學生 | 邏輯思維強、冷靜理性、深度思考 |
| 游庭瑄 | 45歲 | 教授 | 博學多聞、循循善誘、關愛學生 |
| 李昇峰 | 50歲 | 藥店老闆 | 細心專業、服務至上、穩重可靠 |
| 魏祺紘 | 18歲 | 大一新生 | 活潑好動、求知慾強、適應力強 |
| 陳冠佑 | 35歲 | 酒吧老闆 | 健談幽默、善於交際、夜生活達人 |
| 蔡宗陞 | 28歲 | 咖啡店老闆 | 溫和親切、追求品質、注重細節 |

居民名單並非寫死：系統會動態掃描 `frontend/static/assets/village/agents/` 下的資料夾。新增居民時在該目錄建立 `<姓名>/agent.json` 即可。

---

## 專案結構

```
aitwon_noemotion/
├── start.py                 # 模擬啟動器（跑 AI 村莊）
├── compress.py              # 檢查點 → 重播資料轉換器
├── replay.py                # Flask Web 伺服器（重播 + 問卷 + 分析）
├── train.py                 # LoRA 對話微調（選用）
├── requirements.txt
├── .env.example             # 環境變數範本（問卷系統的 Ollama 設定）
├── data/
│   ├── config.json          # 模擬核心設定（LLM/嵌入/情感）
│   └── prompts/             # 問卷 AI 填寫提示詞模板（survey_*.txt）
├── modules/                 # 模擬核心
│   ├── agent.py             # AI 代理邏輯（行程、對話、情緒）
│   ├── game.py              # 遊戲主控制器
│   ├── maze.py              # 村莊地圖系統
│   ├── memory/              # 記憶系統（向量記憶用 LlamaIndex + Ollama 嵌入）
│   └── emotion/             # 情緒模組（SACF 載入 + LLM 後備）
├── survey_system/           # 問卷系統
│   ├── models.py            # 問卷/回應資料模型與管理器
│   ├── ai_filler.py         # AI 居民填寫器
│   ├── ollama_generator.py  # Ollama 答案生成器
│   ├── simulation_context.py# 模擬記憶上下文（活動/情緒/向量記憶）
│   ├── importers.py         # 匯入器（JSON / Google Forms / URL）
│   ├── exporters.py         # 匯出器（CSV / JSON / Excel）
│   ├── test_survey.py       # 系統測試
│   └── data/                # 問卷資料儲存
├── emotion_system/          # SACF 多模態情感模型
│   ├── training/scaf_final.py   # 訓練入口（無 CLI 參數，改 cfg）
│   ├── sacf_final_loader.py     # 權重載入 + 測試集評估 CLI
│   ├── SCAF_FLOWCHART.md        # 模型方法詳解
│   ├── models/                  # 權重輸出目錄（.pt 不入版控）
│   └── data/mosi/               # CMU-MOSI 資料集放置處（unaligned_50.pkl）
├── frontend/                # Web 前端（模板 + 2D 素材 + 居民設定）
└── results/
    ├── checkpoints/         # 模擬檢查點
    └── compressed/          # 重播資料
```
