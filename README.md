# 生成式AI代理社區模擬系統 + SACF 多模態情感模型

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

整合兩大子系統的研究專案：

1. **生成式AI代理社區模擬**：9個具有獨特性格的AI居民、完整問卷調查系統
2. **SACFFinalModel (Sentiment-Aware Cross-modal Fusion)**：多模態情感分析模型，於 CMU-MOSI 達到 **Acc-7 = 53.06%**（無外部教師、單一 .pt 權重檔）

## ✨ 主要功能

### 🤖 AI代理模擬
- **9個獨特AI居民**：每個都有不同的年齡、性格、職業和興趣
- **智能行為模擬**：AI居民會根據個性進行社交互動、移動和活動
- **實時可視化**：2D遊戲引擎顯示AI居民的即時活動狀態
- **互動分析**：追蹤和分析AI居民之間的對話和互動模式

### 🧠 SACFFinalModel 多模態情感模型
- **多分支單一模型**：DeBERTa-v3-large + BiLSTM 音訊/視覺編碼器，4 個並行分支端對端訓練
- **極性增強注意力（PEA）**：為每個 DeBERTa 詞元學習情感顯著性閘值
- **階層式跨模態融合（Hierarchical SACF）**：兩階段堆疊之跨模態注意力，相較單階段 +0.6% Acc-7
- **跨模態 InfoNCE 對比輔助（CMC）**：訓練第二階段對齊三模態語義空間，+0.7% Acc-7
- **兩階段訓練 + 多執行快照集成**：Stage 1 基底（60 ep）→ Stage 2 CMC 精修（20 ep），三次獨立執行於參數層加權平均合併為單一 .pt
- **無外部教師、零資料洩漏**：所有訓練訊號內生，推斷融合超參數於訓練前固定
- **完整訓練/推斷管線**：`scaf_final.py` 統一入口，`sacf_final_loader.py` 對外載入
- **詳細方法說明**：見 `emotion_system/SCAF_FLOWCHART.md` 與 `docs/SACF_Methodology_Chapter3_v2.docx`

### 📊 問卷調查系統
- **多種匯入方式**：支援Google Forms URL、JSON格式或手動創建
- **智能AI填寫**：AI居民根據個性特徵自動生成合理的問卷回應
- **多種問題類型**：單選題、多選題、評分題、開放文字題
- **多格式匯出**：CSV、JSON、Excel格式，包含統計分析

### 📈 數據分析與可視化
- **互動力導向圖**：展示AI居民之間的社交關係網路
- **物品交互分析**：追蹤AI居民與環境物品的互動頻率
- **問卷統計報告**：自動生成完整的調查結果分析

## 🎯 AI居民角色介紹

| 姓名 | 年齡 | 職業/身份 | 性格特點 | 主要活動 |
|------|------|-----------|----------|----------|
| 盧品蓉 | 20歲 | 學生 | 友善、樂於助人、社交活躍 | 參與社團活動 |
| 鄭傑丞 | 21歲 | 學生 | 認真負責、注重細節、理性思考 | 程式設計研究 |
| 莊于萱 | 19歲 | 學生 | 創意豐富、藝術天賦、自由奔放 | 藝術創作 |
| 施宇鴻 | 22歲 | 學生 | 邏輯思維強、冷靜理性、深度思考 | 工程技術研究 |
| 游庭瑄 | 45歲 | 教授 | 博學多聞、循循善誘、關愛學生 | 教學與研究 |
| 李昇峰 | 50歲 | 藥店老闆 | 細心專業、服務至上、穩重可靠 | 健康諮詢服務 |
| 魏祺紘 | 18歲 | 大一新生 | 活潑好動、求知慾強、適應力強 | 探索大學生活 |
| 陳冠佑 | 35歲 | 酒吧老闆 | 健談幽默、善於交際、夜生活達人 | 經營酒吧社交 |
| 蔡宗陞 | 28歳 | 咖啡店老闆 | 溫和親切、追求品質、注重細節 | 咖啡調製研發 |

## 🚀 快速開始

### 環境需求
```bash
Python 3.8+
Flask 2.0+
requests
openpyxl (Excel匯出功能，可選)
```

### 安裝步驟

1. **克隆專案**
```bash
git clone [your-repo-url]
cd generative_agents0910
```

2. **安裝依賴套件**
```bash
pip install flask requests python-dotenv
pip install openpyxl  # Excel匯出功能 (可選)
```

3. **設置API密鑰**
```bash
# 複製環境變數檔案
cp .env.example .env

# 編輯 .env 檔案，加入您的API密鑰
# OPENAI_API_KEY=your_openai_api_key
```

4. **配置設定**
```bash
# 編輯 data/config.json 檔案
# 設定API密鑰和模型參數
```

### 運行系統

#### 方法一：啟動模擬系統
```bash
# 啟動AI社區模擬
python start.py --name my_simulation --step 10

# 啟動Web界面服務
python replay.py
```
訪問 http://localhost:5001 查看模擬結果

#### 方法二：直接使用問卷系統
```bash
# 直接啟動Web服務
python replay.py
```
訪問 http://localhost:5001/surveys 進入問卷系統

## 📋 問卷系統使用指南

### 創建問卷

#### 1. 手動創建
- 訪問 `/surveys/create`
- 選擇「手動創建」標籤
- 填寫問卷標題和描述
- 逐一添加問題（支援4種類型）
- 點擊「創建問卷」

#### 2. Google Forms匯入
```bash
# 支援的URL格式
https://docs.google.com/forms/d/[FORM_ID]/viewform
https://forms.gle/[SHORT_ID]
```

#### 3. JSON格式匯入
```json
{
  "title": "滿意度調查",
  "description": "針對服務滿意度的調查",
  "questions": [
    {
      "type": "single_choice",
      "text": "您對我們的服務滿意嗎？",
      "options": ["非常滿意", "滿意", "普通", "不滿意"],
      "required": true
    },
    {
      "type": "rating",
      "text": "請為我們的服務評分（1-5分）",
      "required": true
    },
    {
      "type": "text",
      "text": "您有什麼建議嗎？",
      "required": false
    }
  ]
}
```

### AI居民填寫
1. 在問卷詳情頁面點擊「讓AI居民填寫」
2. 系統會根據每個AI居民的性格特徵生成回應
3. 所有9個AI居民會批次完成問卷填寫
4. 填寫完成後可立即查看統計結果

### 數據匯出
支援三種格式匯出：
- **CSV格式**：適合Excel或數據分析軟體
- **JSON格式**：適合程式化處理
- **Excel格式**：包含統計摘要工作表

## 🛠️ 技術架構

```
aitown_addsacf/
├── modules/                 # 核心AI模擬模組
│   ├── agent.py            # AI代理邏輯
│   ├── game.py             # 遊戲主控制器
│   ├── maze.py             # 環境地圖系統
│   └── memory/             # AI記憶系統
├── emotion_system/          # SACFFinalModel 多模態情感模型
│   ├── training/
│   │   ├── scaf_final.py           # 兩階段訓練主腳本（PEA + Hier-SACF + CMC）
│   │   └── mmaffin_exp/            # MMAffBen / SemEval-2018 評估
│   ├── sacf_final_loader.py        # 推斷時對外載入介面（含 Reg-Cls 融合）
│   ├── models/sacf_final.pt        # 最終權重（53.06% Acc-7，1.65 GB）
│   ├── data/                       # MOSI / MMAffBen / SemEval 資料
│   ├── SCAF_FLOWCHART.md           # 詳細流程圖（Mermaid 圖、超參數表）
│   ├── core.py                     # 情感模組核心
│   ├── emotion_memory.py           # 情感記憶
│   └── agent_emotion_mixin.py      # 與 AI 代理整合
├── survey_system/          # 問卷系統模組
│   ├── models.py           # 資料模型
│   ├── ai_filler.py        # AI填寫器
│   ├── importers.py        # 問卷匯入器
│   ├── exporters.py        # 結果匯出器
│   └── test_survey.py      # 系統測試
├── frontend/               # Web前端
│   ├── templates/          # HTML模板
│   └── static/             # 靜態資源
├── docs/                   # 論文與圖表
│   ├── SACF_Methodology_Chapter3_v2.docx   # 論文方法章節（中文，~3MB）
│   ├── generate_paper_v2.py                # 圖文生成腳本
│   ├── paper_v2_data.npz                   # 真實測試集 logits 快取
│   └── figures/v2_fig*.svg / .png          # 11 張章節圖檔
├── start.py                # 模擬系統啟動器
├── replay.py               # Web服務器
└── data/                   # 配置和提示詞
```

### 核心技術
- **AI 模擬後端**：Flask、OpenAI GPT
- **情感模型**：PyTorch、HuggingFace Transformers (DeBERTa-v3-large)
- **前端**：Bootstrap + jQuery + D3.js
- **資料存儲**：JSON 檔案系統
- **可視化**：Phaser.js 遊戲引擎、matplotlib

## 🧪 SACFFinalModel 快速使用

### 訓練（兩階段，60+20 ep，~90 min on RTX A6000）
```bash
python emotion_system/training/scaf_final.py
# 自動執行 Stage 1（基底訓練）→ Stage 1 SWA 平均載回 → Stage 2（CMC fine-tune）
# 輸出單一 .pt 至 emotion_system/models/sacf_final.pt
```

### 推斷
```python
from emotion_system.sacf_final_loader import load_sacf_final

model = load_sacf_final(ckpt_path="emotion_system/models/sacf_final.pt", device="cuda:0")
# forward: cls7_logits, cls2_logits, reg = model(input_ids, attention_mask,
#                                                audio, audio_mask, vision, vision_mask)
```

### CLI 評估
```bash
python emotion_system/sacf_final_loader.py \
    --ckpt emotion_system/models/sacf_final.pt --n_tta 1 --device cuda:0
# → Acc-7 = 53.06%, Acc-2 = 85.42%, F1 = 85.41%, MAE = 0.5840, Corr = 0.8691
```

### 完整方法說明
- **流程圖**：`emotion_system/SCAF_FLOWCHART.md`（Mermaid 圖、超參數對照表）
- **學術章節**：`docs/SACF_Methodology_Chapter3_v2.docx`（中文，9 節）
- **章節圖檔**：`docs/figures/v2_fig*.svg`（架構、PEA、SACF、訓練、結果共 11 張）

# MMAffBen 多語純文字評估
python emotion_system/training/mmaffin_exp/eval_text_mmaffben.py

# SemEval-2018 Task 1 評估
python emotion_system/training/mmaffin_exp/eval_semeval2018.py
```

## 🧪 測試系統

運行完整測試：
```bash
python survey_system/test_survey.py
```

測試涵蓋：
- ✅ 問卷創建和管理
- ✅ AI居民自動填寫
- ✅ 多格式匯出功能
- ✅ 問卷匯入功能
- ✅ 數據統計分析

## 📊 使用案例

### 1. 市場調研模擬
```python
from survey_system import SurveyManager, AIResidentSurveyFiller

# 創建問卷
manager = SurveyManager()
survey = manager.load_survey('product_survey_id')

# AI居民填寫
filler = AIResidentSurveyFiller(manager)
responses = filler.fill_survey_for_all_residents(survey.survey_id)

print(f"收集到 {len(responses)} 份回應")
```

### 2. 用戶體驗測試
- 測試不同年齡層對產品的接受度
- 分析不同性格用戶的使用偏好
- 獲得多樣化的反饋意見

### 3. 教育研究
- 模擬學生對課程的反應
- 分析不同學習風格的效果
- 收集教學改進建議

## 🔧 自定義擴展

### 添加新AI居民
```python
# 在 ai_filler.py 中添加
new_resident_info = {
    "新居民姓名": {
        "age": 25,
        "personality": ["特性1", "特性2"],
        "interests": ["興趣1", "興趣2"],
        "lifestyle": "生活方式",
        "current_activity": "當前活動"
    }
}
```

### 自定義問題類型
```python
# 擴展 models.py 中的問題類型
QUESTION_TYPES = [
    'single_choice',    # 單選題
    'multiple_choice',  # 多選題
    'rating',          # 評分題
    'text',            # 文字題
    'date',            # 日期題 (自定義)
    'number'           # 數字題 (自定義)
]
```

## 📈 性能指標

- **AI回應生成**：平均每個居民 2-5秒
- **問卷處理**：支援 100+ 問題的大型問卷
- **數據匯出**：千份回應級別的快速匯出
- **並發支援**：Web界面支援多用戶同時操作

## 🤝 貢獻指南

1. Fork此專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 開啟Pull Request

### 開發環境設置
```bash
# 安裝開發依賴
pip install -r requirements-dev.txt

# 運行測試
python -m pytest survey_system/test_survey.py

# 代碼格式化
black survey_system/
```

## 📝 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 🙋 常見問題

### Q: 如何更改AI居民的回應風格？
A: 編輯 `survey_system/ai_filler.py` 中的回應生成邏輯，可以調整每個居民的性格權重和回應模板。

### Q: 是否支援中文以外的語言？
A: 目前主要支援繁體中文，可以通過修改提示詞模板來支援其他語言。

### Q: 如何增加更多問題類型？
A: 在 `models.py` 中擴展問題類型，並在 `ai_filler.py` 中添加對應的回應邏輯。

### Q: 系統的AI回應有多真實？
A: AI回應基於每個居民的詳細性格設定，包含一致性邏輯，可產生符合角色特徵的合理回應。

### Q: 是否可以匯出原始的AI模擬數據？
A: 可以，模擬數據保存在 `results/checkpoints/` 目錄下，包含完整的互動和行為記錄。

## 📞 支援與聯絡

- 📧 Email: [your-email@domain.com]
- 🐛 Bug回報: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 討論區: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**⭐ 如果這個專案對您有幫助，請給我們一個星星！**

---

*最後更新：2026年5月4日*