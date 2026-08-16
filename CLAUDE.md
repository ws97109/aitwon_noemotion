# 生成式AI代理社區模擬系統 - 專案管理指南

## 專案概覽

這是一個基於生成式AI的虛擬社區模擬系統，包含9個具有獨特性格的AI居民，並內建完整的問卷調查系統。

### 核心功能模組
- **AI代理模擬**: 9個不同性格的AI居民進行社交互動
- **問卷調查系統**: 支援多種匯入方式和AI自動填寫
- **2D可視化**: 實時顯示AI居民活動狀態
- **數據分析**: 互動分析和問卷統計報告

## 開發環境設置

### 必要依賴
```bash
pip install flask requests python-dotenv pandas numpy openpyxl llama-index chromadb python-dateutil
```

### LLM 設定（本系統使用 Ollama 本地推論，不需要雲端 API 密鑰）
```bash
# 複製環境變數檔案
cp .env.example .env

# .env 中實際會被讀取的設定（問卷系統用）
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b   # 必須已 ollama pull

# 模擬系統的 LLM 設定在 data/config.json（base_url 與 model 需與本機 Ollama 一致）
```

### 啟動指令
```bash
# 啟動AI社區模擬
python start.py --name my_simulation --step 10

# 啟動Web界面服務
python replay.py
```

## 常用開發工作流程

### 運行測試
```bash
python survey_system/test_survey.py
```

### 類型檢查
```bash
# 使用 mypy 進行類型檢查（如果已安裝）
mypy modules/ survey_system/
```

### 代碼格式化
```bash
# 使用 black 格式化代碼（如果已安裝）
black survey_system/ modules/
```

## AI居民角色配置

### 當前居民列表
- 盧品蓉 (20歲, 學生): 友善、樂於助人、社交活躍
- 鄭傑丞 (21歲, 學生): 認真負責、注重細節、理性思考
- 莊于萱 (19歲, 學生): 創意豐富、藝術天賦、自由奔放
- 施宇鴻 (22歲, 學生): 邏輯思維強、冷靜理性、深度思考
- 游庭瑄 (45歲, 教授): 博學多聞、循循善誘、關愛學生
- 李昇峰 (50歲, 藥店老闆): 細心專業、服務至上、穩重可靠
- 魏祺紘 (18歲, 大一新生): 活潑好動、求知慾強、適應力強
- 陳冠佑 (35歲, 酒吧老闆): 健談幽默、善於交際、夜生活達人
- 蔡宗陞 (28歲, 咖啡店老闆): 溫和親切、追求品質、注重細節

### 居民配置檔案位置
```
frontend/static/assets/village/agents/[居民姓名]/agent.json
```

## 專案結構說明

```
generative_agents0910/
├── modules/                 # 核心AI模擬模組
│   ├── agent.py            # AI代理邏輯
│   ├── game.py             # 遊戲主控制器
│   ├── maze.py             # 環境地圖系統
│   └── memory/             # AI記憶系統
├── survey_system/          # 問卷系統模組
│   ├── models.py           # 資料模型
│   ├── ai_filler.py        # AI填寫器
│   ├── importers.py        # 問卷匯入器
│   ├── exporters.py        # 結果匯出器
│   └── test_survey.py      # 系統測試
├── frontend/               # Web前端
│   ├── templates/          # HTML模板
│   └── static/             # 靜態資源
├── start.py               # 模擬系統啟動器
├── replay.py              # Web服務器
└── data/                  # 配置和提示詞
```

## 問卷系統功能

### 支援的問題類型
- single_choice: 單選題
- multiple_choice: 多選題
- rating: 評分題
- text: 開放文字題

### 匯入方式
1. Google Forms URL匯入
2. JSON格式匯入
3. 手動創建

### 匯出格式
- CSV: 適合Excel分析
- JSON: 程式化處理
- Excel: 包含統計摘要

## 開發注意事項

### 代碼品質
- 保持代碼整潔和模組化
- 添加適當的錯誤處理
- 遵循Python PEP 8編碼規範

### 安全考量
- 不要在代碼中硬編碼API密鑰
- 使用環境變數管理敏感資訊
- 定期更新依賴套件

### 效能優化
- AI回應生成時間控制在2-5秒內
- 支援100+問題的大型問卷處理
- 確保Web界面支援多用戶並發操作

## 常見問題排解

### API密鑰相關
- 確保.env檔案正確設置
- 檢查API密鑰是否有效且有足夠額度

### 依賴套件問題
- 使用虛擬環境避免套件衝突
- 確保Python版本為3.8+

### 模擬運行問題
- 檢查存檔目錄權限
- 確認配置檔案格式正確

## 新增功能 (2025-09-18)

### 動態配置系統
- 移除硬編碼的AI居民列表
- 支援從配置文件和agents目錄動態載入居民信息
- 自動掃描agent配置文件

### 強化問卷管理功能
- **問卷複製**: 快速複製現有問卷並創建變體
- **批量操作**: 支援批量刪除多個問卷
- **問卷搜索**: 根據標題和描述搜索問卷
- **問卷歸檔**: 將舊問卷移至歸檔目錄
- **問卷更新**: 修改問卷標題和描述

### 高級分析功能
- **詳細統計分析**: 每個問題的詳細回應分析
- **單選題分析**: 選項分布和百分比統計
- **多選題分析**: 各選項被選擇的頻率
- **評分題分析**: 平均分、最高分、最低分和分數分布
- **文字題分析**: 回應長度分析和詞頻統計

### API功能擴展
```python
# 新增的管理功能
manager.duplicate_survey(survey_id, "新問卷標題")
manager.batch_delete_surveys([id1, id2, id3])
manager.search_surveys("滿意度")
manager.archive_survey(survey_id)
manager.get_response_analytics(survey_id)
```

## 更新記錄

### 2025-09-18
- 移除硬編碼配置，實現動態載入
- 新增問卷管理高級功能
- 加強統計分析能力
- 優化AI填寫器配置載入機制

### 初始版本
- 初始化專案管理文件
- 建立開發環境指南  
- 記錄AI居民角色特徵