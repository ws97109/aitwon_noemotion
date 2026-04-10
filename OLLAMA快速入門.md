# Ollama 問卷系統快速入門

## 系統已完成重構！

✅ **完全動態化** - 不再使用預設資料
✅ **模板化管理** - Prompt 存放在獨立文件
✅ **真實資料驅動** - 直接讀取 agent.json 和 simulation.md

---

## 5 分鐘快速開始

### 步驟 1: 啟動 Ollama 服務

```bash
# 啟動 Ollama（如果尚未運行）
ollama serve

# 確認模型已下載
ollama list
# 應該看到 qwen2.5:7b
```

### 步驟 2: 檢查資料文件

```bash
# 檢查 AI 居民配置（應該有 9 個目錄）
ls frontend/static/assets/village/agents/

# 檢查活動歷史
ls results/compressed/test_0513/simulation.md

# 檢查 prompt 模板（應該有 4 個文件）
ls data/prompts/survey_*.txt
```

### 步驟 3: 運行測試

```bash
python test_ollama_survey.py
```

**預期輸出**:
```
============================================================
Ollama LLM 問卷填寫測試
============================================================

📋 初始化問卷管理器...
🤖 初始化Ollama AI填寫器...
   - 從 agent.json 動態載入 AI 居民資料
   - 從 simulation.md 載入活動歷史
   - 從 data/prompts/ 載入 prompt 模板

✓ 已載入 9 位居民的活動歷史
✓ 已載入 9 位AI居民
✓ Ollama模式: 啟用
...
```

### 步驟 4: 在程式中使用

```python
from survey_system.models import SurveyManager
from survey_system.ai_filler import AIResidentSurveyFiller

# 初始化
manager = SurveyManager()
filler = AIResidentSurveyFiller(
    survey_manager=manager,
    simulation_md_path="results/compressed/test_0513/simulation.md",
    use_ollama=True  # 使用 Ollama LLM
)

# 讓特定居民填寫問卷
response = filler.fill_survey_for_resident(survey_id, "李昇峰")

# 或讓所有居民填寫
responses = filler.fill_survey_for_all_residents(survey_id)
```

---

## 系統運作原理

```
問卷問題
    ↓
從 agent.json 載入背景資訊
    ↓
從 simulation.md 載入活動歷史
    ↓
從 data/prompts/ 載入對應模板
    ↓
替換模板變數，生成完整 prompt
    ↓
調用 Ollama API (qwen2.5:7b)
    ↓
處理 LLM 輸出，格式化回答
    ↓
保存問卷回應
```

---

## 關鍵文件位置

| 文件類型 | 路徑 | 用途 |
|----------|------|------|
| AI 居民資料 | `frontend/static/assets/village/agents/[姓名]/agent.json` | 背景、家庭、經濟狀況 |
| 活動歷史 | `results/compressed/test_0513/simulation.md` | 近期活動記錄 |
| Prompt 模板 | `data/prompts/survey_*.txt` | 各類問題的提示詞 |
| 核心程式 | `survey_system/ollama_generator.py` | Ollama 生成器 |
| 填寫器 | `survey_system/ai_filler.py` | AI 填寫邏輯 |

---

## 自訂 Prompt

要修改 AI 的回答風格，直接編輯模板文件：

```bash
# 修改單選題的 prompt
vim data/prompts/survey_single_choice.txt

# 修改文字題的 prompt
vim data/prompts/survey_text.txt
```

### 模板變數說明

在模板中可以使用以下變數：

```
${agent_name}          - 居民姓名
${age}                 - 年齡
${personality}         - 性格特質
${interests}           - 興趣專長
${lifestyle}           - 生活習慣
${current_activity}    - 當前活動
${family_background}   - 家庭背景
${wealth_level}        - 經濟狀況
${activity_history}    - 近期活動記錄
${question_text}       - 問題內容
${options}             - 選項列表（選擇題）
```

### 模板範例

**data/prompts/survey_text.txt**:
```
你是 ${agent_name}，請根據你的背景資訊和生活經歷來回答以下問卷問題。

=== 你的背景資訊 ===
年齡：${age}歲
性格特質：${personality}
...

=== 家庭背景 ===
${family_background}

=== 經濟狀況 ===
${wealth_level}

=== 近期活動記錄 ===
${activity_history}

=== 問卷問題 ===
問題：${question_text}

=== 回答要求 ===
請根據你的實際情況和經歷，用第一人稱回答這個問題。
回答要真實、具體，符合你的背景和經濟狀況，長度約50-150字。

請回答：
```

修改後立即生效，無需重啟程式！

---

## 常見問題

### Q1: 如何停用 Ollama，使用規則引擎？

```python
filler = AIResidentSurveyFiller(
    survey_manager=manager,
    use_ollama=False  # 停用 Ollama
)
```

### Q2: 如何查看生成的完整 prompt？

在 `ollama_generator.py` 中添加除錯輸出：

```python
def _build_prompt(self, ...):
    prompt = template.safe_substitute(template_vars)
    print("=" * 60)
    print("生成的 Prompt:")
    print(prompt)
    print("=" * 60)
    return prompt
```

### Q3: 如何修改 LLM 生成參數？

編輯 `ollama_generator.py` 的 `generate_response()` 方法：

```python
"options": {
    "temperature": 0.7,    # 創造性 (0.0-1.0，越高越創意)
    "top_p": 0.9,          # 採樣範圍
    "num_predict": 500     # 最大 token 數
}
```

### Q4: 如何添加新的 AI 居民？

1. 在 `frontend/static/assets/village/agents/` 創建新目錄
2. 添加 `agent.json` 文件（參考現有格式）
3. 系統會自動識別並載入

### Q5: 回答格式不正確怎麼辦？

檢查並修改對應的 prompt 模板，特別是「回答要求」部分。

---

## 進階功能

### 批量生成

```python
# 為所有居民生成回應
for resident_name in filler.residents_info.keys():
    response = filler.fill_survey_for_resident(survey_id, resident_name)
    print(f"✓ {resident_name} 完成")
```

### 自訂 agents 目錄

```python
from survey_system.ollama_generator import OllamaSurveyGenerator

generator = OllamaSurveyGenerator(
    simulation_md_path="path/to/simulation.md",
    agents_dir="/custom/path/to/agents"  # 自訂路徑
)
```

### 快取管理

Agent 資料會自動快取，如需清除：

```python
generator.agents_cache.clear()  # 清除快取
```

---

## 效能優化

### 回應時間

- 單個問題: **2-5 秒**
- 完整問卷 (10 題): **20-50 秒**
- 建議: 使用背景任務或批次處理

### 記憶體使用

- qwen2.5:7b 需要約 **8GB GPU 記憶體**
- Agent 資料快取約 **1-2MB**

### Token 限制

- Activity history 限制: **1000 字符**
- 總 prompt 長度: 約 **2000-3000 token**
- LLM 輸出上限: **500 token**

---

## 系統優勢

| 特性 | 說明 |
|------|------|
| 🎯 真實性 | 基於真實資料，不是硬編碼 |
| 🔧 易維護 | 模板化管理，修改不需改程式碼 |
| 📈 可擴展 | 新增居民/問題類型都很簡單 |
| 💾 效能優化 | 資料快取，避免重複載入 |
| 🔄 自動容錯 | Ollama 失敗自動切換規則引擎 |

---

## 相關文件

- **[Ollama問卷系統架構說明.md](Ollama問卷系統架構說明.md)** - 詳細技術文件
- **[CLAUDE.md](CLAUDE.md)** - 專案管理指南
- **[問卷系統使用指南.md](問卷系統使用指南.md)** - 問卷系統完整文件

---

## 下一步

1. ✅ 運行 `python test_ollama_survey.py` 確認系統正常
2. ✅ 根據需要調整 prompt 模板
3. ✅ 整合到你的應用程式
4. ✅ 查看詳細技術文件了解更多

**開始使用吧！** 🚀
