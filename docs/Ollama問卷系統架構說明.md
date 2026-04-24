# Ollama LLM 問卷系統架構說明

## 系統概述

系統已完成重構，實現完全動態化的資料載入和 prompt 管理。

### 核心特性

✅ **無預設資料** - 直接從 JSON 文件動態載入
✅ **Prompt 模板化** - 使用 data/prompts/ 管理提示詞
✅ **真實資料驅動** - 基於 agent.json 和 simulation.md
✅ **快取優化** - agent 資料會被快取以提高效能

---

## 系統架構

```
問卷填寫流程
│
├─ AIResidentSurveyFiller (survey_system/ai_filler.py)
│  ├─ 載入居民基本資訊
│  └─ 調用 OllamaSurveyGenerator
│
├─ OllamaSurveyGenerator (survey_system/ollama_generator.py)
│  │
│  ├─ [資料來源]
│  │  ├─ frontend/static/assets/village/agents/[姓名]/agent.json
│  │  └─ results/compressed/test_0513/simulation.md
│  │
│  ├─ [Prompt 模板]
│  │  ├─ data/prompts/survey_single_choice.txt
│  │  ├─ data/prompts/survey_multiple_choice.txt
│  │  ├─ data/prompts/survey_rating.txt
│  │  └─ data/prompts/survey_text.txt
│  │
│  └─ [生成流程]
│     ├─ 1. 從 agent.json 載入居民資料
│     ├─ 2. 從 simulation.md 提取活動歷史
│     ├─ 3. 載入對應的 prompt 模板
│     ├─ 4. 替換模板變數
│     └─ 5. 調用 Ollama API 生成回答
│
└─ Ollama LLM (本地服務)
   └─ Model: qwen3:14b
```

---

## 資料來源

### 1. Agent 配置文件

**路徑**: `frontend/static/assets/village/agents/[居民姓名]/agent.json`

**內容**:
```json
{
  "name": "李昇峰",
  "currently": "當前活動描述...",
  "scratch": {
    "age": 45,
    "innate": "耐心、善良、有條理",
    "learned": "職業描述...",
    "lifestyle": "生活習慣...",
    "family_background": "家庭背景詳細描述...",
    "wealth_level": "經濟狀況詳細描述..."
  }
}
```

**載入方式**:
- OllamaSurveyGenerator 會自動掃描 agents 目錄
- 根據居民姓名動態載入對應的 agent.json
- 首次載入後會快取，避免重複讀取

### 2. 活動歷史文件

**路徑**: `results/compressed/test_0513/simulation.md`

**格式**:
```markdown
### 李昇峰
位置：the Ville，柳樹市場和藥店，商店，藥店櫃檯後面
活動：准備顧客可能需要的常用藥品

### 李昇峰
位置：the Ville，柳樹市場和藥店，商店，雜貨店櫃檯後面
活動：檢查店內安全措施
...
```

**載入方式**:
- 解析 markdown 文件
- 為每位居民保留最近 100 條活動記錄
- 在生成 prompt 時只使用最後 1000 字符（避免超過 token 限制）

---

## Prompt 模板系統

### 模板位置

`data/prompts/survey_[question_type].txt`

### 模板變數

所有模板都支持以下變數（使用 Python Template 語法 `${variable}`）:

| 變數名 | 來源 | 說明 |
|--------|------|------|
| `${agent_name}` | - | 居民姓名 |
| `${age}` | agent.json → scratch.age | 年齡 |
| `${personality}` | agent.json → scratch.innate | 性格特質 |
| `${interests}` | agent.json → scratch.learned | 興趣與專長 |
| `${lifestyle}` | agent.json → scratch.lifestyle | 生活習慣 |
| `${current_activity}` | agent.json → currently | 當前活動 |
| `${family_background}` | agent.json → scratch.family_background | 家庭背景 |
| `${wealth_level}` | agent.json → scratch.wealth_level | 經濟狀況 |
| `${activity_history}` | simulation.md | 近期活動記錄（最後1000字符） |
| `${question_text}` | 問卷問題 | 問題內容 |
| `${options}` | 問卷選項 | 格式化的選項列表 |

### 模板範例

**survey_single_choice.txt**:
```
你是 ${agent_name}，請根據你的背景資訊和生活經歷來回答以下問卷問題。

=== 你的背景資訊 ===
年齡：${age}歲
性格特質：${personality}
...

=== 問卷問題 ===
問題：${question_text}

選項：
${options}

=== 回答要求 ===
請從上述選項中選擇最符合你情況的一個選項。
只需回答選項的完整內容，不要加上編號或其他說明。

請直接回答選項內容：
```

### 自訂 Prompt 模板

要修改 prompt，只需編輯對應的 txt 文件：

```bash
# 編輯單選題 prompt
vim data/prompts/survey_single_choice.txt

# 編輯文字題 prompt
vim data/prompts/survey_text.txt
```

變更會立即生效，無需重啟程式。

---

## 程式實作

### OllamaSurveyGenerator 核心方法

#### 1. `_load_agent_data(agent_name)`

從 agent.json 載入居民資料並快取。

```python
agent_data = self._load_agent_data("李昇峰")
# 返回: {'name': '李昇峰', 'scratch': {...}, ...}
```

#### 2. `_load_simulation_history(file_path)`

解析 simulation.md，為每位居民建立活動歷史。

```python
self.simulation_history = self._load_simulation_history("results/.../simulation.md")
# 返回: {'李昇峰': '位置：...\n活動：...', ...}
```

#### 3. `_load_prompt_template(question_type)`

從文件載入 prompt 模板。

```python
template = self._load_prompt_template("single_choice")
# 返回: Template 物件
```

#### 4. `_build_prompt(...)`

結合 agent 資料、活動歷史和模板，生成完整 prompt。

```python
prompt = self._build_prompt(
    agent_name="李昇峰",
    agent_data=agent_data,
    question_text="您每月的收入大約是多少?",
    question_type="single_choice",
    options=["20,000-30,000元", ...]
)
```

#### 5. `generate_response(...)`

調用 Ollama API 生成回答。

```python
response = generator.generate_response(
    resident_name="李昇峰",
    resident_info={},  # 不再使用，保留以兼容舊接口
    question_text="您每月的收入大約是多少?",
    question_type="single_choice",
    options=[...]
)
```

---

## 使用方式

### 1. 確保資料文件存在

```bash
# 檢查 agent 配置
ls frontend/static/assets/village/agents/李昇峰/agent.json

# 檢查活動歷史
ls results/compressed/test_0513/simulation.md

# 檢查 prompt 模板
ls data/prompts/survey_*.txt
```

### 2. 初始化並使用

```python
from survey_system.models import SurveyManager
from survey_system.ai_filler import AIResidentSurveyFiller

# 初始化
manager = SurveyManager()
filler = AIResidentSurveyFiller(
    survey_manager=manager,
    simulation_md_path="results/compressed/test_0513/simulation.md",
    use_ollama=True
)

# 讓所有居民填寫問卷
responses = filler.fill_survey_for_all_residents(survey_id)

# 或讓特定居民填寫
response = filler.fill_survey_for_resident(survey_id, "李昇峰")
```

### 3. 運行測試

```bash
python test_ollama_survey.py
```

---

## 資料流程範例

### 當李昇峰回答「您每月的收入大約是多少?」

**步驟 1**: 載入 agent.json
```json
{
  "name": "李昇峰",
  "scratch": {
    "age": 45,
    "wealth_level": "中上。藥店經營穩定，月收入約10-15萬元..."
  }
}
```

**步驟 2**: 提取活動歷史
```
位置：the Ville，柳樹市場和藥店，商店，藥店櫃檯後面
活動：准備顧客可能需要的常用藥品
```

**步驟 3**: 載入模板 (survey_single_choice.txt)
```
你是 ${agent_name}，請根據你的背景資訊...
經濟狀況：${wealth_level}
問題：${question_text}
```

**步驟 4**: 替換變數，生成最終 prompt
```
你是 李昇峰，請根據你的背景資訊...

=== 經濟狀況 ===
中上。藥店經營穩定，月收入約10-15萬元...

=== 近期活動記錄 ===
位置：the Ville，柳樹市場和藥店...

=== 問卷問題 ===
問題：您每月的收入大約是多少?
選項：
1. 20,000-30,000元
2. 30,000-50,000元
3. 100,000元以上
```

**步驟 5**: 調用 Ollama 生成回答
```
100,000元以上
```

---

## 優勢

### 相比舊版本

| 項目 | 舊版本 | 新版本 |
|------|--------|--------|
| 資料來源 | 硬編碼預設值 | 動態從 JSON 載入 |
| Prompt 管理 | 寫死在程式碼 | 使用獨立模板文件 |
| 擴展性 | 需修改程式碼 | 只需編輯模板文件 |
| 維護性 | 困難 | 簡單 |
| 資料一致性 | 可能不同步 | 確保使用最新資料 |

---

## 配置檔案

### .env

```bash
# Ollama 服務設定
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:14b

# 其他設定
DEBUG=True
```

---

## 故障排除

### Q: 找不到 agent.json
**A**: 檢查路徑是否正確，確保居民姓名與目錄名稱一致

### Q: 找不到 prompt 模板
**A**: 確認 data/prompts/ 目錄下有對應的模板文件

### Q: Ollama 連線失敗
**A**: 確認 Ollama 服務正在運行 (`ollama serve`)

### Q: 生成的回答格式不對
**A**: 檢查並修改對應的 prompt 模板，調整「回答要求」部分

---

## 擴展與自訂

### 新增問題類型

1. 創建新的 prompt 模板:
```bash
cp data/prompts/survey_text.txt data/prompts/survey_new_type.txt
vim data/prompts/survey_new_type.txt
```

2. 系統會自動識別新模板（檔名格式: `survey_[type].txt`）

### 添加新的模板變數

在 `ollama_generator.py` 的 `_build_prompt()` 方法中添加:

```python
template_vars = {
    ...
    "new_variable": some_data,
}
```

然後在模板中使用 `${new_variable}`

---

## 總結

新架構實現了:
- ✅ 資料與邏輯分離
- ✅ 模板化 prompt 管理
- ✅ 動態資料載入
- ✅ 高度可擴展性
- ✅ 易於維護和調試

所有設定都是配置驅動，無需修改程式碼即可調整行為。
