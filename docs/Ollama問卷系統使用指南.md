# Ollama LLM 問卷系統使用指南

## 概述

系統現已整合 Ollama 本地 LLM，AI居民將根據其背景資訊和活動歷史，使用自然語言生成真實、個性化的問卷回答。

## 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                 問卷填寫系統架構                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐    ┌──────────────────┐                │
│  │   問卷題目   │───>│ AIResidentFiller │                │
│  └─────────────┘    └────────┬─────────┘                │
│                               │                           │
│                               v                           │
│                    ┌──────────────────┐                  │
│                    │  使用Ollama?     │                  │
│                    └────┬─────────┬───┘                  │
│                         │ Yes     │ No                   │
│                         v         v                       │
│              ┌──────────────┐  ┌──────────┐             │
│              │Ollama生成器  │  │規則引擎  │             │
│              └──────┬───────┘  └─────┬────┘             │
│                     │                 │                   │
│                     │ 失敗時fallback  │                   │
│                     └────────>────────┘                   │
│                               │                           │
│                               v                           │
│                    ┌──────────────────┐                  │
│                    │   格式化回答     │                  │
│                    └──────────────────┘                  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 核心功能

### 1. Ollama LLM 生成器

**檔案**: `survey_system/ollama_generator.py`

**功能**:
- 讀取 AI 居民的 agent.json 背景資訊
- 解析 simulation.md 活動歷史
- 根據問題類型生成符合格式的回答
- 支援四種問題類型: 單選、多選、評分、文字

**主要方法**:
```python
# 初始化生成器
generator = OllamaSurveyGenerator(
    simulation_md_path="results/compressed/test_0513/simulation.md"
)

# 生成單個回答
response = generator.generate_response(
    resident_name="李昇峰",
    resident_info=resident_info_dict,
    question_text="您每月的收入大約是多少?",
    question_type="single_choice",
    options=["20,000-30,000元", "30,000-50,000元", ...]
)
```

### 2. 提示詞工程

系統為每個問題構建完整的提示詞，包含:

#### 基本身份
```
你是 李昇峰，請根據你的背景資訊和生活經歷來回答以下問卷問題。
```

#### 背景資訊
- 年齡
- 性格特質
- 興趣與專長
- 生活習慣
- 當前活動

#### 家庭背景
```
出身於嘉義市的醫藥世家，祖父和父親都是藥師...
```

#### 經濟狀況
```
中上。藥店經營穩定，月收入約10-15萬元...
```

#### 近期活動記錄
```
位置：the Ville，柳樹市場和藥店，商店，藥店櫃檯後面
活動：准備顧客可能需要的常用藥品
...
```

#### 回答要求（依問題類型）
- **單選題**: "請從上述選項中選擇最符合你情況的一個選項..."
- **多選題**: "請從上述選項中選擇所有符合你情況的選項..."
- **評分題**: "請給出一個1-10之間的數字評分..."
- **文字題**: "請根據你的實際情況和經歷，用第一人稱回答..."

## 使用方式

### 方法一: 使用測試腳本

```bash
python test_ollama_survey.py
```

此腳本會:
1. 自動載入 simulation.md
2. 初始化 Ollama 生成器
3. 選擇第一個可用問卷
4. 為測試居民生成回答
5. 顯示生成結果

### 方法二: Python 程式調用

```python
from survey_system.models import SurveyManager
from survey_system.ai_filler import AIResidentSurveyFiller

# 初始化（啟用Ollama）
manager = SurveyManager()
filler = AIResidentSurveyFiller(
    survey_manager=manager,
    simulation_md_path="results/compressed/test_0513/simulation.md",
    use_ollama=True  # 啟用Ollama LLM
)

# 讓所有居民填寫問卷
responses = filler.fill_survey_for_all_residents(survey_id)

# 或讓特定居民填寫
response = filler.fill_survey_for_resident(survey_id, "李昇峰")
```

### 方法三: 停用 Ollama（使用規則引擎）

```python
# 停用Ollama，使用原本的規則引擎
filler = AIResidentSurveyFiller(
    survey_manager=manager,
    use_ollama=False  # 停用Ollama
)
```

## 配置檔案

### .env 設定

```bash
# Ollama 本地服務設定
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:14b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
```

### 檢查 Ollama 服務狀態

```bash
# 檢查Ollama是否運行
curl http://127.0.0.1:11434/api/tags

# 檢查可用模型
ollama list
```

## 回答格式處理

### 單選題
LLM 輸出: "30,000-50,000元"
系統處理: 匹配選項列表，找到完全符合的選項

### 多選題
LLM 輸出: "飲食、交通、娛樂"
系統處理: 分割字串，匹配所有選項，返回列表

### 評分題
LLM 輸出: "評分：8，理由：藥店經營穩定，收入可觀"
系統處理: 提取數字 8，驗證範圍 1-10

### 文字題
LLM 輸出: "我每個月的收入約10-15萬元，主要來自藥店營業收入..."
系統處理: 直接使用原始文字

## 錯誤處理

### Fallback 機制

```python
try:
    # 嘗試使用 Ollama 生成
    response = ollama_generator.generate_response(...)
except Exception as e:
    print(f"Ollama生成失敗，使用規則引擎: {e}")
    # 自動切換到規則引擎
    response = _generate_with_rules(...)
```

### 常見錯誤

| 錯誤 | 原因 | 解決方案 |
|------|------|----------|
| Connection refused | Ollama 服務未運行 | 執行 `ollama serve` |
| Model not found | 模型未下載 | 執行 `ollama pull qwen3:14b` |
| Timeout | 生成時間過長 | 調整 timeout 參數或使用更小的模型 |
| Invalid format | LLM 輸出格式錯誤 | 系統會自動處理並 fallback |

## 優勢對比

### Ollama LLM 模式
✅ 自然語言回答，更真實
✅ 考慮完整背景和活動歷史
✅ 回答具有個性化特徵
✅ 適應各種問題類型
✅ 可處理複雜、開放式問題

### 規則引擎模式
✅ 快速、可預測
✅ 不需要額外服務
✅ 適合結構化問題
✅ 作為 fallback 機制
❌ 回答較機械化

## 效能考量

### 回答生成時間
- 單個問題: 2-5 秒（取決於問題複雜度和模型大小）
- 完整問卷（10題）: 20-50 秒
- 建議: 使用批量處理或背景任務

### 記憶體使用
- qwen3:14b 模型約需 8GB GPU 記憶體
- 建議: 在有足夠資源的機器上運行

### API 調用頻率
- Ollama 是本地服務，無 API 限制
- 可並行處理多個請求

## 最佳實踐

1. **首次使用前確保 Ollama 正常運行**
   ```bash
   ollama serve
   ollama pull qwen3:14b
   ```

2. **使用適當的 simulation.md 路徑**
   - 確保路徑指向最新的模擬記錄
   - 活動歷史越豐富，回答越真實

3. **監控生成品質**
   - 檢查 LLM 輸出是否符合預期
   - 必要時調整提示詞

4. **合理使用 fallback**
   - 保留規則引擎作為備用方案
   - 在 Ollama 不可用時自動切換

5. **優化提示詞**
   - 根據實際使用情況調整提示詞
   - 可在 `ollama_generator.py` 中修改 `_build_prompt()` 方法

## 範例輸出

### 文字題範例

**問題**: "請描述您的家庭經濟狀況"

**李昇峰（藥店老闆）的回答**:
```
我經營柳樹市場的藥店已經十多年了，收入還算穩定，每個月大概有10-15萬元
的營業額。加上太太在大學當教授，家庭經濟算是寬裕。我們擁有自己的店面
和住宅，也有一些儲蓄和投資。雖然不算富裕，但生活舒適，能夠支持兒子魏
祺紘追求音樂夢想，我覺得很滿足。
```

**蔡宗陞（咖啡店老闆）的回答**:
```
我的咖啡館開業五年了，經營漸入佳境。每個月收入約7-10萬元，扣除咖啡豆
成本、房租和人事費用後，淨利大概4-6萬元。店面是租的，我自己也租住公寓。
創業初期有貸款壓力，但現在已經好轉。我會把賺到的錢投資在咖啡豆品質和
設備升級上，生活簡樸但對事業充滿熱情。
```

## 進階功能

### 自訂生成參數

```python
# 在 ollama_generator.py 中調整
response = requests.post(
    self.api_url,
    json={
        "model": self.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,    # 創造性 (0.0-1.0)
            "top_p": 0.9,          # 採樣範圍
            "max_tokens": 500      # 最大長度
        }
    }
)
```

### 批量生成

```python
# 為多個居民批量生成回答
responses = []
for resident_name in filler.residents_info.keys():
    response = filler.fill_survey_for_resident(survey_id, resident_name)
    responses.append(response)
```

## 故障排除

### 檢查清單

- [ ] Ollama 服務是否運行？(`curl http://127.0.0.1:11434`)
- [ ] 模型是否已下載？(`ollama list`)
- [ ] simulation.md 文件是否存在？
- [ ] agent.json 配置是否完整？
- [ ] .env 配置是否正確？

### 除錯模式

在 ai_filler.py 中添加除錯輸出:

```python
print(f"使用Ollama: {self.use_ollama}")
print(f"問題類型: {question_type}")
print(f"LLM回應: {llm_response}")
```

## 未來擴展

可能的改進方向:
1. 支援更多 LLM 模型（如 Llama 3, Mistral）
2. 實現回答品質評分機制
3. 添加對話式問卷支援
4. 整合向量資料庫進行語義搜尋
5. 支援多語言問卷

## 相關文件

- [問卷系統使用指南.md](問卷系統使用指南.md)
- [CLAUDE.md](CLAUDE.md) - 專案管理指南
- [Ollama 官方文件](https://ollama.ai/docs)

## 聯絡與支援

如有問題或建議，請查閱專案文件或測試腳本範例。
