# 🎭 情緒評分員完整工作流程說明

## 📋 概述

MGT 情緒評分員（MGTEmotionRater）是每位 AI 居民的「情緒顧問」，它會：
1. **觀察分析**：分析居民遇到的情境和對話
2. **評估情緒**：判斷應該有什麼情緒反應
3. **提供建議**：告訴居民如何表達情緒
4. **追蹤演化**：記錄情緒變化歷史

---

## 🔄 情緒評分員的完整工作週期

### 階段 1️⃣: 初始化 (Agent 啟動時)

**時機**: 當 AI 居民第一次被創建時

**發生什麼**:
```python
# 在 modules/agent.py 的 __init__ 中
emotion_config = {
    "emotional_stability": 0.7,    # 盧品蓉的情緒穩定性
    "empathy_level": 0.8,          # 同理心水平
    "optimism": 0.8,               # 樂觀程度
    "anxiety_proneness": 0.3       # 焦慮傾向
}

# 自動創建專屬的情緒評分員
self._init_emotion_system(emotion_config)
```

**結果**:
```
[盧品蓉] 初始化情緒系統
[盧品蓉] 創建 MGT 情緒評分員 (hidden_dim=768)
[盧品蓉] 載入訓練權重: emotion_system/models/mgt/mgt_weights.npz
[盧品蓉] ✅ 情緒系統準備就緒
[盧品蓉] 當前情緒: 😐 中性 (強度: 0.50)
```

---

### 階段 2️⃣: 事件情緒評估 (重要事件發生時)

**時機**:
- AI 居民經歷重要事件 (poignancy > 5)
- 例如：完成重要任務、遇到朋友、收到消息等

**工作流程**:

#### 步驟 1: 捕捉事件
```python
# 在 modules/agent.py 的 _add_concept 中
def _add_concept(self, e_type, event, ...):
    # 計算事件重要性
    poignancy = self.completion("poignancy_event", event)

    # 觸發情緒更新
    self.update_emotion_from_event(
        event_description=event.get_describe(),
        poignancy=poignancy,
        event_type=e_type
    )
```

#### 步驟 2: 情緒評分員分析

**輸入資訊**:
```python
事件描述: "盧品蓉在霍布斯咖啡館喝咖啡時，收到了教授的好評"
事件重要性: 8/10
事件類型: "chat"
當前情緒: 😐 中性 (強度: 0.50)
```

**MGT 評分員處理**:
```python
# 1. 檢索相關記憶
retrieved_memories = [
    "上次收到教授稱讚時感到很開心",
    "我一直努力想獲得認可",
    "咖啡館是我放鬆的地方"
]

# 2. 準備多模態輸入
multimodal_input = MultimodalEmotionInput(
    text_content="收到了教授的好評",
    context_description="在霍布斯咖啡館喝咖啡",
    retrieved_memories=retrieved_memories,
    current_emotion=EmotionState(emotion="neutral", intensity=0.50)
)

# 3. MGT 模型評估
output = mgt_rater.rate_emotion(multimodal_input)
```

**MGT 內部運作** (Multimodal Gating Transformer):

```
Step 1: 文本嵌入
├─ "收到了教授的好評" → [768 維向量]
├─ "在霍布斯咖啡館喝咖啡" → [768 維向量]
└─ 記憶1, 記憶2, 記憶3 → [768 維向量]

Step 2: Parallel Multimodal Flow (平行多模態流)
├─ Text Projection: 處理文本資訊
├─ Context Projection: 處理情境資訊
└─ Memory Projection: 處理記憶資訊

Step 3: Cross-modal Attention (跨模態注意力)
├─ 計算各模態的重要性
├─ 注意力權重:
│   ├─ 文本: 0.45
│   ├─ 情境: 0.30
│   └─ 記憶: 0.25
└─ 融合多模態資訊

Step 4: Gating Mechanism (閘控機制)
├─ 決定使用多少多模態資訊
├─ 閘控值: 0.65 (65% 多模態, 35% 純文本)
└─ 生成最終表徵

Step 5: Emotion Classification (情緒分類)
├─ Layer 1: 768 → 384 (ReLU)
├─ Layer 2: 384 → 7 (各情緒分數)
└─ Softmax 歸一化
```

**輸出結果**:
```python
EmotionRaterOutput(
    predicted_emotion="joy",           # 預測情緒: 喜悅
    emotion_intensity=0.72,            # 強度: 72%
    confidence=0.85,                   # 信心度: 85%

    emotion_distribution={             # 7種情緒的機率分佈
        "joy": 0.68,       # 😊 喜悅 (最高)
        "surprise": 0.15,  # 😲 驚訝
        "neutral": 0.10,   # 😐 中性
        "sadness": 0.03,   # 😢 悲傷
        "fear": 0.02,      # 😨 恐懼
        "anger": 0.01,     # 😠 憤怒
        "disgust": 0.01    # 🤢 厭惡
    },

    interaction_suggestions=[          # 互動建議
        "以愉悅的語氣分享你的喜悅（強度：高）",
        "主動表達積極情緒，增進互動親密度",
        "可以提議一起做些有趣的事情"
    ],

    reasoning="預測情緒為 喜悅（強度 0.72）；主要影響因素：當前情境（注意力權重 0.45）；多模態資訊融合度：0.65",

    attention_weights={
        "context": 0.45,
        "memory": 0.25
    },

    gating_values={
        "multimodal_gate": 0.65,
        "language_preserve": 0.35
    }
)
```

#### 步驟 3: 更新居民情緒狀態

```python
# 情緒分析器更新狀態
self.emotion_analyzer.update_emotion(
    new_emotion="joy",
    intensity=0.72,
    trigger="收到教授好評"
)
```

**居民內部變化**:
```
盧品蓉的情緒狀態更新:
  之前: 😐 中性 (0.50)
  現在: 😊 喜悅 (0.72)

情緒記憶新增:
  時間: 2024-11-20 14:30
  情緒: joy
  強度: 0.72
  觸發: 收到教授好評
  情境: 霍布斯咖啡館
```

---

### 階段 3️⃣: 對話情緒增強 (與其他居民對話時)

**時機**: 盧品蓉與魏祺紘在咖啡館相遇並對話

**工作流程**:

#### 步驟 1: 捕捉對話情境

```python
# 在 modules/agent.py 的 _chat_with 中
curr_context = "盧品蓉 正在霍布斯咖啡館喝咖啡，遇到了 魏祺紘，魏祺紘 正在學習音樂理論"

chat_history = [
    # 之前的對話輪次（如果有）
]
```

#### 步驟 2: 情緒評分員生成對話建議

**呼叫情緒感知對話生成**:
```python
text = self.generate_emotional_chat(
    other_agent_name="魏祺紘",
    chat_context=curr_context,
    chat_history=chat_history,
    use_mgt_rater=True  # 使用 MGT 評分員
)
```

**內部處理** (在 agent_emotion_mixin.py 中):

```python
# 1. 準備多模態輸入
multimodal_input = MultimodalEmotionInput(
    text_content=curr_context,
    context_description=curr_context,
    retrieved_memories=[
        "上次和魏祺紘聊音樂很開心",
        "我們都喜歡在咖啡館學習",
        "魏祺紘很有音樂天賦"
    ],
    current_emotion=EmotionState(emotion="joy", intensity=0.72),
    target_agent_name="魏祺紘"
)

# 2. MGT 評分員評估對話情緒
mgt_output = self.mgt_rater.rate_emotion(
    multimodal_input,
    agent_completion_func=self.completion  # 可選的 LLM 增強
)
```

**MGT 評分結果**:
```python
預測情緒: joy (維持)
情緒強度: 0.75 (略微提升)
信心度: 0.88

互動建議:
1. "以愉悅的語氣分享你的喜悅"
2. "主動表達對魏祺紘音樂的興趣"
3. "可以分享你最近的好事"
```

#### 步驟 3: 構建情緒增強的對話提示詞

**使用 MGT 增強提示詞** (在 mgt_prompts.py 中):

```python
prompt = f"""
你是 盧品蓉，一位友善、樂於助人的學生。

## 當前情緒狀態
😊 喜悅 (強度: 0.75)
你現在感到開心和興奮，因為剛剛收到了教授的好評。

## 你的情緒特質
- 情緒穩定性: 70% (較能控制情緒起伏)
- 同理心水平: 80% (能深刻理解他人感受)
- 樂觀程度: 80% (傾向正面看待事物)
- 焦慮傾向: 30% (不太容易焦慮)

## 當前情境
{curr_context}

## MGT 情緒評分員的建議
{mgt_output.interaction_suggestions}

## 相關記憶
- 上次和魏祺紘聊音樂很開心
- 我們都喜歡在咖啡館學習
- 魏祺紘很有音樂天賦

## 對話歷史
{chat_history}

## 任務
請以符合你當前情緒狀態（喜悅）的方式，自然地回應 魏祺紘。
記得：
1. 讓你的喜悅自然流露在對話中
2. 考慮 MGT 評分員的建議
3. 保持你的性格特質（友善、樂於助人）
4. 用溫暖、積極的語氣

你的回應:
"""
```

#### 步驟 4: LLM 生成情緒豐富的對話

**LLM 輸出** (基於上述提示詞):
```
盧品蓉: (帶著明顯的笑容，眼睛發亮) 魏祺紘！太好了在這裡
遇到你！我剛剛收到教授的好評，心情超好的！(語氣輕快)
你在學音樂理論啊？最近有什麼有趣的發現嗎？我記得你上次
跟我分享的那個和聲理論，我覺得超酷的！
```

**對比原始對話** (無情緒系統):
```
盧品蓉: 嗨，魏祺紘。你在學音樂理論嗎？
```

---

### 階段 4️⃣: 情緒反思 (定期反思時)

**時機**: 當居民的 poignancy 累積達到閾值時觸發反思

**工作流程**:

#### 步驟 1: 觸發反思

```python
# 在 modules/agent.py 的 reflect 中
if self.status["poignancy"] >= self.think_config["poignancy_max"]:
    # 執行一般反思...

    # 執行情緒反思
    if hasattr(self, 'reflect_on_emotions'):
        emotion_insights = self.reflect_on_emotions(
            use_mgt_feedback=True
        )
```

#### 步驟 2: 情緒評分員分析情緒歷史

**收集最近情緒記錄**:
```python
recent_emotions = [
    {時間: "14:00", 情緒: "neutral", 強度: 0.50},
    {時間: "14:30", 情緒: "joy", 強度: 0.72},
    {時間: "15:00", 情緒: "joy", 強度: 0.75},
    {時間: "15:30", 情緒: "joy", 強度: 0.68},
    {時間: "16:00", 情緒: "neutral", 強度: 0.55}
]

recent_mgt_ratings = [
    MGT評分1, MGT評分2, MGT評分3, ...
]
```

**MGT 情緒模式分析**:
```python
# 統計情緒分佈
emotion_counts = {
    "joy": 3,      # 出現3次
    "neutral": 2,  # 出現2次
}

# 識別主導情緒
dominant_emotions = ["joy", "neutral"]

# 分析情緒觸發因素
triggers = {
    "joy": ["收到教授好評", "與魏祺紘對話", "完成作業"],
    "neutral": ["日常活動", "休息時間"]
}
```

#### 步驟 3: 生成情緒洞察

**使用 MGT 反饋的反思提示詞**:
```python
prompt = f"""
作為 盧品蓉，請反思最近的情緒變化和互動經驗。

## 最近情緒統計 (過去 5 次評分)
- 主導情緒: 喜悅(3次), 中性(2次)
- 平均強度: 0.64
- 情緒穩定度: 高 (符合你70%的情緒穩定性)

## MGT 評分員觀察
1. 收到教授好評時 → 喜悅提升顯著 (0.50 → 0.72)
2. 與魏祺紘對話時 → 喜悅持續維持 (0.75)
3. 你對學業成就的情緒反應比對社交的更強烈

## 互動結果
- 與魏祺紘的對話很愉快，雙方都感到開心
- 你的積極情緒感染了周圍的人
- 同理心發揮良好，能感受到他人的需求

請生成 3-5 個情緒洞察，幫助 盧品蓉 更好地理解自己的情緒模式。
"""
```

**LLM 生成的情緒洞察**:
```python
emotion_insights = [
    "我發現當我的學業得到認可時，會感到特別的滿足和喜悅，這可能是因為我重視自己的成長",

    "與朋友分享好消息時，我的喜悅會持續更久，說明社交支持對我的情緒穩定很重要",

    "我注意到自己的樂觀特質幫助我快速從中性狀態轉換到積極情緒，這是我的優勢",

    "魏祺紘對音樂的熱情總是能引起我的興趣和好奇，我們的對話讓我感到充滿活力"
]
```

#### 步驟 4: 儲存洞察到記憶

```python
# 每個洞察都會成為一個「思考」記憶節點
for insight in emotion_insights:
    _add_thought(insight)  # 添加到聯想記憶中
```

**結果**:
```
[盧品蓉] 完成情緒反思，生成 4 個洞察
[盧品蓉] 情緒洞察已儲存到記憶系統
[盧品蓉] 重置 poignancy: 150 → 0
```

---

### 階段 5️⃣: 持續監控和記錄

**整個運行期間，情緒評分員持續工作**:

```python
# 緩存管理
if self.embedding_cache:
    stats = self.embedding_cache.get_stats()
    # 命中率: 75%
    # 記憶體: 85MB
    # 總請求: 150

# 評分歷史
len(self.mgt_rater.rating_history)  # 已評分 47 次

# 情緒記憶
len(self.emotion_memory.emotion_history)  # 記錄 32 次情緒變化
```

---

## 📊 情緒評分員的數據結構

### 輸入數據

```python
MultimodalEmotionInput(
    text_content: str,           # 主要文本 (對話/事件描述)
    context_description: str,    # 情境描述
    retrieved_memories: List,    # 相關記憶 (3-5條)
    current_emotion: EmotionState,  # 當前情緒
    target_agent_name: str,      # 對話對象 (可選)
    additional_context: Dict     # 額外資訊 (可選)
)
```

### 輸出數據

```python
EmotionRaterOutput(
    predicted_emotion: str,              # 預測情緒 (7選1)
    emotion_intensity: float,            # 強度 (0-1)
    confidence: float,                   # 信心度 (0-1)
    emotion_distribution: Dict,          # 7種情緒機率分佈
    interaction_suggestions: List[str],  # 互動建議 (3-5條)
    reasoning: str,                      # 推理說明
    attention_weights: Dict,             # 注意力權重
    gating_values: Dict,                 # 閘控值
    timestamp: datetime                  # 評分時間
)
```

---

## 🎯 真實案例追蹤

### 案例: 盧品蓉的一天

```
時間    | 事件                | MGT評估        | 情緒變化        | 行為影響
--------|--------------------|--------------|--------------|-----------------
08:00   | 起床               | neutral 0.50 | 😐 → 😐      | 正常對話
09:30   | 上課聽到新觀點      | surprise 0.55| 😐 → 😲      | 積極提問
11:00   | 與同學討論         | joy 0.60     | 😲 → 😊      | 熱情分享
12:30   | 午餐時間           | neutral 0.50 | 😊 → 😐      | 平靜放鬆
14:30   | 收到教授好評       | joy 0.72     | 😐 → 😊      | 興奮分享
15:00   | 遇到魏祺紘         | joy 0.75     | 😊 → 😊      | 熱情對話
16:30   | 完成作業           | joy 0.68     | 😊 → 😊      | 滿足感
18:00   | 反思一天           | neutral 0.60 | 😊 → 😐      | 深度思考
```

**MGT 評分員工作統計**:
- 總評分次數: 8 次
- 情緒變化次數: 5 次
- 主導情緒: joy (佔 50%)
- 緩存命中率: 62.5%
- 生成建議數: 24 條
- 情緒洞察: 4 個

---

## 💡 MGT 評分員的獨特價值

### 1. **多模態整合**
不只看文本，還考慮：
- 當前情境 (在哪裡、做什麼)
- 歷史記憶 (過去經驗)
- 個性特質 (情緒穩定性等)

### 2. **個性化評估**
根據每位居民的配置調整：
```python
盧品蓉 (樂觀0.8) 遇到小挫折 → 輕微失望
施宇鴻 (樂觀0.5) 遇到小挫折 → 顯著沮喪
```

### 3. **動態適應**
考慮當前情緒狀態：
```python
當前開心 + 收到好消息 → 更開心
當前悲傷 + 收到好消息 → 漸漸轉好
```

### 4. **可解釋性**
提供推理說明：
```
"預測情緒為 喜悅（強度 0.72）；
主要影響因素：當前情境（注意力權重 0.45）；
多模態資訊融合度：0.65"
```

### 5. **持續學習**
隨著訓練完成，準確率提升：
```
未訓練: 14% (隨機)
訓練中: 46% (Epoch 1/20)
訓練後: 55-65% (預期)
```

---

## 🔬 技術細節

### MGT 模型架構

```
輸入層 (768 維嵌入)
    ↓
┌───────────────────────────────┐
│ Parallel Multimodal Flow      │
│  - Text Projection (768→768)  │
│  - Context Projection (768→768)│
│  - Memory Projection (768→768)│
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│ Cross-modal Attention         │
│  - Query/Key/Value Transform  │
│  - Additive Attention         │
│  - Attention Weighting        │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│ Gating Mechanism              │
│  - Gate(1536→768→Sigmoid)    │
│  - Adaptive Fusion            │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│ Emotion Classifier            │
│  - Layer1 (768→384, ReLU)    │
│  - Layer2 (384→7, Softmax)   │
└───────────────────────────────┘
    ↓
輸出: 7 種情緒的機率分佈
```

### 訓練數據

```
MELD 數據集: 13,708 樣本
- 訓練集: 9,989 (73%)
- 驗證集: 1,109 (8%)
- 測試集: 2,610 (19%)

7 種情緒分佈:
- neutral: 4,710 (34%)
- joy: 2,308 (17%)
- surprise: 1,636 (12%)
- sadness: 1,002 (7%)
- anger: 1,607 (12%)
- disgust: 271 (2%)
- fear: 268 (2%)
```

---

## 📈 性能指標

### 實際運行性能

| 操作 | 時間 | 備註 |
|------|------|------|
| 初始化評分員 | ~100ms | 首次載入權重 |
| 單次情緒評估 | 50-200ms | 視緩存命中率 |
| 生成對話建議 | 2-5秒 | 包含 LLM 調用 |
| 情緒反思 | 3-6秒 | 包含 LLM 調用 |
| 緩存查詢 | <1ms | 命中時 |

### 準確率

| 階段 | 準確率 | 說明 |
|------|--------|------|
| 未訓練 | 14% | 隨機猜測 |
| Epoch 1 | 46% | 訓練初期 |
| 預期 (Epoch 20) | 55-65% | 訓練完成 |
| 人類標註 | ~70% | MELD 數據集 |

---

## 🎊 總結

情緒評分員為每位 AI 居民提供：

1. **觀察能力** 👁️
   - 分析情境、對話和記憶

2. **評估能力** 🧠
   - 預測應有的情緒反應

3. **建議能力** 💡
   - 提供具體的互動建議

4. **記憶能力** 📝
   - 追蹤情緒歷史和模式

5. **學習能力** 📚
   - 透過訓練提升準確率

**結果**: 每位 AI 居民都擁有了真實、動態、個性化的情緒能力！

---

**相關文檔**:
- [EMOTION_INTEGRATION_GUIDE.md](EMOTION_INTEGRATION_GUIDE.md)
- [emotion_system/MGT_RATER_GUIDE.md](emotion_system/MGT_RATER_GUIDE.md)
- [emotion_system/README.md](emotion_system/README.md)
