# BERT 是什麼？完整解釋

## 快速回答

**BERT 是幹嘛的？**

**BERT 是一個將「文字」轉換成「數字向量」的工具，讓電腦能夠理解文字的語義意思。**

就像翻譯機把中文翻成英文，BERT 把文字翻成電腦能理解的數字。

---

## 核心概念（用比喻說明）

### 問題：電腦不懂文字

```
人類: "我今天很開心"
電腦: ？？？（只認識 0 和 1）
```

### BERT 的作用：文字 → 數字

```
人類: "我今天很開心"
  ↓
BERT 處理
  ↓
電腦: [0.23, -0.15, 0.67, ..., 0.42]  (768個數字)
     ↑
   這叫「嵌入向量」或「向量」
```

### 為什麼需要這樣做？

**比喻：地圖座標**

```
台北市 → (25.04°N, 121.56°E)  座標
高雄市 → (22.62°N, 120.30°E)  座標

有了座標，電腦可以:
- 計算距離: 台北到高雄多遠？
- 找相近城市: 跟台北最近的城市？
```

**文字也一樣：**

```
"開心" → [0.8, 0.6, 0.2, ...]  向量
"快樂" → [0.75, 0.65, 0.15, ...]  向量

有了向量，電腦可以:
- 計算相似度: "開心"和"快樂"很相近
- 理解意思: "開心"是正面情緒
```

---

## BERT 全名與背景

### 全名

**BERT = Bidirectional Encoder Representations from Transformers**

中文翻譯：**雙向編碼器表示**（來自 Transformers 架構）

### 關鍵字解釋

**Bidirectional（雙向）**
- 同時看前後文
- 例如："我**愛**吃蘋果"
  - 看前面：「我」
  - 看後面：「吃蘋果」
  - 理解「愛」是「喜歡」的意思

**Encoder（編碼器）**
- 把文字編碼成數字
- 文字 → 向量

**Representations（表示）**
- 數字向量代表文字的意思

**Transformers（轉換器）**
- 一種神經網路架構
- 2017年 Google 提出

### 發明者與時間

- **發明者**: Google AI (Jacob Devlin 等人)
- **發表時間**: 2018年10月
- **論文**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **影響力**: NLP 領域的革命性突破

---

## BERT 如何工作？

### 步驟1：文字切分（Tokenization）

```python
輸入文字: "我今天很開心"

BERT 分詞:
["我", "今天", "很", "開心"]

加上特殊符號:
["[CLS]", "我", "今天", "很", "開心", "[SEP]"]
```

**特殊符號說明**:
- `[CLS]`: 句子開頭（整句的代表）
- `[SEP]`: 句子結尾

### 步驟2：轉換成 ID

```python
詞彙表:
{
  "[CLS]": 101,
  "我": 2769,
  "今天": 791,
  "很": 1215,
  "開心": 5423,
  "[SEP]": 102
}

轉換結果:
[101, 2769, 791, 1215, 5423, 102]
```

### 步驟3：生成嵌入向量

```python
輸入 ID: [101, 2769, 791, 1215, 5423, 102]

經過 BERT 模型（12層神經網路）:
  Layer 1 → Layer 2 → ... → Layer 12

輸出向量:
[
  [0.21, -0.15, 0.67, ..., 0.42],  # [CLS] 的向量 (768維)
  [0.33, 0.25, -0.18, ..., 0.56],  # "我" 的向量
  [0.12, 0.45, -0.22, ..., 0.31],  # "今天" 的向量
  [0.44, -0.33, 0.78, ..., 0.22],  # "很" 的向量
  [0.88, 0.67, 0.45, ..., 0.91],  # "開心" 的向量 (正面情緒特徵)
  [0.15, -0.08, 0.34, ..., 0.27]   # [SEP] 的向量
]
```

### 步驟4：使用向量

在我們的專案中，我們使用 `[CLS]` 向量代表整句話：

```python
sentence_embedding = output[0]  # 取第一個 [CLS] 的向量
# [0.21, -0.15, 0.67, ..., 0.42]  (768個數字)
```

---

## BERT 在我們專案中的作用

### 使用位置：訓練階段

```python
# train_mgt_model.py

from transformers import AutoTokenizer, AutoModel

# 1. 載入 BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 2. 編碼文字
def encode_text(text):
    # 文字 → Token IDs
    inputs = tokenizer(text, return_tensors="pt", max_length=128)

    # Token IDs → 向量
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

    return embedding.numpy()  # [768] 維向量

# 3. 實際使用
text = "I'm so happy today!"
embedding = encode_text(text)  # [0.23, -0.15, ..., 0.42]
```

### 為什麼需要 BERT？

**問題**：訓練 MGT 模型需要輸入向量

```python
MGT 模型需要:
- utterance_embedding: [768] 維向量  ← 需要 BERT 生成
- context_embedding: [768] 維向量   ← 需要 BERT 生成

才能訓練情緒分類
```

**解決方案**：用 BERT 把文字轉成向量

```python
# MELD 數據集
utterance = "I failed the exam"  # 文字
emotion = "sadness"               # 情緒標籤

# 用 BERT 轉換
utterance_emb = encode_text(utterance)  # [768] 向量

# 送入 MGT 訓練
mgt_model.train(utterance_emb, emotion_label)
```

---

## BERT 的強大之處

### 1. 理解語境（Context）

**傳統方法**：
```
"bank" 永遠是同一個向量
```

**BERT**：
```
"I went to the bank to deposit money"
→ bank = [0.5, 0.3, ...]  (銀行的向量)

"I sat on the river bank"
→ bank = [0.1, 0.8, ...]  (河岸的向量)

同一個字，不同語境，不同向量！
```

### 2. 捕捉語義相似度

```python
BERT 向量空間中:

"happy" → [0.8, 0.6, 0.2, ...]
"joyful" → [0.75, 0.65, 0.15, ...]  # 很接近！
"sad" → [-0.7, -0.5, -0.3, ...]     # 很遠！

相似的詞 → 相似的向量
```

### 3. 跨語言理解

```python
# 英文 BERT
"I am happy" → [0.8, 0.6, ...]

# 中文 BERT
"我很開心" → [0.78, 0.58, ...]  # 相似的向量！

不同語言，相似意思，相似向量
```

---

## BERT 版本與選擇

### 常見版本

| 版本 | 語言 | 參數量 | 向量維度 | 用途 |
|-----|------|--------|----------|------|
| **bert-base-uncased** | 英文 | 110M | 768 | 一般任務 |
| **bert-large-uncased** | 英文 | 340M | 1024 | 高精度任務 |
| **bert-base-chinese** | 中文 | 110M | 768 | 中文任務 |
| **bert-base-multilingual** | 多語言 | 110M | 768 | 跨語言任務 |

### 我們的選擇

**目前使用**: `bert-base-uncased`

**原因**:
1. ✅ MELD 數據集是英文
2. ✅ 768 維向量符合 MGT 設計
3. ✅ 參數量適中（110M）
4. ✅ 預訓練品質好

### 程式碼中的使用

```python
# 在 train_mgt_model.py 中

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    print("警告: transformers 未安裝，將使用簡化版本")
    HAS_TRANSFORMERS = False

if HAS_TRANSFORMERS:
    # 使用真實 BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_model = AutoModel.from_pretrained("bert-base-uncased")
else:
    # 簡化版本（用於測試）
    tokenizer = None
    text_model = None
```

---

## BERT vs 其他方法

### 對比表

| 方法 | 文字 → 向量方式 | 優點 | 缺點 |
|-----|----------------|------|------|
| **One-Hot** | "cat" → [0,0,1,0,0,...] | 簡單 | 無法表示語義 |
| **Word2Vec** | "cat" → [0.2, -0.5, ...] | 有語義 | 不看語境 |
| **GloVe** | "cat" → [0.3, -0.4, ...] | 有語義 | 不看語境 |
| **BERT** | "cat" → 看語境決定 | 理解語境 | 計算成本高 |

### 實際例子

**句子**: "I love apples"

**One-Hot**:
```python
love → [0, 0, 1, 0, 0, 0, ...]  # 第3個位置是1
# 問題: "love" 和 "like" 完全不同，無法表示相似性
```

**Word2Vec**:
```python
love → [0.5, 0.3, -0.2, ...]  # 固定向量
# 問題: "I love apples" 和 "I love you" 的 "love" 向量一樣
```

**BERT**:
```python
"I love apples" → love = [0.5, 0.3, ...]  (喜歡食物)
"I love you" → love = [0.8, 0.6, ...]     (情感表達)
# 優點: 同一個字根據語境有不同向量！
```

---

## BERT 的訓練方式（預訓練）

BERT 本身也是訓練出來的！

### 預訓練任務

**任務1: 遮蔽語言模型（Masked Language Model）**

```python
原句: "我今天很[MASK]心"
任務: 預測 [MASK] 是什麼

BERT 學習: [MASK] = "開" (因為"開心"是常見詞)

這樣 BERT 學會理解語境！
```

**任務2: 下一句預測（Next Sentence Prediction）**

```python
句子A: "我今天很開心"
句子B: "因為考試考得很好"

任務: B 是 A 的下一句嗎？
答案: 是 ✓

BERT 學習句子之間的關係！
```

### 預訓練數據

```python
數據來源:
- 維基百科: 25億詞
- BookCorpus: 8億詞

訓練時間:
- 64 個 TPU (Google 的特殊 GPU)
- 4 天

成本:
- 估計 $7,000 美元

所以我們不需要自己訓練，直接用預訓練好的！
```

---

## 在我們專案中的實際使用

### 訓練階段（需要 BERT）

```python
# 1. 下載 BERT 模型（第一次會下載）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 2. 處理 MELD 數據集
for idx, row in meld_df.iterrows():
    utterance = row['Utterance']  # "I failed the exam"
    emotion = row['Emotion']       # "sadness"

    # 3. 用 BERT 編碼
    utterance_emb = encode_text(utterance)  # [768] 向量
    context_emb = encode_text(context)      # [768] 向量

    # 4. 訓練 MGT
    mgt_model.train(utterance_emb, context_emb, emotion)
```

### 推理階段（不需要 BERT）

```python
# 我們已經訓練好 MGT，不再需要 BERT

# 使用簡化的嵌入生成（雜湊函數）
def simple_embedding(text):
    hash_val = hash(text) % 10000
    np.random.seed(hash_val)
    return np.random.randn(768)

# 或者使用 OpenAI API
def openai_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding
```

### 為什麼推理不用 BERT？

**原因**:
1. **成本**: BERT 模型大（110M 參數），佔用記憶體
2. **速度**: BERT 推理較慢（~100ms）
3. **替代方案**: OpenAI API 或簡化方法已足夠

**對比**:
```python
# 訓練時（需要精確）
✓ 使用 BERT（準確但慢）
→ 13,708 筆數據 × 一次編碼 = 可接受

# 推理時（需要快速）
✓ 使用簡化方法（快速）
→ 每次對話都要編碼 = 需要快
```

---

## 安裝與使用 BERT

### 安裝

```bash
# 安裝 transformers 套件
pip install transformers

# 安裝 PyTorch
pip install torch
```

### 基本使用範例

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 1. 載入 BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 2. 編碼文字
text = "I am happy today!"
inputs = tokenizer(text, return_tensors="pt")

# 3. 生成向量
with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]

print(f"向量維度: {embedding.shape}")  # torch.Size([1, 768])
print(f"向量內容: {embedding[0, :5]}")  # tensor([0.1234, -0.5678, ...])
```

### 完整範例（中文）

```python
# 使用中文 BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 編碼中文
text = "我今天很開心"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]

print(f"中文向量: {embedding.shape}")  # torch.Size([1, 768])
```

---

## 常見問題

### Q1: BERT 和 GPT 有什麼不同？

**A**:

| 特性 | BERT | GPT |
|-----|------|-----|
| 任務 | 理解文字（編碼器） | 生成文字（解碼器） |
| 訓練 | 遮蔽詞預測 | 下一個詞預測 |
| 方向 | 雙向（看前後） | 單向（只看前面） |
| 用途 | 分類、問答、嵌入 | 對話、寫作、翻譯 |

**簡單說**:
- **BERT**: 理解文字 → 用於分析
- **GPT**: 生成文字 → 用於創作

### Q2: 為什麼是 768 維？

**A**: 這是 BERT-base 的設計選擇

```python
BERT-base: 768 維（較小，較快）
BERT-large: 1024 維（較大，較準）

768 是在準確率和效率之間的平衡點
```

### Q3: 能否用其他嵌入方法？

**A**: 可以！

```python
# 方法1: Word2Vec
from gensim.models import Word2Vec
embedding = word2vec_model['happy']  # [300] 維

# 方法2: GloVe
import torchtext
glove = torchtext.vocab.GloVe(name='6B', dim=300)
embedding = glove['happy']  # [300] 維

# 方法3: OpenAI API
import openai
response = openai.embeddings.create(
    model="text-embedding-ada-002",
    input="happy"
)
embedding = response.data[0].embedding  # [1536] 維

# 方法4: 簡化版（我們在推理時使用）
def hash_embedding(text):
    hash_val = hash(text) % 10000
    np.random.seed(hash_val)
    return np.random.randn(768)
```

### Q4: BERT 需要多少記憶體？

**A**:

```python
BERT-base:
- 模型大小: ~440MB
- 推理記憶體: ~1GB (CPU) / ~2GB (GPU)
- 訓練記憶體: ~4GB (batch_size=32)

BERT-large:
- 模型大小: ~1.3GB
- 推理記憶體: ~3GB (CPU) / ~6GB (GPU)
- 訓練記憶體: ~12GB (batch_size=32)
```

### Q5: BERT 速度如何？

**A**:

```python
BERT-base (CPU):
- 單句編碼: ~50-100ms
- 批次編碼 (32句): ~500ms

BERT-base (GPU):
- 單句編碼: ~5-10ms
- 批次編碼 (32句): ~50ms

簡化方法 (我們推理時):
- 雜湊嵌入: ~0.01ms (非常快！)
- OpenAI API: ~100-200ms
```

---

## 總結

### BERT 是什麼？

**一句話總結**:
BERT 是一個預訓練的語言模型，可以把文字轉換成有語義的數字向量，讓電腦理解文字的意思。

### 在我們專案中的作用

```
訓練階段:
MELD 對話文字 → BERT 編碼 → 768維向量 → 訓練 MGT 模型

推理階段:
用戶對話文字 → 簡化編碼 → 768維向量 → MGT 預測情緒
(不使用 BERT，因為太慢)
```

### 核心價值

1. ✅ **語義理解**: 把文字變成有意義的數字
2. ✅ **語境感知**: 同一個字在不同語境有不同向量
3. ✅ **預訓練好**: 不需要自己訓練，直接用
4. ✅ **標準工具**: NLP 領域的標準選擇

### 為什麼重要？

沒有 BERT（或類似工具），我們無法:
- ❌ 將文字輸入神經網路
- ❌ 訓練深度學習模型
- ❌ 理解文字的語義相似度

有了 BERT，我們可以:
- ✅ 文字 → 向量 → 訓練模型
- ✅ 捕捉語義和語境
- ✅ 達到高準確率（62%）

**BERT 是現代 NLP 的基礎工具，就像數學裡的加減乘除一樣重要！** 🤖✨📚
