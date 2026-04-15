"""
MGT 情緒評分員模組

基於論文 "Injecting Multimodal Information Into Pre-Trained Language Model
for Multimodal Sentiment Analysis" 的 MGT (Multimodal Gating Transformer) 架構

核心功能:
1. Parallel Multimodal Flow Module - 平行多模態流模組
2. Cross-modal Additive Attention - 跨模態加法注意力
3. Gating Mechanism - 閘控機制動態控制情緒資訊流
4. 提供情緒評估和互動建議給 AI 居民
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import math
import numpy as np
import os

from .core import EmotionState, EMOTION_TYPES, EMOTION_DISPLAY

# 嘗試載入嵌入緩存（可選）
try:
    from .embedding_cache import EmbeddingCache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False
    print("⚠️  未找到 embedding_cache，將不使用緩存功能")


def _get_emotion_name(emotion_code: str) -> str:
    """從 EMOTION_DISPLAY 中提取情緒中文名稱"""
    display = EMOTION_DISPLAY.get(emotion_code, emotion_code)
    # 移除表情符號，只保留中文名稱
    if " " in display:
        return display.split(" ")[1]
    return display


@dataclass
class MultimodalEmotionInput:
    """多模態情緒輸入數據結構"""

    # 語言模態 (Language modality)
    text_content: str
    text_embeddings: Optional[List[float]] = None

    # 情境模態 (Context modality)
    context_description: str = ""
    context_embeddings: Optional[List[float]] = None

    # 記憶模態 (Memory modality)
    retrieved_memories: List[str] = field(default_factory=list)
    memory_embeddings: Optional[List[List[float]]] = None

    # 當前情緒狀態 (Current emotion state)
    current_emotion: Optional[EmotionState] = None

    # 互動對象資訊 (Interaction target)
    target_agent_name: Optional[str] = None
    target_emotion_state: Optional[EmotionState] = None


@dataclass
class EmotionRaterOutput:
    """情緒評分員輸出結果"""

    # 評估的情緒
    predicted_emotion: str
    emotion_intensity: float
    confidence: float  # 預測信心度

    # 各情緒的機率分佈
    emotion_distribution: Dict[str, float] = field(default_factory=dict)

    # 互動建議
    interaction_suggestions: List[str] = field(default_factory=list)

    # 推理過程
    reasoning: str = ""

    # 注意力權重（用於解釋）
    attention_weights: Dict[str, float] = field(default_factory=dict)

    # 閘控值（各模態的重要性）
    gating_values: Dict[str, float] = field(default_factory=dict)


class ParallelMultimodalFlow:
    """
    平行多模態流模組

    論文核心創新：並行處理多模態資訊而非順序注入
    避免破壞預訓練語言模型的內在表徵
    """

    def __init__(self, hidden_dim: int = 768):
        """
        初始化平行多模態流模組

        Args:
            hidden_dim: 隱藏層維度（與 LLM embedding 維度對齊）
        """
        self.hidden_dim = hidden_dim

        # 模擬的投影層權重（實際應用會使用 PyTorch/TensorFlow）
        self._text_projection = self._init_projection_weights()
        self._context_projection = self._init_projection_weights()
        self._memory_projection = self._init_projection_weights()

    def _init_projection_weights(self) -> np.ndarray:
        """初始化投影權重（Xavier initialization）"""
        limit = math.sqrt(6.0 / (self.hidden_dim * 2))
        return np.random.uniform(-limit, limit, (self.hidden_dim, self.hidden_dim))

    def project_modality(self, embeddings: np.ndarray, projection_weights: np.ndarray) -> np.ndarray:
        """
        將模態嵌入投影到共同空間

        Args:
            embeddings: 輸入嵌入向量
            projection_weights: 投影權重矩陣

        Returns:
            投影後的向量
        """
        return np.dot(embeddings, projection_weights)

    def forward(self,
                text_emb: np.ndarray,
                context_emb: np.ndarray,
                memory_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        前向傳播 - 並行處理三個模態

        Args:
            text_emb: 文本嵌入
            context_emb: 情境嵌入
            memory_emb: 記憶嵌入

        Returns:
            投影後的三個模態表徵
        """
        text_proj = self.project_modality(text_emb, self._text_projection)
        context_proj = self.project_modality(context_emb, self._context_projection)
        memory_proj = self.project_modality(memory_emb, self._memory_projection)

        return text_proj, context_proj, memory_proj


class CrossModalAdditiveAttention:
    """
    跨模態加法注意力機制

    論文關鍵：使用加法注意力而非點積注意力
    優勢：更好地捕捉模態間的非線性交互
    """

    def __init__(self, hidden_dim: int = 768):
        self.hidden_dim = hidden_dim

        # 注意力權重參數
        self._W_q = self._init_weights()  # Query 權重
        self._W_k = self._init_weights()  # Key 權重
        self._v = np.random.randn(hidden_dim)  # 注意力向量

    def _init_weights(self) -> np.ndarray:
        limit = math.sqrt(6.0 / (self.hidden_dim * 2))
        return np.random.uniform(-limit, limit, (self.hidden_dim, self.hidden_dim))

    def compute_attention(self,
                         query: np.ndarray,
                         keys: List[np.ndarray],
                         values: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算加法注意力

        Args:
            query: 查詢向量（通常是語言模態）
            keys: 鍵向量列表（其他模態）
            values: 值向量列表

        Returns:
            (加權融合結果, 注意力權重)
        """
        # Query 投影
        q_proj = np.dot(query, self._W_q)

        # 計算每個 key 的注意力分數
        attention_scores = []
        for key in keys:
            k_proj = np.dot(key, self._W_k)
            # 加法注意力：tanh(W_q * q + W_k * k)
            combined = np.tanh(q_proj + k_proj)
            score = np.dot(combined, self._v)
            attention_scores.append(score)

        # Softmax 歸一化
        attention_weights = self._softmax(np.array(attention_scores))

        # 加權融合
        weighted_sum = np.zeros_like(query)
        for i, (weight, value) in enumerate(zip(attention_weights, values)):
            weighted_sum += weight * value

        return weighted_sum, attention_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函數"""
        exp_x = np.exp(x - np.max(x))  # 數值穩定性
        return exp_x / np.sum(exp_x)


class GatingMechanism:
    """
    閘控機制

    論文核心：動態控制非語言資訊流入語言模型的量
    避免過度注入導致的資訊污染
    """

    def __init__(self, hidden_dim: int = 768):
        self.hidden_dim = hidden_dim

        # 閘控網絡參數
        self._W_gate = self._init_weights()
        self._b_gate = np.zeros(hidden_dim)

    def _init_weights(self) -> np.ndarray:
        limit = math.sqrt(6.0 / (self.hidden_dim * 2))
        return np.random.uniform(-limit, limit, (self.hidden_dim * 2, self.hidden_dim))

    def compute_gate(self,
                     language_rep: np.ndarray,
                     multimodal_rep: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        計算閘控值

        Args:
            language_rep: 語言模態表徵
            multimodal_rep: 融合的多模態表徵

        Returns:
            (閘控後的表徵, 閘控值)
        """
        # 拼接兩個表徵
        concat = np.concatenate([language_rep, multimodal_rep])

        # 計算閘控值 g = sigmoid(W * [h_l; h_m] + b)
        gate_logit = np.dot(concat, self._W_gate) + self._b_gate
        gate_value = self._sigmoid(np.mean(gate_logit))  # 平均作為總閘控值

        # 閘控融合：h_output = g * h_m + (1 - g) * h_l
        gated_output = gate_value * multimodal_rep + (1 - gate_value) * language_rep

        return gated_output, gate_value

    def _sigmoid(self, x: float) -> float:
        """Sigmoid 函數"""
        return 1.0 / (1.0 + math.exp(-x))


class MGTEmotionRater:
    """
    MGT 情緒評分員

    整合 Parallel Multimodal Flow, Cross-modal Attention, Gating Mechanism
    為 AI 居民提供情緒評估和互動建議
    """

    def __init__(self,
                 agent_name: str,
                 hidden_dim: int = 768,
                 enable_llm_enhancement: bool = True,
                 model_weights_path: Optional[str] = None,
                 enable_embedding_cache: bool = True,
                 cache_size: int = 1000):
        """
        初始化情緒評分員

        Args:
            agent_name: AI 居民名稱
            hidden_dim: 隱藏層維度
            enable_llm_enhancement: 是否使用 LLM 增強建議生成
            model_weights_path: 訓練好的模型權重路徑 (.npz)
            enable_embedding_cache: 是否啟用嵌入緩存
            cache_size: 緩存大小
        """
        self.agent_name = agent_name
        self.hidden_dim = hidden_dim
        self.enable_llm_enhancement = enable_llm_enhancement
        self.model_weights_path = model_weights_path
        self.is_trained = False

        # 初始化三個核心組件
        self.parallel_flow = ParallelMultimodalFlow(hidden_dim)
        self.cross_attention = CrossModalAdditiveAttention(hidden_dim)
        self.gating = GatingMechanism(hidden_dim)

        # 情緒分類器權重（簡化版本，實際應用會更複雜）
        self.emotion_classifier = self._init_emotion_classifier()

        # 初始化嵌入緩存
        self.embedding_cache = None
        if enable_embedding_cache and HAS_CACHE:
            self.embedding_cache = EmbeddingCache(
                max_size=cache_size,
                embedding_dim=hidden_dim,
                enable_persistence=True,
                cache_file=f"emotion_system/cache/{agent_name}_embeddings.pkl"
            )

        # 評分歷史
        self.rating_history: List[EmotionRaterOutput] = []

        # 載入訓練好的權重（如果提供）
        if model_weights_path and os.path.exists(model_weights_path):
            self.load_trained_weights(model_weights_path)

    def _init_emotion_classifier(self) -> Dict[str, np.ndarray]:
        """初始化情緒分類器（每個情緒一個權重向量）"""
        classifier = {}
        for emotion in EMOTION_TYPES:
            # 隨機初始化分類權重
            classifier[emotion] = np.random.randn(self.hidden_dim) * 0.01
        return classifier

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        將文本轉換為嵌入向量

        優先級：
        1. 檢查緩存
        2. 生成新嵌入（簡化版本使用哈希）
        3. 緩存結果
        """
        # 1. 檢查緩存
        if self.embedding_cache is not None:
            cached = self.embedding_cache.get(text)
            if cached is not None:
                return cached

        # 2. 生成嵌入（簡化版本，實際應該調用 BERT/LLM API）
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        embedding = np.random.randn(self.hidden_dim)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # 歸一化

        # 3. 緩存結果
        if self.embedding_cache is not None:
            self.embedding_cache.put(text, embedding)

        return embedding

    def rate_emotion(self,
                     multimodal_input: MultimodalEmotionInput,
                     agent_completion_func: Optional[Any] = None) -> EmotionRaterOutput:
        """
        評估情緒並提供建議

        Args:
            multimodal_input: 多模態輸入
            agent_completion_func: Agent 的 LLM 調用函數（用於生成建議）

        Returns:
            情緒評分和建議
        """
        # 1. 準備嵌入向量
        text_emb = (multimodal_input.text_embeddings
                   if multimodal_input.text_embeddings
                   else self._text_to_embedding(multimodal_input.text_content))

        context_emb = (multimodal_input.context_embeddings
                      if multimodal_input.context_embeddings
                      else self._text_to_embedding(multimodal_input.context_description))

        # 記憶嵌入（平均多個記憶）
        if multimodal_input.memory_embeddings:
            memory_emb = np.mean(multimodal_input.memory_embeddings, axis=0)
        else:
            memory_texts = " ".join(multimodal_input.retrieved_memories[:3])
            memory_emb = self._text_to_embedding(memory_texts)

        # 轉換為 numpy 陣列
        text_emb = np.array(text_emb) if isinstance(text_emb, list) else text_emb
        context_emb = np.array(context_emb) if isinstance(context_emb, list) else context_emb
        memory_emb = np.array(memory_emb) if isinstance(memory_emb, list) else memory_emb

        # 2. 平行多模態流
        text_proj, context_proj, memory_proj = self.parallel_flow.forward(
            text_emb, context_emb, memory_emb
        )

        # 3. 跨模態注意力（語言為 Query）
        multimodal_fused, attention_weights = self.cross_attention.compute_attention(
            query=text_proj,
            keys=[context_proj, memory_proj],
            values=[context_proj, memory_proj]
        )

        # 4. 閘控機制
        final_rep, gate_value = self.gating.compute_gate(text_proj, multimodal_fused)

        # 5. 情緒分類
        emotion_scores = {}

        # 檢查是否使用訓練好的分類器
        if self.is_trained and isinstance(list(self.emotion_classifier.values())[0], dict):
            # 使用訓練好的兩層分類器
            for emotion, weights in self.emotion_classifier.items():
                # Layer 1: 768 -> 384 with ReLU
                hidden = np.dot(final_rep, weights['layer1_W'].T) + weights['layer1_b']
                hidden = np.maximum(0, hidden)  # ReLU

                # Layer 2: 384 -> 1 (該情緒的分數)
                score = np.dot(hidden, weights['layer2_w']) + weights['layer2_b']
                emotion_scores[emotion] = score
        else:
            # 使用簡化的單層分類器（未訓練）
            for emotion, weight_vector in self.emotion_classifier.items():
                score = np.dot(final_rep, weight_vector)
                emotion_scores[emotion] = score

        # Softmax 歸一化得到機率分佈
        emotion_probs = self._softmax_dict(emotion_scores)

        # 選擇最高機率的情緒
        predicted_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[predicted_emotion]

        # 計算情緒強度（基於機率和當前情緒狀態）
        intensity = self._compute_intensity(
            emotion_probs,
            multimodal_input.current_emotion
        )

        # 6. 生成互動建議
        suggestions = self._generate_interaction_suggestions(
            predicted_emotion=predicted_emotion,
            intensity=intensity,
            multimodal_input=multimodal_input,
            agent_completion_func=agent_completion_func
        )

        # 7. 生成推理說明
        reasoning = self._generate_reasoning(
            predicted_emotion,
            intensity,
            attention_weights,
            gate_value,
            multimodal_input
        )

        # 構建輸出
        output = EmotionRaterOutput(
            predicted_emotion=predicted_emotion,
            emotion_intensity=intensity,
            confidence=confidence,
            emotion_distribution=emotion_probs,
            interaction_suggestions=suggestions,
            reasoning=reasoning,
            attention_weights={
                "context": float(attention_weights[0]),
                "memory": float(attention_weights[1])
            },
            gating_values={
                "multimodal_gate": float(gate_value),
                "language_preserve": float(1 - gate_value)
            }
        )

        # 記錄歷史
        self.rating_history.append(output)

        return output

    def _softmax_dict(self, scores: Dict[str, float]) -> Dict[str, float]:
        """對字典值應用 softmax"""
        values = np.array(list(scores.values()))
        exp_values = np.exp(values - np.max(values))
        softmax_values = exp_values / np.sum(exp_values)

        return {k: float(v) for k, v in zip(scores.keys(), softmax_values)}

    def _compute_intensity(self,
                          emotion_probs: Dict[str, float],
                          current_emotion: Optional[EmotionState]) -> float:
        """
        計算情緒強度

        考慮因素：
        1. 預測機率
        2. 與當前情緒的連續性
        """
        max_prob = max(emotion_probs.values())
        base_intensity = max_prob

        # 如果有當前情緒，考慮情緒轉換的平滑性
        if current_emotion:
            predicted = max(emotion_probs, key=emotion_probs.get)
            if predicted == current_emotion.emotion:
                # 同一情緒，強度變化較平滑
                base_intensity = 0.7 * base_intensity + 0.3 * current_emotion.intensity
            else:
                # 情緒轉換，強度較激烈
                base_intensity = min(base_intensity * 1.2, 1.0)

        return min(max(base_intensity, 0.0), 1.0)

    def _generate_interaction_suggestions(self,
                                         predicted_emotion: str,
                                         intensity: float,
                                         multimodal_input: MultimodalEmotionInput,
                                         agent_completion_func: Optional[Any] = None) -> List[str]:
        """
        生成互動建議

        根據預測的情緒和情境提供具體的對話建議
        """
        suggestions = []

        # 基於規則的建議
        rule_based = self._get_rule_based_suggestions(predicted_emotion, intensity)
        suggestions.extend(rule_based)

        # 如果啟用 LLM 增強且提供了 completion 函數
        if self.enable_llm_enhancement and agent_completion_func:
            llm_suggestions = self._get_llm_enhanced_suggestions(
                predicted_emotion,
                intensity,
                multimodal_input,
                agent_completion_func
            )
            if llm_suggestions:
                suggestions.extend(llm_suggestions)

        return suggestions[:5]  # 最多返回 5 個建議

    def _get_rule_based_suggestions(self, emotion: str, intensity: float) -> List[str]:
        """基於規則的建議生成（MELD 7種情緒）"""
        intensity_level = "高" if intensity > 0.7 else "中" if intensity > 0.4 else "低"

        suggestion_templates = {
            "anger": [
                f"直接但理性地表達你的不滿（強度：{intensity_level}）",
                "明確指出問題所在，但避免人身攻擊",
                "建議設定清楚的界限或提出改進方案"
            ],
            "disgust": [
                f"表達你的不認同或反感（強度：{intensity_level}）",
                "可以委婉地表示拒絕或迴避",
                "說明你不能接受的原因"
            ],
            "fear": [
                f"表達你的擔憂並尋求確認（強度：{intensity_level}）",
                "可以請求對方提供保證或支持",
                "詢問是否有解決方案來緩解擔憂"
            ],
            "joy": [
                f"以愉悅的語氣分享你的喜悅（強度：{intensity_level}）",
                "主動表達積極情緒，增進互動親密度",
                "可以提議一起做些有趣的事情"
            ],
            "neutral": [
                "保持客觀平衡的態度",
                "專注於事實和資訊交流",
                "適時觀察對方的情緒反應"
            ],
            "sadness": [
                f"用溫和的方式表達你的感受（強度：{intensity_level}）",
                "可以尋求對方的理解和支持",
                "避免過度隱藏情緒，適當表達有助於獲得同理"
            ],
            "surprise": [
                f"表達你的驚訝或意外（強度：{intensity_level}）",
                "積極尋求更多資訊來理解情況",
                "分享你的反應並確認事實"
            ]
        }

        return suggestion_templates.get(emotion, suggestion_templates["neutral"])

    def _get_llm_enhanced_suggestions(self,
                                     emotion: str,
                                     intensity: float,
                                     multimodal_input: MultimodalEmotionInput,
                                     completion_func: Any) -> List[str]:
        """使用 LLM 生成個性化建議（可選）"""
        # 這裡可以調用 LLM 生成更個性化的建議
        # 為了避免過多 LLM 調用，可以設置條件觸發
        # 此處返回空列表，實際整合時可以啟用
        return []

    def _generate_reasoning(self,
                           emotion: str,
                           intensity: float,
                           attention_weights: np.ndarray,
                           gate_value: float,
                           multimodal_input: MultimodalEmotionInput) -> str:
        """生成推理說明"""
        emotion_name = _get_emotion_name(emotion)

        # 分析主要影響因素
        context_weight = attention_weights[0]
        memory_weight = attention_weights[1]

        main_factor = "當前情境" if context_weight > memory_weight else "過往記憶"

        reasoning_parts = [
            f"預測情緒為 {emotion_name}（強度 {intensity:.2f}）",
            f"主要影響因素：{main_factor}（注意力權重 {max(context_weight, memory_weight):.2f}）",
            f"多模態資訊融合度：{gate_value:.2f}"
        ]

        # 添加情境說明
        if multimodal_input.context_description:
            reasoning_parts.append(f"考慮情境：{multimodal_input.context_description[:50]}")

        # 添加記憶影響
        if multimodal_input.retrieved_memories:
            reasoning_parts.append(f"參考了 {len(multimodal_input.retrieved_memories)} 條相關記憶")

        return "；".join(reasoning_parts)

    def load_trained_weights(self, weights_path: str) -> bool:
        """
        載入訓練好的模型權重

        Args:
            weights_path: Numpy 權重檔案路徑 (.npz)

        Returns:
            是否載入成功
        """
        try:
            print(f"[{self.agent_name}] 載入訓練權重: {weights_path}")

            # 載入 numpy 權重
            weights = np.load(weights_path, allow_pickle=True)

            # 1. 更新 Parallel Multimodal Flow 權重
            if 'text_projection_weight' in weights:
                self.parallel_flow._text_projection = weights['text_projection_weight']
                self.parallel_flow._text_projection_bias = weights['text_projection_bias']
                print(f"  ✓ Text Projection: {weights['text_projection_weight'].shape}")

            if 'context_projection_weight' in weights:
                self.parallel_flow._context_projection = weights['context_projection_weight']
                self.parallel_flow._context_projection_bias = weights['context_projection_bias']
                print(f"  ✓ Context Projection: {weights['context_projection_weight'].shape}")

            # 2. 更新 Cross-modal Attention 權重
            if 'attention_query_weight' in weights:
                self.cross_attention._W_q = weights['attention_query_weight']
                self.cross_attention._b_q = weights['attention_query_bias']
                print(f"  ✓ Attention Query: {weights['attention_query_weight'].shape}")

            if 'attention_key_weight' in weights:
                self.cross_attention._W_k = weights['attention_key_weight']
                self.cross_attention._b_k = weights['attention_key_bias']
                print(f"  ✓ Attention Key: {weights['attention_key_weight'].shape}")

            if 'attention_value_weight' in weights:
                self.cross_attention._W_v = weights['attention_value_weight']
                self.cross_attention._b_v = weights['attention_value_bias']
                print(f"  ✓ Attention Value: {weights['attention_value_weight'].shape}")

            if 'attention_vector' in weights:
                self.cross_attention._v = weights['attention_vector']
                print(f"  ✓ Attention Vector: {weights['attention_vector'].shape}")

            # 3. 更新 Gating Mechanism 權重
            if 'gate_weight' in weights:
                self.gating._W_gate = weights['gate_weight']
                self.gating._b_gate = weights['gate_bias']
                print(f"  ✓ Gate: {weights['gate_weight'].shape}")

            # 4. 更新 Emotion Classifier 權重
            if 'classifier_layer1_weight' in weights and 'classifier_layer2_weight' in weights:
                # 重新構建分類器以使用訓練好的權重
                W1 = weights['classifier_layer1_weight']  # (384, 768)
                b1 = weights['classifier_layer1_bias']    # (384,)
                W2 = weights['classifier_layer2_weight']  # (7, 384)
                b2 = weights['classifier_layer2_bias']    # (7,)

                # 為每種情緒創建權重（使用 W2 的對應行）
                self.emotion_classifier = {}
                for i, emotion in enumerate(EMOTION_TYPES):
                    # 組合兩層權重為單一權重向量（簡化）
                    # 實際推理時需要完整的兩層前向傳播
                    self.emotion_classifier[emotion] = {
                        'layer1_W': W1,  # 共享第一層
                        'layer1_b': b1,
                        'layer2_w': W2[i],  # 該情緒的輸出權重
                        'layer2_b': b2[i]
                    }

                print(f"  ✓ Classifier: {W1.shape} -> {W2.shape}")

            # 5. 載入元數據
            if 'metadata' in weights:
                metadata = weights['metadata'].item()
                val_acc = metadata.get('val_acc', 0.0)
                epoch = metadata.get('epoch', 0)
                print(f"  ✓ 模型元數據: epoch={epoch}, val_acc={val_acc:.2f}%")

            self.is_trained = True
            self.model_weights_path = weights_path
            print(f"[{self.agent_name}] ✅ 訓練權重載入完成！\n")

            return True

        except Exception as e:
            print(f"[{self.agent_name}] ❌ 載入權重失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_cache_stats(self) -> Optional[Dict]:
        """獲取緩存統計資訊"""
        if self.embedding_cache is not None:
            return self.embedding_cache.get_stats()
        return None

    def get_rating_summary(self, last_n: int = 5) -> str:
        """獲取最近的評分摘要"""
        if not self.rating_history:
            return "尚無評分記錄"

        recent = self.rating_history[-last_n:]
        lines = [f"\n[{self.agent_name}] MGT 情緒評分員 - 最近 {len(recent)} 次評分：\n"]

        for i, rating in enumerate(recent, 1):
            emotion_name = _get_emotion_name(rating.predicted_emotion)
            lines.append(
                f"{i}. {emotion_name} (強度: {rating.emotion_intensity:.2f}, "
                f"信心: {rating.confidence:.2f})"
            )
            if rating.interaction_suggestions:
                lines.append(f"   建議: {rating.interaction_suggestions[0]}")

        return "\n".join(lines)


def create_mgt_rater_for_agent(agent_name: str,
                               config: Optional[Dict] = None) -> MGTEmotionRater:
    """
    為 Agent 創建 MGT 情緒評分員的工廠函數

    Args:
        agent_name: AI 居民名稱
        config: 可選配置
            - hidden_dim: 隱藏層維度
            - enable_llm_enhancement: 是否啟用 LLM 增強

    Returns:
        MGTEmotionRater 實例
    """
    config = config or {}

    return MGTEmotionRater(
        agent_name=agent_name,
        hidden_dim=config.get("hidden_dim", 768),
        enable_llm_enhancement=config.get("enable_llm_enhancement", True)
    )
