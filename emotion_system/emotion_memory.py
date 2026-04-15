"""
情緒記憶系統

擴展現有的 Associate 記憶系統以支援情緒感知的檢索
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from .core import EmotionState, EmotionAnalyzer


@dataclass
class EmotionalConcept:
    """
    帶有情緒的概念（擴展原有 Concept）

    這個類用於包裝原有的 Concept，添加情緒資訊
    """

    # 原有屬性
    node_id: str
    description: str
    poignancy: int
    created: datetime
    last_access: datetime
    expire: Optional[datetime] = None

    # 新增情緒屬性
    created_emotion: Optional[str] = None        # 創建時的情緒
    mutable_emotion: Optional[str] = None        # 當前對該記憶的情緒（可變）
    emotion_intensity: float = 0.5               # 情緒強度
    success_score: float = 0.5                   # 成功評分（用於學習）
    retrieval_count: int = 0                     # 被檢索次數

    def to_metadata(self) -> Dict:
        """轉換為 metadata 字典（用於 LlamaIndex）"""
        metadata = {
            "node_id": self.node_id,
            "description": self.description,
            "poignancy": self.poignancy,
            "created": self.created.isoformat(),
            "last_access": self.last_access.isoformat(),
        }

        if self.expire:
            metadata["expire"] = self.expire.isoformat()

        # 添加情緒資訊
        if self.created_emotion:
            metadata["created_emotion"] = self.created_emotion
        if self.mutable_emotion:
            metadata["mutable_emotion"] = self.mutable_emotion

        metadata["emotion_intensity"] = self.emotion_intensity
        metadata["success_score"] = self.success_score
        metadata["retrieval_count"] = self.retrieval_count

        return metadata

    @classmethod
    def from_concept_and_emotion(
        cls,
        concept: Any,  # 原有的 Concept 物件
        emotion_state: Optional[EmotionState] = None
    ) -> 'EmotionalConcept':
        """
        從原有 Concept 和情緒狀態創建

        Args:
            concept: 原有的 Concept 物件
            emotion_state: 當前情緒狀態

        Returns:
            EmotionalConcept
        """
        created_emotion = None
        emotion_intensity = 0.5

        if emotion_state:
            created_emotion = emotion_state.emotion
            emotion_intensity = emotion_state.intensity

        return cls(
            node_id=concept.node_id,
            description=concept.description,
            poignancy=concept.poignancy,
            created=concept.created,
            last_access=concept.last_access,
            expire=concept.expire,
            created_emotion=created_emotion,
            mutable_emotion=created_emotion,  # 初始時相同
            emotion_intensity=emotion_intensity,
            success_score=0.5,
            retrieval_count=0
        )


class EmotionMemory:
    """
    情緒記憶管理器

    與現有的 Associate 系統協同工作，提供情緒感知的檢索功能
    """

    def __init__(self, emotion_analyzer: EmotionAnalyzer):
        self.emotion_analyzer = emotion_analyzer
        self.emotional_concepts: Dict[str, EmotionalConcept] = {}

    def add_emotional_concept(
        self,
        concept: Any,
        emotion_state: Optional[EmotionState] = None
    ) -> EmotionalConcept:
        """
        添加帶情緒的概念

        Args:
            concept: 原有的 Concept 物件
            emotion_state: 情緒狀態（如果為 None，使用當前情緒）

        Returns:
            EmotionalConcept
        """
        if emotion_state is None:
            emotion_state = self.emotion_analyzer.current_emotion

        emotional_concept = EmotionalConcept.from_concept_and_emotion(
            concept,
            emotion_state
        )

        self.emotional_concepts[concept.node_id] = emotional_concept
        return emotional_concept

    def get_emotional_concept(self, node_id: str) -> Optional[EmotionalConcept]:
        """獲取情緒概念"""
        return self.emotional_concepts.get(node_id)

    def apply_emotion_bias_to_retrieval(
        self,
        retrieved_nodes: List[Tuple[Any, float]],  # (node, score) pairs
        top_k: int = 5
    ) -> List[Tuple[Any, float]]:
        """
        對檢索結果應用情緒偏差

        Args:
            retrieved_nodes: 原始檢索結果 [(node, score), ...]
            top_k: 返回前 k 個結果

        Returns:
            調整後的檢索結果
        """
        # 獲取當前情緒的偏差規則
        emotion_bias = self.emotion_analyzer.get_emotion_bias_for_retrieval()

        if not emotion_bias:
            # 無偏差，返回原始結果
            return retrieved_nodes[:top_k]

        adjusted_results = []

        for node, base_score in retrieved_nodes:
            node_id = node.node_id if hasattr(node, 'node_id') else str(node)
            emotional_concept = self.emotional_concepts.get(node_id)

            if not emotional_concept:
                # 無情緒資訊，使用原始分數
                adjusted_results.append((node, base_score))
                continue

            # 計算調整係數
            multiplier = 1.0

            # 1. 情緒相容性加成
            node_emotion = emotional_concept.mutable_emotion
            if node_emotion and node_emotion in emotion_bias:
                multiplier *= emotion_bias[node_emotion]

            # 2. 成功評分加成
            success_score = emotional_concept.success_score
            if success_score > 0.7:
                multiplier *= 1.2
            elif success_score < 0.4:
                multiplier *= 0.8

            # 3. 情緒強度影響
            intensity_factor = emotional_concept.emotion_intensity
            multiplier *= (0.8 + 0.4 * intensity_factor)

            # 4. 重要性加成
            if emotional_concept.poignancy >= 8:
                multiplier *= 1.1

            adjusted_score = base_score * multiplier
            adjusted_results.append((node, adjusted_score))

        # 重新排序
        adjusted_results.sort(key=lambda x: x[1], reverse=True)
        return adjusted_results[:top_k]

    def update_concept_emotion(
        self,
        node_id: str,
        new_emotion: str,
        intensity: float
    ):
        """
        更新概念的可變情緒

        當對某個記憶的情緒發生變化時調用（例如，一件曾經高興的事現在感到後悔）
        """
        if node_id in self.emotional_concepts:
            concept = self.emotional_concepts[node_id]
            concept.mutable_emotion = new_emotion
            concept.emotion_intensity = intensity
            concept.last_access = datetime.now()

    def update_success_score(
        self,
        node_ids: List[str],
        feedback: str
    ):
        """
        根據反饋更新成功評分

        Args:
            node_ids: 使用過的記憶節點 ID 列表
            feedback: "positive", "negative", "neutral"
        """
        adjustments = {
            "positive": 0.05,
            "negative": -0.1,
            "neutral": 0.0
        }

        adjustment = adjustments.get(feedback, 0.0)

        for node_id in node_ids:
            if node_id in self.emotional_concepts:
                concept = self.emotional_concepts[node_id]
                old_score = concept.success_score
                concept.success_score = max(0.0, min(1.0, old_score + adjustment))
                concept.retrieval_count += 1
                concept.last_access = datetime.now()

    def get_emotion_weighted_importance(
        self,
        node_id: str,
        base_importance: float
    ) -> float:
        """
        獲取情緒加權的重要性分數

        結合原有的重要性（poignancy）和情緒因素

        Args:
            node_id: 節點 ID
            base_importance: 基礎重要性分數

        Returns:
            調整後的重要性分數
        """
        concept = self.emotional_concepts.get(node_id)
        if not concept:
            return base_importance

        # 情緒強度加成
        emotion_factor = 1.0 + (concept.emotion_intensity - 0.5) * 0.3

        # 成功評分加成
        success_factor = 0.8 + concept.success_score * 0.4

        # 檢索頻率衰減（避免過度依賴同一記憶）
        retrieval_penalty = 1.0 / (1.0 + concept.retrieval_count * 0.05)

        weighted_importance = (
            base_importance *
            emotion_factor *
            success_factor *
            retrieval_penalty
        )

        return weighted_importance

    def get_emotion_statistics(self) -> Dict[str, Any]:
        """
        獲取情緒記憶統計

        Returns:
            統計資訊字典
        """
        if not self.emotional_concepts:
            return {
                "total_concepts": 0,
                "emotion_distribution": {},
                "avg_success_score": 0.0,
                "avg_retrieval_count": 0.0
            }

        concepts = list(self.emotional_concepts.values())

        # 情緒分布
        emotion_counts: Dict[str, int] = {}
        for concept in concepts:
            if concept.mutable_emotion:
                emotion_counts[concept.mutable_emotion] = (
                    emotion_counts.get(concept.mutable_emotion, 0) + 1
                )

        # 計算平均值
        avg_success = sum(c.success_score for c in concepts) / len(concepts)
        avg_retrieval = sum(c.retrieval_count for c in concepts) / len(concepts)

        return {
            "total_concepts": len(concepts),
            "emotion_distribution": emotion_counts,
            "avg_success_score": round(avg_success, 3),
            "avg_retrieval_count": round(avg_retrieval, 2),
            "high_importance_count": sum(1 for c in concepts if c.poignancy >= 7)
        }

    def cleanup_low_value_memories(
        self,
        min_success_score: float = 0.2,
        max_age_days: int = 30
    ) -> List[str]:
        """
        清理低價值記憶

        Args:
            min_success_score: 最低成功評分閾值
            max_age_days: 最大年齡（天）

        Returns:
            被清理的節點 ID 列表
        """
        now = datetime.now()
        to_remove = []

        for node_id, concept in self.emotional_concepts.items():
            # 保護高重要性記憶
            if concept.poignancy >= 8:
                continue

            age_days = (now - concept.created).days

            # 清理條件：低成功率且很久沒被檢索
            if (concept.success_score < min_success_score and
                age_days > max_age_days and
                concept.retrieval_count < 3):
                to_remove.append(node_id)

        # 執行清理
        for node_id in to_remove:
            del self.emotional_concepts[node_id]

        return to_remove

    def to_dict(self) -> Dict:
        """序列化（用於保存）"""
        return {
            "emotional_concepts": {
                node_id: {
                    "node_id": concept.node_id,
                    "description": concept.description,
                    "poignancy": concept.poignancy,
                    "created": concept.created.isoformat(),
                    "last_access": concept.last_access.isoformat(),
                    "expire": concept.expire.isoformat() if concept.expire else None,
                    "created_emotion": concept.created_emotion,
                    "mutable_emotion": concept.mutable_emotion,
                    "emotion_intensity": concept.emotion_intensity,
                    "success_score": concept.success_score,
                    "retrieval_count": concept.retrieval_count
                }
                for node_id, concept in self.emotional_concepts.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict, emotion_analyzer: EmotionAnalyzer) -> 'EmotionMemory':
        """從字典反序列化"""
        memory = cls(emotion_analyzer)

        for node_id, concept_data in data.get("emotional_concepts", {}).items():
            concept = EmotionalConcept(
                node_id=concept_data["node_id"],
                description=concept_data["description"],
                poignancy=concept_data["poignancy"],
                created=datetime.fromisoformat(concept_data["created"]),
                last_access=datetime.fromisoformat(concept_data["last_access"]),
                expire=(datetime.fromisoformat(concept_data["expire"])
                       if concept_data.get("expire") else None),
                created_emotion=concept_data.get("created_emotion"),
                mutable_emotion=concept_data.get("mutable_emotion"),
                emotion_intensity=concept_data.get("emotion_intensity", 0.5),
                success_score=concept_data.get("success_score", 0.5),
                retrieval_count=concept_data.get("retrieval_count", 0)
            )
            memory.emotional_concepts[node_id] = concept

        return memory


def calculate_emotion_recency_score(
    concept: EmotionalConcept,
    current_time: datetime,
    decay_factor: float = 0.995
) -> float:
    """
    計算情緒感知的 recency 分數

    與原有的 recency 計算類似，但考慮情緒因素

    Args:
        concept: 情緒概念
        current_time: 當前時間
        decay_factor: 衰減因子

    Returns:
        recency 分數
    """
    # 計算時間差（小時）
    time_diff = (current_time - concept.last_access).total_seconds() / 3600

    # 基礎 recency 分數
    base_recency = decay_factor ** time_diff

    # 情緒強度加成（強烈情緒更容易被記起）
    emotion_boost = 1.0 + (concept.emotion_intensity - 0.5) * 0.2

    return base_recency * emotion_boost
