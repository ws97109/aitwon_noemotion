"""
核心情緒系統模組

提供情緒狀態管理、情緒分析和情緒配置功能
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class EmotionState:
    """情緒狀態數據類"""

    emotion: str = "neutral"  # 情緒類型
    intensity: float = 0.5    # 強度 (0.0-1.0)
    reasoning: str = ""       # 產生此情緒的原因
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 情緒持續時間（分鐘），用於情緒衰減
    duration: int = 30

    def __post_init__(self):
        """驗證數據有效性"""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"Intensity must be between 0.0 and 1.0, got {self.intensity}")

        if self.emotion not in EMOTION_TYPES:
            raise ValueError(f"Unknown emotion type: {self.emotion}")

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "duration": self.duration
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionState':
        """從字典創建"""
        return cls(**data)

    def get_display_name(self) -> str:
        """獲取顯示名稱（中文+表情符號）"""
        return EMOTION_DISPLAY.get(self.emotion, self.emotion)

    def decay(self, minutes_passed: int) -> 'EmotionState':
        """
        情緒衰減

        隨時間流逝，情緒強度會逐漸降低，最終回歸中性
        """
        if self.emotion == "neutral":
            return self

        decay_rate = 0.02  # 每分鐘衰減 2%
        new_intensity = max(0.1, self.intensity - (decay_rate * minutes_passed))

        # 如果強度低於閾值，回歸中性
        if new_intensity < 0.3:
            return EmotionState(
                emotion="neutral",
                intensity=0.5,
                reasoning=f"從 {self.emotion} 衰減回歸中性",
                timestamp=datetime.now().isoformat()
            )

        return EmotionState(
            emotion=self.emotion,
            intensity=new_intensity,
            reasoning=self.reasoning,
            timestamp=datetime.now().isoformat(),
            duration=self.duration
        )


# 定義 7 種情緒類型（基於 MELD 數據集）
EMOTION_TYPES = [
    "anger",      # 憤怒
    "disgust",    # 厭惡
    "fear",       # 恐懼
    "joy",        # 喜悅
    "neutral",    # 中性
    "sadness",    # 悲傷
    "surprise"    # 驚訝
]

# 情緒顯示名稱（中文+表情符號）
EMOTION_DISPLAY = {
    "anger": "😠 憤怒",
    "disgust": "🤢 厭惡",
    "fear": "😨 恐懼",
    "joy": "😊 喜悅",
    "neutral": "😐 中性",
    "sadness": "😢 悲傷",
    "surprise": "😲 驚訝"
}

# 情緒回應風格配置（基於 MELD 的 7 種情緒）
EMOTION_STYLES = {
    "anger": {
        "tone": "使用強烈、直接的語氣，表達不滿或憤怒",
        "action_tendency": "對抗、表達異議、設定界限",
        "energy_level": "high",
        "behavior": "直接表達不滿，可能提高音調或使用強烈措辭"
    },
    "disgust": {
        "tone": "表達厭惡、排斥的態度",
        "action_tendency": "迴避、拒絕、表達反感",
        "energy_level": "medium",
        "behavior": "顯示厭惡感，可能使用否定或排斥性語言"
    },
    "fear": {
        "tone": "使用謹慎、不安的語氣，表達恐懼或擔憂",
        "action_tendency": "尋求安全、避免威脅、請求幫助",
        "energy_level": "high",
        "behavior": "表現出不安或擔心，尋求確認或保護"
    },
    "joy": {
        "tone": "使用積極、熱情的語氣，適當表達喜悅",
        "action_tendency": "主動社交，分享好消息",
        "energy_level": "high",
        "behavior": "展現積極情緒，主動互動和分享"
    },
    "neutral": {
        "tone": "使用平衡、客觀的語氣",
        "action_tendency": "正常互動，保持中立",
        "energy_level": "medium",
        "behavior": "保持理性和客觀，不帶強烈情緒色彩"
    },
    "sadness": {
        "tone": "使用低落、沉穩的語氣，傾向於內省",
        "action_tendency": "減少社交，尋求獨處或支持",
        "energy_level": "low",
        "behavior": "表現出悲傷或失落，可能尋求安慰"
    },
    "surprise": {
        "tone": "表達驚訝、意外的反應",
        "action_tendency": "尋求資訊、確認情況",
        "energy_level": "medium-high",
        "behavior": "顯示意外或驚訝，積極尋求更多資訊"
    }
}


@dataclass
class EmotionConfig:
    """情緒配置（從 agent.json 讀取）"""

    # 情緒特質（0.0-1.0）
    emotional_stability: float = 0.5  # 情緒穩定性（低=情緒波動大）
    empathy_level: float = 0.5        # 同理心水平
    optimism: float = 0.5             # 樂觀程度
    anxiety_proneness: float = 0.3    # 焦慮傾向

    # 情緒反應閾值
    emotion_trigger_threshold: float = 0.5  # 觸發情緒變化的閾值

    # 記憶中的情緒權重
    emotion_memory_weight: float = 0.3  # 情緒在記憶檢索中的權重

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            "emotional_stability": self.emotional_stability,
            "empathy_level": self.empathy_level,
            "optimism": self.optimism,
            "anxiety_proneness": self.anxiety_proneness,
            "emotion_trigger_threshold": self.emotion_trigger_threshold,
            "emotion_memory_weight": self.emotion_memory_weight
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionConfig':
        """從字典創建（過濾掉不支持的字段）"""
        # 只提取 EmotionConfig 支持的字段
        valid_fields = {
            'emotional_stability',
            'empathy_level',
            'optimism',
            'anxiety_proneness',
            'emotion_trigger_threshold',
            'emotion_memory_weight'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class EmotionAnalyzer:
    """
    情緒分析器

    負責分析情境並決定 Agent 的情緒狀態
    """

    def __init__(self, agent_name: str, emotion_config: EmotionConfig):
        self.agent_name = agent_name
        self.config = emotion_config
        self.current_emotion = EmotionState()
        self.emotion_history: List[EmotionState] = []

    def analyze_event_emotion(
        self,
        event_description: str,
        poignancy: int,
        context: Optional[str] = None
    ) -> EmotionState:
        """
        分析事件應產生的情緒

        這是一個簡化版本，實際使用時會調用 LLM
        在整合到 Agent 時，會使用 Agent.completion() 方法

        Args:
            event_description: 事件描述
            poignancy: 事件重要性 (1-10)
            context: 額外上下文

        Returns:
            EmotionState: 新的情緒狀態
        """
        # 這裡返回佔位符，實際會在 Agent 中使用 LLM
        return EmotionState(
            emotion="neutral",
            intensity=0.5,
            reasoning="待 LLM 分析",
            timestamp=datetime.now().isoformat()
        )

    def update_emotion(self, new_emotion: EmotionState):
        """
        更新當前情緒狀態

        考慮情緒穩定性，平滑過渡
        """
        # 記錄歷史
        if self.current_emotion.emotion != "neutral":
            self.emotion_history.append(self.current_emotion)

        # 根據情緒穩定性調整強度
        stability_factor = self.config.emotional_stability
        adjusted_intensity = new_emotion.intensity * (1 - stability_factor * 0.3)

        self.current_emotion = EmotionState(
            emotion=new_emotion.emotion,
            intensity=max(0.1, min(1.0, adjusted_intensity)),
            reasoning=new_emotion.reasoning,
            timestamp=datetime.now().isoformat(),
            duration=new_emotion.duration
        )

    def get_emotion_bias_for_retrieval(self) -> Dict[str, float]:
        """
        獲取情緒偏差用於記憶檢索

        Returns:
            Dict[str, float]: 情緒類型到權重倍數的映射
        """
        current = self.current_emotion.emotion

        bias_rules = {
            "anxiety": {
                "anxiety": 1.5,
                "caution": 1.3,
                "frustration": 1.2
            },
            "joy": {
                "joy": 1.5,
                "excitement": 1.4,
                "confidence": 1.3,
                "gratitude": 1.2
            },
            "empathy": {
                "empathy": 1.8,
                "sadness": 1.5,
                "gratitude": 1.3
            },
            "confidence": {
                "confidence": 1.5,
                "excitement": 1.3,
                "joy": 1.2
            },
            "sadness": {
                "sadness": 1.5,
                "empathy": 1.3,
                "confusion": 1.2
            },
            "curiosity": {
                "curiosity": 1.5,
                "excitement": 1.3,
                "confusion": 1.2
            }
        }

        return bias_rules.get(current, {})

    def get_emotion_style(self) -> Dict:
        """獲取當前情緒的回應風格"""
        return EMOTION_STYLES.get(
            self.current_emotion.emotion,
            EMOTION_STYLES["neutral"]
        )

    def check_emotion_decay(self, current_time: datetime) -> bool:
        """
        檢查情緒是否需要衰減

        Returns:
            bool: 是否發生了衰減
        """
        emotion_time = datetime.fromisoformat(self.current_emotion.timestamp)
        minutes_passed = (current_time - emotion_time).total_seconds() / 60

        if minutes_passed > 5:  # 每5分鐘檢查一次
            old_emotion = self.current_emotion.emotion
            self.current_emotion = self.current_emotion.decay(int(minutes_passed))
            return old_emotion != self.current_emotion.emotion

        return False

    def get_recent_emotion_summary(self, n: int = 5) -> str:
        """
        獲取最近的情緒歷史摘要

        Args:
            n: 獲取最近 n 個情緒

        Returns:
            str: 情緒歷史摘要
        """
        if not self.emotion_history:
            return "無情緒歷史"

        recent = self.emotion_history[-n:]
        summary_parts = []

        for emotion in recent:
            time_str = datetime.fromisoformat(emotion.timestamp).strftime("%H:%M")
            summary_parts.append(
                f"{time_str} {emotion.get_display_name()} "
                f"(強度:{emotion.intensity:.1f}) - {emotion.reasoning}"
            )

        return "\n".join(summary_parts)

    def to_dict(self) -> Dict:
        """序列化為字典（用於保存）"""
        return {
            "agent_name": self.agent_name,
            "config": self.config.to_dict(),
            "current_emotion": self.current_emotion.to_dict(),
            "emotion_history": [e.to_dict() for e in self.emotion_history[-20:]]  # 只保留最近20個
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionAnalyzer':
        """從字典反序列化"""
        config = EmotionConfig.from_dict(data["config"])
        analyzer = cls(data["agent_name"], config)
        analyzer.current_emotion = EmotionState.from_dict(data["current_emotion"])
        analyzer.emotion_history = [
            EmotionState.from_dict(e) for e in data.get("emotion_history", [])
        ]
        return analyzer


# 工具函數

def calculate_emotion_compatibility(emotion1: str, emotion2: str) -> float:
    """
    計算兩個情緒的相容性（用於社交互動）

    Returns:
        float: 0.0-1.0，值越高表示越相容
    """
    # 相容性矩陣
    compatibility_matrix = {
        ("joy", "joy"): 1.0,
        ("joy", "excitement"): 0.9,
        ("joy", "gratitude"): 0.9,
        ("joy", "confidence"): 0.8,
        ("joy", "sadness"): 0.2,
        ("joy", "anxiety"): 0.3,

        ("empathy", "sadness"): 0.9,
        ("empathy", "anxiety"): 0.8,
        ("empathy", "frustration"): 0.8,

        ("confidence", "anxiety"): 0.4,
        ("confidence", "confusion"): 0.5,

        ("curiosity", "curiosity"): 0.9,
        ("curiosity", "excitement"): 0.8,

        ("sadness", "sadness"): 0.7,
        ("sadness", "empathy"): 0.9,

        ("anxiety", "anxiety"): 0.6,
        ("anxiety", "caution"): 0.8,
    }

    # 嘗試直接查找
    key = (emotion1, emotion2)
    if key in compatibility_matrix:
        return compatibility_matrix[key]

    # 嘗試反向查找
    reverse_key = (emotion2, emotion1)
    if reverse_key in compatibility_matrix:
        return compatibility_matrix[reverse_key]

    # 默認中等相容性
    return 0.6


def get_emotion_from_poignancy_and_context(
    poignancy: int,
    event_type: str,
    is_positive: bool
) -> Tuple[str, float]:
    """
    根據重要性和上下文快速判斷情緒（不使用 LLM 的簡化版本）

    Args:
        poignancy: 重要性 (1-10)
        event_type: 事件類型 ("event", "chat", "thought")
        is_positive: 是否正面事件

    Returns:
        Tuple[str, float]: (情緒類型, 強度)
    """
    intensity = min(1.0, poignancy / 10.0)

    if event_type == "chat":
        if is_positive:
            if poignancy >= 7:
                return ("joy", intensity)
            else:
                return ("gratitude", intensity * 0.7)
        else:
            if poignancy >= 7:
                return ("sadness", intensity)
            else:
                return ("empathy", intensity * 0.7)

    elif event_type == "event":
        if is_positive:
            if poignancy >= 8:
                return ("excitement", intensity)
            elif poignancy >= 5:
                return ("joy", intensity)
            else:
                return ("neutral", 0.5)
        else:
            if poignancy >= 8:
                return ("anxiety", intensity)
            elif poignancy >= 5:
                return ("frustration", intensity)
            else:
                return ("caution", intensity * 0.6)

    else:  # thought
        if poignancy >= 7:
            return ("curiosity", intensity)
        else:
            return ("neutral", 0.5)
