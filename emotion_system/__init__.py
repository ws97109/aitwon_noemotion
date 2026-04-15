"""
情緒感知AI系統

整合到 Generative Agents 專案的情緒系統模組
"""

from .core import EmotionState, EmotionAnalyzer, EmotionConfig
from .emotion_memory import EmotionMemory, EmotionalConcept
from .emotion_prompts import EmotionPrompts

__all__ = [
    'EmotionState',
    'EmotionAnalyzer',
    'EmotionConfig',
    'EmotionMemory',
    'EmotionalConcept',
    'EmotionPrompts'
]

__version__ = '1.0.0'
