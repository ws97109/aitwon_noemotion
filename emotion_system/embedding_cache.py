"""
嵌入向量緩存系統

使用 LRU (Least Recently Used) 策略緩存文本嵌入，
減少重複計算，提升 MGT 情緒評分效率
"""

import numpy as np
from typing import Optional, Dict, Tuple
from collections import OrderedDict
import hashlib
import pickle
import os
from pathlib import Path


class EmbeddingCache:
    """
    LRU 嵌入緩存

    特性：
    - 自動驅逐最少使用的條目
    - 持久化支持（可選）
    - 記憶體管理
    """

    def __init__(self,
                 max_size: int = 1000,
                 embedding_dim: int = 768,
                 enable_persistence: bool = False,
                 cache_file: Optional[str] = None):
        """
        初始化緩存

        Args:
            max_size: 最大緩存條目數
            embedding_dim: 嵌入向量維度
            enable_persistence: 是否啟用持久化
            cache_file: 緩存檔案路徑
        """
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.enable_persistence = enable_persistence
        self.cache_file = cache_file or "emotion_system/cache/embedding_cache.pkl"

        # LRU 緩存 (OrderedDict 自動維護插入順序)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # 統計信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }

        # 如果啟用持久化，嘗試載入
        if self.enable_persistence:
            self._load_from_disk()

    def _compute_key(self, text: str) -> str:
        """
        計算文本的緩存鍵（使用 hash）

        Args:
            text: 輸入文本

        Returns:
            緩存鍵
        """
        # 使用 SHA256 確保唯一性和穩定性
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        從緩存獲取嵌入

        Args:
            text: 輸入文本

        Returns:
            嵌入向量（如果存在），否則 None
        """
        key = self._compute_key(text)

        if key in self._cache:
            # 命中：移動到末尾（標記為最近使用）
            self._cache.move_to_end(key)
            self.stats['hits'] += 1
            return self._cache[key].copy()  # 返回副本避免修改
        else:
            # 未命中
            self.stats['misses'] += 1
            return None

    def put(self, text: str, embedding: np.ndarray):
        """
        添加嵌入到緩存

        Args:
            text: 輸入文本
            embedding: 嵌入向量
        """
        # 驗證維度
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"嵌入維度不匹配: {embedding.shape[0]} vs {self.embedding_dim}"
            )

        key = self._compute_key(text)

        # 如果已存在，更新並移到末尾
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = embedding.copy()
        else:
            # 新增條目
            if len(self._cache) >= self.max_size:
                # 驅逐最舊的條目（LRU）
                self._cache.popitem(last=False)
                self.stats['evictions'] += 1

            self._cache[key] = embedding.copy()
            self.stats['size'] = len(self._cache)

    def clear(self):
        """清空緩存"""
        self._cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }

    def get_stats(self) -> Dict:
        """
        獲取緩存統計信息

        Returns:
            統計字典
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_mb': self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """
        估算記憶體使用量（MB）

        Returns:
            記憶體使用量（MB）
        """
        # 每個 float32 向量：embedding_dim * 4 bytes
        # 加上字典開銷（約 200 bytes per entry）
        bytes_per_entry = self.embedding_dim * 4 + 200
        total_bytes = len(self._cache) * bytes_per_entry
        return total_bytes / (1024 * 1024)

    def _load_from_disk(self):
        """從磁碟載入緩存"""
        if not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self._cache = data['cache']
                self.stats = data['stats']
                print(f"✓ 從磁碟載入緩存: {len(self._cache)} 條目")
        except Exception as e:
            print(f"⚠️  載入緩存失敗: {e}")

    def save_to_disk(self):
        """保存緩存到磁碟"""
        if not self.enable_persistence:
            return

        # 確保目錄存在
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        try:
            data = {
                'cache': self._cache,
                'stats': self.stats
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ 緩存已保存到磁碟: {self.cache_file}")
        except Exception as e:
            print(f"⚠️  保存緩存失敗: {e}")

    def __del__(self):
        """析構時自動保存（如果啟用持久化）"""
        if self.enable_persistence and len(self._cache) > 0:
            self.save_to_disk()


class PrecomputedEmbeddings:
    """
    預計算嵌入管理器

    為常見短語預先計算並存儲嵌入，進一步提升效能
    """

    def __init__(self, precomputed_file: Optional[str] = None):
        """
        初始化預計算嵌入管理器

        Args:
            precomputed_file: 預計算嵌入檔案路徑 (.npz)
        """
        self.precomputed_file = precomputed_file or \
            "emotion_system/cache/precomputed_embeddings.npz"
        self.embeddings: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self):
        """載入預計算嵌入"""
        if not os.path.exists(self.precomputed_file):
            return

        try:
            data = np.load(self.precomputed_file, allow_pickle=True)
            self.embeddings = {key: data[key] for key in data.files}
            print(f"✓ 載入 {len(self.embeddings)} 個預計算嵌入")
        except Exception as e:
            print(f"⚠️  載入預計算嵌入失敗: {e}")

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        獲取預計算嵌入

        Args:
            text: 輸入文本

        Returns:
            嵌入向量（如果存在），否則 None
        """
        return self.embeddings.get(text)

    def add(self, text: str, embedding: np.ndarray):
        """
        添加預計算嵌入

        Args:
            text: 文本
            embedding: 嵌入向量
        """
        self.embeddings[text] = embedding

    def save(self):
        """保存預計算嵌入"""
        # 確保目錄存在
        cache_dir = os.path.dirname(self.precomputed_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        try:
            np.savez_compressed(self.precomputed_file, **self.embeddings)
            print(f"✓ 預計算嵌入已保存: {self.precomputed_file}")
        except Exception as e:
            print(f"⚠️  保存預計算嵌入失敗: {e}")

    @staticmethod
    def create_common_phrases() -> list:
        """
        創建常見情緒短語列表（中文和英文）

        Returns:
            短語列表
        """
        common_phrases = [
            # 喜悅相關
            "我很開心", "我感到快樂", "這真是太好了", "我很興奮",
            "I'm happy", "I feel joyful", "This is great", "I'm excited",

            # 悲傷相關
            "我很難過", "我感到悲傷", "這讓我失望", "我很沮喪",
            "I'm sad", "I feel down", "I'm disappointed", "I'm upset",

            # 憤怒相關
            "我很生氣", "這讓我憤怒", "太過分了", "我受夠了",
            "I'm angry", "This makes me mad", "I'm furious", "I've had enough",

            # 恐懼相關
            "我很擔心", "我感到害怕", "這讓我緊張", "我很焦慮",
            "I'm worried", "I'm scared", "This makes me nervous", "I'm anxious",

            # 驚訝相關
            "太驚訝了", "我沒想到", "真意外", "這真不可思議",
            "I'm surprised", "I didn't expect that", "That's unexpected", "Amazing",

            # 厭惡相關
            "這太噁心了", "我不喜歡這個", "這讓我反感", "太糟糕了",
            "This is disgusting", "I don't like this", "This is awful", "Gross",

            # 中性相關
            "我明白了", "我理解", "好的", "我知道了",
            "I understand", "I see", "Okay", "I got it",

            # 常見對話
            "你好嗎", "最近怎麼樣", "謝謝你", "不客氣",
            "How are you", "What's new", "Thank you", "You're welcome",

            # 情境描述
            "在家裡", "在辦公室", "和朋友聊天", "收到好消息",
            "at home", "at work", "chatting with friends", "received good news",
        ]

        return common_phrases


def precompute_embeddings_for_mgt(
    embedding_func,
    phrases: list,
    output_file: str = "emotion_system/cache/precomputed_embeddings.npz"
):
    """
    為 MGT 評分員預計算常見短語的嵌入

    Args:
        embedding_func: 嵌入函數（接受文本，返回 numpy 數組）
        phrases: 短語列表
        output_file: 輸出檔案路徑
    """
    print(f"預計算 {len(phrases)} 個短語的嵌入...")

    embeddings = {}
    for i, phrase in enumerate(phrases, 1):
        try:
            embedding = embedding_func(phrase)
            embeddings[phrase] = embedding
            if i % 10 == 0:
                print(f"  進度: {i}/{len(phrases)}")
        except Exception as e:
            print(f"  ⚠️  跳過 '{phrase}': {e}")

    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, **embeddings)

    print(f"✓ 預計算完成！保存到: {output_file}")
    print(f"  成功: {len(embeddings)}/{len(phrases)}")


if __name__ == "__main__":
    # 測試緩存
    print("測試嵌入緩存系統\n")

    # 創建緩存
    cache = EmbeddingCache(max_size=5, embedding_dim=768)

    # 測試添加和獲取
    for i in range(10):
        text = f"測試文本 {i}"
        embedding = np.random.randn(768).astype(np.float32)

        cache.put(text, embedding)
        retrieved = cache.get(text)

        if retrieved is not None:
            print(f"✓ 文本 {i}: 成功緩存和檢索")

    # 打印統計
    print("\n緩存統計:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if key == 'hit_rate':
            print(f"  {key}: {value:.2%}")
        elif key == 'memory_mb':
            print(f"  {key}: {value:.2f} MB")
        else:
            print(f"  {key}: {value}")
