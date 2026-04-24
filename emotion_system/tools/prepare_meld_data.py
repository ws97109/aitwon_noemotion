"""
MELD 數據集準備腳本

MELD (Multimodal EmotionLines Dataset) 包含：
- 7種情緒: anger, disgust, fear, joy, neutral, sadness, surprise
- 多模態數據: 文本、音頻、視頻
- 對話上下文

數據集來源: https://github.com/declare-lab/MELD
"""

import os
import sys
import pandas as pd
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

# MELD 數據集 URL
MELD_URLS = {
    "train": "https://github.com/declare-lab/MELD/raw/master/data/MELD/train_sent_emo.csv",
    "dev": "https://github.com/declare-lab/MELD/raw/master/data/MELD/dev_sent_emo.csv",
    "test": "https://github.com/declare-lab/MELD/raw/master/data/MELD/test_sent_emo.csv"
}

# MELD 情緒類別（7種）
MELD_EMOTIONS = [
    "anger",      # 憤怒
    "disgust",    # 厭惡
    "fear",       # 恐懼
    "joy",        # 喜悅
    "neutral",    # 中性
    "sadness",    # 悲傷
    "surprise"    # 驚訝
]

# 情緒中文映射
EMOTION_ZH_MAPPING = {
    "anger": "憤怒",
    "disgust": "厭惡",
    "fear": "恐懼",
    "joy": "喜悅",
    "neutral": "中性",
    "sadness": "悲傷",
    "surprise": "驚訝"
}

# 情緒表情映射
EMOTION_EMOJI_MAPPING = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😊",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲"
}


class MELDDatasetPreparer:
    """MELD 數據集準備器"""

    def __init__(self, data_dir: str = "./data/meld"):
        """
        初始化數據集準備器

        Args:
            data_dir: 數據存放目錄
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.datasets = {}

    def download_dataset(self) -> bool:
        """
        下載 MELD 數據集

        Returns:
            是否下載成功
        """
        print("開始下載 MELD 數據集...")

        for split, url in MELD_URLS.items():
            filepath = self.data_dir / f"{split}_sent_emo.csv"

            if filepath.exists():
                print(f"✓ {split} 數據已存在: {filepath}")
                continue

            try:
                print(f"下載 {split} 數據...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                print(f"✓ {split} 數據下載完成")

            except Exception as e:
                print(f"✗ {split} 數據下載失敗: {e}")
                return False

        print("\n✅ MELD 數據集下載完成！")
        return True

    def load_dataset(self, split: str = "train") -> pd.DataFrame:
        """
        載入數據集

        Args:
            split: 數據分割 (train/dev/test)

        Returns:
            DataFrame
        """
        filepath = self.data_dir / f"{split}_sent_emo.csv"

        if not filepath.exists():
            print(f"數據文件不存在: {filepath}")
            print("請先運行 download_dataset()")
            return None

        df = pd.read_csv(filepath)
        self.datasets[split] = df

        print(f"\n載入 {split} 數據集:")
        print(f"  樣本數: {len(df)}")
        print(f"  欄位: {list(df.columns)}")

        # 顯示情緒分佈
        if 'Emotion' in df.columns:
            print(f"\n  情緒分佈:")
            emotion_counts = df['Emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                print(f"    {emotion}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        準備訓練數據

        Returns:
            (train_df, dev_df, test_df)
        """
        print("\n準備訓練數據...")

        train_df = self.load_dataset("train")
        dev_df = self.load_dataset("dev")
        test_df = self.load_dataset("test")

        if train_df is None or dev_df is None or test_df is None:
            raise ValueError("數據載入失敗，請先下載數據集")

        # 數據清理
        for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            print(f"\n清理 {name} 數據...")

            # 移除缺失值
            original_len = len(df)
            df.dropna(subset=['Utterance', 'Emotion'], inplace=True)
            removed = original_len - len(df)
            if removed > 0:
                print(f"  移除 {removed} 個缺失樣本")

            # 確保情緒標籤有效
            valid_emotions = set(MELD_EMOTIONS)
            invalid_mask = ~df['Emotion'].isin(valid_emotions)
            if invalid_mask.sum() > 0:
                print(f"  移除 {invalid_mask.sum()} 個無效情緒標籤")
                df = df[~invalid_mask]

        print("\n✅ 訓練數據準備完成！")
        return train_df, dev_df, test_df

    def get_context_window(self, df: pd.DataFrame, idx: int, window_size: int = 3) -> List[str]:
        """
        獲取對話上下文窗口

        Args:
            df: DataFrame
            idx: 當前樣本索引
            window_size: 上下文窗口大小

        Returns:
            上下文語句列表
        """
        dialogue_id = df.iloc[idx]['Dialogue_ID']
        utterance_id = df.iloc[idx]['Utterance_ID']

        # 獲取同一對話中的前幾句
        context = []
        for i in range(max(0, idx - window_size), idx):
            if df.iloc[i]['Dialogue_ID'] == dialogue_id:
                context.append(df.iloc[i]['Utterance'])

        return context

    def export_preprocessed_data(self, output_dir: str = "./data/meld_processed"):
        """
        導出預處理後的數據

        Args:
            output_dir: 輸出目錄
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_df, dev_df, test_df = self.prepare_training_data()

        # 保存處理後的數據
        for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            filepath = output_path / f"{name}_processed.csv"
            df.to_csv(filepath, index=False)
            print(f"✓ 保存 {name} 數據: {filepath}")

        # 保存情緒映射
        emotion_info = {
            "emotions": MELD_EMOTIONS,
            "zh_mapping": EMOTION_ZH_MAPPING,
            "emoji_mapping": EMOTION_EMOJI_MAPPING
        }

        import json
        with open(output_path / "emotion_info.json", 'w', encoding='utf-8') as f:
            json.dump(emotion_info, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 預處理數據已保存到: {output_path}")

    def print_statistics(self):
        """打印數據集統計信息"""
        print("\n" + "="*70)
        print("MELD 數據集統計")
        print("="*70)

        for split in ["train", "dev", "test"]:
            if split in self.datasets:
                df = self.datasets[split]
                print(f"\n{split.upper()}:")
                print(f"  總樣本數: {len(df)}")
                print(f"  對話數: {df['Dialogue_ID'].nunique()}")
                print(f"  說話人數: {df['Speaker'].nunique() if 'Speaker' in df.columns else 'N/A'}")

                print(f"\n  情緒分佈:")
                for emotion in MELD_EMOTIONS:
                    count = (df['Emotion'] == emotion).sum()
                    percentage = count / len(df) * 100
                    emoji = EMOTION_EMOJI_MAPPING[emotion]
                    zh = EMOTION_ZH_MAPPING[emotion]
                    print(f"    {emoji} {emotion:10s} ({zh:4s}): {count:5d} ({percentage:5.2f}%)")

        print("\n" + "="*70)


def main():
    """主函數"""
    print("MELD 數據集準備工具\n")

    preparer = MELDDatasetPreparer()

    # 下載數據
    if not preparer.download_dataset():
        print("數據下載失敗")
        return

    # 準備訓練數據
    preparer.prepare_training_data()

    # 導出預處理數據
    preparer.export_preprocessed_data()

    # 打印統計信息
    preparer.print_statistics()

    print("\n✅ MELD 數據集準備完成！")
    print("\n下一步:")
    print("  1. 查看數據: data/meld_processed/")
    print("  2. 運行訓練: python emotion_system/train_mgt_model.py")


if __name__ == "__main__":
    main()
