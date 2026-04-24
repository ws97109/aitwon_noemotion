"""
自動下載 MOSI 原始文本轉錄（無需用戶交互）
"""

import sys
import os
from pathlib import Path

# 添加父目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.download_mosi_text import download_mosi_text_transcripts, extract_text_from_ids

def main():
    print("="*70)
    print("🚀 自動下載 MOSI 原始文本")
    print("="*70)

    # Step 1: 下載文本轉錄
    print("\n步驟 1: 下載文本轉錄...")
    download_mosi_text_transcripts(
        output_dir="./emotion_system/data/mosi/transcripts"
    )

    # Step 2: 檢查是否下載成功
    transcript_file = Path("./emotion_system/data/mosi/transcripts/transcripts.json")
    if transcript_file.exists():
        print("\n✅ 文本轉錄下載成功！")

        # Step 3: 提取文本並匹配 ID
        print("\n步驟 2: 提取文本並匹配數據...")
        extract_text_from_ids(
            raw_pkl_path="./emotion_system/data/mosi/mosi_raw.pkl",
            transcript_json=str(transcript_file),
            output_dir="./emotion_system/data/mosi/text_data"
        )

        print("\n" + "="*70)
        print("✅ 完成！原始文本已準備好用於 BERT 訓練")
        print("="*70)
        print("\n下一步：")
        print("python emotion_system/training/train_bert_finetuned.py \\")
        print("    --data_dir emotion_system/data/mosi/text_data \\")
        print("    --model_name roberta-base")

    else:
        print("\n⚠️  下載失敗，使用備用方案...")
        print("\n您可以：")
        print("1. 使用現有 GloVe 特徵訓練（準確率 60-70%）：")
        print("   python emotion_system/training/train_mosi_glove.py")
        print("\n2. 手動下載 MOSI 原始數據")

if __name__ == "__main__":
    main()
