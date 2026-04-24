"""
從 CMU-MOSI 官方獲取原始文本轉錄

使用您數據中的視頻 ID，下載對應的原始文本
"""

import os
import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# 嘗試導入 CMU-MultimodalSDK
try:
    from mmsdk import mmdatasdk
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("⚠️  CMU-MultimodalSDK 未安裝")
    print("請運行: pip install CMU-MultimodalSDK")


def download_mosi_text_transcripts(output_dir: str = "./emotion_system/data/mosi/transcripts"):
    """
    下載 MOSI 文本轉錄

    Args:
        output_dir: 輸出目錄
    """
    if not HAS_SDK:
        print("\n❌ 錯誤: 需要先安裝 CMU-MultimodalSDK")
        print("運行: pip install CMU-MultimodalSDK")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("📥 下載 MOSI 文本轉錄")
    print("="*70)

    # 下載 MOSI 數據集
    print("\n1️⃣ 下載 MOSI 數據集...")
    try:
        # 定義 MOSI 的各個部分
        DATASET = mmdatasdk.mmdataset(
            mmdatasdk.cmu_mosi.highlevel,
            destination='./mosi_sdk_data'
        )
        print("✅ MOSI 數據集下載成功")
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        print("\n嘗試備用方法...")

        # 備用：直接下載文本轉錄
        try:
            text_url = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/language/CMU_MOSI_TimestampedWords.csd"
            print(f"下載文本數據: {text_url}")
            text_data = mmdatasdk.mmdataset(text_url)
            print("✅ 文本數據下載成功")
        except Exception as e2:
            print(f"❌ 備用方法也失敗: {e2}")
            return

    # 提取文本
    print("\n2️⃣ 提取文本轉錄...")
    try:
        # 獲取文本特徵
        text_field = 'CMU_MOSI_TimestampedWords'  # 或 'CMU_MOSI_TimestampedWordVectors'

        if text_field in DATASET.computational_sequences:
            text_data = DATASET.computational_sequences[text_field]

            # 保存文本
            transcripts = {}
            for video_id in tqdm(text_data.data.keys(), desc="提取文本"):
                segments = text_data.data[video_id]['features']
                intervals = text_data.data[video_id]['intervals']

                # 提取每個片段的文本
                video_transcripts = []
                for i, (seg, interval) in enumerate(zip(segments, intervals)):
                    # seg 可能是詞向量或原始文本
                    if isinstance(seg, str):
                        text = seg
                    else:
                        # 如果是向量，需要從其他地方獲取文本
                        text = ""  # 待處理

                    video_transcripts.append({
                        'segment_id': i,
                        'interval': interval.tolist(),
                        'text': text
                    })

                transcripts[video_id] = video_transcripts

            # 保存
            output_file = output_path / "transcripts.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcripts, f, indent=2, ensure_ascii=False)

            print(f"✅ 文本轉錄已保存: {output_file}")
            print(f"   共 {len(transcripts)} 個視頻")

        else:
            print(f"❌ 找不到文本字段: {text_field}")
            print(f"可用字段: {list(DATASET.computational_sequences.keys())}")

    except Exception as e:
        print(f"❌ 提取失敗: {e}")
        import traceback
        traceback.print_exc()


def extract_text_from_ids(
    raw_pkl_path: str = "./emotion_system/data/mosi/mosi_raw.pkl",
    transcript_json: str = "./emotion_system/data/mosi/transcripts/transcripts.json",
    output_dir: str = "./emotion_system/data/mosi/text_data"
):
    """
    使用數據中的 ID 提取對應的原始文本

    Args:
        raw_pkl_path: mosi_raw.pkl 路徑
        transcript_json: 下載的文本轉錄 JSON
        output_dir: 輸出目錄
    """
    print("\n3️⃣ 匹配文本與數據...")

    # 載入原始數據
    with open(raw_pkl_path, 'rb') as f:
        raw_data = pickle.load(f)

    # 載入文本轉錄
    with open(transcript_json, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 處理每個 split
    for split_name in ['train', 'valid', 'test']:
        split_data = raw_data[split_name]
        ids = split_data['id']
        labels = split_data['labels']

        # 提取文本
        texts = []
        valid_indices = []

        for i, video_seg_id in enumerate(tqdm(ids, desc=f"處理 {split_name}")):
            # 解析 ID: '03bSnISJMiM[0]' -> video_id='03bSnISJMiM', seg_id=0
            try:
                video_id, seg_id_str = video_seg_id.split('[')
                seg_id = int(seg_id_str.rstrip(']'))

                # 查找對應的文本
                if video_id in transcripts:
                    video_trans = transcripts[video_id]
                    if seg_id < len(video_trans):
                        text = video_trans[seg_id]['text']
                        texts.append(text)
                        valid_indices.append(i)
                    else:
                        print(f"⚠️  片段不存在: {video_seg_id}")
                else:
                    print(f"⚠️  視頻不存在: {video_id}")

            except Exception as e:
                print(f"⚠️  解析失敗: {video_seg_id}, 錯誤: {e}")

        # 保存
        output_data = {
            'texts': texts,
            'labels': labels[valid_indices].tolist() if len(valid_indices) > 0 else [],
            'ids': [ids[i] for i in valid_indices]
        }

        output_file = output_path / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ {split_name}: {len(texts)} 個樣本已保存")

    print(f"\n✅ 所有文本數據已保存到: {output_dir}")


def main():
    """主流程"""
    print("="*70)
    print("🎯 MOSI 原始文本下載器")
    print("="*70)

    # Step 1: 下載文本轉錄
    print("\n【方法 1】從 CMU-MultimodalSDK 下載")
    print("這個方法需要安裝 SDK 並可能需要較長時間")

    choice = input("\n是否嘗試下載？(y/n): ").strip().lower()

    if choice == 'y':
        download_mosi_text_transcripts()

        # Step 2: 匹配文本
        if Path("./emotion_system/data/mosi/transcripts/transcripts.json").exists():
            extract_text_from_ids()
        else:
            print("\n⚠️  未找到下載的文本轉錄")
            print("請先成功下載文本數據")

    else:
        print("\n【方法 2】手動下載（推薦）")
        print("\n由於 CMU-MultimodalSDK 可能有版本問題，建議手動下載：")
        print("\n步驟：")
        print("1. 訪問: https://github.com/A2Zadeh/CMU-MultimodalSDK")
        print("2. 下載 MOSI 數據集的文本部分")
        print("3. 或者聯繫我，我可以提供備用下載連結")
        print("\n備註：")
        print("- MOSI 原始數據集包含視頻、音頻和文本轉錄")
        print("- 您只需要文本轉錄部分")
        print("- 文本轉錄是 .csd 格式，需要用 SDK 解析")


if __name__ == "__main__":
    main()
