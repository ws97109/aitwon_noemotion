"""
載入本地 MOSI 數據並轉換為比較實驗所需格式

你的數據格式:
- vision: (N, 50, 20)  時間序列，20維視覺特徵
- text: (N, 50, 300)   時間序列，300維文本特徵 (GloVe)
- audio: (N, 50, 5)    時間序列，5維音頻特徵
- labels: (N, 1, 1)    標籤 [-3, 3]

轉換策略:
- 對時間序列進行平均池化，得到固定維度向量
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple
import json


class LocalMOSIDataLoader:
    """本地 MOSI 數據加載器"""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: mosi_data.pkl 文件路徑
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"數據文件不存在: {data_path}")

        # 載入數據
        print(f"📦 載入數據: {data_path}")
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        print("✓ 數據載入成功")

        # 檢查數據結構
        self._check_data_structure()

        # 轉換數據
        self.processed_data = self._process_data()

    def _check_data_structure(self):
        """檢查數據結構"""
        required_splits = ['train', 'valid', 'test']
        for split in required_splits:
            if split not in self.raw_data:
                raise ValueError(f"缺少 {split} 數據集")

        # 檢查必要的鍵
        sample_split = self.raw_data['train']
        required_keys = ['text', 'audio', 'labels']  # vision 可能叫 vision 或 visual

        # 視覺特徵可能叫不同名字
        visual_key = None
        for key in ['vision', 'visual', 'video']:
            if key in sample_split:
                visual_key = key
                break

        if visual_key is None:
            raise ValueError("找不到視覺特徵鍵（vision/visual/video）")

        self.visual_key = visual_key

        print(f"✓ 數據結構檢查通過")
        print(f"   視覺特徵鍵: {visual_key}")

    def _process_data(self) -> Dict[str, Dict]:
        """
        處理數據：將時間序列轉換為固定維度向量

        策略: 對時間維度進行平均池化
        """
        print("\n🔄 處理數據...")
        processed = {}

        for split_name in ['train', 'valid', 'test']:
            raw_split = self.raw_data[split_name]

            # 提取各模態特徵（時間序列）
            text_seq = raw_split['text']        # (N, T, 300)
            visual_seq = raw_split[self.visual_key]  # (N, T, 20)
            audio_seq = raw_split['audio']      # (N, T, 5)
            labels = raw_split['labels']        # (N, 1, 1)

            # 平均池化：對時間維度求平均
            text_feat = text_seq.mean(axis=1)      # (N, 300)
            visual_feat = visual_seq.mean(axis=1)  # (N, 20)
            audio_feat = audio_seq.mean(axis=1)    # (N, 5)

            # 標籤
            if len(labels.shape) == 3:
                labels = labels.squeeze()  # (N, 1, 1) -> (N,)
            elif len(labels.shape) == 2:
                labels = labels.flatten()  # (N, 1) -> (N,)

            # 保存處理後的數據
            processed[split_name] = {
                'text': text_feat.astype(np.float32),
                'visual': visual_feat.astype(np.float32),
                'audio': audio_feat.astype(np.float32),
                'labels': labels.astype(np.float32)
            }

            print(f"   {split_name:6s}: {len(labels)} 樣本")
            print(f"      text:   {text_feat.shape}")
            print(f"      visual: {visual_feat.shape}")
            print(f"      audio:  {audio_feat.shape}")
            print(f"      labels: {labels.shape}, 範圍=[{labels.min():.2f}, {labels.max():.2f}]")

        print("✓ 數據處理完成\n")
        return processed

    def get_data(self) -> Tuple[Dict, Dict, Dict]:
        """
        獲取處理後的數據

        Returns:
            (train_data, val_data, test_data)
        """
        return (
            self.processed_data['train'],
            self.processed_data['valid'],
            self.processed_data['test']
        )

    def get_feature_dims(self) -> Dict[str, int]:
        """獲取特徵維度"""
        sample = self.processed_data['train']
        return {
            'text_dim': sample['text'].shape[1],
            'visual_dim': sample['visual'].shape[1],
            'audio_dim': sample['audio'].shape[1]
        }

    def save_processed_data(self, save_dir: str = None):
        """保存處理後的數據"""
        if save_dir is None:
            save_dir = self.data_path.parent / "processed"
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"💾 保存處理後的數據到: {save_dir}")

        # 保存每個分割
        for split_name, split_data in self.processed_data.items():
            save_path = save_dir / f"{split_name}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"   ✓ {split_name}.pkl")

        # 保存統計信息
        dims = self.get_feature_dims()
        stats = {
            'train_size': len(self.processed_data['train']['labels']),
            'valid_size': len(self.processed_data['valid']['labels']),
            'test_size': len(self.processed_data['test']['labels']),
            **dims
        }

        with open(save_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"   ✓ stats.json")

        print("✓ 數據保存完成")


class LocalMOSIDataset(Dataset):
    """本地 MOSI PyTorch 數據集"""

    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        modalities: list = ['text', 'visual', 'audio']
    ):
        """
        Args:
            data_dict: 包含 'text', 'visual', 'audio', 'labels' 的字典
            modalities: 要使用的模態列表
        """
        self.data_dict = data_dict
        self.modalities = modalities
        self.length = len(data_dict['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}

        # 添加各模態特徵
        for modality in self.modalities:
            if modality in self.data_dict:
                sample[modality] = torch.FloatTensor(self.data_dict[modality][idx])

        # 標籤
        sample['label'] = torch.FloatTensor([self.data_dict['labels'][idx]])

        return sample


# ========== 測試 ==========
if __name__ == "__main__":
    print("="*80)
    print("本地 MOSI 數據加載測試")
    print("="*80 + "\n")

    # 載入數據
    data_path = "/Users/lishengfeng/Desktop/generative_agents1118_add_emotion/emotion_system/data/mosi/mosi_data.pkl"

    loader = LocalMOSIDataLoader(data_path)

    # 獲取數據
    train_data, val_data, test_data = loader.get_data()

    # 獲取特徵維度
    dims = loader.get_feature_dims()
    print(f"📊 特徵維度:")
    for key, value in dims.items():
        print(f"   {key}: {value}")

    # 創建 PyTorch 數據集
    print(f"\n📦 創建 PyTorch 數據集...")
    train_dataset = LocalMOSIDataset(train_data, modalities=['text', 'visual', 'audio'])
    print(f"   訓練集: {len(train_dataset)} 樣本")

    # 測試加載一個樣本
    sample = train_dataset[0]
    print(f"\n🧪 測試樣本:")
    print(f"   text shape:   {sample['text'].shape}")
    print(f"   visual shape: {sample['visual'].shape}")
    print(f"   audio shape:  {sample['audio'].shape}")
    print(f"   label:        {sample['label'].item():.4f}")

    # 保存處理後的數據
    loader.save_processed_data()

    print("\n" + "="*80)
    print("✅ 測試完成！")
    print("="*80)
