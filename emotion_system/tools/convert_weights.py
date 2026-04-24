"""
PyTorch 到 Numpy 權重轉換器

將訓練好的 PyTorch MGT 模型權重轉換為 Numpy 格式，
以便在 MGTEmotionRater 中使用
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

# 添加專案路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_system.train_mgt_model import MGTModel, MELD_EMOTIONS


class WeightConverter:
    """PyTorch 權重轉換器"""

    def __init__(self, pytorch_model_path: str):
        """
        初始化轉換器

        Args:
            pytorch_model_path: PyTorch 模型檔案路徑 (.pth)
        """
        self.pytorch_model_path = pytorch_model_path
        self.checkpoint = None
        self.state_dict = None

    def load_pytorch_model(self) -> Dict[str, Any]:
        """載入 PyTorch 模型"""
        print(f"載入 PyTorch 模型: {self.pytorch_model_path}")

        if not os.path.exists(self.pytorch_model_path):
            raise FileNotFoundError(f"模型檔案不存在: {self.pytorch_model_path}")

        self.checkpoint = torch.load(self.pytorch_model_path, map_location='cpu')

        # 提取 state_dict
        if 'model_state_dict' in self.checkpoint:
            self.state_dict = self.checkpoint['model_state_dict']
        else:
            self.state_dict = self.checkpoint

        print(f"✓ 模型載入成功")
        print(f"  包含 {len(self.state_dict)} 個權重張量")

        # 打印模型資訊
        if 'val_acc' in self.checkpoint:
            print(f"  驗證準確率: {self.checkpoint['val_acc']:.2f}%")
        if 'epoch' in self.checkpoint:
            print(f"  訓練輪數: {self.checkpoint['epoch']}")

        return self.checkpoint

    def convert_to_numpy(self) -> Dict[str, np.ndarray]:
        """
        轉換所有權重為 Numpy 格式

        Returns:
            包含所有 numpy 權重的字典
        """
        print("\n轉換權重為 Numpy 格式...")

        numpy_weights = {}

        # 1. Parallel Multimodal Flow - Text Projection
        if 'text_projection.weight' in self.state_dict:
            numpy_weights['text_projection_weight'] = \
                self.state_dict['text_projection.weight'].cpu().numpy()
            numpy_weights['text_projection_bias'] = \
                self.state_dict['text_projection.bias'].cpu().numpy()
            print("✓ Text Projection: {} -> {}".format(
                self.state_dict['text_projection.weight'].shape,
                numpy_weights['text_projection_weight'].shape
            ))

        # 2. Parallel Multimodal Flow - Context Projection
        if 'context_projection.weight' in self.state_dict:
            numpy_weights['context_projection_weight'] = \
                self.state_dict['context_projection.weight'].cpu().numpy()
            numpy_weights['context_projection_bias'] = \
                self.state_dict['context_projection.bias'].cpu().numpy()
            print("✓ Context Projection: {} -> {}".format(
                self.state_dict['context_projection.weight'].shape,
                numpy_weights['context_projection_weight'].shape
            ))

        # 3. Cross-modal Additive Attention - Query
        if 'attention_query.weight' in self.state_dict:
            numpy_weights['attention_query_weight'] = \
                self.state_dict['attention_query.weight'].cpu().numpy()
            numpy_weights['attention_query_bias'] = \
                self.state_dict['attention_query.bias'].cpu().numpy()
            print("✓ Attention Query: {} -> {}".format(
                self.state_dict['attention_query.weight'].shape,
                numpy_weights['attention_query_weight'].shape
            ))

        # 4. Cross-modal Additive Attention - Key
        if 'attention_key.weight' in self.state_dict:
            numpy_weights['attention_key_weight'] = \
                self.state_dict['attention_key.weight'].cpu().numpy()
            numpy_weights['attention_key_bias'] = \
                self.state_dict['attention_key.bias'].cpu().numpy()
            print("✓ Attention Key: {} -> {}".format(
                self.state_dict['attention_key.weight'].shape,
                numpy_weights['attention_key_weight'].shape
            ))

        # 5. Cross-modal Additive Attention - Value
        if 'attention_value.weight' in self.state_dict:
            numpy_weights['attention_value_weight'] = \
                self.state_dict['attention_value.weight'].cpu().numpy()
            numpy_weights['attention_value_bias'] = \
                self.state_dict['attention_value.bias'].cpu().numpy()
            print("✓ Attention Value: {} -> {}".format(
                self.state_dict['attention_value.weight'].shape,
                numpy_weights['attention_value_weight'].shape
            ))

        # 6. Cross-modal Additive Attention - Attention Vector
        if 'attention_vector' in self.state_dict:
            numpy_weights['attention_vector'] = \
                self.state_dict['attention_vector'].cpu().numpy()
            print("✓ Attention Vector: {} -> {}".format(
                self.state_dict['attention_vector'].shape,
                numpy_weights['attention_vector'].shape
            ))

        # 7. Gating Mechanism
        if 'gate.0.weight' in self.state_dict:
            numpy_weights['gate_weight'] = \
                self.state_dict['gate.0.weight'].cpu().numpy()
            numpy_weights['gate_bias'] = \
                self.state_dict['gate.0.bias'].cpu().numpy()
            print("✓ Gate: {} -> {}".format(
                self.state_dict['gate.0.weight'].shape,
                numpy_weights['gate_weight'].shape
            ))

        # 8. Emotion Classifier
        # Layer 1: hidden_dim -> hidden_dim // 2
        if 'classifier.0.weight' in self.state_dict:
            numpy_weights['classifier_layer1_weight'] = \
                self.state_dict['classifier.0.weight'].cpu().numpy()
            numpy_weights['classifier_layer1_bias'] = \
                self.state_dict['classifier.0.bias'].cpu().numpy()
            print("✓ Classifier Layer 1: {} -> {}".format(
                self.state_dict['classifier.0.weight'].shape,
                numpy_weights['classifier_layer1_weight'].shape
            ))

        # Layer 2: hidden_dim // 2 -> num_emotions
        if 'classifier.3.weight' in self.state_dict:
            numpy_weights['classifier_layer2_weight'] = \
                self.state_dict['classifier.3.weight'].cpu().numpy()
            numpy_weights['classifier_layer2_bias'] = \
                self.state_dict['classifier.3.bias'].cpu().numpy()
            print("✓ Classifier Layer 2: {} -> {}".format(
                self.state_dict['classifier.3.weight'].shape,
                numpy_weights['classifier_layer2_weight'].shape
            ))

        # 9. 保存元數據
        numpy_weights['metadata'] = np.array({
            'hidden_dim': 768,
            'num_emotions': len(MELD_EMOTIONS),
            'emotions': MELD_EMOTIONS,
            'val_acc': self.checkpoint.get('val_acc', 0.0),
            'epoch': self.checkpoint.get('epoch', 0)
        }, dtype=object)

        print(f"\n✓ 成功轉換 {len(numpy_weights)} 個權重張量")

        return numpy_weights

    def save_numpy_weights(self, numpy_weights: Dict[str, np.ndarray], output_path: str):
        """
        保存 Numpy 權重到檔案

        Args:
            numpy_weights: Numpy 權重字典
            output_path: 輸出檔案路徑 (.npz)
        """
        print(f"\n保存 Numpy 權重到: {output_path}")

        # 確保輸出目錄存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存為 .npz 格式（壓縮）
        np.savez_compressed(output_path, **numpy_weights)

        # 檢查檔案大小
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ 權重已保存")
        print(f"  檔案大小: {file_size_mb:.2f} MB")

    def verify_conversion(self, numpy_weights: Dict[str, np.ndarray]) -> bool:
        """
        驗證轉換正確性

        Args:
            numpy_weights: 轉換後的 numpy 權重

        Returns:
            驗證是否通過
        """
        print("\n驗證權重轉換...")

        checks_passed = 0
        checks_total = 0

        # 檢查 1: 形狀一致性
        checks_total += 1
        if 'text_projection_weight' in numpy_weights:
            expected_shape = (768, 768)
            actual_shape = numpy_weights['text_projection_weight'].shape
            if actual_shape == expected_shape:
                print(f"✓ Text Projection 形狀正確: {actual_shape}")
                checks_passed += 1
            else:
                print(f"✗ Text Projection 形狀錯誤: {actual_shape} vs {expected_shape}")

        # 檢查 2: 數值範圍合理
        checks_total += 1
        if 'classifier_layer2_weight' in numpy_weights:
            weights = numpy_weights['classifier_layer2_weight']
            mean = np.mean(weights)
            std = np.std(weights)
            if -1 < mean < 1 and 0 < std < 2:
                print(f"✓ 分類器權重範圍合理: mean={mean:.4f}, std={std:.4f}")
                checks_passed += 1
            else:
                print(f"✗ 分類器權重範圍異常: mean={mean:.4f}, std={std:.4f}")

        # 檢查 3: 沒有 NaN 或 Inf
        checks_total += 1
        has_invalid = False
        for key, value in numpy_weights.items():
            if key != 'metadata' and isinstance(value, np.ndarray):
                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    print(f"✗ {key} 包含 NaN 或 Inf")
                    has_invalid = True
        if not has_invalid:
            print("✓ 所有權重數值有效（無 NaN/Inf）")
            checks_passed += 1

        # 檢查 4: 必要權重存在
        checks_total += 1
        required_weights = [
            'text_projection_weight',
            'context_projection_weight',
            'attention_query_weight',
            'attention_key_weight',
            'gate_weight',
            'classifier_layer1_weight',
            'classifier_layer2_weight'
        ]
        missing_weights = [w for w in required_weights if w not in numpy_weights]
        if not missing_weights:
            print(f"✓ 所有必要權重已轉換 ({len(required_weights)} 個)")
            checks_passed += 1
        else:
            print(f"✗ 缺少權重: {missing_weights}")

        print(f"\n驗證結果: {checks_passed}/{checks_total} 項檢查通過")

        return checks_passed == checks_total

    def convert(self, output_path: str) -> bool:
        """
        執行完整轉換流程

        Args:
            output_path: 輸出檔案路徑

        Returns:
            轉換是否成功
        """
        try:
            # 載入 PyTorch 模型
            self.load_pytorch_model()

            # 轉換為 Numpy
            numpy_weights = self.convert_to_numpy()

            # 驗證轉換
            if not self.verify_conversion(numpy_weights):
                print("\n⚠️  驗證失敗，但仍會保存權重")

            # 保存權重
            self.save_numpy_weights(numpy_weights, output_path)

            print("\n✅ 權重轉換完成！")
            print(f"\n使用方式：")
            print(f"  from emotion_system.mgt_emotion_rater import MGTEmotionRater")
            print(f"  rater = MGTEmotionRater(...)")
            print(f"  rater.load_trained_weights('{output_path}')")

            return True

        except Exception as e:
            print(f"\n❌ 轉換失敗: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='將 PyTorch MGT 模型權重轉換為 Numpy 格式')
    parser.add_argument(
        '--input',
        type=str,
        default='models/mgt/best_model.pth',
        help='PyTorch 模型檔案路徑 (.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/mgt/mgt_weights.npz',
        help='輸出 Numpy 權重檔案路徑 (.npz)'
    )

    args = parser.parse_args()

    print("="*70)
    print("PyTorch 到 Numpy 權重轉換器")
    print("="*70)
    print(f"\n輸入模型: {args.input}")
    print(f"輸出權重: {args.output}\n")

    # 創建轉換器並執行
    converter = WeightConverter(args.input)
    success = converter.convert(args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
