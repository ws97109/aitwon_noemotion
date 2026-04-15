"""
MAG 模型權重轉換工具

將訓練好的 PyTorch MAG 模型轉換為 numpy 格式
使其可以被現有的 MGT 情緒評分系統使用
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any
import json


def extract_mag_weights(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """
    從 PyTorch checkpoint 提取權重並轉換為 numpy 格式

    Args:
        checkpoint_path: 訓練好的模型路徑 (.pt 文件)

    Returns:
        numpy 格式的權重字典
    """
    print(f"\n📦 載入模型權重: {checkpoint_path}")

    # 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})
    val_f1 = checkpoint.get('val_f1', 0.0)
    epoch = checkpoint.get('epoch', 0)

    print(f"   訓練輪數: {epoch}")
    print(f"   驗證 F1: {val_f1:.2f}%")
    print(f"   模型配置: {config}")

    # 提取權重
    numpy_weights = {}

    # ========== 1. MAG 門控網絡權重 ==========
    print("\n🔧 提取 MAG 門控網絡權重...")

    # Gate Transform (2層)
    if 'mag.gate_transform.0.weight' in model_state:
        numpy_weights['gate_layer1_weight'] = model_state['mag.gate_transform.0.weight'].cpu().numpy()
        numpy_weights['gate_layer1_bias'] = model_state['mag.gate_transform.0.bias'].cpu().numpy()
        print(f"   ✓ Gate Layer 1: {numpy_weights['gate_layer1_weight'].shape}")

    if 'mag.gate_transform.3.weight' in model_state:
        numpy_weights['gate_layer2_weight'] = model_state['mag.gate_transform.3.weight'].cpu().numpy()
        numpy_weights['gate_layer2_bias'] = model_state['mag.gate_transform.3.bias'].cpu().numpy()
        print(f"   ✓ Gate Layer 2: {numpy_weights['gate_layer2_weight'].shape}")

    # Fusion Transform
    if 'mag.fusion_transform.0.weight' in model_state:
        numpy_weights['fusion_weight'] = model_state['mag.fusion_transform.0.weight'].cpu().numpy()
        numpy_weights['fusion_bias'] = model_state['mag.fusion_transform.0.bias'].cpu().numpy()
        print(f"   ✓ Fusion Layer: {numpy_weights['fusion_weight'].shape}")

    # ========== 2. Context Attention 權重 ==========
    print("\n🔧 提取 Context Attention 權重...")

    # Multi-head Attention
    if 'context_attention.multihead_attn.in_proj_weight' in model_state:
        # in_proj 包含 Q, K, V 的權重（拼接在一起）
        in_proj_weight = model_state['context_attention.multihead_attn.in_proj_weight'].cpu().numpy()
        in_proj_bias = model_state['context_attention.multihead_attn.in_proj_bias'].cpu().numpy()

        hidden_dim = in_proj_weight.shape[1]
        # 分離 Q, K, V
        numpy_weights['attention_query_weight'] = in_proj_weight[:hidden_dim, :]
        numpy_weights['attention_key_weight'] = in_proj_weight[hidden_dim:2*hidden_dim, :]
        numpy_weights['attention_value_weight'] = in_proj_weight[2*hidden_dim:, :]

        numpy_weights['attention_query_bias'] = in_proj_bias[:hidden_dim]
        numpy_weights['attention_key_bias'] = in_proj_bias[hidden_dim:2*hidden_dim]
        numpy_weights['attention_value_bias'] = in_proj_bias[2*hidden_dim:]

        print(f"   ✓ Attention Q: {numpy_weights['attention_query_weight'].shape}")
        print(f"   ✓ Attention K: {numpy_weights['attention_key_weight'].shape}")
        print(f"   ✓ Attention V: {numpy_weights['attention_value_weight'].shape}")

    # Attention Output Projection
    if 'context_attention.multihead_attn.out_proj.weight' in model_state:
        numpy_weights['attention_out_weight'] = model_state['context_attention.multihead_attn.out_proj.weight'].cpu().numpy()
        numpy_weights['attention_out_bias'] = model_state['context_attention.multihead_attn.out_proj.bias'].cpu().numpy()
        print(f"   ✓ Attention Out: {numpy_weights['attention_out_weight'].shape}")

    # ========== 3. 情緒分類器權重 ==========
    print("\n🔧 提取情緒分類器權重...")

    # Layer 1: hidden_dim -> hidden_dim//2
    if 'emotion_classifier.0.weight' in model_state:
        numpy_weights['classifier_layer1_weight'] = model_state['emotion_classifier.0.weight'].cpu().numpy()
        numpy_weights['classifier_layer1_bias'] = model_state['emotion_classifier.0.bias'].cpu().numpy()
        print(f"   ✓ Classifier Layer 1: {numpy_weights['classifier_layer1_weight'].shape}")

    # Layer 2: hidden_dim//2 -> hidden_dim//4
    if 'emotion_classifier.4.weight' in model_state:
        numpy_weights['classifier_layer2_weight'] = model_state['emotion_classifier.4.weight'].cpu().numpy()
        numpy_weights['classifier_layer2_bias'] = model_state['emotion_classifier.4.bias'].cpu().numpy()
        print(f"   ✓ Classifier Layer 2: {numpy_weights['classifier_layer2_weight'].shape}")

    # Layer 3 (Output): hidden_dim//4 -> num_emotions
    if 'emotion_classifier.8.weight' in model_state:
        numpy_weights['classifier_output_weight'] = model_state['emotion_classifier.8.weight'].cpu().numpy()
        numpy_weights['classifier_output_bias'] = model_state['emotion_classifier.8.bias'].cpu().numpy()
        print(f"   ✓ Classifier Output: {numpy_weights['classifier_output_weight'].shape}")

    # ========== 4. BERT 權重（可選） ==========
    # BERT 權重太大，通常不包含在轉換中
    # 如果需要，可以單獨保存
    bert_keys = [k for k in model_state.keys() if k.startswith('bert.')]
    if bert_keys:
        print(f"\n⚠️  檢測到 {len(bert_keys)} 個 BERT 權重層")
        print("   注意: BERT 權重未包含在轉換中（體積太大）")
        print("   如需使用，請直接載入完整的 PyTorch 模型")

    # ========== 元數據 ==========
    metadata = {
        'model_name': config.get('model_name', 'bert-base-uncased'),
        'hidden_dim': config.get('hidden_dim', 768),
        'num_emotions': config.get('num_emotions', 7),
        'val_f1': float(val_f1),
        'val_acc': float(checkpoint.get('val_acc', 0.0)),
        'epoch': int(epoch),
        'train_f1': float(checkpoint.get('train_f1', 0.0))
    }

    numpy_weights['metadata'] = np.array([metadata])

    print(f"\n✅ 權重提取完成！共 {len(numpy_weights)} 個權重張量")

    return numpy_weights


def save_numpy_weights(weights: Dict[str, np.ndarray],
                       output_path: str,
                       compressed: bool = True):
    """
    保存 numpy 權重到文件

    Args:
        weights: 權重字典
        output_path: 輸出路徑
        compressed: 是否壓縮（使用 .npz）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        # 壓縮格式
        np.savez_compressed(output_path, **weights)
        print(f"\n💾 權重已保存（壓縮）: {output_path}")
    else:
        # 非壓縮格式
        np.savez(output_path, **weights)
        print(f"\n💾 權重已保存（非壓縮）: {output_path}")

    # 顯示文件大小
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   文件大小: {file_size_mb:.2f} MB")


def verify_weights(weights_path: str):
    """驗證權重文件"""
    print(f"\n🔍 驗證權重文件: {weights_path}")

    weights = np.load(weights_path, allow_pickle=True)

    print(f"\n📊 權重清單:")
    for key in weights.files:
        if key == 'metadata':
            metadata = weights[key].item()
            print(f"\n   📋 元數據:")
            for k, v in metadata.items():
                print(f"      {k}: {v}")
        else:
            array = weights[key]
            print(f"   {key:40s}: {array.shape} ({array.dtype})")

    print(f"\n✅ 驗證完成！")


def create_lightweight_classifier(weights_path: str,
                                   output_path: str):
    """
    創建輕量級分類器（只包含分類器層，不包含 BERT）

    這個版本可以與現有的 MGT 系統整合
    使用系統自己的文本嵌入，只使用訓練好的分類頭
    """
    print(f"\n🔧 創建輕量級分類器...")

    weights = np.load(weights_path, allow_pickle=True)

    # 只保留分類器和 MAG 相關權重
    lightweight_weights = {}

    # MAG 權重
    for key in ['gate_layer1_weight', 'gate_layer1_bias',
                'gate_layer2_weight', 'gate_layer2_bias',
                'fusion_weight', 'fusion_bias']:
        if key in weights:
            lightweight_weights[key] = weights[key]

    # 分類器權重
    for key in ['classifier_layer1_weight', 'classifier_layer1_bias',
                'classifier_layer2_weight', 'classifier_layer2_bias',
                'classifier_output_weight', 'classifier_output_bias']:
        if key in weights:
            lightweight_weights[key] = weights[key]

    # 元數據
    if 'metadata' in weights:
        lightweight_weights['metadata'] = weights['metadata']

    # 保存
    np.savez_compressed(output_path, **lightweight_weights)

    original_size = Path(weights_path).stat().st_size / (1024 * 1024)
    lightweight_size = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"   原始大小: {original_size:.2f} MB")
    print(f"   輕量級大小: {lightweight_size:.2f} MB")
    print(f"   壓縮比: {lightweight_size/original_size*100:.1f}%")
    print(f"\n✅ 輕量級分類器已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MAG 模型權重轉換工具")

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./emotion_system/models/mag/best_mag_model.pt',
        help='訓練好的 PyTorch 模型路徑'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./emotion_system/models/mag/mag_weights.npz',
        help='輸出的 numpy 權重路徑'
    )

    parser.add_argument(
        '--compressed',
        action='store_true',
        default=True,
        help='是否壓縮輸出文件'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='驗證權重文件'
    )

    parser.add_argument(
        '--lightweight',
        action='store_true',
        help='創建輕量級版本（只包含分類器，不包含 BERT）'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MAG 模型權重轉換工具")
    print("=" * 80)

    if args.verify:
        # 只驗證現有文件
        verify_weights(args.output)
    else:
        # 轉換權重
        checkpoint_path = Path(args.checkpoint)

        if not checkpoint_path.exists():
            print(f"\n❌ 錯誤: 找不到模型文件 {checkpoint_path}")
            print("請先訓練模型或指定正確的路徑")
            return

        # 提取權重
        weights = extract_mag_weights(str(checkpoint_path))

        # 保存權重
        save_numpy_weights(weights, args.output, compressed=args.compressed)

        # 驗證
        verify_weights(args.output)

        # 創建輕量級版本
        if args.lightweight:
            lightweight_path = args.output.replace('.npz', '_lightweight.npz')
            create_lightweight_classifier(args.output, lightweight_path)

    print("\n" + "=" * 80)
    print("✅ 轉換完成！")
    print("=" * 80)

    # 使用說明
    print("\n📖 使用說明:")
    print("\n1. 在 MGT 情緒評分系統中載入權重:")
    print("   ```python")
    print("   from emotion_system.mgt_emotion_rater import MGTEmotionRater")
    print("")
    print("   rater = MGTEmotionRater(")
    print("       agent_name='李昇峰',")
    print(f"       model_weights_path='{args.output}'")
    print("   )")
    print("   ```")

    print("\n2. 或使用輕量級版本:")
    if args.lightweight:
        lightweight_path = args.output.replace('.npz', '_lightweight.npz')
        print(f"   model_weights_path='{lightweight_path}'")

    print("\n3. 完整的 PyTorch 模型推理:")
    print("   參考 MAG_TRAINING_GUIDE.md 中的「模型載入和使用」章節")
    print()


if __name__ == "__main__":
    main()
