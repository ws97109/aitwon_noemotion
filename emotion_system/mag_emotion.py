"""
MAG (Multimodal Adaptation Gate) 情緒分類訓練系統 - 純文本版本

基於論文: "Integrating Multimodal Information in Large Pretrained Transformers"
專注於純文本情緒分類，使用 MAG 架構的核心思想

核心組件:
1. BERT/RoBERTa 作為預訓練基礎
2. MAG (Multimodal Adaptation Gate) 控制情緒信息流
3. Context-aware Attention 機制
4. 完整的訓練管道（數據增強、學習率調度、早停）

重要: 此實現優先考慮精準度和效果，不做簡化
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
import pickle
import random
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
    mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr

# Transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoConfig,
        get_linear_schedule_with_warmup
    )
    HAS_TRANSFORMERS = True
except ImportError as e:
    print(f"❌ transformers 未安裝或導入失敗！")
    print(f"錯誤信息: {e}")
    print("請運行: pip install transformers torch")
    HAS_TRANSFORMERS = False
    sys.exit(1)


# ========== MELD 情緒類別 ==========
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(MELD_EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

# 情緒中文映射
EMOTION_CN_MAP = {
    "anger": "憤怒",
    "disgust": "厭惡",
    "fear": "恐懼",
    "joy": "喜悅",
    "neutral": "中性",
    "sadness": "悲傷",
    "surprise": "驚訝"
}


# ========== 數據增強 ==========
class TextAugmentation:
    """文本數據增強"""

    @staticmethod
    def synonym_replacement(text: str, n: int = 1) -> str:
        """同義詞替換（簡化版本）"""
        # 實際應用可以使用 WordNet 或 BERT 進行同義詞替換
        return text

    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
        """隨機刪除單詞"""
        words = text.split()
        if len(words) == 1:
            return text

        new_words = [word for word in words if random.random() > p]

        if len(new_words) == 0:
            return random.choice(words)

        return ' '.join(new_words)

    @staticmethod
    def random_swap(text: str, n: int = 1) -> str:
        """隨機交換單詞位置"""
        words = text.split()
        length = len(words)

        if length < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(length), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)


# ========== 數據集 ==========
class MELDEmotionDataset(Dataset):
    """MELD 情緒數據集 - 純文本版本"""

    def __init__(self,
                 csv_path: str,
                 tokenizer: Any,
                 max_length: int = 128,
                 context_window: int = 3,
                 use_augmentation: bool = False,
                 augmentation_prob: float = 0.3):
        """
        初始化數據集

        Args:
            csv_path: CSV 文件路徑
            tokenizer: BERT/RoBERTa tokenizer
            max_length: 最大序列長度
            context_window: 上下文窗口大小
            use_augmentation: 是否使用數據增強
            augmentation_prob: 數據增強概率
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob
        self.augmenter = TextAugmentation()

        # 計算類別權重（用於處理不平衡數據）
        self.class_weights = self._compute_class_weights()

        print(f"📊 載入數據集: {csv_path}")
        print(f"   樣本數: {len(self.df)}")
        print(f"   情緒分佈: {dict(self.df['Emotion'].value_counts())}")
        print(f"   數據增強: {'啟用' if use_augmentation else '禁用'}")

    def _compute_class_weights(self) -> torch.Tensor:
        """計算類別權重（處理不平衡數據）"""
        emotion_counts = self.df['Emotion'].value_counts()
        total = len(self.df)

        weights = []
        for emotion in MELD_EMOTIONS:
            count = emotion_counts.get(emotion, 1)
            weight = total / (len(MELD_EMOTIONS) * count)
            weights.append(weight)

        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.df)

    def _get_context(self, idx: int) -> str:
        """獲取對話上下文"""
        context = []
        dialogue_id = self.df.iloc[idx]['Dialogue_ID']

        # 獲取前 N 個同對話的語句
        for i in range(max(0, idx - self.context_window), idx):
            if self.df.iloc[i]['Dialogue_ID'] == dialogue_id:
                speaker = self.df.iloc[i]['Speaker']
                utterance = self.df.iloc[i]['Utterance']
                context.append(f"{speaker}: {utterance}")

        return " [SEP] ".join(context) if context else ""

    def _augment_text(self, text: str) -> str:
        """應用數據增強"""
        if not self.use_augmentation or random.random() > self.augmentation_prob:
            return text

        # 隨機選擇增強方法
        aug_method = random.choice([
            self.augmenter.random_deletion,
            self.augmenter.random_swap
        ])

        return aug_method(text)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 獲取語句和情緒標籤
        utterance = row['Utterance']
        emotion = row['Emotion']
        emotion_idx = EMOTION_TO_IDX[emotion]
        speaker = row['Speaker']

        # 獲取對話上下文
        context = self._get_context(idx)

        # 數據增強
        utterance = self._augment_text(utterance)

        # 構建輸入文本: [CLS] context [SEP] speaker: utterance [SEP]
        if context:
            input_text = f"{context} [SEP] {speaker}: {utterance}"
        else:
            input_text = f"{speaker}: {utterance}"

        # Tokenization
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'emotion_label': torch.tensor(emotion_idx, dtype=torch.long),
            'utterance_text': utterance,
            'emotion_name': emotion,
            'has_context': len(context) > 0
        }


# ========== MAG 核心模組 ==========
class MAG(nn.Module):
    """
    MAG (Multimodal Adaptation Gate) - 原始論文實現

    論文: "Integrating Multimodal Information in Large Pretrained Transformers"
    核心思想: 使用門控機制動態調節非語言信息注入到預訓練語言模型

    純文本適配:
    - text_h: 當前語句的 BERT 表徵（主要信息）
    - aux_h: 對話上下文的表徵（輔助信息）

    MAG 公式:
    H_m = α ⊙ (W_m * [text_h; aux_h] + b_m)
    α = sigmoid(W_g * [text_h; aux_h] + b_g) * β_shift
    output = text_h + α ⊙ H_m

    參數:
    - β_shift: 控制門的開合程度（通常 0.5-2.0）
    - dropout: 正則化
    """

    def __init__(self,
                 text_dim: int,
                 aux_dim: int,
                 output_dim: int,
                 beta_shift: float = 1.0,
                 dropout: float = 0.1):
        """
        Args:
            text_dim: 文本表徵維度（BERT hidden size）
            aux_dim: 輔助信息維度（上下文表徵維度）
            output_dim: 輸出維度（通常與 text_dim 相同）
            beta_shift: 門控偏移參數（論文推薦 1.0）
            dropout: Dropout 率
        """
        super().__init__()

        self.text_dim = text_dim
        self.aux_dim = aux_dim
        self.output_dim = output_dim
        self.beta_shift = beta_shift

        # 融合層: W_m * [text_h; aux_h] + b_m
        self.W_m = nn.Linear(text_dim + aux_dim, output_dim, bias=True)

        # 門控層: W_g * [text_h; aux_h] + b_g
        self.W_g = nn.Linear(text_dim + aux_dim, output_dim, bias=True)

        # Layer Normalization（穩定訓練）
        self.layer_norm = nn.LayerNorm(output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.W_m.weight)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_m.bias)
        nn.init.zeros_(self.W_g.bias)

    def forward(self,
                text_h: torch.Tensor,
                aux_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAG 前向傳播

        Args:
            text_h: 文本表徵 [batch_size, text_dim] 或 [batch_size, seq_len, text_dim]
            aux_h: 輔助表徵 [batch_size, aux_dim] 或 [batch_size, seq_len, aux_dim]

        Returns:
            (output, gate_values)
            - output: MAG 輸出 [batch_size, output_dim]
            - gate_values: 門控值（用於監控） [batch_size]
        """
        # 處理不同的輸入形狀
        if text_h.dim() == 3:  # [batch_size, seq_len, dim]
            # 只使用 [CLS] token（第一個位置）
            text_h = text_h[:, 0, :]  # [batch_size, text_dim]

        if aux_h.dim() == 3:
            aux_h = aux_h[:, 0, :]  # [batch_size, aux_dim]

        # 1. 拼接文本和輔助表徵
        concat_h = torch.cat([text_h, aux_h], dim=-1)  # [batch_size, text_dim + aux_dim]

        # 2. 融合變換: H_m = W_m * [text_h; aux_h] + b_m
        H_m = torch.tanh(self.W_m(concat_h))  # [batch_size, output_dim]

        # 3. 門控: α = sigmoid(W_g * [text_h; aux_h] + b_g) * β_shift
        gate_logits = self.W_g(concat_h)  # [batch_size, output_dim]
        gate = torch.sigmoid(gate_logits) * self.beta_shift  # [batch_size, output_dim]

        # 4. 門控融合: H_m_gated = α ⊙ H_m
        H_m_gated = gate * H_m  # Element-wise multiplication

        # 5. 殘差連接: output = text_h + H_m_gated
        output = text_h + H_m_gated  # [batch_size, output_dim]

        # 6. Layer Normalization + Dropout
        output = self.layer_norm(output)
        output = self.dropout(output)

        # 計算平均門控值（用於監控）
        avg_gate = gate.mean(dim=-1)  # [batch_size]

        return output, avg_gate


class ContextAwareAttention(nn.Module):
    """
    上下文感知注意力機制

    讓模型學習當前語句和上下文的相關性
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: 查詢向量 [batch_size, seq_len, hidden_dim]
            key: 鍵向量 [batch_size, seq_len, hidden_dim]
            value: 值向量 [batch_size, seq_len, hidden_dim]
            key_padding_mask: Padding mask [batch_size, seq_len]

        Returns:
            (注意力輸出, 注意力權重)
        """
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            query, key, value,
            key_padding_mask=key_padding_mask
        )

        # Residual connection + Layer norm
        output = self.layer_norm(query + self.dropout(attn_output))

        return output, attn_weights


# ========== MAG 情緒分類模型 ==========
class MAGEmotionClassifier(nn.Module):
    """
    MAG 情緒分類器（純文本版本）

    架構:
    1. BERT/RoBERTa 編碼器
    2. Context-aware Attention
    3. MAG 門控融合
    4. 情緒分類頭
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_emotions: int = 7,
                 hidden_dim: int = 768,
                 dropout: float = 0.3,
                 beta_shift: float = 1.0,
                 freeze_bert: bool = False):
        """
        Args:
            model_name: 預訓練模型名稱
            num_emotions: 情緒類別數
            hidden_dim: 隱藏層維度
            dropout: Dropout 率
            beta_shift: MAG beta 參數
            freeze_bert: 是否凍結 BERT 參數
        """
        super().__init__()

        self.model_name = model_name
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim

        # 載入預訓練 BERT/RoBERTa
        print(f"📦 載入預訓練模型: {model_name}")
        self.bert_config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # 凍結 BERT（可選）
        if freeze_bert:
            print("❄️  凍結 BERT 參數")
            for param in self.bert.parameters():
                param.requires_grad = False

        # Context-aware Attention
        self.context_attention = ContextAwareAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )

        # MAG 門控模組（使用原始論文實現）
        self.mag = MAG(
            text_dim=hidden_dim,
            aux_dim=hidden_dim,
            output_dim=hidden_dim,
            beta_shift=beta_shift,
            dropout=dropout
        )

        # 情緒分類頭（多層）
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, num_emotions)
        )

        # 初始化分類頭
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """初始化分類器權重"""
        for module in self.emotion_classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向傳播

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_features: 是否返回中間特徵

        Returns:
            {
                'logits': 分類 logits [batch_size, num_emotions],
                'gate_values': MAG 門控值 [batch_size],
                'features': (可選) 特徵表徵
            }
        """
        batch_size = input_ids.size(0)

        # 1. BERT 編碼
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 獲取所有 hidden states 和 [CLS] token
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        cls_output = sequence_output[:, 0, :]  # [CLS] token [batch_size, hidden_dim]

        # 2. Context-aware Attention
        # 讓模型學習序列內的依賴關係
        attn_output, attn_weights = self.context_attention(
            query=sequence_output,
            key=sequence_output,
            value=sequence_output,
            key_padding_mask=(attention_mask == 0)
        )

        # 獲取注意力加權後的 [CLS]
        context_aware_cls = attn_output[:, 0, :]  # [batch_size, hidden_dim]

        # 3. MAG 門控融合
        # text_h: 原始 [CLS] token（主要信息）
        # aux_h: context-aware [CLS]（輔助上下文信息）
        mag_output, gate_values = self.mag(
            text_h=cls_output,
            aux_h=context_aware_cls
        )

        # 4. 情緒分類
        logits = self.emotion_classifier(mag_output)  # [batch_size, num_emotions]

        output = {
            'logits': logits,
            'gate_values': gate_values
        }

        if return_features:
            output['features'] = {
                'cls_output': cls_output,
                'context_aware_cls': context_aware_cls,
                'mag_output': mag_output,
                'attention_weights': attn_weights
            }

        return output


# ========== 評估指標（論文標準）==========
def compute_metrics(predictions: np.ndarray,
                   labels: np.ndarray,
                   num_classes: int = 7) -> Dict[str, float]:
    """
    計算論文標準評估指標

    指標說明（基於 MAG 論文）:
    - Acc7: 7-class accuracy（7 類準確率）
    - Acc2: 2-class accuracy（二分類準確率：正面 vs 負面）
    - F1: Weighted F1-score
    - MAE: Mean Absolute Error（平均絕對誤差）*
    - Corr: Pearson Correlation（皮爾遜相關係數）*

    * 注：MAE 和 Corr 通常用於情感強度回歸任務（如 CMU-MOSI）
      對於分類任務（MELD），這些指標可能不適用

    Args:
        predictions: 預測標籤 [N]
        labels: 真實標籤 [N]
        num_classes: 類別數量

    Returns:
        指標字典
    """
    metrics = {}

    # 1. Acc7: 7-class Accuracy
    acc7 = accuracy_score(labels, predictions)
    metrics['Acc7'] = acc7 * 100.0

    # 2. Acc2: Binary Accuracy（正面 vs 負面）
    # MELD 情緒映射到二分類：
    # 正面：joy (3)
    # 負面：anger (0), disgust (1), fear (2), sadness (5)
    # 中性：neutral (4), surprise (6) → 視為正面
    def to_binary(emotion_idx):
        """將 7 類情緒映射到 2 類"""
        negative_emotions = {0, 1, 2, 5}  # anger, disgust, fear, sadness
        return 0 if emotion_idx in negative_emotions else 1  # 0: negative, 1: non-negative

    binary_preds = np.array([to_binary(p) for p in predictions])
    binary_labels = np.array([to_binary(l) for l in labels])
    acc2 = accuracy_score(binary_labels, binary_preds)
    metrics['Acc2'] = acc2 * 100.0

    # 3. F1: Weighted F1-score
    f1_weighted = f1_score(labels, predictions, average='weighted')
    metrics['F1_weighted'] = f1_weighted * 100.0

    # 4. F1: Macro F1-score（所有類別平均）
    f1_macro = f1_score(labels, predictions, average='macro')
    metrics['F1_macro'] = f1_macro * 100.0

    # 5. MAE: Mean Absolute Error
    # 對於分類任務，MAE 可以表示預測類別與真實類別的平均距離
    # 這在有序情緒強度任務中更有意義
    mae = mean_absolute_error(labels, predictions)
    metrics['MAE'] = mae

    # 6. Correlation: Pearson & Spearman
    # 對於分類任務，相關係數可能不太有意義
    # 但如果情緒類別有隱含的順序（如強度），可以計算
    try:
        pearson_corr, _ = pearsonr(labels, predictions)
        metrics['Pearson'] = pearson_corr
    except:
        metrics['Pearson'] = 0.0

    try:
        spearman_corr, _ = spearmanr(labels, predictions)
        metrics['Spearman'] = spearman_corr
    except:
        metrics['Spearman'] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    打印論文標準格式的指標

    Args:
        metrics: 指標字典
        prefix: 前綴（如 "Train", "Val", "Test"）
    """
    print(f"\n📊 {prefix} 評估指標（論文標準）:")
    print(f"   Acc7 (7-class):     {metrics['Acc7']:.2f}%")
    print(f"   Acc2 (binary):      {metrics['Acc2']:.2f}%")
    print(f"   F1 (weighted):      {metrics['F1_weighted']:.2f}%")
    print(f"   F1 (macro):         {metrics['F1_macro']:.2f}%")
    print(f"   MAE:                {metrics['MAE']:.4f}")
    print(f"   Pearson Corr:       {metrics.get('Pearson', 0.0):.4f}")
    print(f"   Spearman Corr:      {metrics.get('Spearman', 0.0):.4f}")


# ========== 損失函數 ==========
class FocalLoss(nn.Module):
    """
    Focal Loss - 處理類別不平衡

    論文: "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # 類別權重
        self.gamma = gamma  # 聚焦參數

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型輸出 logits [batch_size, num_classes]
            targets: 真實標籤 [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# ========== 訓練器 ==========
class MAGEmotionTrainer:
    """MAG 情緒分類器訓練器"""

    def __init__(self,
                 model: MAGEmotionClassifier,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 device: str = None,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 0,
                 max_grad_norm: float = 1.0,
                 use_focal_loss: bool = True,
                 class_weights: Optional[torch.Tensor] = None):
        """
        初始化訓練器

        Args:
            model: MAG 情緒分類模型
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            test_loader: 測試數據加載器（可選）
            device: 訓練設備
            learning_rate: 學習率
            weight_decay: 權重衰減
            warmup_steps: Warmup 步數
            max_grad_norm: 梯度裁剪閾值
            use_focal_loss: 是否使用 Focal Loss
            class_weights: 類別權重
        """
        # 自動選擇設備：CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                print("🚀 使用 CUDA (NVIDIA GPU) 加速")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                print("🚀 使用 MPS (Apple Silicon GPU) 加速")
            else:
                device = "cpu"
                print("⚠️  使用 CPU（訓練較慢）")

        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_grad_norm = max_grad_norm

        # 優化器（使用 AdamW）
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )

        # 學習率調度器
        total_steps = len(train_loader) * 50  # 假設最多訓練 50 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 損失函數
        if use_focal_loss:
            self.criterion = FocalLoss(
                alpha=class_weights.to(device) if class_weights is not None else None,
                gamma=2.0
            )
            print("📊 使用 Focal Loss（處理類別不平衡）")
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device) if class_weights is not None else None
            )

        # 訓練歷史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'best_val_f1': 0.0,
            'best_epoch': 0
        }

        # 早停
        self.patience = 5
        self.patience_counter = 0
        self.best_val_f1 = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """訓練一個 epoch"""
        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []
        gate_values_list = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['emotion_label'].to(self.device)

            # 前向傳播
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            gate_values = outputs['gate_values']

            # 計算損失
            loss = self.criterion(logits, labels)

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # 更新參數
            self.optimizer.step()
            self.scheduler.step()

            # 統計
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            gate_values_list.extend(gate_values.cpu().detach().numpy())

            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gate': f'{gate_values.mean().item():.3f}'
            })

        # 計算指標
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted') * 100
        avg_gate = np.mean(gate_values_list)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'avg_gate': avg_gate
        }

    def validate(self) -> Dict[str, float]:
        """驗證模型"""
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['emotion_label'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']

                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 計算基本指標
        avg_loss = total_loss / len(self.val_loader)

        # 計算論文標準指標
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        metrics = compute_metrics(all_preds_np, all_labels_np, num_classes=7)

        # 詳細報告（每個類別的 F1）
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=list(range(7))
        )

        return {
            'loss': avg_loss,
            'accuracy': metrics['Acc7'],  # 使用 Acc7
            'f1': metrics['F1_weighted'],  # 使用 weighted F1
            'acc2': metrics['Acc2'],  # 二分類準確率
            'f1_macro': metrics['F1_macro'],  # Macro F1
            'mae': metrics['MAE'],  # MAE
            'pearson': metrics.get('Pearson', 0.0),  # Pearson 相關
            'spearman': metrics.get('Spearman', 0.0),  # Spearman 相關
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'metrics': metrics  # 完整指標字典
        }

    def train(self,
              num_epochs: int,
              save_dir: str = "./emotion_system/models/mag",
              log_interval: int = 1):
        """
        完整訓練流程

        Args:
            num_epochs: 訓練輪數
            save_dir: 模型保存目錄
            log_interval: 日誌記錄間隔
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"🚀 開始訓練 MAG 情緒分類模型")
        print("=" * 80)
        print(f"📦 模型: {self.model.model_name}")
        print(f"🖥️  設備: {self.device}")
        print(f"📊 訓練樣本: {len(self.train_loader.dataset)}")
        print(f"📊 驗證樣本: {len(self.val_loader.dataset)}")
        print(f"🔄 訓練輪數: {num_epochs}")
        print(f"📈 學習率: {self.optimizer.param_groups[0]['lr']}")
        print(f"⚖️  損失函數: {type(self.criterion).__name__}")
        print("=" * 80 + "\n")

        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*80)

            # 訓練
            train_metrics = self.train_epoch(epoch)

            # 驗證
            val_metrics = self.validate()

            # 記錄歷史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            # 打印結果
            print(f"\n📊 訓練結果:")
            print(f"   Loss: {train_metrics['loss']:.4f}")
            print(f"   Accuracy: {train_metrics['accuracy']:.2f}%")
            print(f"   F1-Score: {train_metrics['f1']:.2f}%")
            print(f"   Avg Gate: {train_metrics['avg_gate']:.3f}")

            # 使用論文標準格式打印驗證指標
            print_metrics(val_metrics['metrics'], prefix="驗證集")

            # 每個情緒的 F1（詳細）
            print(f"\n📊 各情緒 F1-Score（詳細）:")
            for i, emotion in enumerate(MELD_EMOTIONS):
                cn_name = EMOTION_CN_MAP[emotion]
                f1_val = val_metrics['f1_per_class'][i] * 100
                print(f"   {cn_name:6s}: {f1_val:.2f}%")

            # 保存最佳模型
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.history['best_val_f1'] = self.best_val_f1
                self.history['best_epoch'] = epoch + 1
                self.patience_counter = 0

                best_model_path = save_path / "best_mag_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'val_acc': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'config': {
                        'model_name': self.model.model_name,
                        'num_emotions': self.model.num_emotions,
                        'hidden_dim': self.model.hidden_dim
                    }
                }, best_model_path)

                print(f"\n✅ 保存最佳模型: {best_model_path}")
                print(f"   Val F1: {val_metrics['f1']:.2f}% (↑ {val_metrics['f1'] - self.history['best_val_f1']:.2f}%)")
            else:
                self.patience_counter += 1
                print(f"\n⏸️  Val F1 未提升 ({self.patience_counter}/{self.patience})")

                if self.patience_counter >= self.patience:
                    print(f"\n🛑 早停！連續 {self.patience} 輪未提升")
                    break

        # 保存最終模型
        final_model_path = save_path / "final_mag_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, final_model_path)

        # 保存訓練歷史
        history_path = save_path / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        # 測試集評估（如果有）
        if self.test_loader:
            print("\n" + "=" * 80)
            print("🧪 測試集評估")
            print("=" * 80)
            self.evaluate_test_set(save_path)

        print("\n" + "=" * 80)
        print("✅ 訓練完成！")
        print("=" * 80)
        print(f"📊 最佳驗證 F1: {self.history['best_val_f1']:.2f}% (Epoch {self.history['best_epoch']})")
        print(f"💾 模型保存在: {save_path}")
        print("=" * 80 + "\n")

    def evaluate_test_set(self, save_path: Path):
        """在測試集上評估"""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['emotion_label'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']

                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 計算論文標準指標
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        metrics = compute_metrics(all_preds_np, all_labels_np, num_classes=7)

        # 打印論文標準指標
        print_metrics(metrics, prefix="測試集")

        # 分類報告（詳細）
        report = classification_report(
            all_labels,
            all_preds,
            target_names=MELD_EMOTIONS,
            digits=4
        )

        print("\n📊 測試集分類報告（詳細）:")
        print(report)

        # 混淆矩陣
        cm = confusion_matrix(all_labels, all_preds)

        # 保存結果（包含所有論文指標）
        test_results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            # 論文標準指標
            'Acc7': metrics['Acc7'],
            'Acc2': metrics['Acc2'],
            'F1_weighted': metrics['F1_weighted'],
            'F1_macro': metrics['F1_macro'],
            'MAE': metrics['MAE'],
            'Pearson': metrics.get('Pearson', 0.0),
            'Spearman': metrics.get('Spearman', 0.0),
            # 兼容舊格式
            'accuracy': metrics['Acc7'],
            'f1_weighted': metrics['F1_weighted']
        }

        # 保存為 JSON
        test_results_json = test_results.copy()
        test_results_json['confusion_matrix'] = cm.tolist()

        with open(save_path / "test_results.json", 'w', encoding='utf-8') as f:
            json.dump(test_results_json, f, indent=2, ensure_ascii=False)

        # 保存簡潔版本（論文表格格式）
        paper_metrics = {
            'Dataset': 'MELD',
            'Acc7': f"{metrics['Acc7']:.2f}",
            'Acc2': f"{metrics['Acc2']:.2f}",
            'F1': f"{metrics['F1_weighted']:.2f}",
            'MAE': f"{metrics['MAE']:.4f}",
            'Corr': f"{metrics.get('Pearson', 0.0):.4f}"
        }

        with open(save_path / "paper_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(paper_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 結果已保存:")
        print(f"   完整結果: {save_path / 'test_results.json'}")
        print(f"   論文指標: {save_path / 'paper_metrics.json'}")


# ========== 主訓練流程 ==========
def main():
    """主訓練流程"""
    print("\n" + "=" * 80)
    print("MAG 情緒分類訓練系統 - 純文本版本")
    print("=" * 80 + "\n")

    # ========== 配置 ==========
    config = {
        # 數據
        'data_dir': './data/meld_processed',
        'batch_size': 16,
        'max_length': 128,
        'context_window': 3,
        'use_augmentation': True,
        'augmentation_prob': 0.3,

        # 模型
        'model_name': 'bert-base-uncased',  # 可選: roberta-base, bert-large-uncased
        'hidden_dim': 768,
        'dropout': 0.3,
        'beta_shift': 1.0,
        'freeze_bert': False,

        # 訓練
        'num_epochs': 30,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'max_grad_norm': 1.0,
        'use_focal_loss': True,

        # 保存
        'save_dir': './emotion_system/models/mag',
        'num_workers': 4,
    }

    print("📋 訓練配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()

    # ========== 檢查數據 ==========
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 數據目錄不存在: {data_dir}")
        print("請先運行: python emotion_system/prepare_meld_data.py")
        return

    # ========== 載入 Tokenizer ==========
    print(f"📦 載入 Tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # ========== 創建數據集 ==========
    print("\n📊 創建數據集...")

    train_dataset = MELDEmotionDataset(
        csv_path=str(data_dir / "train_processed.csv"),
        tokenizer=tokenizer,
        max_length=config['max_length'],
        context_window=config['context_window'],
        use_augmentation=config['use_augmentation'],
        augmentation_prob=config['augmentation_prob']
    )

    val_dataset = MELDEmotionDataset(
        csv_path=str(data_dir / "dev_processed.csv"),
        tokenizer=tokenizer,
        max_length=config['max_length'],
        context_window=config['context_window'],
        use_augmentation=False
    )

    test_dataset = MELDEmotionDataset(
        csv_path=str(data_dir / "test_processed.csv"),
        tokenizer=tokenizer,
        max_length=config['max_length'],
        context_window=config['context_window'],
        use_augmentation=False
    )

    # 類別權重
    class_weights = train_dataset.class_weights
    print(f"\n⚖️  類別權重: {class_weights.numpy()}")

    # ========== 創建 DataLoader ==========
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # ========== 創建模型 ==========
    print("\n🏗️  創建 MAG 情緒分類模型...")
    model = MAGEmotionClassifier(
        model_name=config['model_name'],
        num_emotions=len(MELD_EMOTIONS),
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        beta_shift=config['beta_shift'],
        freeze_bert=config['freeze_bert']
    )

    print(f"📊 模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 可訓練參數: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ========== 創建訓練器 ==========
    print("\n🎯 創建訓練器...")
    trainer = MAGEmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        max_grad_norm=config['max_grad_norm'],
        use_focal_loss=config['use_focal_loss'],
        class_weights=class_weights
    )

    # ========== 開始訓練 ==========
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )

    print("\n✅ 訓練流程完成！")


if __name__ == "__main__":
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main()
