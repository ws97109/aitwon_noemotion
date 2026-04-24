"""
CMU-MOSI 單模態和多模態模型

實現論文比較中的不同網絡架構:
1. Text-only Network (僅文本)
2. Visual-only Network (僅視覺)
3. Audio-only Network (僅音頻)
4. Multimodal Fusion Network (多模態融合)
   - Early Fusion (早期融合)
   - Late Fusion (後期融合)
   - MAG Fusion (門控自適應融合)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


# ========== 單模態網絡基類 ==========
class UnimodalNetwork(nn.Module):
    """單模態網絡基類"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 1,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Args:
            input_dim: 輸入特徵維度
            hidden_dims: 隱藏層維度列表
            output_dim: 輸出維度（回歸任務為1）
            dropout: Dropout 率
            activation: 激活函數 ('relu', 'tanh', 'gelu')
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 選擇激活函數
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # 構建網絡層
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 輸出層
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # 初始化權重
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入特徵 [batch_size, input_dim]

        Returns:
            輸出 [batch_size, output_dim]
        """
        return self.network(x)


# ========== Text-only Network ==========
class TextOnlyNetwork(UnimodalNetwork):
    """僅文本網絡

    輸入: GloVe 詞嵌入 (300維)
    """

    def __init__(
        self,
        text_dim: int = 300,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 1,
        dropout: float = 0.3
    ):
        super().__init__(
            input_dim=text_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation='relu'
        )
        self.modality = 'text'


# ========== Visual-only Network ==========
class VisualOnlyNetwork(UnimodalNetwork):
    """僅視覺網絡

    輸入: FACET 面部表情特徵 (47維)
    """

    def __init__(
        self,
        visual_dim: int = 47,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.3
    ):
        super().__init__(
            input_dim=visual_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation='tanh'  # 視覺特徵用 tanh
        )
        self.modality = 'visual'


# ========== Audio-only Network ==========
class AudioOnlyNetwork(UnimodalNetwork):
    """僅音頻網絡

    輸入: COVAREP 聲學特徵 (74維)
    """

    def __init__(
        self,
        audio_dim: int = 74,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.3
    ):
        super().__init__(
            input_dim=audio_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation='relu'
        )
        self.modality = 'audio'


# ========== Early Fusion Network ==========
class EarlyFusionNetwork(nn.Module):
    """早期融合網絡

    將所有模態的特徵在輸入層連接後進行處理
    """

    def __init__(
        self,
        text_dim: int = 300,
        visual_dim: int = 47,
        audio_dim: int = 74,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()

        # 總輸入維度 = 所有模態特徵拼接
        total_dim = text_dim + visual_dim + audio_dim

        # 共享網絡
        layers = []
        prev_dim = total_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            text: 文本特徵 [batch_size, text_dim]
            visual: 視覺特徵 [batch_size, visual_dim]
            audio: 音頻特徵 [batch_size, audio_dim]

        Returns:
            輸出 [batch_size, output_dim]
        """
        # 早期融合：直接拼接
        fused = torch.cat([text, visual, audio], dim=-1)
        return self.network(fused)


# ========== Late Fusion Network ==========
class LateFusionNetwork(nn.Module):
    """後期融合網絡

    每個模態獨立處理，然後融合各自的輸出
    """

    def __init__(
        self,
        text_dim: int = 300,
        visual_dim: int = 47,
        audio_dim: int = 74,
        hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.3,
        fusion_method: str = 'concat'  # 'concat', 'average', 'weighted'
    ):
        """
        Args:
            fusion_method: 融合方法
                - 'concat': 拼接後通過FC層
                - 'average': 簡單平均
                - 'weighted': 加權平均（可學習權重）
        """
        super().__init__()

        self.fusion_method = fusion_method

        # 每個模態的獨立網絡
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.visual_net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Tanh()
        )

        self.audio_net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )

        # 融合層
        if fusion_method == 'concat':
            fusion_input_dim = (hidden_dim // 2) * 3
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        elif fusion_method == 'weighted':
            # 可學習的權重
            self.modality_weights = nn.Parameter(torch.ones(3) / 3.0)
            self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        else:  # average
            self.output_layer = nn.Linear(hidden_dim // 2, output_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            text: 文本特徵 [batch_size, text_dim]
            visual: 視覺特徵 [batch_size, visual_dim]
            audio: 音頻特徵 [batch_size, audio_dim]

        Returns:
            輸出 [batch_size, output_dim]
        """
        # 各模態獨立處理
        text_repr = self.text_net(text)      # [batch_size, hidden_dim//2]
        visual_repr = self.visual_net(visual)  # [batch_size, hidden_dim//2]
        audio_repr = self.audio_net(audio)    # [batch_size, hidden_dim//2]

        # 後期融合
        if self.fusion_method == 'concat':
            fused = torch.cat([text_repr, visual_repr, audio_repr], dim=-1)
            output = self.fusion_layer(fused)

        elif self.fusion_method == 'weighted':
            # 加權平均（可學習權重）
            weights = F.softmax(self.modality_weights, dim=0)
            fused = (
                weights[0] * text_repr +
                weights[1] * visual_repr +
                weights[2] * audio_repr
            )
            output = self.output_layer(fused)

        else:  # average
            # 簡單平均
            fused = (text_repr + visual_repr + audio_repr) / 3.0
            output = self.output_layer(fused)

        return output


# ========== MAG Fusion Network ==========
class MAGFusionNetwork(nn.Module):
    """MAG (Multimodal Adaptation Gate) 融合網絡

    論文: "Integrating Multimodal Information in Large Pretrained Transformers"

    核心思想: 使用門控機制動態調節非語言信息注入到語言模型
    """

    def __init__(
        self,
        text_dim: int = 300,
        visual_dim: int = 47,
        audio_dim: int = 74,
        hidden_dim: int = 128,
        output_dim: int = 1,
        beta_shift: float = 1.0,
        dropout: float = 0.3
    ):
        super().__init__()

        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.beta_shift = beta_shift

        # 文本編碼器（主要模態）
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 視覺編碼器
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # 音頻編碼器
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # MAG 視覺門控
        self.mag_visual = MAGModule(
            text_dim=hidden_dim,
            aux_dim=hidden_dim,
            output_dim=hidden_dim,
            beta_shift=beta_shift,
            dropout=dropout
        )

        # MAG 音頻門控
        self.mag_audio = MAGModule(
            text_dim=hidden_dim,
            aux_dim=hidden_dim,
            output_dim=hidden_dim,
            beta_shift=beta_shift,
            dropout=dropout
        )

        # 輸出層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
        audio: torch.Tensor,
        return_gates: bool = False
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            text: 文本特徵 [batch_size, text_dim]
            visual: 視覺特徵 [batch_size, visual_dim]
            audio: 音頻特徵 [batch_size, audio_dim]
            return_gates: 是否返回門控值

        Returns:
            輸出 [batch_size, output_dim]
            或 (輸出, 門控值字典) if return_gates=True
        """
        # 1. 編碼各模態
        text_h = self.text_encoder(text)       # [batch_size, hidden_dim]
        visual_h = self.visual_encoder(visual)  # [batch_size, hidden_dim]
        audio_h = self.audio_encoder(audio)    # [batch_size, hidden_dim]

        # 2. MAG 門控融合視覺信息
        text_with_visual, visual_gate = self.mag_visual(text_h, visual_h)

        # 3. MAG 門控融合音頻信息
        fused_repr, audio_gate = self.mag_audio(text_with_visual, audio_h)

        # 4. 輸出層
        output = self.output_layer(fused_repr)

        if return_gates:
            gates = {
                'visual_gate': visual_gate.mean().item(),
                'audio_gate': audio_gate.mean().item()
            }
            return output, gates

        return output


class MAGModule(nn.Module):
    """MAG 門控模組"""

    def __init__(
        self,
        text_dim: int,
        aux_dim: int,
        output_dim: int,
        beta_shift: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.text_dim = text_dim
        self.aux_dim = aux_dim
        self.output_dim = output_dim
        self.beta_shift = beta_shift

        # 融合層
        self.W_m = nn.Linear(text_dim + aux_dim, output_dim, bias=True)

        # 門控層
        self.W_g = nn.Linear(text_dim + aux_dim, output_dim, bias=True)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.xavier_uniform_(self.W_m.weight)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_m.bias)
        nn.init.zeros_(self.W_g.bias)

    def forward(
        self,
        text_h: torch.Tensor,
        aux_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAG 前向傳播

        Args:
            text_h: 文本表徵 [batch_size, text_dim]
            aux_h: 輔助表徵 [batch_size, aux_dim]

        Returns:
            (output, gate_values)
        """
        # 1. 拼接
        concat_h = torch.cat([text_h, aux_h], dim=-1)

        # 2. 融合變換
        H_m = torch.tanh(self.W_m(concat_h))

        # 3. 門控
        gate = torch.sigmoid(self.W_g(concat_h)) * self.beta_shift

        # 4. 門控融合
        H_m_gated = gate * H_m

        # 5. 殘差連接
        output = text_h + H_m_gated

        # 6. Layer Norm + Dropout
        output = self.layer_norm(output)
        output = self.dropout(output)

        # 平均門控值
        avg_gate = gate.mean(dim=-1)

        return output, avg_gate


# ========== 模型工廠 ==========
def create_model(
    model_type: str,
    text_dim: int = 300,
    visual_dim: int = 47,
    audio_dim: int = 74,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    **kwargs
) -> nn.Module:
    """
    創建模型

    Args:
        model_type: 模型類型
            - 'text_only': 僅文本
            - 'visual_only': 僅視覺
            - 'audio_only': 僅音頻
            - 'early_fusion': 早期融合
            - 'late_fusion': 後期融合
            - 'mag_fusion': MAG 融合

    Returns:
        模型實例
    """
    if model_type == 'text_only':
        return TextOnlyNetwork(
            text_dim=text_dim,
            hidden_dims=[256, 128, 64],
            dropout=dropout
        )

    elif model_type == 'visual_only':
        return VisualOnlyNetwork(
            visual_dim=visual_dim,
            hidden_dims=[64, 32],
            dropout=dropout
        )

    elif model_type == 'audio_only':
        return AudioOnlyNetwork(
            audio_dim=audio_dim,
            hidden_dims=[64, 32],
            dropout=dropout
        )

    elif model_type == 'early_fusion':
        return EarlyFusionNetwork(
            text_dim=text_dim,
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            hidden_dims=[256, 128, 64],
            dropout=dropout
        )

    elif model_type == 'late_fusion':
        fusion_method = kwargs.get('fusion_method', 'concat')
        return LateFusionNetwork(
            text_dim=text_dim,
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            fusion_method=fusion_method
        )

    elif model_type == 'mag_fusion':
        return MAGFusionNetwork(
            text_dim=text_dim,
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            beta_shift=kwargs.get('beta_shift', 1.0),
            dropout=dropout
        )

    else:
        raise ValueError(f"未知的模型類型: {model_type}")


if __name__ == "__main__":
    """測試模型"""
    print("=" * 80)
    print("CMU-MOSI 模型測試")
    print("=" * 80)

    batch_size = 4
    text_dim = 300
    visual_dim = 47
    audio_dim = 74

    # 生成假數據
    text = torch.randn(batch_size, text_dim)
    visual = torch.randn(batch_size, visual_dim)
    audio = torch.randn(batch_size, audio_dim)

    models_to_test = [
        'text_only',
        'visual_only',
        'audio_only',
        'early_fusion',
        'late_fusion',
        'mag_fusion'
    ]

    for model_type in models_to_test:
        print(f"\n測試 {model_type}...")

        model = create_model(model_type, text_dim, visual_dim, audio_dim)

        # 前向傳播
        if model_type in ['text_only']:
            output = model(text)
        elif model_type in ['visual_only']:
            output = model(visual)
        elif model_type in ['audio_only']:
            output = model(audio)
        else:
            output = model(text, visual, audio)

        print(f"   輸出形狀: {output.shape}")
        print(f"   參數量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✅ 測試完成！")
