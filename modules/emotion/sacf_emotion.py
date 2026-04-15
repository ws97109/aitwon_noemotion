"""modules.emotion.sacf_emotion
============================
SACF 情緒偵測後端

以 emotion_system/training/scaf_final.py 的 SACFModel 作為推論引擎。
SACFModel 是多模態架構（文字 + 音訊 + 視覺），在純文字場景下
以零向量補全音訊／視覺通道，讓 DeBERTa 語言主幹主導情緒判斷。

情感分數 → 情緒標籤映射
──────────────────────────
  reg ≥  1.5  →  興奮（強度高）
  0.3 ≤ reg < 1.5  →  快樂
 -0.3 < reg < 0.3  →  平靜（強度上限 5）
 -1.5 < reg ≤ -0.3 →  焦慮
  reg ≤ -1.5  →  悲傷（強度高）

使用方式（emotion 欄位設定）：
  {
    "emotion": {
      "enabled": true,
      "sacf_model_path": "/path/to/sacf_weights.pt",
      "sacf_lang_model": "microsoft/deberta-v3-large"   // 可選，預設同上
    }
  }
"""

import sys
from pathlib import Path

import torch

# ── 將 scaf_final.py 所在目錄加入 sys.path ────────────────────────
_SCAF_DIR = Path(__file__).parent.parent.parent / "emotion_system" / "training"
if str(_SCAF_DIR) not in sys.path:
    sys.path.insert(0, str(_SCAF_DIR))

# MOSI 任務提示詞（與 scaf_final.py 保持一致）
_TASK_PROMPT = (
    "Predict the sentiment intensity (-3 to 3, negative to positive) "
    "of the following text: "
)

# 情感分數 → 情緒標籤
_SCORE_EMOTION_MAP = [
    (1.5,  "興奮"),   # reg ≥ 1.5
    (0.3,  "快樂"),   # 0.3 ≤ reg < 1.5
    (-0.3, "平靜"),   # -0.3 ≤ reg < 0.3（中性區間）
    (-1.5, "焦慮"),   # -1.5 ≤ reg < -0.3
]
_SCORE_NEG_LABEL = "悲傷"  # reg < -1.5

# 各標籤的預設原因
_DEFAULT_REASONS = {
    "興奮": "強烈正面情感",
    "快樂": "正面情感",
    "平靜": "情緒中性穩定",
    "焦慮": "輕度負面情感",
    "悲傷": "強烈負面情感",
}


def _reg_to_emotion(reg: float):
    """
    將 SACF 回歸分數 (-3 ~ +3) 映射至情緒標籤和強度。

    強度換算：abs(reg) 0→1, 1→4, 2→7, 3→10
    平靜強度上限為 5（符合 EmotionState 約定）。

    Returns:
        (label, intensity, reason): 情緒標籤、強度(1-10)、原因文字
    """
    abs_r = abs(reg)
    intensity = max(1, min(10, round(1 + abs_r * 3)))

    label = _SCORE_NEG_LABEL
    for threshold, lbl in _SCORE_EMOTION_MAP:
        if reg >= threshold:
            label = lbl
            break

    if label == "平靜":
        intensity = min(5, intensity)

    reason = _DEFAULT_REASONS.get(label, "例行活動")
    return label, intensity, reason


class SACFEmotionBackend:
    """
    SACF 推論後端，供 EmotionModel 在 "sacf" 模式下呼叫。

    支援兩種 checkpoint 格式：
      - 新格式（scaf_final.py v59+）：dict 含 model_config + model_state_dict
      - 舊格式：直接為 state_dict（向下相容）

    Public API（與 local/ollama 後端保持一致）：
        is_available() -> bool
        predict(text)  -> (label, intensity, reason)
    """

    def __init__(self, model_path: str, lang_model: str = "microsoft/deberta-v3-large"):
        self._available  = False
        self._model      = None
        self._tokenizer  = None
        # 優先使用 cuda:0（RTX PRO 6000，記憶體最大）；否則 cpu
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # audio_dim / vision_dim 與 seq_len 在載入後填入，供 predict() 使用
        self._audio_dim  = 5
        self._vision_dim = 20

        self._load(model_path, lang_model)

    # ── 載入 ─────────────────────────────────────────────────────

    def _load(self, model_path: str, lang_model_arg: str):
        try:
            from scaf_final import SACFModel
            from transformers import DebertaV2Tokenizer

            print(f"[SACFEmotionBackend] 讀取 checkpoint：{model_path}")
            # weights_only=False：checkpoint 為含 metadata 的 dict，需完整 unpickle
            checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

            # ── 解析 checkpoint ──────────────────────────────────
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # 新格式：含 model_config meta
                model_cfg   = checkpoint.get("model_config", {})
                state_dict  = checkpoint["model_state_dict"]
                seed        = checkpoint.get("seed", "?")
                version     = checkpoint.get("version", "?")
                val_acc7    = checkpoint.get("val_acc7", None)
                # lang_model 以 checkpoint 內紀錄為主，config 中的設定為備援
                lang_model  = model_cfg.get("lang_model", lang_model_arg)
                self._audio_dim  = model_cfg.get("audio_dim",  5)
                self._vision_dim = model_cfg.get("vision_dim", 20)
                audio_dim   = self._audio_dim
                vision_dim  = self._vision_dim
                modal_hidden = model_cfg.get("modal_hidden", 128)
                fusion_dim   = model_cfg.get("fusion_dim",   512)
                top_k        = model_cfg.get("top_k",         5)
                num_classes  = model_cfg.get("num_classes",   7)
                info_str = (f"version={version}, seed={seed}"
                            + (f", val_acc7={val_acc7:.2f}%" if val_acc7 else ""))
            else:
                # 舊格式：整個 checkpoint 即為 state_dict
                state_dict   = checkpoint
                lang_model   = lang_model_arg
                audio_dim    = self._audio_dim
                vision_dim   = self._vision_dim
                modal_hidden = 128
                fusion_dim   = 512
                top_k        = 5
                num_classes  = 7
                info_str     = "legacy format"

            print(f"[SACFEmotionBackend] 載入 tokenizer：{lang_model}")
            self._tokenizer = DebertaV2Tokenizer.from_pretrained(lang_model)

            print(f"[SACFEmotionBackend] 建立 SACFModel（{info_str}），裝置：{self._device}")
            # dropout=0.0：推論時完全關閉 dropout
            model = SACFModel(
                lang_model   = lang_model,
                audio_dim    = audio_dim,
                vision_dim   = vision_dim,
                modal_hidden = modal_hidden,
                fusion_dim   = fusion_dim,
                top_k        = top_k,
                num_classes  = num_classes,
                dropout      = 0.0,
            )
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            self._model     = model
            self._available = True
            print("[SACFEmotionBackend] 載入完成 ✓"
                  "（純文字模式：音訊/視覺以零向量補全）")

        except FileNotFoundError:
            print(
                f"[SACFEmotionBackend] 找不到模型權重：{model_path}\n"
                "  請先執行：python emotion_system/training/scaf_final.py\n"
                "  訓練完成後權重會自動儲存至 emotion_system/models/。"
            )
        except Exception as e:
            print(f"[SACFEmotionBackend] 載入失敗，退回主要 LLM 模式：{e}")

    # ── 推論 ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, text: str):
        """
        對純文字進行情感預測。

        音訊通道 (5-dim) 與視覺通道 (20-dim) 以零序列補全，
        模型內部的 F.normalize 會將全零向量保持為全零（nan_to_num 處理），
        情緒判斷完全由 DeBERTa 語言主幹主導。

        Returns:
            (label, intensity, reason)
        """
        # Tokenize（沿用 scaf_final.py 的 TASK_PROMPT 前綴）
        enc = self._tokenizer(
            _TASK_PROMPT + text[:800],
            add_special_tokens=True,
            max_length=80,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self._device)       # [1, 80]
        mask = enc["attention_mask"].to(self._device)  # [1, 80]

        # 零向量補全：batch=1, seq_len=1（避免 pack_padded_sequence 的空序列問題）
        # audio_dim / vision_dim 從 checkpoint 的 model_config 取得，確保與訓練一致
        audio  = torch.zeros(1, 1, self._audio_dim,  device=self._device)
        amask  = torch.ones(1,  1,                   device=self._device)
        vision = torch.zeros(1, 1, self._vision_dim, device=self._device)
        vmask  = torch.ones(1,  1,                   device=self._device)

        # 前向傳播
        _cls7, _cls2, reg = self._model(ids, mask, audio, amask, vision, vmask)
        reg_score = float(reg.item())  # 範圍約 -3.0 ~ +3.0

        label, intensity, reason = _reg_to_emotion(reg_score)
        return label, intensity, reason

    # ── 狀態查詢 ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._available
