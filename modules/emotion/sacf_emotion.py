"""modules.emotion.sacf_emotion
============================
SACF Ensemble 情緒偵測後端

以 emotion_system/training/scaf_final.py 的 SACFModel 作為推論引擎。
自動掃描同版本的所有 seed 模型，載入為 Ensemble 進行推論
（regression 輸出取平均），確保部署效能 = 訓練報告的 Ensemble 指標。

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
      "sacf_lang_model": "microsoft/deberta-v3-large"
    }
  }
"""

import sys
import re as _re
import glob as _glob
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
    SACF Ensemble 推論後端，供 EmotionModel 在 "sacf" 模式下呼叫。

    自動掃描同版本的所有 seed checkpoint，載入多模型 Ensemble。
    推論時對所有 seed 模型的 regression 輸出取平均，
    確保部署效能與訓練報告的 Ensemble 指標一致。

    Public API（與 local/ollama 後端保持一致）：
        is_available() -> bool
        predict(text)  -> (label, intensity, reason)
    """

    def __init__(self, model_path: str, lang_model: str = "microsoft/deberta-v3-large"):
        self._available  = False
        self._models     = []       # 所有 seed 模型
        self._tokenizer  = None
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._audio_dim  = 5
        self._vision_dim = 20

        self._load_ensemble(model_path, lang_model)

    # ── Ensemble 掃描與載入 ──────────────────────────────────────

    @staticmethod
    def _discover_seed_models(model_path: str):
        """從給定的 checkpoint 路徑推斷版本號，掃描同版本所有 seed 模型。"""
        p = Path(model_path)
        match = _re.search(r'sacf_(v\d+)_', p.name)
        if match:
            version = match.group(1)
            pattern = str(p.parent / f"sacf_{version}_seed*.pt")
            seeds = sorted(_glob.glob(pattern))
            if seeds:
                return seeds
        # 備援：僅使用給定路徑
        return [model_path]

    def _load_ensemble(self, model_path: str, lang_model_arg: str):
        try:
            from scaf_final import SACFFinalModel as SACFModel
            from transformers import DebertaV2Tokenizer

            seed_paths = self._discover_seed_models(model_path)
            n_seeds = len(seed_paths)
            print(f"[SACFEmotionBackend] 發現 {n_seeds} 個 seed 模型 → Ensemble 推論")

            # 從第一個 checkpoint 取得模型架構設定
            first_ckpt = torch.load(seed_paths[0], map_location=self._device, weights_only=False)
            if isinstance(first_ckpt, dict) and "model_config" in first_ckpt:
                model_cfg    = first_ckpt.get("model_config", {})
                lang_model   = model_cfg.get("lang_model", lang_model_arg)
                self._audio_dim  = model_cfg.get("audio_dim",  5)
                self._vision_dim = model_cfg.get("vision_dim", 20)
                audio_dim    = self._audio_dim
                vision_dim   = self._vision_dim
                modal_hidden = model_cfg.get("modal_hidden", 128)
                fusion_dim   = model_cfg.get("fusion_dim",   512)
                top_k        = model_cfg.get("top_k",         5)
                num_classes  = model_cfg.get("num_classes",   7)
                dropout      = model_cfg.get("dropout",      0.15)
                num_branches = model_cfg.get("num_branches",  4)
            else:
                lang_model   = lang_model_arg
                audio_dim    = self._audio_dim
                vision_dim   = self._vision_dim
                modal_hidden = 128
                fusion_dim   = 512
                top_k        = 5
                num_classes  = 7
                dropout      = 0.15
                num_branches = 4

            print(f"[SACFEmotionBackend] 載入 tokenizer：{lang_model}")
            self._tokenizer = DebertaV2Tokenizer.from_pretrained(lang_model)

            for path in seed_paths:
                ckpt = torch.load(path, map_location=self._device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                    seed_id    = ckpt.get("seed", "?")
                else:
                    state_dict = ckpt
                    seed_id    = "?"

                model = SACFModel(
                    lang_model   = lang_model,
                    audio_dim    = audio_dim,
                    vision_dim   = vision_dim,
                    modal_hidden = modal_hidden,
                    fusion_dim   = fusion_dim,
                    top_k        = top_k,
                    num_classes  = num_classes,
                    dropout      = dropout,
                    num_branches = num_branches,
                )
                model.load_state_dict(state_dict)
                model.to(self._device)
                model.eval()
                self._models.append(model)
                print(f"  Seed {seed_id}: {Path(path).name}")

            self._available = len(self._models) > 0
            print(f"[SACFEmotionBackend] Ensemble 載入完成 ✓"
                  f"（{len(self._models)} 模型，純文字模式，"
                  f"num_branches={num_branches}）")

        except FileNotFoundError:
            print(
                f"[SACFEmotionBackend] 找不到模型權重：{model_path}\n"
                "  請先執行：python emotion_system/training/scaf_final.py\n"
                "  訓練完成後權重會自動儲存至 emotion_system/models/。"
            )
        except Exception as e:
            print(f"[SACFEmotionBackend] 載入失敗，退回主要 LLM 模式：{e}")

    # ── Ensemble 推論 ────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, text: str):
        """
        Ensemble 推論：對純文字進行情感預測。

        每個 seed 模型各自前向傳播（音訊/視覺以零向量補全），
        取 regression 輸出的平均值後映射至情緒標籤。

        Returns:
            (label, intensity, reason)
        """
        enc = self._tokenizer(
            _TASK_PROMPT + text[:800],
            add_special_tokens=True,
            max_length=80,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self._device)
        mask = enc["attention_mask"].to(self._device)

        audio  = torch.zeros(1, 1, self._audio_dim,  device=self._device)
        amask  = torch.ones(1,  1,                   device=self._device)
        vision = torch.zeros(1, 1, self._vision_dim, device=self._device)
        vmask  = torch.ones(1,  1,                   device=self._device)

        reg_sum = 0.0
        for model in self._models:
            _cls7, _cls2, reg = model(ids, mask, audio, amask, vision, vmask)
            reg_sum += reg.item()
        avg_reg = reg_sum / len(self._models)

        label, intensity, reason = _reg_to_emotion(avg_reg)
        return label, intensity, reason

    # ── 狀態查詢 ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._available
