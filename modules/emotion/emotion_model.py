"""generative_agents.emotion.emotion_model

EmoLLM 情緒偵測模組
====================
支援兩種專用後端（當兩者皆未設定時，agent.py 會自動用主要 LLM 搭配
prompt_emotion_detect 提示詞作為備援）：

  1. 本地 HuggingFace 模型（EmoLLM InternLM2.5-7B 等）
       config: {"model_path": "/path/to/emollm"}

  2. Ollama 本地服務（若已 pull 支援情緒的模型）
       config: {"ollama_model": "emollm:latest", "base_url": "..."}

情緒標籤（對應 EmoLLM CPsyCounD 資料集類別，繁體中文）：
  快樂、悲傷、憤怒、恐懼、厭惡、驚訝、平靜、焦慮、興奮、疲憊
"""

import re
import json
import requests


# ─────────────────────────────────────────────────────────────
# EmotionState：情緒狀態資料物件（含慣性融合機制）
#
# 情緒傳遞邏輯：
#   情緒不會瞬間切換。每次 EmoLLM 偵測到新情緒後，
#   會與目前的情緒做「加權融合」，由慣性係數 MOMENTUM 控制：
#
#   ┌─────────────────────────────────────────────────────────┐
#   │  新情緒強度低（1-3）→ 幾乎不影響當前情緒（衰減為主）   │
#   │  新情緒強度中（4-6）→ 漸進過渡，2-3 輪才完成切換       │
#   │  新情緒強度高（7-10）→ 較快切換，但仍帶有舊情緒殘留     │
#   │  新舊情緒相同 → 直接強化，無需切換                     │
#   └─────────────────────────────────────────────────────────┘
#
#   例：難過(8) → 快樂(3) → 當前仍為難過，強度降至 6-7
#   例：難過(8) → 快樂(9) → 漸進切換，當前呈「難過帶有快樂」
#   例：難過(8) → 難過(6) → 難過強化或微幅下降
# ─────────────────────────────────────────────────────────────

class EmotionState:
    LABELS   = ["快樂", "悲傷", "憤怒", "恐懼", "厭惡", "驚訝", "平靜", "焦慮", "興奮", "疲憊"]
    DEFAULT  = "平靜"
    MOMENTUM = 0.65   # 情緒慣性係數（0=完全跟隨新情緒，1=完全不變）

    def __init__(self, label="平靜", intensity=3, reason=""):
        self.label     = label if label in self.LABELS else self.DEFAULT
        self.intensity = max(1, min(10, int(intensity)))
        self.reason    = str(reason)
        self._prev_label = None   # 上一個情緒標籤（用於描述過渡狀態）
        self._history    = []     # 最近 5 次原始偵測結果

    # ── 核心：帶慣性的情緒融合 ────────────────────────────────

    def blend_update(self, new_label, new_intensity, new_reason):
        """
        將 EmoLLM 偵測到的新情緒與當前情緒做慣性融合。
        直接修改 self，不回傳新物件。
        """
        new_label     = new_label if new_label in self.LABELS else self.DEFAULT
        new_intensity = max(1, min(10, int(new_intensity)))

        # 保存歷史（最多 5 筆）
        self._history.append({"label": self.label, "intensity": self.intensity})
        if len(self._history) > 5:
            self._history = self._history[-5:]

        prev_label     = self.label
        prev_intensity = self.intensity

        if new_label == self.label:
            # ── 同一情緒：強度加權平均（漸進強化或衰減）
            blended = self.MOMENTUM * prev_intensity + (1 - self.MOMENTUM) * new_intensity
            self.intensity = max(1, min(10, round(blended)))
            self.reason    = new_reason
            self._prev_label = None
        else:
            # ── 不同情緒：依新情緒強度決定切換速度
            #   switch_rate：本次切換的比例（0.0 ~ 0.6）
            switch_rate = (new_intensity / 10.0) * (1 - self.MOMENTUM)

            if switch_rate >= 0.28:
                # 高強度（≈ 7-10）：切換標籤，強度快速過渡
                blended = (1 - switch_rate) * prev_intensity + switch_rate * new_intensity
                self.label     = new_label
                self.intensity = max(1, min(10, round(blended)))
                self.reason    = new_reason
                self._prev_label = prev_label   # 記錄過渡來源
            elif switch_rate >= 0.14:
                # 中強度（≈ 4-6）：標籤切換但強度保守融合
                blended = self.MOMENTUM * prev_intensity * 0.7 + new_intensity * 0.3
                self.label     = new_label
                self.intensity = max(1, min(10, round(blended)))
                self.reason    = new_reason
                self._prev_label = prev_label
            else:
                # 低強度（≈ 1-3）：忽略標籤切換，僅讓舊情緒自然衰減
                self.intensity = max(1, round(prev_intensity * self.MOMENTUM))
                self.reason    = self.reason   # 保留原因不變
                self._prev_label = None

    # ── 描述（注入提示詞用）───────────────────────────────────

    def describe(self):
        base = f"{self.label}（強度：{self.intensity}/10）"
        if self._prev_label and self._prev_label != self.label:
            return f"{base}，由{self._prev_label}轉變中"
        return base

    # ── 序列化（checkpoint 持久化）────────────────────────────

    def to_dict(self):
        return {
            "label":      self.label,
            "intensity":  self.intensity,
            "reason":     self.reason,
            "prev_label": self._prev_label,
            "history":    self._history,
        }

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        obj              = cls(
            label     = d.get("label",     cls.DEFAULT),
            intensity = d.get("intensity", 3),
            reason    = d.get("reason",    ""),
        )
        obj._prev_label  = d.get("prev_label")
        obj._history     = d.get("history", [])
        return obj


# ─────────────────────────────────────────────────────────────
# EmoLLM 系統提示詞（讓心理諮詢模型輸出結構化情緒 JSON）
# ─────────────────────────────────────────────────────────────

_EMOTION_SYSTEM_PROMPT = (
    "你是一位專業的情緒心理分析師。"
    "你的任務是分析用戶提供的文字中所蘊含的情緒狀態，"
    "並以嚴格的 JSON 格式輸出分析結果，不得加入任何額外說明。\n"
    "可選情緒類別：快樂、悲傷、憤怒、恐懼、厭惡、驚訝、平靜、焦慮、興奮、疲憊\n"
    "輸出格式：{\"label\": \"<情緒標籤>\", \"intensity\": <1-10>, \"reason\": \"<15字內原因>\"}"
)


# ─────────────────────────────────────────────────────────────
# EmotionModel：負責呼叫 EmoLLM 並解析結果
# ─────────────────────────────────────────────────────────────

class EmotionModel:
    """
    專用情緒偵測後端。

    當 model_path 或 ollama_model 有效時 is_available() 回傳 True；
    否則回傳 False，讓 agent.py 退回主要 LLM 路徑。
    """

    def __init__(self, config):
        self._config       = config or {}
        self._local_handle = None
        self._ollama_url   = None
        self._ollama_model = None
        self._mode         = None   # "local" | "ollama" | None

        self._setup(self._config)

    # ── 初始化 ────────────────────────────────────────────────

    def _setup(self, config):
        model_path   = config.get("model_path", "").strip()
        ollama_model = config.get("ollama_model", "").strip()

        if model_path:
            self._mode = "local"
            self._load_local(model_path)
        elif ollama_model:
            self._mode        = "ollama"
            self._ollama_url  = config.get("base_url", "http://127.0.0.1:11434/v1").rstrip("/")
            self._ollama_model = ollama_model

    def _load_local(self, model_path):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32
            print(f"[EmotionModel] 載入 EmoLLM：{model_path}，裝置：{device}")

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                model = model.to(device)
            model.eval()

            self._local_handle = {"tokenizer": tokenizer, "model": model, "device": device}
            print("[EmotionModel] EmoLLM 載入完成 ✓")
        except Exception as e:
            print(f"[EmotionModel] 本地模型載入失敗，切換為主要 LLM 模式：{e}")
            self._mode         = None
            self._local_handle = None

    # ── 後端呼叫 ──────────────────────────────────────────────

    def _call_local(self, user_prompt):
        import torch
        tokenizer = self._local_handle["tokenizer"]
        model     = self._local_handle["model"]
        device    = self._local_handle["device"]

        # EmoLLM (InternLM2.5) 使用 system + user 訊息
        messages = [
            {"role": "system", "content": _EMOTION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            text      = f"{_EMOTION_SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=80,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _call_ollama(self, user_prompt):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self._ollama_model,
            "messages": [
                {"role": "system", "content": _EMOTION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.3,
            "stream":      False,
        }
        resp = requests.post(
            f"{self._ollama_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # 移除 Qwen3 等模型的 <think> 區塊
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()

    # ── 解析 ──────────────────────────────────────────────────

    @staticmethod
    def _parse(response):
        if not response:
            return EmotionState()
        try:
            match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return EmotionState(
                    label     = data.get("label",     EmotionState.DEFAULT),
                    intensity = data.get("intensity", 5),
                    reason    = data.get("reason",    ""),
                )
        except Exception:
            pass
        # 備援：關鍵字掃描
        for label in EmotionState.LABELS:
            if label in response:
                return EmotionState(label, 5, "")
        return EmotionState()

    # ── 公開介面 ──────────────────────────────────────────────

    def detect(self, text, agent_name="", context=""):
        """
        偵測文字中的情緒，回傳 EmotionState。

        Args:
            text:       行動描述或對話內容
            agent_name: 代理人姓名（提供上下文）
            context:    額外背景
        """
        if not text or not text.strip():
            return EmotionState()

        agent_str   = f"角色：{agent_name}\n" if agent_name else ""
        context_str = f"背景：{context}\n"    if context    else ""
        user_prompt = (
            f"{agent_str}{context_str}"
            f"請分析以下文字的情緒狀態：\n「{text[:800]}」"
        )

        try:
            if self._mode == "local":
                raw = self._call_local(user_prompt)
            elif self._mode == "ollama":
                raw = self._call_ollama(user_prompt)
            else:
                return EmotionState()
        except Exception as e:
            print(f"[EmotionModel] 情緒偵測呼叫失敗：{e}")
            return EmotionState()

        return self._parse(raw)

    def is_available(self):
        """只有在專用後端（本地/Ollama）正確設定時才回傳 True"""
        if self._mode == "local":
            return self._local_handle is not None
        if self._mode == "ollama":
            return bool(self._ollama_model)
        return False
