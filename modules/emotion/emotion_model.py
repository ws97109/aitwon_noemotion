"""generative_agents.emotion.emotion_model

情緒偵測模組
====================
支援三種專用後端（當三者皆未設定時，agent.py 會自動用主要 LLM 搭配
prompt_emotion_detect 提示詞作為備援）：

  1. SACF 模型（scaf_final.py 訓練的多模態情感模型，純文字模式）
       config: {"sacf_model_path": "/path/to/sacf_weights.pt"}

  2. 本地 HuggingFace 模型（EmoLLM InternLM2.5-7B 等）
       config: {"model_path": "/path/to/emollm"}

  3. Ollama 本地服務（若已 pull 支援情緒的模型）
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

        # 平靜強度上限（無論哪個分支，平靜 intensity 不得超過 5）
        if self.label == "平靜" and self.intensity > 5:
            self.intensity = 5

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
    "重要限制：「平靜」的 intensity 最高為 5；若活動令人感到充實或愉悅，請改選「快樂」或「興奮」。"
    "reason 欄位必須填寫 15 字以內的說明，不得為空字串。\n"
    "輸出格式：{\"label\": \"<情緒標籤>\", \"intensity\": <1-10>, \"reason\": \"<15字內原因>\"}"
)


# ─────────────────────────────────────────────────────────────
# EmotionModel：負責呼叫 EmoLLM 並解析結果
# ─────────────────────────────────────────────────────────────

class EmotionModel:
    """
    專用情緒偵測後端。

    優先順序：sacf_model_path > model_path > ollama_model
    以上皆未設定時 is_available() 回傳 False，
    讓 agent.py 退回主要 LLM 路徑。
    """

    def __init__(self, config):
        self._config        = config or {}
        self._local_handle  = None
        self._ollama_url    = None
        self._ollama_model  = None
        self._sacf_backend  = None
        self._mode          = None   # "sacf" | "local" | "ollama" | None

        self._setup(self._config)

    # ── 初始化 ────────────────────────────────────────────────

    def _setup(self, config):
        sacf_path    = config.get("sacf_model_path", "").strip()
        model_path   = config.get("model_path", "").strip()
        ollama_model = config.get("ollama_model", "").strip()

        # sacf_model_path 未設定時，自動掃描 emotion_system/models/ 取最新 best checkpoint
        if not sacf_path:
            sacf_path = self._discover_sacf_checkpoint()

        if sacf_path:
            self._mode = "sacf"
            self._load_sacf(sacf_path, config.get("sacf_lang_model", "microsoft/deberta-v3-large"))
        elif model_path:
            self._mode = "local"
            self._load_local(model_path)
        elif ollama_model:
            self._mode        = "ollama"
            self._ollama_url  = config.get("base_url", "http://127.0.0.1:11434/v1").rstrip("/")
            self._ollama_model = ollama_model

    @staticmethod
    def _discover_sacf_checkpoint():
        """
        自動掃描 emotion_system/models/ 找最新的 sacf_*_best.pt 檔案。
        找到則回傳路徑字串，否則回傳空字串。
        """
        import glob
        from pathlib import Path

        # 從本模組路徑推算專案根目錄
        project_root = Path(__file__).parent.parent.parent
        pattern = str(project_root / "emotion_system" / "models" / "sacf_*_best.pt")
        candidates = sorted(glob.glob(pattern))
        if candidates:
            # 取最新（按字典序最大 = 版本號最高）
            chosen = candidates[-1]
            print(f"[EmotionModel] 自動發現 SACF checkpoint：{chosen}")
            return chosen
        return ""

    def _load_sacf(self, model_path: str, lang_model: str):
        try:
            from modules.emotion.sacf_emotion import SACFEmotionBackend
            self._sacf_backend = SACFEmotionBackend(model_path, lang_model)
            if not self._sacf_backend.is_available():
                self._mode         = None
                self._sacf_backend = None
        except Exception as e:
            print(f"[EmotionModel] SACF 後端初始化失敗，退回主要 LLM 模式：{e}")
            self._mode         = None
            self._sacf_backend = None

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
            gen_kwargs = dict(
                max_new_tokens=80,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            # transformers >= 4.44 預設使用 DynamicCache，
            # 部分舊版 InternLM 模型不相容，強制停用 KV cache 以確保相容
            try:
                output_ids = model.generate(input_ids, **gen_kwargs)
            except (AttributeError, TypeError):
                gen_kwargs["use_cache"] = False
                output_ids = model.generate(input_ids, **gen_kwargs)
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

    # 各情緒標籤的預設原因（當模型未輸出原因時使用）
    _DEFAULT_REASONS = {
        "快樂": "正向活動帶來滿足感",
        "悲傷": "感受到失落或低落",
        "憤怒": "遇到令人不滿的情況",
        "恐懼": "感受到威脅或不安",
        "厭惡": "遇到令人反感的事",
        "驚訝": "遭遇出乎意料的事",
        "平靜": "例行活動，情緒穩定",
        "焦慮": "對未來感到不確定",
        "興奮": "對事物充滿期待",
        "疲憊": "消耗精力，感到疲倦",
    }

    @staticmethod
    def _parse(response):
        if not response:
            return EmotionState()
        try:
            match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if match:
                data  = json.loads(match.group())
                label = data.get("label",     EmotionState.DEFAULT)
                intens = int(data.get("intensity", 5))
                reason = str(data.get("reason", "")).strip()
                # 平靜的強度不應超過 5（極度平靜在日常生活中幾乎不存在；
                # 若模型給出高強度平靜，通常是誤判，應重新解讀為低強度正面情緒）
                if label == "平靜" and intens > 5:
                    intens = 5
                # reason 不得為空：用預設說明補足
                if not reason:
                    reason = EmotionModel._DEFAULT_REASONS.get(label, "例行活動")
                return EmotionState(label=label, intensity=intens, reason=reason)
        except Exception:
            pass
        # 備援：關鍵字掃描
        for label in EmotionState.LABELS:
            if label in response:
                default_reason = EmotionModel._DEFAULT_REASONS.get(label, "例行活動")
                return EmotionState(label, 5, default_reason)
        return EmotionState(reason="無法解析情緒")

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

        # ── SACF 模式：直接推論，不需 JSON 解析 ─────────────────
        if self._mode == "sacf":
            try:
                label, intensity, reason = self._sacf_backend.predict(text)
                return EmotionState(label=label, intensity=intensity, reason=reason)
            except Exception as e:
                print(f"[EmotionModel] SACF 情緒偵測失敗：{e}")
                return EmotionState(reason=self._DEFAULT_REASONS.get(EmotionState.DEFAULT, "例行活動"))

        # ── Local / Ollama 模式：需建立提示詞並解析 JSON ─────────
        agent_str   = f"角色：{agent_name}\n" if agent_name else ""
        context_str = f"背景：{context}\n"    if context    else ""
        user_prompt = (
            f"{agent_str}{context_str}"
            f"請分析以下文字的情緒狀態：\n「{text[:800]}」"
        )

        raw = ""
        try:
            if self._mode == "local":
                raw = self._call_local(user_prompt)
            elif self._mode == "ollama":
                raw = self._call_ollama(user_prompt)
            else:
                return EmotionState(reason=self._DEFAULT_REASONS.get(EmotionState.DEFAULT, "例行活動"))
        except Exception as e:
            print(f"[EmotionModel] 情緒偵測呼叫失敗：{e}")

        result = self._parse(raw)
        # 確保 reason 不為空：用預設說明補足（含呼叫失敗的備援路徑）
        if not result.reason:
            result.reason = self._DEFAULT_REASONS.get(result.label, "例行活動")
        return result

    def is_available(self):
        """只有在專用後端（SACF/本地/Ollama）正確設定時才回傳 True"""
        if self._mode == "sacf":
            return self._sacf_backend is not None and self._sacf_backend.is_available()
        if self._mode == "local":
            return self._local_handle is not None
        if self._mode == "ollama":
            return bool(self._ollama_model)
        return False
