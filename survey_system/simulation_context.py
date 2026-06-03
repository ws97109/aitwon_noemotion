"""
SimulationContext — 模擬世界資料的統一存取層

把同一份模擬產出的三類資料整合在一起，供問卷生成使用：
  1. simulation.md（活動歷史摘要，由 compress.py 產出）
  2. 最新 simulate-*.json checkpoint（情緒狀態、行動、排程）
  3. storage/{agent}/associate/（LlamaIndex 向量記憶庫）

為避免影響模擬資料，記憶查詢直接走 LlamaIndex（read-only），
不經過 modules.memory.Associate 的 cleanup_index() 路徑。
"""

import os
import sys
import json
import warnings
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any

# 加入專案根目錄以便載入 modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class SimulationContext:
    """封裝單一 simulation 名稱對應的所有資料來源。"""

    def __init__(self, simulation_name: str, project_root: Optional[Path] = None):
        self.simulation_name = simulation_name
        self.project_root = Path(project_root) if project_root else _PROJECT_ROOT

        # 主要路徑
        self.compressed_folder = self.project_root / "results" / "compressed" / simulation_name
        self.checkpoints_folder = self.project_root / "results" / "checkpoints" / simulation_name
        self.simulation_md_path = self.compressed_folder / "simulation.md"
        self.storage_root = self.checkpoints_folder / "storage"

        # 解析資料
        self._activity_history: Dict[str, str] = self._load_activity_history()
        self._latest_checkpoint: Dict[str, Any] = self._load_latest_checkpoint()
        self._embedding_config: Optional[Dict[str, Any]] = self._extract_embedding_config()

        # Associate 索引快取：{agent_name: LlamaIndex or False}
        self._index_cache: Dict[str, Any] = {}
        self._embedding_available: Optional[bool] = None  # 延遲檢查

    # ------------------------------------------------------------
    # 列出可用的 simulation
    # ------------------------------------------------------------
    @staticmethod
    def list_available(project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
        root = Path(project_root) if project_root else _PROJECT_ROOT
        compressed_dir = root / "results" / "compressed"
        checkpoints_dir = root / "results" / "checkpoints"

        names = set()
        if compressed_dir.exists():
            for p in compressed_dir.iterdir():
                if p.is_dir() and (p / "simulation.md").exists():
                    names.add(p.name)
        if checkpoints_dir.exists():
            for p in checkpoints_dir.iterdir():
                if p.is_dir():
                    names.add(p.name)

        result = []
        for name in sorted(names):
            md_exists = (compressed_dir / name / "simulation.md").exists()
            ckpt_dir = checkpoints_dir / name
            has_storage = (ckpt_dir / "storage").exists()
            has_checkpoint = ckpt_dir.exists() and any(
                f.name.startswith("simulate-") for f in ckpt_dir.iterdir()
            ) if ckpt_dir.exists() else False
            result.append({
                "name": name,
                "has_simulation_md": md_exists,
                "has_checkpoint": has_checkpoint,
                "has_associate": has_storage,
            })
        return result

    # ------------------------------------------------------------
    # simulation.md 解析
    # ------------------------------------------------------------
    def _load_activity_history(self) -> Dict[str, str]:
        if not self.simulation_md_path.exists():
            return {}

        try:
            content = self.simulation_md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[SimulationContext] 讀取 simulation.md 失敗: {e}")
            return {}

        history: Dict[str, List[str]] = {}
        current = None
        for line in content.splitlines():
            stripped = line.strip()
            if line.startswith("### "):
                header = line.replace("### ", "").strip()
                # 跳過對話標題（含 " -> "），由下方第二個迴圈處理
                if " -> " in header:
                    current = None
                    continue
                current = header
                history.setdefault(current, [])
            elif line.startswith("## ") or line.startswith("# "):
                # 時間戳或大標題
                current = None
            elif current and stripped:
                history[current].append(stripped)

        # 也解析對話紀錄：「### A -> B @ location」格式
        # 將對話內容歸給雙方
        in_conversation = False
        conv_participants: list = []
        conv_lines: list = []
        for line in content.splitlines():
            stripped = line.strip()
            if line.startswith("### ") and " -> " in line:
                # 儲存上一段對話
                if conv_participants and conv_lines:
                    conv_text = "\n".join(conv_lines)
                    for p in conv_participants:
                        history.setdefault(p, [])
                        history[p].append(conv_text)
                # 解析新對話參與者
                header = line.replace("### ", "").strip()
                parts = header.split(" @ ")[0]
                conv_participants = [p.strip() for p in parts.split(" -> ")]
                conv_lines = [f"[對話] {header}"]
                in_conversation = True
            elif in_conversation and stripped:
                if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
                    # 對話結束
                    if conv_participants and conv_lines:
                        conv_text = "\n".join(conv_lines)
                        for p in conv_participants:
                            history.setdefault(p, [])
                            history[p].append(conv_text)
                    conv_participants = []
                    conv_lines = []
                    in_conversation = False
                else:
                    conv_lines.append(stripped)
        # 最後一段對話
        if conv_participants and conv_lines:
            conv_text = "\n".join(conv_lines)
            for p in conv_participants:
                history.setdefault(p, [])
                history[p].append(conv_text)

        # 去重並只保留最後 200 條
        result: Dict[str, str] = {}
        for name, lines in history.items():
            if lines:
                # 去重但保持順序
                seen: set = set()
                unique: list = []
                for ln in lines:
                    if ln not in seen:
                        seen.add(ln)
                        unique.append(ln)
                result[name] = "\n".join(unique[-200:])
        return result

    def get_activity_history(self, agent_name: str, limit_chars: int = 1000) -> str:
        text = self._activity_history.get(agent_name, "")
        if not text:
            return "無活動記錄"
        return text[-limit_chars:] if len(text) > limit_chars else text

    # ------------------------------------------------------------
    # checkpoint / 情緒狀態
    # ------------------------------------------------------------
    def _load_latest_checkpoint(self) -> Dict[str, Any]:
        if not self.checkpoints_folder.exists():
            return {}
        candidates = sorted(
            f for f in self.checkpoints_folder.iterdir()
            if f.name.startswith("simulate-") and f.suffix == ".json"
        )
        if not candidates:
            return {}
        try:
            return json.loads(candidates[-1].read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[SimulationContext] 載入 checkpoint 失敗: {e}")
            return {}

    def get_emotion_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        agents = self._latest_checkpoint.get("agents", {})
        agent = agents.get(agent_name)
        if not agent:
            return None
        status = agent.get("status", {}) if isinstance(agent.get("status"), dict) else {}
        return status.get("emotion")

    def format_emotion_state(self, agent_name: str) -> str:
        emo = self.get_emotion_state(agent_name)
        if not emo:
            return "無情緒記錄"

        label = emo.get("label", "未知")
        intensity = emo.get("intensity", 0)
        reason = emo.get("reason", "")
        history = emo.get("history", []) or []

        parts = [f"當前情緒：{label}（強度 {intensity}/3）"]
        if reason:
            parts.append(f"原因：{reason}")
        if history:
            recent = "、".join(
                f"{h.get('label','')}({h.get('intensity',0)})"
                for h in history[:5]
                if isinstance(h, dict)
            )
            if recent:
                parts.append(f"近期情緒序列：{recent}")
        return "；".join(parts)

    # ------------------------------------------------------------
    # 向量記憶查詢（read-only）
    # ------------------------------------------------------------
    def _extract_embedding_config(self) -> Optional[Dict[str, Any]]:
        agent_base = self._latest_checkpoint.get("agent_base", {})
        associate = agent_base.get("associate", {}) if isinstance(agent_base, dict) else {}
        cfg = dict(associate.get("embedding") or {})
        if not cfg:
            return None
        # 用 .env 的 OLLAMA_BASE_URL 覆蓋 checkpoint 中的舊 URL（checkpoint 可能來自不同 port）
        env_url = os.getenv("OLLAMA_BASE_URL")
        if cfg.get("type") == "ollama" and env_url:
            if cfg.get("base_url") != env_url:
                print(f"[SimulationContext] embedding base_url 已從 checkpoint 的 "
                      f"{cfg.get('base_url')} 覆蓋為 .env 的 {env_url}")
            cfg["base_url"] = env_url
        return cfg

    def _is_embedding_available(self) -> bool:
        if self._embedding_available is not None:
            return self._embedding_available

        cfg = self._embedding_config
        if not cfg or cfg.get("type") != "ollama":
            # huggingface 不需要外部服務檢查；保守視為可用
            self._embedding_available = bool(cfg)
            return self._embedding_available

        base_url = cfg.get("base_url", "http://127.0.0.1:11434")
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=2)
            self._embedding_available = (r.status_code == 200)
        except Exception:
            self._embedding_available = False

        if not self._embedding_available:
            print(f"[SimulationContext] Embedding 服務 {base_url} 無法連線，跳過向量記憶查詢")
        return self._embedding_available

    def _get_index_for(self, agent_name: str):
        """延遲載入 LlamaIndex；不可用時回傳 None。"""
        if agent_name in self._index_cache:
            return self._index_cache[agent_name] or None

        if not self._embedding_config:
            self._index_cache[agent_name] = False
            return None

        storage_path = self.storage_root / agent_name / "associate"
        if not storage_path.exists():
            self._index_cache[agent_name] = False
            return None

        if not self._is_embedding_available():
            self._index_cache[agent_name] = False
            return None

        try:
            from modules.storage.index import LlamaIndex
            index = LlamaIndex(
                embedding=self._embedding_config,
                path=str(storage_path),
            )
            self._index_cache[agent_name] = index
            return index
        except Exception as e:
            print(f"[SimulationContext] 載入 {agent_name} 向量庫失敗: {e}")
            self._index_cache[agent_name] = False
            return None

    def retrieve_memories(self, agent_name: str, query: str, top_k: int = 5) -> List[str]:
        """根據查詢語意檢索與該居民相關的記憶描述。失敗時回傳空 list。"""
        index = self._get_index_for(agent_name)
        if index is None:
            return []

        # LlamaIndex.retrieve 內部有無限重試迴圈，這裡用較小的 timeout 包一層保護。
        # 為簡化實作，仍直接呼叫；在外部已確認服務可用。
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                nodes = index.retrieve(query, similarity_top_k=top_k)
        except Exception as e:
            print(f"[SimulationContext] {agent_name} 記憶檢索失敗: {e}")
            return []

        descriptions: List[str] = []
        for node in nodes:
            text = getattr(node, "text", None) or getattr(getattr(node, "node", None), "text", "")
            if text:
                descriptions.append(text.strip())
        return descriptions

    def format_memories(self, agent_name: str, query: str, top_k: int = 5) -> str:
        memories = self.retrieve_memories(agent_name, query, top_k=top_k)
        if not memories:
            return "（無相關記憶）"
        return "\n".join(f"- {m}" for m in memories)

    # ------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "simulation_name": self.simulation_name,
            "simulation_md_exists": self.simulation_md_path.exists(),
            "checkpoint_loaded": bool(self._latest_checkpoint),
            "checkpoint_step": self._latest_checkpoint.get("step"),
            "embedding_config": self._embedding_config,
            "agents_with_history": list(self._activity_history.keys()),
        }
