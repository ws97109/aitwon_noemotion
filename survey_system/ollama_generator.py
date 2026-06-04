"""
Ollama-based survey response generator for AI residents.
Uses local Ollama LLM to generate contextual survey responses based on:
- Agent background information from agent.json
- Activity history from simulation.md
- Emotion state from latest checkpoint
- Vector-retrieved memories from Associate index
- Prompt templates from data/prompts/
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
from string import Template
from dotenv import load_dotenv

from .simulation_context import SimulationContext

load_dotenv()


LANGUAGE_SUFFIX = (
    "\n\n【語言要求】請一律使用繁體中文（台灣用字）作答，"
    "避免使用簡體中文字。"
)


class OllamaSurveyGenerator:
    """Generate survey responses using Ollama LLM with prompt templates."""

    def __init__(
        self,
        simulation_md_path: Optional[str] = None,
        agents_dir: Optional[str] = None,
        simulation_context: Optional[SimulationContext] = None,
    ):
        """
        Initialize Ollama generator.

        Args:
            simulation_md_path: (Deprecated) 為兼容舊呼叫保留；若有提供且未給 context，
                會嘗試用其父目錄名作為 simulation_name 建立 SimulationContext。
            agents_dir: Path to agents directory (default: frontend/static/assets/village/agents)
            simulation_context: 預先建好的 SimulationContext（推薦使用）。
        """
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.api_url = f"{self.base_url}/api/generate"

        self.project_root = Path(__file__).parent.parent
        self.prompts_dir = self.project_root / "data" / "prompts"

        if agents_dir:
            self.agents_dir = Path(agents_dir)
        else:
            self.agents_dir = self.project_root / "frontend" / "static" / "assets" / "village" / "agents"

        # 解析 SimulationContext
        self.simulation_context: Optional[SimulationContext] = simulation_context
        if self.simulation_context is None and simulation_md_path:
            sim_path = Path(simulation_md_path)
            if sim_path.exists():
                # results/compressed/{name}/simulation.md → name = parent.name
                sim_name = sim_path.parent.name
                try:
                    self.simulation_context = SimulationContext(sim_name, project_root=self.project_root)
                except Exception as e:
                    print(f"[OllamaGenerator] 從 simulation_md_path 推斷 SimulationContext 失敗: {e}")

        if self.simulation_context:
            summary = self.simulation_context.summary()
            print(f"✓ SimulationContext 已就緒: {summary['simulation_name']} "
                  f"(md={summary['simulation_md_exists']}, "
                  f"checkpoint={summary['checkpoint_loaded']}, "
                  f"agents={len(summary['agents_with_history'])})")
        else:
            print("⚠️  未提供 SimulationContext — 居民將以靜態人設回答（無活動歷史/記憶/情緒）")

        # Cache for agent.json data
        self.agents_cache: Dict[str, Dict[str, Any]] = {}

    def _load_agent_data(self, agent_name: str) -> Optional[Dict[str, Any]]:
        if agent_name in self.agents_cache:
            return self.agents_cache[agent_name]

        agent_json_path = self.agents_dir / agent_name / "agent.json"
        if not agent_json_path.exists():
            print(f"❌ 找不到 agent.json: {agent_json_path}")
            return None

        try:
            with open(agent_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.agents_cache[agent_name] = data
            return data
        except Exception as e:
            print(f"❌ 載入 {agent_name} 的資料時發生錯誤: {e}")
            return None

    def _load_prompt_template(self, question_type: str) -> Optional[Template]:
        template_file = self.prompts_dir / f"survey_{question_type}.txt"
        if not template_file.exists():
            print(f"⚠️  找不到 prompt 模板: {template_file}")
            return None
        try:
            return Template(template_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"❌ 載入 prompt 模板時發生錯誤: {e}")
            return None

    @staticmethod
    def _format_options(options: List[str]) -> str:
        if not options:
            return ""
        return "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, 1))

    @staticmethod
    def _extract_rating_scale(question_text: str):
        """從問題文字中提取評分範圍，例如「1-5分」→ (1, 5)。預設 (1, 5)。"""
        import re
        m = re.search(r"(\d+)\s*[-~到至]\s*(\d+)\s*分", question_text)
        if m:
            return int(m.group(1)), int(m.group(2))
        return 1, 5

    def _build_prompt(
        self,
        agent_name: str,
        agent_data: Dict[str, Any],
        question_text: str,
        question_type: str,
        options: Optional[List[str]] = None,
        prev_answers: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        template = self._load_prompt_template(question_type)
        if not template:
            return f"無法載入 {question_type} 的 prompt 模板"

        scratch = agent_data.get("scratch", {})
        ctx = self.simulation_context

        if ctx:
            activity_history = ctx.get_activity_history(agent_name, limit_chars=3000)
            emotion_state = ctx.format_emotion_state(agent_name)
            memories = ctx.format_memories(agent_name, question_text, top_k=5)
        else:
            activity_history = "無活動記錄"
            emotion_state = "無情緒記錄"
            memories = "（無相關記憶）"

        # 評分題：從問題文字提取範圍
        rating_min, rating_max = self._extract_rating_scale(question_text)

        # 組合先前回答的上下文（只保留最近 5 題避免 prompt 過長）
        prev_context = ""
        if prev_answers:
            recent = prev_answers[-5:]
            lines = []
            for pa in recent:
                ans = pa["answer"]
                if isinstance(ans, list):
                    ans = "、".join(str(a) for a in ans)
                lines.append(f"問：{pa['text']}\n答：{ans}")
            prev_context = "\n\n".join(lines)

        template_vars = {
            "agent_name": agent_name,
            "age": scratch.get("age", "未知"),
            "personality": scratch.get("innate", "未知"),
            "interests": scratch.get("learned", "未知"),
            "lifestyle": scratch.get("lifestyle", "未知"),
            "current_activity": agent_data.get("currently", "未知"),
            "family_background": scratch.get("family_background", "無相關資訊"),
            "wealth_level": scratch.get("wealth_level", "無相關資訊"),
            "question_text": question_text,
            "activity_history": activity_history,
            "emotion_state": emotion_state,
            "memories": memories,
            "options": self._format_options(options) if options and question_type in ("single_choice", "multiple_choice") else "",
            "rating_min": str(rating_min),
            "rating_max": str(rating_max),
            "prev_answers": prev_context,
        }

        try:
            result = template.safe_substitute(template_vars) + LANGUAGE_SUFFIX
            # 如果模板沒有 ${prev_answers} 佔位符但有上下文，手動插入
            if prev_context and "${prev_answers}" not in template.template:
                result = result.replace(
                    "=== 問卷問題 ===",
                    f"=== 你在本問卷中的先前回答 ===\n{prev_context}\n\n=== 問卷問題 ===",
                )
            return result
        except Exception as e:
            print(f"❌ 替換 prompt 變數時發生錯誤: {e}")
            return f"模板處理錯誤: {e}"

    def _verify_model_available(self) -> Optional[str]:
        """確認 OLLAMA_MODEL 在伺服器上存在；不存在或無法連線時回傳錯誤訊息。"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return f"Ollama /api/tags HTTP {r.status_code}"
            installed = [m.get("name", "") for m in r.json().get("models", [])]
            if self.model not in installed:
                return (
                    f"模型 '{self.model}' 不在 Ollama 已安裝清單中。\n"
                    f"  已安裝: {installed}\n"
                    f"  解法: 修改 .env 的 OLLAMA_MODEL，或執行 `ollama pull {self.model}`"
                )
            return None
        except Exception as e:
            return f"無法連線 Ollama ({self.base_url}): {e}"

    def generate_response(
        self,
        resident_name: str,
        resident_info: Dict[str, Any],  # 保留兼容；不使用
        question_text: str,
        question_type: str,
        options: Optional[List[str]] = None,
        prev_answers: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        agent_data = self._load_agent_data(resident_name)
        if not agent_data:
            print(f"❌ [generate_response] {resident_name} 缺少 agent.json，回傳 fallback")
            return self._fallback_response(question_type)

        # 第一次呼叫先驗證模型是否真的存在；之後快取
        if not hasattr(self, "_model_check_done"):
            err = self._verify_model_available()
            object.__setattr__(self, "_model_check_done", True)
            object.__setattr__(self, "_model_check_error", err)
            if err:
                print(f"❌ [Ollama 預檢失敗] {err}")
        if getattr(self, "_model_check_error", None):
            return self._fallback_response(question_type)

        prompt = self._build_prompt(
            agent_name=resident_name,
            agent_data=agent_data,
            question_text=question_text,
            question_type=question_type,
            options=options,
            prev_answers=prev_answers,
        )

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 500,
                    },
                },
                timeout=300,
            )

            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                if not text:
                    print(f"⚠️  [{resident_name}] Ollama 回傳空字串，prompt 長度={len(prompt)}")
                return text or self._fallback_response(question_type)
            print(f"❌ Ollama API HTTP {response.status_code}: {response.text[:200]}")
            return self._fallback_response(question_type)

        except requests.Timeout:
            print(f"⏱️  [{resident_name}] Ollama 請求逾時（300s）— 模型可能太慢或還在 load")
            return self._fallback_response(question_type)
        except Exception as e:
            print(f"❌ [{resident_name}] 調用 Ollama API 時發生錯誤: {type(e).__name__}: {e}")
            return self._fallback_response(question_type)

    def _fallback_response(self, question_type: str) -> str:
        fallbacks = {
            "single_choice": "其他",
            "multiple_choice": "其他",
            "rating": "5",
            "text": "抱歉，目前無法提供詳細回答。",
        }
        return fallbacks.get(question_type, "無回答")

    def batch_generate_responses(
        self,
        resident_name: str,
        resident_info: Dict[str, Any],
        questions: List[Dict[str, Any]],
    ) -> List[str]:
        return [
            self.generate_response(
                resident_name=resident_name,
                resident_info=resident_info,
                question_text=q.get("question_text", ""),
                question_type=q.get("question_type", "text"),
                options=q.get("options", []),
            )
            for q in questions
        ]
