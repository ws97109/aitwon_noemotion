"""
AI居民問卷填寫器 - 純 Ollama LLM 版本
所有回答均由 Ollama LLM 根據以下資料生成：
  - agent.json（靜態人設）
  - simulation.md（活動歷史）
  - 最新 checkpoint（情緒狀態）
  - storage/{agent}/associate（向量記憶）
不使用任何硬編碼或規則引擎。
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加父目錄到路徑，以便導入遊戲模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import Survey, SurveyResponse, SurveyManager
from .ollama_generator import OllamaSurveyGenerator
from .simulation_context import SimulationContext


class AIResidentSurveyFiller:
    """AI居民問卷填寫器 - 純 LLM 驅動，無硬編碼規則"""

    def __init__(
        self,
        survey_manager: SurveyManager,
        simulation_name: Optional[str] = None,
        simulation_md_path: Optional[str] = None,
    ):
        """
        初始化AI居民問卷填寫器

        Args:
            survey_manager: 問卷管理器
            simulation_name: 模擬名稱（會自動解析 results/compressed/{name}/simulation.md
                以及 results/checkpoints/{name}/）。推薦使用此參數。
            simulation_md_path: （舊參數，保留兼容）simulation.md 路徑。若同時提供
                simulation_name，以 simulation_name 為準。
        """
        self.survey_manager = survey_manager
        self.game = None

        # 動態載入AI居民基本信息（用於列表和驗證）
        self.residents_info = self._load_residents_info()

        # 建立 SimulationContext
        self.simulation_context: Optional[SimulationContext] = None
        if simulation_name:
            try:
                self.simulation_context = SimulationContext(simulation_name)
                print(f"✓ 已載入模擬上下文: {simulation_name}")
            except Exception as e:
                print(f"⚠️  載入 SimulationContext 失敗: {e}")

        # 初始化 Ollama 生成器
        self.ollama_generator = OllamaSurveyGenerator(
            simulation_md_path=simulation_md_path,
            simulation_context=self.simulation_context,
        )
        print("✓ Ollama 生成器已初始化（純 LLM 模式，無硬編碼規則）")

    def _load_residents_info(self) -> Dict[str, Dict]:
        """動態載入AI居民基本信息（僅用於列表/驗證）。"""
        try:
            project_root = Path(__file__).resolve().parent.parent
            agents_path = project_root / "frontend" / "static" / "assets" / "village" / "agents"

            residents_info: Dict[str, Dict] = {}

            if agents_path.exists():
                for agent_dir in agents_path.iterdir():
                    if not agent_dir.is_dir():
                        continue
                    agent_config_path = agent_dir / "agent.json"
                    if not agent_config_path.exists():
                        continue
                    try:
                        with open(agent_config_path, "r", encoding="utf-8") as f:
                            agent_config = json.load(f)
                        agent_name = agent_config.get("name", agent_dir.name)
                        scratch = agent_config.get("scratch", {})
                        residents_info[agent_name] = {
                            "age": scratch.get("age", 25),
                            "personality": scratch.get("innate", "").split("、") if scratch.get("innate") else [],
                            "current_activity": agent_config.get("currently", ""),
                        }
                    except json.JSONDecodeError:
                        print(f"⚠️  無法解析 {agent_config_path}")
                        continue

            if not residents_info:
                print("❌ 未找到任何 AI 居民配置文件")
                print(f"   請確認路徑: {agents_path}")

            return residents_info

        except Exception as e:
            print(f"❌ 載入AI居民信息失敗: {e}")
            return {}

    def set_game_context(self):
        """設置遊戲上下文（保留接口兼容性）"""
        pass

    def fill_survey_for_all_residents(self, survey_id: str) -> List[SurveyResponse]:
        survey = self.survey_manager.load_survey(survey_id)
        if not survey:
            raise ValueError(f"問卷 {survey_id} 不存在")

        print(f"\n📋 開始為所有居民填寫問卷: {survey.title}")
        print(f"   共 {len(self.residents_info)} 位居民")
        print(f"   問卷包含 {len(survey.questions)} 個問題")
        if self.simulation_context:
            print(f"   使用模擬上下文: {self.simulation_context.simulation_name}")
        print("   使用 Ollama LLM 生成所有回答\n")

        responses: List[SurveyResponse] = []
        for i, resident_name in enumerate(self.residents_info.keys(), 1):
            print(f"[{i}/{len(self.residents_info)}] 正在為 {resident_name} 生成回答...")
            try:
                response = self.fill_survey_for_resident(survey_id, resident_name)
                if response:
                    responses.append(response)
                    print(f"   ✓ {resident_name} 完成")
            except Exception as e:
                print(f"   ❌ {resident_name} 失敗: {e}")
                continue

        print(f"\n✅ 完成！成功生成 {len(responses)}/{len(self.residents_info)} 份問卷回應")
        return responses

    def fill_survey_for_resident(self, survey_id: str, resident_name: str) -> Optional[SurveyResponse]:
        survey = self.survey_manager.load_survey(survey_id)
        if not survey:
            raise ValueError(f"問卷 {survey_id} 不存在")
        if resident_name not in self.residents_info:
            raise ValueError(f"AI居民 {resident_name} 不存在")

        response = self._generate_resident_response_via_llm(survey, resident_name)
        if response:
            # 先清掉同問卷同居民的舊回應，避免重複累積（確保每人最多一份最新回應）
            removed = self.survey_manager.delete_resident_responses(survey_id, resident_name)
            if removed > 0:
                print(f"      🧹 已清除 {resident_name} 的 {removed} 份舊回應")
            self.survey_manager.save_response(response)
        return response

    def _generate_resident_response_via_llm(self, survey: Survey, resident_name: str) -> SurveyResponse:
        response = SurveyResponse(survey.survey_id, resident_name)

        print(f"   📝 問卷: {survey.title}")
        print(f"   🤖 使用 Ollama LLM 生成 {len(survey.questions)} 個回答...")

        for i, question in enumerate(survey.questions, 1):
            try:
                answer = self._generate_answer_via_ollama(question, resident_name)
                response.add_response(question["id"], answer)
                print(f"      [{i}/{len(survey.questions)}] {question['text'][:30]}... ✓")
            except Exception as e:
                print(f"      [{i}/{len(survey.questions)}] {question['text'][:30]}... ❌ 錯誤: {e}")
                response.add_response(question["id"], self._get_error_fallback_answer(question["type"]))

        response.complete()
        return response

    def _generate_answer_via_ollama(self, question: Dict, resident_name: str) -> Any:
        question_type = question["type"]
        question_text = question["text"]
        options = question.get("options", [])

        llm_response = self.ollama_generator.generate_response(
            resident_name=resident_name,
            resident_info={},
            question_text=question_text,
            question_type=question_type,
            options=options,
        )
        return self._process_llm_response(llm_response, question_type, options)

    def _process_llm_response(self, llm_response: str, question_type: str, options: List[str]) -> Any:
        if not llm_response:
            return self._get_error_fallback_answer(question_type)

        llm_response = llm_response.strip()

        if question_type == "single_choice":
            for option in options:
                if option == llm_response or option in llm_response or llm_response in option:
                    return option
            for option in options:
                option_words = set(option.replace("、", " ").replace("，", " ").split())
                response_words = set(llm_response.replace("、", " ").replace("，", " ").split())
                if option_words & response_words:
                    return option
            print(f"         ⚠️  無法匹配選項，使用 LLM 原始回應: {llm_response}")
            return llm_response

        elif question_type == "multiple_choice":
            selected: List[str] = []
            for option in options:
                if option in llm_response:
                    selected.append(option)
            if selected:
                return selected
            parts = llm_response.replace("、", ",").replace("，", ",").split(",")
            for part in parts:
                part = part.strip()
                for option in options:
                    if (part in option or option in part) and option not in selected:
                        selected.append(option)
            if selected:
                return selected
            print("         ⚠️  無法匹配選項，使用 LLM 原始回應")
            return [llm_response]

        elif question_type == "rating":
            import re
            match = re.search(r"(\d+)", llm_response)
            if match:
                rating = int(match.group(1))
                return rating if 1 <= rating <= 10 else max(1, min(10, rating))
            print("         ⚠️  無法提取評分，使用預設值 5")
            return 5

        elif question_type == "text":
            return llm_response

        return llm_response

    def _get_error_fallback_answer(self, question_type: str) -> Any:
        fallbacks = {
            "single_choice": "無法回答",
            "multiple_choice": [],
            "rating": 5,
            "text": "抱歉，目前無法提供回答。",
        }
        return fallbacks.get(question_type, "無法回答")
