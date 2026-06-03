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

        # 「不能填寫自己」：將自己從選項中移除
        if "不能填寫自己" in question_text and resident_name in options:
            options = [o for o in options if o != resident_name]

        llm_response = self.ollama_generator.generate_response(
            resident_name=resident_name,
            resident_info={},
            question_text=question_text,
            question_type=question_type,
            options=options,
        )

        # 評分題：提取實際範圍用於驗證
        rating_min, rating_max = 1, 5
        if question_type == "rating":
            from .ollama_generator import OllamaSurveyGenerator
            rating_min, rating_max = OllamaSurveyGenerator._extract_rating_scale(question_text)

        return self._process_llm_response(llm_response, question_type, options,
                                          rating_min=rating_min, rating_max=rating_max)

    def _process_llm_response(self, llm_response: str, question_type: str, options: List[str],
                              rating_min: int = 1, rating_max: int = 5) -> Any:
        if not llm_response:
            return self._get_error_fallback_answer(question_type)

        llm_response = llm_response.strip()

        if question_type == "single_choice":
            matched = self._match_to_option(llm_response, options)
            if matched is not None:
                return matched
            # 真的找不到 → 回 fallback（不再保留出格的 LLM 原文）
            print(f"         ⚠️  選項全無匹配，使用 fallback: 原回應='{llm_response[:40]}'")
            return self._get_error_fallback_answer(question_type)

        elif question_type == "multiple_choice":
            return self._match_multiple_to_options(llm_response, options)

        elif question_type == "rating":
            import re
            match = re.search(r"(\d+)", llm_response)
            if match:
                rating = int(match.group(1))
                return max(rating_min, min(rating_max, rating))
            mid = (rating_min + rating_max) // 2
            print(f"         ⚠️  無法提取評分，使用預設值 {mid}")
            return mid

        elif question_type == "text":
            return llm_response

        return llm_response

    # ------------------------------------------------------------
    # 選項匹配輔助：保證輸出一定落在 options 內
    # ------------------------------------------------------------
    @staticmethod
    def _parse_numeric_range(option: str):
        """從選項文字抽取數字範圍。回傳 (low, high) 或 None。

        支援格式：
          "20,000元以下"            → (None, 20000)
          "20,000-40,000元"         → (20000, 40000)
          "40%以上" / "100,000元以上" → (40, None) / (100000, None)
          "10%-20%"                  → (10, 20)
        """
        import re
        cleaned = option.replace(",", "").replace("，", "")
        nums = [int(n) for n in re.findall(r"\d+", cleaned)]
        if not nums:
            return None
        if "以下" in option or "以內" in option:
            return (None, nums[0])
        if "以上" in option:
            return (nums[0], None)
        if len(nums) >= 2:
            return (nums[0], nums[1])
        return (nums[0], nums[0])

    @classmethod
    def _match_numeric_range(cls, value: int, options: List[str]):
        """根據單一數值，挑出包含它的範圍選項。"""
        best = None
        best_dist = None
        for opt in options:
            rng = cls._parse_numeric_range(opt)
            if rng is None:
                continue
            low, high = rng
            in_range = (low is None or value >= low) and (high is None or value <= high)
            if in_range:
                return opt
            # 不在範圍 → 計算到最近邊界的距離，留作 fallback
            edge = low if (low is not None and value < low) else high
            if edge is None:
                continue
            dist = abs(value - edge)
            if best_dist is None or dist < best_dist:
                best_dist, best = dist, opt
        return best

    @classmethod
    def _match_to_option(cls, response: str, options: List[str]):
        """把 LLM 回應匹配到合法選項。找不到回 None。"""
        if not options:
            return response
        resp = response.strip().strip("「」\"'`。 ")

        # 1) 完全相等
        for opt in options:
            if opt == resp:
                return opt
        # 2) 雙向包含
        for opt in options:
            if opt in resp or resp in opt:
                return opt
        # 3) 範圍題：抽出回應中的數字 → 找包含它的範圍
        import re
        nums = [int(n) for n in re.findall(r"\d+", resp.replace(",", "").replace("，", ""))]
        if nums and any(cls._parse_numeric_range(o) for o in options):
            # 若 LLM 回應本身也是個範圍（兩個數字），取中位以避免落在邊界端點
            value = (nums[0] + nums[1]) // 2 if len(nums) >= 2 else nums[0]
            matched = cls._match_numeric_range(value, options)
            if matched:
                return matched
        # 4) difflib 模糊比對（相似度 >= 0.5）
        import difflib
        matches = difflib.get_close_matches(resp, options, n=1, cutoff=0.5)
        if matches:
            return matches[0]
        # 5) 詞袋交集
        resp_words = set(resp.replace("、", " ").replace("，", " ").split())
        for opt in options:
            opt_words = set(opt.replace("、", " ").replace("，", " ").split())
            if opt_words & resp_words:
                return opt
        return None

    @classmethod
    def _match_multiple_to_options(cls, response: str, options: List[str]) -> List[str]:
        """多選題：把回應拆解後逐一匹配，全部落在合法選項內。"""
        if not options:
            return [response]

        selected: List[str] = []

        # 先看是否有整段子字串就是某個選項
        for opt in options:
            if opt in response and opt not in selected:
                selected.append(opt)

        # 用分隔符拆分後逐一匹配
        import re
        parts = re.split(r"[、,，;；\n/]+", response)
        for part in parts:
            part = part.strip().strip("「」\"'`。 ")
            if not part:
                continue
            matched = cls._match_to_option(part, options)
            if matched and matched not in selected:
                selected.append(matched)

        if not selected:
            # 整段交給單選匹配器試一次
            matched = cls._match_to_option(response, options)
            if matched:
                selected.append(matched)

        if not selected:
            print(f"         ⚠️  多選題無匹配，回傳空 list: 原回應='{response[:40]}'")
            return []
        return selected

    def _get_error_fallback_answer(self, question_type: str) -> Any:
        fallbacks = {
            "single_choice": "無法回答",
            "multiple_choice": [],
            "rating": 5,
            "text": "抱歉，目前無法提供回答。",
        }
        return fallbacks.get(question_type, "無法回答")
