"""
Agent 情緒擴展 Mixin

提供給 Agent 類別的情緒能力擴展
無需直接修改原有 agent.py，通過 mixin 方式整合
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import os

from .core import EmotionState, EmotionAnalyzer, EmotionConfig, get_emotion_from_poignancy_and_context
from .emotion_memory import EmotionMemory, EmotionalConcept
from .emotion_prompts import EmotionPrompts
from .mgt_emotion_rater import MGTEmotionRater, MultimodalEmotionInput, create_mgt_rater_for_agent
from . import mgt_prompts


class AgentEmotionMixin:
    """
    Agent 情緒擴展 Mixin

    為 Agent 添加情緒感知能力
    使用方法：在 Agent.__init__() 中調用 self._init_emotion_system()
    """

    def _init_emotion_system(self, config: Optional[Dict] = None):
        """
        初始化情緒系統

        Args:
            config: 情緒配置字典（從 agent.json 讀取）
                可包含：
                - emotional_stability, empathy_level, optimism, anxiety_proneness
                - enable_mgt_rater: 是否啟用 MGT 評分員（預設 True）
                - mgt_hidden_dim: MGT 隱藏層維度（預設 768）
        """
        # 創建情緒配置
        if config:
            self.emotion_config = EmotionConfig.from_dict(config)
        else:
            self.emotion_config = EmotionConfig()

        # 創建情緒分析器
        self.emotion_analyzer = EmotionAnalyzer(self.name, self.emotion_config)

        # 創建情緒記憶管理器
        self.emotion_memory = EmotionMemory(self.emotion_analyzer)

        # 創建情緒提示詞管理器
        self.emotion_prompts = EmotionPrompts(self.name)

        # 創建 MGT 情緒評分員（新增）
        enable_mgt = config.get("enable_mgt_rater", True) if config else True
        if enable_mgt:
            mgt_config = {
                "hidden_dim": config.get("mgt_hidden_dim", 768) if config else 768,
                "enable_llm_enhancement": config.get("mgt_llm_enhancement", True) if config else True
            }
            self.mgt_rater = create_mgt_rater_for_agent(self.name, mgt_config)
            print(f"[{self.name}] MGT 情緒評分員已啟用")
        else:
            self.mgt_rater = None
            print(f"[{self.name}] MGT 情緒評分員未啟用")

        # 追蹤最近使用的記憶（用於反饋學習）
        self._last_used_memory_ids: List[str] = []

        # 追蹤互動結果（用於 MGT 反思）
        self._interaction_outcomes: List[str] = []

        print(f"[{self.name}] 情緒系統已初始化")

    # ========== 情緒評估方法 ==========

    def appraise_emotion(
        self,
        event_description: str,
        event_type: str = "event",
        poignancy: int = 5,
        use_llm: bool = True
    ) -> EmotionState:
        """
        評估事件應產生的情緒

        Args:
            event_description: 事件描述
            event_type: 事件類型 ("event", "chat", "thought")
            poignancy: 重要性 (1-10)
            use_llm: 是否使用 LLM（否則使用規則）

        Returns:
            EmotionState: 新的情緒狀態
        """
        if not use_llm:
            # 使用簡單規則
            is_positive = any(word in event_description for word in
                            ["好", "開心", "成功", "完成", "讚", "喜歡"])
            emotion, intensity = get_emotion_from_poignancy_and_context(
                poignancy, event_type, is_positive
            )
            return EmotionState(
                emotion=emotion,
                intensity=intensity,
                reasoning=f"基於事件: {event_description[:30]}",
                timestamp=datetime.now().isoformat()
            )

        # 使用 LLM 評估
        try:
            recent_emotions = self.emotion_analyzer.get_recent_emotion_summary(3)
            # 修復：使用 self.scratch.config['innate'] 而非 self.scratch.innate
            innate = self.scratch.config.get('innate', '友善且樂於助人')
            context = f"{self.scratch.name} 的性格: {innate}"

            prompt_data = self.emotion_prompts.prompt_emotion_appraisal(
                event_description=event_description,
                event_type=event_type,
                poignancy=poignancy,
                context=context,
                recent_emotions=recent_emotions
            )

            # 使用 Agent 的 completion 方法調用 LLM
            # 注意：Agent.completion() 需要 func_hint 參數
            emotion_state = self._llm.completion(
                prompt=prompt_data["prompt"],
                callback=prompt_data["callback"],
                failsafe=prompt_data["failsafe"],
                retry=prompt_data.get("retry", 2),
                caller="emotion_appraisal"
            )

            return emotion_state

        except Exception as e:
            print(f"[{self.name}] 情緒評估失敗: {e}")
            # 修復：返回預設的 EmotionState 而非未定義的 prompt_data["failsafe"]
            return EmotionState(
                emotion="neutral",
                intensity=0.3,
                reasoning=f"評估失敗，返回預設情緒: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def update_emotion_from_event(
        self,
        event_description: str,
        poignancy: int,
        event_type: str = "event"
    ):
        """
        從事件更新情緒狀態

        在 Agent._add_concept() 或 percept() 中調用

        Args:
            event_description: 事件描述
            poignancy: 重要性
            event_type: 事件類型
        """
        # 只對重要事件進行情緒評估（節省 API 調用）
        if poignancy >= 5:
            new_emotion = self.appraise_emotion(
                event_description,
                event_type,
                poignancy,
                use_llm=True
            )
            self.emotion_analyzer.update_emotion(new_emotion)

            print(f"[{self.name}] 情緒更新: {new_emotion.get_display_name()} "
                  f"(強度:{new_emotion.intensity:.1f}) - {new_emotion.reasoning}")

    # ========== 記憶系統整合 ==========

    def add_concept_with_emotion(
        self,
        node_id: str,
        concept: Any,
        current_time: datetime
    ):
        """
        添加帶情緒的記憶概念

        在 Agent._add_concept() 中調用

        Args:
            node_id: 節點 ID
            concept: Concept 物件
            current_time: 當前時間
        """
        # 創建情緒概念
        emotional_concept = self.emotion_memory.add_emotional_concept(
            concept,
            self.emotion_analyzer.current_emotion
        )

        # 更新節點 metadata（如果使用 LlamaIndex）
        try:
            if hasattr(concept, 'metadata'):
                concept.metadata.update(emotional_concept.to_metadata())
        except Exception as e:
            print(f"[{self.name}] 更新 metadata 失敗: {e}")

    def retrieve_with_emotion_bias(
        self,
        query: str,
        retrieve_func: callable,
        top_k: int = 5
    ) -> List[Any]:
        """
        帶情緒偏差的記憶檢索

        包裝原有的檢索方法，應用情緒偏差

        Args:
            query: 查詢文字
            retrieve_func: 原有的檢索函數（如 self.associate.retrieve_events）
            top_k: 返回數量

        Returns:
            檢索結果列表
        """
        # 調用原有檢索（獲取更多候選）
        candidates = retrieve_func(query, top_k * 2)

        # 應用情緒偏差
        adjusted_results = self.emotion_memory.apply_emotion_bias_to_retrieval(
            candidates,
            top_k
        )

        # 追蹤使用的記憶 ID
        self._last_used_memory_ids = [
            node.node_id if hasattr(node, 'node_id') else str(node)
            for node, _ in adjusted_results
        ]

        return [node for node, _ in adjusted_results]

    # ========== 對話系統整合 ==========

    def generate_emotional_chat(
        self,
        other_agent_name: str,
        chat_context: str,
        chat_history: List[Tuple[str, str]] = None,
        retrieved_memories: List[Any] = None,
        use_mgt_rater: bool = True
    ) -> str:
        """
        生成帶情緒的對話（整合 MGT 評分員）

        在 Agent._chat_with() 中調用

        Args:
            other_agent_name: 對方姓名
            chat_context: 對話情境
            chat_history: 對話歷史 (可選)
            retrieved_memories: 檢索到的記憶 (可選，如果未提供則自動檢索)
            use_mgt_rater: 是否使用 MGT 評分員（預設 True）

        Returns:
            對話內容
        """
        try:
            # 如果沒有提供記憶，則自動檢索
            if retrieved_memories is None:
                # 構建檢索查詢
                query_parts = [other_agent_name, chat_context]
                if chat_history and len(chat_history) > 0:
                    # 添加最近的對話內容作為查詢
                    recent_chats = "; ".join([f"{n}: {t}" for n, t in chat_history[-2:]])
                    query_parts.append(recent_chats)

                query = " ".join(query_parts)

                # 使用 associate 檢索相關記憶
                if hasattr(self, 'associate'):
                    retrieved_memories = self.associate.retrieve_focus([query], 5)
                else:
                    retrieved_memories = []

            # 格式化記憶
            memories_text = []
            for mem in (retrieved_memories or [])[:5]:
                if hasattr(mem, 'description'):
                    memories_text.append(mem.description)
                elif hasattr(mem, 'describe'):
                    memories_text.append(mem.describe)
                else:
                    memories_text.append(str(mem))

            # 轉換 chat_history 為提示詞需要的格式（兩個路徑都需要）
            formatted_chat_history = None
            if chat_history:
                formatted_chat_history = [
                    {"speaker": speaker, "content": content}
                    for speaker, content in chat_history
                ]

            # ✅ 收集原始系統的所有必要資訊
            # 1. 基本描述（年齡、性格特質等）
            base_desc = None
            if hasattr(self, 'scratch') and hasattr(self.scratch, '_base_desc'):
                base_desc = self.scratch._base_desc()

            # 2. 當前位置
            current_location = None
            if hasattr(self, 'get_tile'):
                try:
                    address = self.get_tile().get_address()
                    if len(address) >= 2:
                        current_location = f"{address[-2]}，{address[-1]}"
                except:
                    pass

            # 3. 當前時間
            current_time = None
            try:
                from modules import utils
                current_time = utils.get_timer().get_date("%H:%M")
            except:
                pass

            # 4. 關係描述
            relation = None
            try:
                relation = self.completion("summarize_relation", self, other_agent_name)
            except:
                pass

            # 5. 歷史對話摘要（8小時內）
            previous_context = None
            if hasattr(self, 'associate'):
                try:
                    from modules import utils
                    chat_nodes = self.associate.retrieve_chats(other_agent_name)
                    prev_context_parts = []
                    for n in chat_nodes:
                        delta = utils.get_timer().get_delta(n.create)
                        if delta > 480:  # 超過8小時
                            continue
                        prev_context_parts.append(
                            f"{delta} 分鐘前，{self.name} 和 {other_agent_name} 進行過對話。{n.describe}"
                        )
                    if prev_context_parts:
                        previous_context = "\n".join(prev_context_parts)
                except:
                    pass

            # 如果啟用 MGT 評分員，使用其建議
            if use_mgt_rater and self.mgt_rater:
                # 準備多模態輸入
                multimodal_input = MultimodalEmotionInput(
                    text_content=chat_context,
                    context_description=chat_context,
                    retrieved_memories=memories_text,
                    current_emotion=self.emotion_analyzer.current_emotion,
                    target_agent_name=other_agent_name
                )

                # 調用 MGT 評分員
                mgt_output = self.mgt_rater.rate_emotion(
                    multimodal_input,
                    agent_completion_func=self.completion
                )

                # 使用 MGT 增強的提示詞（包含所有原始系統資訊）
                prompt_data = mgt_prompts.prompt_mgt_enhanced_chat(
                    agent_name=self.name,
                    other_agent_name=other_agent_name,
                    chat_context=chat_context,
                    current_emotion=self.emotion_analyzer.current_emotion,
                    mgt_suggestions=mgt_output.interaction_suggestions,
                    retrieved_memories=memories_text,
                    chat_history=formatted_chat_history,
                    # ✅ 傳遞原始系統的所有資訊
                    base_desc=base_desc,
                    current_location=current_location,
                    current_time=current_time,
                    relation=relation,
                    previous_context=previous_context
                )

                # 可選：打印 MGT 分析（調試用）
                if hasattr(self, '_debug_mgt') and self._debug_mgt:
                    print(f"\n[MGT 評分員分析 - {self.name}]")
                    print(f"預測情緒: {mgt_output.predicted_emotion} ({mgt_output.emotion_intensity:.2f})")
                    print(f"建議: {mgt_output.interaction_suggestions[:2]}")
                    print(f"推理: {mgt_output.reasoning}\n")

            else:
                # 使用原本的情緒感知對話（包含所有原始系統資訊）
                memories_formatted = "\n".join([f"- {mem}" for mem in memories_text[:5]])
                prompt_data = self.emotion_prompts.prompt_emotion_aware_chat(
                    other_agent_name=other_agent_name,
                    current_emotion=self.emotion_analyzer.current_emotion,
                    chat_context=chat_context,
                    retrieved_memories=memories_formatted,
                    chat_history=formatted_chat_history,
                    # ✅ 傳遞原始系統的所有資訊
                    base_desc=base_desc,
                    current_location=current_location,
                    current_time=current_time,
                    relation=relation,
                    previous_context=previous_context
                )

            # 使用 LLM 的 completion 方法（因為我們已經有完整的 prompt_data）
            chat_response = self._llm.completion(
                prompt=prompt_data["prompt"],
                callback=prompt_data["callback"],
                failsafe=prompt_data["failsafe"],
                retry=prompt_data.get("retry", 2),
                caller="generate_emotional_chat"
            )

            # 記錄互動結果（用於後續反思）
            if use_mgt_rater and self.mgt_rater:
                outcome = f"與{other_agent_name}對話: {chat_response[:50]}"
                self._interaction_outcomes.append(outcome)
                # 保持最多 20 條記錄
                if len(self._interaction_outcomes) > 20:
                    self._interaction_outcomes = self._interaction_outcomes[-20:]

            return chat_response

        except Exception as e:
            print(f"[{self.name}] 情緒對話生成失敗: {e}")
            import traceback
            traceback.print_exc()

            # 使用更自然的錯誤恢復回應
            import random
            fallback_responses = [
                f"抱歉，我剛才走神了。",
                f"不好意思，請再說一遍？",
                f"嗯，我在想別的事情。",
                f"對不起，我沒聽清楚。"
            ]
            return random.choice(fallback_responses)

    # ========== 反思系統整合 ==========

    def reflect_on_emotions(self, use_mgt_feedback: bool = True) -> Optional[str]:
        """
        情緒反思（整合 MGT 評分員反饋）

        定期調用（例如在 Agent.reflect() 中）

        Args:
            use_mgt_feedback: 是否使用 MGT 評分員反饋（預設 True）

        Returns:
            反思洞察文字
        """
        if len(self.emotion_analyzer.emotion_history) < 3:
            return None

        try:
            # 如果啟用 MGT 且有評分歷史，使用 MGT 反思
            if use_mgt_feedback and self.mgt_rater and len(self.mgt_rater.rating_history) >= 3:
                prompt_data = mgt_prompts.prompt_reflect_on_mgt_feedback(
                    agent_name=self.name,
                    recent_mgt_ratings=self.mgt_rater.rating_history[-10:],
                    interaction_outcomes=self._interaction_outcomes[-5:]
                )
            else:
                # 使用原本的情緒反思
                prompt_data = self.emotion_prompts.prompt_emotion_reflection(
                    emotion_history=self.emotion_analyzer.emotion_history[-10:],
                    time_period="最近"
                )

            reflection = self.completion(
                prompt=prompt_data["prompt"],
                callback=prompt_data["callback"],
                failsafe=prompt_data["failsafe"],
                retry=1
            )

            return reflection

        except Exception as e:
            print(f"[{self.name}] 情緒反思失敗: {e}")
            return None

    def get_mgt_rating_summary(self, last_n: int = 5) -> str:
        """
        獲取 MGT 評分員的摘要報告

        Args:
            last_n: 最近幾次評分

        Returns:
            摘要文字
        """
        if not self.mgt_rater:
            return "MGT 評分員未啟用"

        return self.mgt_rater.get_rating_summary(last_n)

    # ========== 狀態更新 ==========

    def update_emotion_status(self):
        """
        更新帶情緒的狀態描述

        在 Agent.think() 每輪循環時調用

        更新 self.currently（當前狀態文字）
        """
        try:
            # 檢查情緒衰減
            self.emotion_analyzer.check_emotion_decay(datetime.now())

            # 生成狀態描述
            current_activity = self.action.event.get_describe(with_subject=False)

            prompt_data = self.emotion_prompts.prompt_emotion_status_description(
                current_emotion=self.emotion_analyzer.current_emotion,
                current_activity=current_activity
            )

            new_status = self.completion(
                prompt=prompt_data["prompt"],
                callback=prompt_data["callback"],
                failsafe=prompt_data["failsafe"],
                retry=1
            )

            # 更新狀態（不覆蓋原有格式，只在必要時更新）
            # self.currently = new_status

        except Exception as e:
            print(f"[{self.name}] 情緒狀態更新失敗: {e}")

    # ========== 學習與反饋 ==========

    def learn_from_feedback(self, feedback: str):
        """
        從反饋中學習

        Args:
            feedback: "positive", "negative", "neutral"
        """
        self.emotion_memory.update_success_score(
            self._last_used_memory_ids,
            feedback
        )

        # 根據反饋調整情緒
        if feedback == "positive":
            new_emotion = EmotionState(
                emotion="gratitude",
                intensity=0.6,
                reasoning="收到正面反饋",
                timestamp=datetime.now().isoformat()
            )
            self.emotion_analyzer.update_emotion(new_emotion)
        elif feedback == "negative":
            new_emotion = EmotionState(
                emotion="frustration",
                intensity=0.5,
                reasoning="收到負面反饋，需要改進",
                timestamp=datetime.now().isoformat()
            )
            self.emotion_analyzer.update_emotion(new_emotion)

    # ========== 數據持久化 ==========

    def save_emotion_state(self, filepath: str):
        """
        保存情緒狀態

        Args:
            filepath: 保存路徑
        """
        data = {
            "emotion_analyzer": self.emotion_analyzer.to_dict(),
            "emotion_memory": self.emotion_memory.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[{self.name}] 情緒狀態已保存: {filepath}")

    def load_emotion_state(self, filepath: str):
        """
        載入情緒狀態

        Args:
            filepath: 保存路徑
        """
        if not os.path.exists(filepath):
            print(f"[{self.name}] 情緒狀態文件不存在: {filepath}")
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.emotion_analyzer = EmotionAnalyzer.from_dict(
                data["emotion_analyzer"]
            )
            self.emotion_memory = EmotionMemory.from_dict(
                data["emotion_memory"],
                self.emotion_analyzer
            )

            print(f"[{self.name}] 情緒狀態已載入: {filepath}")
            print(f"  當前情緒: {self.emotion_analyzer.current_emotion.get_display_name()}")
            print(f"  情緒記憶數: {len(self.emotion_memory.emotional_concepts)}")

        except Exception as e:
            print(f"[{self.name}] 載入情緒狀態失敗: {e}")

    # ========== 工具方法 ==========

    def get_emotion_display(self) -> str:
        """獲取當前情緒顯示文字"""
        return self.emotion_analyzer.current_emotion.get_display_name()

    def get_emotion_statistics(self) -> Dict:
        """獲取情緒統計資訊"""
        return {
            "current_emotion": self.emotion_analyzer.current_emotion.to_dict(),
            "emotion_history_count": len(self.emotion_analyzer.emotion_history),
            "memory_stats": self.emotion_memory.get_emotion_statistics()
        }

    def print_emotion_status(self):
        """打印情緒狀態（用於調試）"""
        print(f"\n{'='*50}")
        print(f"[{self.name}] 情緒狀態報告")
        print(f"{'='*50}")

        current = self.emotion_analyzer.current_emotion
        print(f"當前情緒: {current.get_display_name()}")
        print(f"強度: {current.intensity:.2f}")
        print(f"原因: {current.reasoning}")
        print(f"時間: {current.timestamp}")

        print(f"\n情緒歷史 (最近5個):")
        print(self.emotion_analyzer.get_recent_emotion_summary(5))

        print(f"\n記憶統計:")
        stats = self.emotion_memory.get_emotion_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"{'='*50}\n")


# ========== 整合輔助函數 ==========

def integrate_emotion_into_agent(agent_class):
    """
    將情緒系統整合到 Agent 類別

    使用裝飾器模式，無需修改原始類別

    使用方法:
    ```python
    from emotion_system.agent_emotion_mixin import integrate_emotion_into_agent

    # 在創建 Agent 後
    agent = Agent(...)
    integrate_emotion_into_agent(agent)
    ```
    """
    # 添加 mixin 方法到 agent 實例
    for method_name in dir(AgentEmotionMixin):
        if not method_name.startswith('_'):
            method = getattr(AgentEmotionMixin, method_name)
            if callable(method):
                setattr(agent_class, method_name, method.__get__(agent_class))

    return agent_class
