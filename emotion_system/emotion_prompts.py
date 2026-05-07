"""
情緒提示詞模組

提供與情緒相關的 LLM 提示詞構建方法
這些方法將整合到 Agent.completion() 工作流中
"""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json

from .core import EmotionState, EMOTION_TYPES, EMOTION_DISPLAY, EMOTION_STYLES


class EmotionPrompts:
    """
    情緒提示詞管理器

    提供各種情緒相關的提示詞構建方法
    每個方法返回格式: {prompt, callback, failsafe, retry}
    與現有的 Scratch.prompt_XXX() 方法格式一致
    """

    LANGUAGE_SUFFIX = (
        "\n\n【語言要求】請一律使用繁體中文（台灣用字）回答，"
        "避免出現簡體中文字。JSON 結構與英文欄位名稱保持原樣，"
        "僅自然語言內容需為繁體中文。"
    )

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    # ========== 1. 情緒評估提示詞 ==========

    def prompt_emotion_appraisal(
        self,
        event_description: str,
        event_type: str,  # "event", "chat", "thought"
        poignancy: int,
        context: str = "",
        recent_emotions: str = ""
    ) -> Dict[str, Any]:
        """
        情緒評估提示詞

        分析事件/對話/思考應產生的情緒

        Returns:
            {prompt, callback, failsafe, retry}
        """

        prompt = f"""你是情緒分析專家。{self.agent_name} 剛經歷了以下情況，請判斷他/她應有的情緒狀態。

## 情況描述
{event_description}

## 情境類型
{event_type} (事件/對話/思考)

## 重要性
{poignancy}/10

## 額外上下文
{context if context else "無"}

## 最近情緒歷史
{recent_emotions if recent_emotions else "無歷史記錄"}

## 可選情緒類型
{', '.join([f"{k}({v})" for k, v in EMOTION_DISPLAY.items()])}

## 判斷標準
1. 事件的性質（正面/負面/中性）
2. 重要性程度（影響情緒強度）
3. {self.agent_name} 的性格特質
4. 最近的情緒趨勢

## 輸出格式
請以 JSON 格式回應（僅輸出 JSON，不要其他文字）：
{{
  "emotion": "選擇的情緒類型（從上述列表中選擇）",
  "intensity": 0.0到1.0之間的浮點數,
  "reasoning": "簡短說明為什麼選擇此情緒（一句話）",
  "duration": 預計持續時間（分鐘，整數）
}}

範例：
{{"emotion": "joy", "intensity": 0.7, "reasoning": "收到好友的生日祝福", "duration": 30}}
"""

        def callback(response: str) -> EmotionState:
            """解析 LLM 回應為 EmotionState"""
            try:
                # 清理回應（移除可能的 markdown 代碼塊）
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1])

                data = json.loads(response)

                return EmotionState(
                    emotion=data["emotion"],
                    intensity=float(data["intensity"]),
                    reasoning=data["reasoning"],
                    timestamp=datetime.now().isoformat(),
                    duration=int(data.get("duration", 30))
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[EmotionPrompts] 解析失敗: {e}, 回應: {response}")
                # 返回 failsafe
                return None

        failsafe = EmotionState(
            emotion="neutral",
            intensity=0.5,
            reasoning="無法分析情緒，保持中性",
            timestamp=datetime.now().isoformat()
        )

        return {
            "prompt": prompt + self.LANGUAGE_SUFFIX,
            "callback": callback,
            "failsafe": failsafe,
            "retry": 2
        }

    # ========== 2. 情緒調整對話風格提示詞 ==========

    def prompt_emotion_aware_chat(
        self,
        other_agent_name: str,
        current_emotion: EmotionState,
        chat_context: str,
        retrieved_memories: str,
        chat_history: List[Any] = None,
        # ✅ 新增：原始系統的所有參數
        base_desc: str = None,              # 基本描述（年齡、性格特質等）
        current_location: str = None,       # 當前位置
        current_time: str = None,           # 當前時間
        relation: str = None,               # 關係描述
        previous_context: str = None        # 歷史對話摘要（8小時內）
    ) -> Dict[str, Any]:
        """
        情緒感知的對話生成提示詞

        根據當前情緒調整對話風格

        Args:
            chat_history: 最近的對話歷史記錄
            base_desc: 基本描述（年齡、性格特質等）
            current_location: 當前位置
            current_time: 當前時間
            relation: 關係描述
            previous_context: 歷史對話摘要（8小時內）

        Returns:
            {prompt, callback, failsafe, retry}
        """

        emotion_style = EMOTION_STYLES.get(
            current_emotion.emotion,
            EMOTION_STYLES["neutral"]
        )

        # ✅ 構建基本資訊（原始系統）
        base_info = ""
        if base_desc:
            base_info = f"\n## {self.agent_name} 的基本資訊\n{base_desc}\n"

        # ✅ 構建位置和時間（原始系統）
        location_time = ""
        if current_location or current_time:
            location_time = "\n## 當前狀況\n"
            if current_location:
                location_time += f"當前位置：{current_location}\n"
            if current_time:
                location_time += f"當前時間：{current_time}\n"

        # ✅ 構建關係描述（原始系統）
        relation_desc = ""
        if relation:
            relation_desc = f"\n## 關係\n{self.agent_name} 和 {other_agent_name} 的關係：{relation}\n"

        # ✅ 構建歷史對話摘要（原始系統）
        previous_conv = ""
        if previous_context:
            previous_conv = f"\n## 背景\n{previous_context}\n"

        # 構建對話歷史上下文
        history_context = ""
        if chat_history:
            recent_history = chat_history[-3:]  # 只取最近3輪對話
            history_lines = []
            for msg in recent_history:
                speaker = msg.get("speaker", "")
                content = msg.get("content", "")
                if speaker and content:
                    history_lines.append(f"{speaker}: {content}")
            if history_lines:
                history_context = "\n## 本次對話記錄\n" + "\n".join(history_lines)

        prompt = f"""你是 {self.agent_name}，正在與 {other_agent_name} 對話。
{base_info}{location_time}{relation_desc}{previous_conv}
## 你當前的情緒狀態
情緒：{current_emotion.get_display_name()}
強度：{current_emotion.intensity:.1f}
原因：{current_emotion.reasoning}

## 對話風格要求
{emotion_style['tone']}

行動傾向：{emotion_style['action_tendency']}
能量水平：{emotion_style['energy_level']}

## 對話情境
{chat_context}

## 相關記憶
{retrieved_memories}
{history_context}

## 要求
1. 根據當前情緒調整語氣和表達方式
2. 自然融入相關記憶（不要說"我記得"之類的）
3. 保持角色一致性和年齡特徵
4. 考慮當前的時間和地點
5. 考慮與 {other_agent_name} 的關係
6. 對話要簡潔（1-2句話）
7. **重要：根據對話歷史中對方最後說的話進行回應，確保對話連貫**
8. **重要：推進話題深度，提供實質內容，避免只是附和或重複對方的意思**
9. **重要：如果對方提出具體話題（如討論音樂、規劃活動），要給出具體的想法或建議，不要只說「好啊我們討論」或「我們等下談」**

請生成 {self.agent_name} 會說的話："""

        def callback(response: str) -> str:
            """清理回應"""
            response = response.strip()
            # 移除可能的引號
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            return response

        failsafe = f"（{self.agent_name} 一時不知道該說什麼）"

        return {
            "prompt": prompt + self.LANGUAGE_SUFFIX,
            "callback": callback,
            "failsafe": failsafe,
            "retry": 2
        }

    # ========== 3. 情緒衝突處理提示詞 ==========

    def prompt_emotion_conflict_resolution(
        self,
        agent1_emotion: EmotionState,
        agent2_emotion: str,  # 對方的情緒
        conflict_description: str
    ) -> Dict[str, Any]:
        """
        情緒衝突處理提示詞

        當兩個 agent 的情緒不相容時，決定如何反應

        Returns:
            {prompt, callback, failsafe, retry}
        """

        prompt = f"""你是 {self.agent_name}。以下是一個社交情境：

## 你的情緒
{agent1_emotion.get_display_name()} (強度: {agent1_emotion.intensity:.1f})
原因：{agent1_emotion.reasoning}

## 對方的情緒
{EMOTION_DISPLAY.get(agent2_emotion, agent2_emotion)}

## 衝突情境
{conflict_description}

## 問題
考慮到雙方的情緒狀態，{self.agent_name} 應該如何反應？

可選反應：
1. empathize（展現同理心，調整自己的情緒）
2. maintain（堅持自己的情緒，但友善表達）
3. withdraw（暫時退出互動）
4. redirect（轉移話題到中性主題）

請以 JSON 格式回應：
{{
  "action": "選擇的反應（從上述4個中選擇）",
  "reasoning": "為什麼選擇這個反應",
  "emotional_adjustment": {{
    "new_emotion": "如果需要調整情緒，新情緒是什麼（或null）",
    "intensity": 調整後的強度（0.0-1.0，或null）
  }}
}}
"""

        def callback(response: str) -> Dict:
            """解析衝突處理決策"""
            try:
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1])

                return json.loads(response)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[EmotionPrompts] 衝突處理解析失敗: {e}")
                return None

        failsafe = {
            "action": "maintain",
            "reasoning": "保持禮貌但堅持自己的立場",
            "emotional_adjustment": {
                "new_emotion": None,
                "intensity": None
            }
        }

        return {
            "prompt": prompt + self.LANGUAGE_SUFFIX,
            "callback": callback,
            "failsafe": failsafe,
            "retry": 1
        }

    # ========== 4. 情緒記憶重要性評估 ==========

    def prompt_emotional_importance(
        self,
        event_description: str,
        event_emotion: str,
        emotion_intensity: float
    ) -> Dict[str, Any]:
        """
        評估帶有情緒的事件的重要性

        擴展原有的 poignancy 評估，考慮情緒因素

        Returns:
            {prompt, callback, failsafe, retry}
        """

        prompt = f"""評估以下事件對 {self.agent_name} 的重要性。

## 事件描述
{event_description}

## 情緒資訊
情緒：{EMOTION_DISPLAY.get(event_emotion, event_emotion)}
強度：{emotion_intensity:.1f}

## 評分標準（1-10）
1-2：日常瑣事（例如：吃早餐、走路）
3-4：普通事件（例如：和朋友閒聊）
5-6：值得注意的事件（例如：學到新知識）
7-8：重要事件（例如：達成小目標、收到讚美）
9-10：極重要事件（例如：重大決定、深刻的情感體驗）

## 情緒加成
- 強烈情緒（intensity > 0.7）應增加1-2分
- 負面高強度情緒（焦慮、悲傷）特別重要
- 正面高強度情緒（喜悅、興奮）也很重要

請僅回答一個1-10之間的整數："""

        def callback(response: str) -> int:
            """解析重要性分數"""
            try:
                # 提取數字
                import re
                match = re.search(r'\b(\d+)\b', response.strip())
                if match:
                    score = int(match.group(1))
                    return max(1, min(10, score))
                return None
            except ValueError:
                return None

        # failsafe：根據情緒強度估算
        failsafe = max(5, min(10, int(5 + emotion_intensity * 5)))

        return {
            "prompt": prompt + self.LANGUAGE_SUFFIX,
            "callback": callback,
            "failsafe": failsafe,
            "retry": 2
        }

    # ========== 5. 情緒反思提示詞 ==========

    def prompt_emotion_reflection(
        self,
        emotion_history: List[EmotionState],
        time_period: str = "今天"
    ) -> Dict[str, Any]:
        """
        情緒反思提示詞

        定期反思情緒變化，生成洞察

        Returns:
            {prompt, callback, failsafe, retry}
        """

        emotion_summary = "\n".join([
            f"- {e.get_display_name()} (強度:{e.intensity:.1f}) - {e.reasoning}"
            for e in emotion_history
        ])

        prompt = f"""你是 {self.agent_name}。回顧{time_period}的情緒經歷，進行反思。

## 情緒經歷
{emotion_summary}

## 反思問題
1. 什麼事件觸發了最強烈的情緒？
2. 我的情緒模式是什麼（例如：容易焦慮、傾向樂觀）？
3. 有什麼值得記住的情緒體驗？
4. 未來應該注意什麼？

請生成1-2句話的反思洞察："""

        def callback(response: str) -> str:
            """清理反思文字"""
            return response.strip()

        failsafe = f"{self.agent_name} 度過了平靜的一天，情緒整體穩定。"

        return {
            "prompt": prompt + self.LANGUAGE_SUFFIX,
            "callback": callback,
            "failsafe": failsafe,
            "retry": 1
        }

    # ========== 6. 情緒狀態描述生成 ==========

    def prompt_emotion_status_description(
        self,
        current_emotion: EmotionState,
        current_activity: str
    ) -> Dict[str, Any]:
        """
        生成帶情緒的狀態描述

        用於更新 agent.currently（當前狀態文字）

        Returns:
            {prompt, callback, failsafe, retry}
        """

        prompt = f"""將以下資訊整合為一句自然的狀態描述。

角色：{self.agent_name}
當前活動：{current_activity}
情緒：{current_emotion.get_display_name()} (強度:{current_emotion.intensity:.1f})

要求：
- 一句話（不超過20字）
- 自然融入情緒表現
- 不要直接說"感到XX情緒"

範例：
- "正在咖啡廳開心地喝咖啡" （活動+喜悅）
- "在圖書館專注地讀書，略感焦慮" （活動+焦慮）
- "悠閒地在公園散步" （活動+中性）

請生成狀態描述："""

        def callback(response: str) -> str:
            """清理狀態描述"""
            desc = response.strip()
            # 移除可能的引號或句號
            desc = desc.strip('"\'。.')
            return desc

        failsafe = f"正在{current_activity}"

        return {
            "prompt": prompt + self.LANGUAGE_SUFFIX,
            "callback": callback,
            "failsafe": failsafe,
            "retry": 1
        }

    # ========== 輔助方法 ==========

    def format_emotion_context(
        self,
        emotion_state: EmotionState,
        include_style: bool = True
    ) -> str:
        """
        格式化情緒上下文（用於其他提示詞）

        Args:
            emotion_state: 情緒狀態
            include_style: 是否包含風格指導

        Returns:
            格式化的情緒上下文文字
        """
        context = f"""當前情緒：{emotion_state.get_display_name()}
強度：{emotion_state.intensity:.1f}
原因：{emotion_state.reasoning}"""

        if include_style:
            style = EMOTION_STYLES.get(
                emotion_state.emotion,
                EMOTION_STYLES["neutral"]
            )
            context += f"""

回應風格：{style['tone']}
能量水平：{style['energy_level']}"""

        return context

    @staticmethod
    def parse_json_from_llm_response(response: str) -> Optional[Dict]:
        """
        從 LLM 回應中解析 JSON

        處理常見的格式問題（markdown 代碼塊等）
        """
        try:
            # 清理回應
            response = response.strip()

            # 移除 markdown 代碼塊
            if response.startswith("```"):
                lines = response.split("\n")
                # 找到 JSON 內容
                start_idx = 1
                end_idx = len(lines) - 1
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        start_idx = i
                        break
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().endswith("}"):
                        end_idx = i + 1
                        break
                response = "\n".join(lines[start_idx:end_idx])

            return json.loads(response)
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"[EmotionPrompts] JSON 解析失敗: {e}")
            print(f"原始回應: {response[:200]}")
            return None


# ========== 提示詞模板字串（用於 data/prompts/ 目錄）==========

# 這些模板可以保存為 .txt 文件，並被 Scratch.build_prompt() 使用

EMOTION_APPRAISAL_TEMPLATE = """你是情緒分析專家。{agent_name} 剛經歷了以下情況，請判斷他/她應有的情緒狀態。

情況描述：{event_description}
情境類型：{event_type}
重要性：{poignancy}/10

可選情緒：joy(喜悅), sadness(悲傷), empathy(同理), anxiety(焦慮), curiosity(好奇),
         frustration(挫折), confidence(自信), excitement(興奮), caution(謹慎),
         gratitude(感激), confusion(困惑), neutral(中性)

請以 JSON 格式回應：
{{"emotion": "情緒類型", "intensity": 0.0-1.0, "reasoning": "原因", "duration": 分鐘數}}
"""

EMOTION_AWARE_CHAT_TEMPLATE = """你是 {agent_name}，正與 {other_agent} 對話。

當前情緒：{emotion_display} (強度:{intensity})
對話風格：{tone}

情境：{context}

請生成簡潔的對話（1-2句）："""

