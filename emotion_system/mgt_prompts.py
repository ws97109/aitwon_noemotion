"""
MGT 情緒評分員專用提示詞擴展

為 EmotionPrompts 提供 MGT 相關的提示詞方法
"""

from typing import Dict, List, Optional, Any
from .core import EmotionState, EMOTION_TYPES, EMOTION_DISPLAY, EMOTION_STYLES


def _get_emotion_name(emotion_code: str) -> str:
    """從 EMOTION_DISPLAY 中提取情緒中文名稱"""
    display = EMOTION_DISPLAY.get(emotion_code, emotion_code)
    if " " in display:
        return display.split(" ")[1]
    return display


def prompt_mgt_enhanced_chat(
    agent_name: str,
    other_agent_name: str,
    chat_context: str,
    current_emotion: EmotionState,
    mgt_suggestions: List[str],
    retrieved_memories: List[str] = None,
    chat_history: List[Any] = None,
    # ✅ 新增：原始系統的所有參數
    base_desc: str = None,              # 基本描述（年齡、性格特質等）
    current_location: str = None,       # 當前位置
    current_time: str = None,           # 當前時間
    relation: str = None,               # 關係描述
    previous_context: str = None        # 歷史對話摘要（8小時內）
) -> Dict[str, Any]:
    """
    整合 MGT 評分員建議的對話生成

    結合原本的 prompt 和 MGT 評分員給出的建議

    Args:
        agent_name: AI 居民名稱
        other_agent_name: 對話對象名稱
        chat_context: 對話情境
        current_emotion: 當前情緒狀態
        mgt_suggestions: MGT 評分員提供的建議列表
        retrieved_memories: 檢索到的記憶
        chat_history: 最近的對話歷史記錄
        base_desc: 基本描述（年齡、性格特質等）
        current_location: 當前位置
        current_time: 當前時間
        relation: 關係描述
        previous_context: 歷史對話摘要（8小時內）

    Returns:
        {prompt, callback, failsafe, retry}
    """
    emotion_display = EMOTION_DISPLAY.get(current_emotion.emotion, '😐 中性')
    emotion_style = EMOTION_STYLES.get(current_emotion.emotion, {})
    tone = emotion_style.get("tone", "平和")

    # ✅ 構建基本資訊（原始系統）
    base_info = ""
    if base_desc:
        base_info = f"\n## {agent_name} 的基本資訊\n{base_desc}\n"

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
        relation_desc = f"\n## 關係\n{agent_name} 和 {other_agent_name} 的關係：{relation}\n"

    # ✅ 構建歷史對話摘要（原始系統）
    previous_conv = ""
    if previous_context:
        previous_conv = f"\n## 背景\n{previous_context}\n"

    # 構建記憶上下文（保留更多記憶，從3條增加到5條）
    memory_context = ""
    if retrieved_memories:
        memory_context = "\n## 相關記憶\n" + "\n".join([f"- {mem}" for mem in retrieved_memories[:5]])

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

    # 構建 MGT 建議部分
    mgt_guidance = ""
    if mgt_suggestions:
        mgt_guidance = "\n## 情緒評分員建議（MGT）\n" + "\n".join([f"{i+1}. {sug}" for i, sug in enumerate(mgt_suggestions[:3])])

    prompt = f"""你是 {agent_name}，正在與 {other_agent_name} 進行對話。
{base_info}{location_time}{relation_desc}{previous_conv}
## 當前情緒狀態
{emotion_display}（強度：{current_emotion.intensity:.2f}）
情緒原因：{current_emotion.reasoning}

## 對話風格
語氣：{tone}
行為特徵：{emotion_style.get('behavior', '自然互動')}

## 對話情境
{chat_context}
{memory_context}
{history_context}
{mgt_guidance}

## 任務
請根據以上所有資訊，生成 {agent_name} 的回應。

要求：
1. 自然融入當前情緒（不要直接說出情緒名稱）
2. 考慮評分員的建議但保持自然
3. 符合 {agent_name} 的性格特質和年齡
4. 考慮當前的時間和地點
5. 考慮與 {other_agent_name} 的關係
6. 簡潔（2-3句話）
7. **重要：根據對話歷史中對方最後說的話進行回應，確保對話連貫**
8. **重要：推進話題深度，提供實質內容，避免只是附和或重複對方的意思**
9. **重要：如果對方提出具體話題（如討論音樂、規劃活動），要給出具體的想法或建議，不要只說「好啊我們討論」或「我們等下談」**

請直接輸出對話內容，不要添加任何格式標記："""

    def callback(response: str) -> str:
        """清理回應"""
        response = response.strip()
        # 移除可能的引號
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        return response

    # 動態生成更自然的 failsafe，根據情緒和情境
    failsafe_responses = {
        "joy": ["真好！", "太棒了！", "聽起來不錯。", "我很高興聽到這個。"],
        "sadness": ["嗯...", "我懂了。", "是這樣啊...", "我理解。"],
        "anger": ["好吧。", "知道了。", "是嗎。", "這樣啊。"],
        "fear": ["哦，明白了。", "我知道了。", "是這樣的話...", "好的，我懂了。"],
        "surprise": ["真的嗎？", "這樣啊！", "沒想到。", "是嗎！"],
        "disgust": ["這樣...", "哦。", "好吧...", "我知道了。"],
        "neutral": ["嗯，好的。", "我明白。", "了解了。", "知道了。"]
    }

    import random
    emotion_key = current_emotion.emotion if current_emotion.emotion in failsafe_responses else "neutral"
    failsafe = random.choice(failsafe_responses[emotion_key])

    return {
        "prompt": prompt,
        "callback": callback,
        "failsafe": failsafe,
        "retry": 2
    }


def prompt_mgt_rater_explanation(
    agent_name: str,
    mgt_output: Any,  # EmotionRaterOutput
    current_context: str
) -> Dict[str, Any]:
    """
    生成 MGT 評分員的詳細解釋（用於調試和分析）

    Args:
        agent_name: AI 居民名稱
        mgt_output: MGT 評分員輸出
        current_context: 當前情境

    Returns:
        {prompt, callback, failsafe, retry}
    """
    emotion_display = EMOTION_DISPLAY.get(mgt_output.predicted_emotion, "😐")
    emotion_name = _get_emotion_name(mgt_output.predicted_emotion)

    prompt = f"""作為情緒分析專家，請幫助 {agent_name} 理解以下情緒評估結果：

## 當前情境
{current_context}

## MGT 評分員評估結果
預測情緒：{emotion_display} {emotion_name}
情緒強度：{mgt_output.emotion_intensity:.2f}
預測信心：{mgt_output.confidence:.2f}

## 評估依據
{mgt_output.reasoning}

## 注意力分析
- 情境注意力：{mgt_output.attention_weights.get('context', 0):.2f}
- 記憶注意力：{mgt_output.attention_weights.get('memory', 0):.2f}

## 資訊融合度
- 多模態融合：{mgt_output.gating_values.get('multimodal_gate', 0):.2f}
- 語言保留：{mgt_output.gating_values.get('language_preserve', 0):.2f}

請用通俗易懂的語言解釋：
1. 為什麼會有這樣的情緒評估？
2. 主要影響因素是什麼？
3. 這個評估結果對接下來的互動有什麼啟示？

請以 2-3 句話簡潔說明："""

    def callback(response: str) -> str:
        return response.strip()

    failsafe = f"根據當前情境，評估為{emotion_name}情緒（強度{mgt_output.emotion_intensity:.1f}），主要考慮了當前情境和過往經驗。"

    return {
        "prompt": prompt,
        "callback": callback,
        "failsafe": failsafe,
        "retry": 1
    }


def prompt_reflect_on_mgt_feedback(
    agent_name: str,
    recent_mgt_ratings: List[Any],  # List[EmotionRaterOutput]
    interaction_outcomes: List[str]
) -> Dict[str, Any]:
    """
    基於 MGT 評分員的反饋進行情緒反思

    Args:
        agent_name: AI 居民名稱
        recent_mgt_ratings: 最近的 MGT 評分記錄
        interaction_outcomes: 互動結果

    Returns:
        {prompt, callback, failsafe, retry}
    """
    # 統計情緒分佈
    emotion_counts = {}
    for rating in recent_mgt_ratings:
        emotion = rating.predicted_emotion
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    dominant_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    emotion_summary = ", ".join([f"{_get_emotion_name(e)}({c}次)" for e, c in dominant_emotions])

    # 構建評分摘要
    rating_summary = []
    for i, rating in enumerate(recent_mgt_ratings[-5:], 1):
        emotion_name = _get_emotion_name(rating.predicted_emotion)
        rating_summary.append(
            f"{i}. {emotion_name}（強度{rating.emotion_intensity:.1f}）- {rating.reasoning[:30]}"
        )

    outcome_summary = "\n".join([f"- {outcome}" for outcome in interaction_outcomes[-3:]])

    prompt = f"""作為 {agent_name}，請反思最近的情緒變化和互動經驗。

## 最近情緒統計
主要情緒：{emotion_summary}

## 詳細評分記錄
{chr(10).join(rating_summary)}

## 互動結果
{outcome_summary}

## 反思問題
1. 你的情緒變化模式是什麼？
2. 哪些情境容易觸發特定情緒？
3. MGT 評分員的建議是否有幫助？
4. 你希望如何調整未來的情緒反應？

請用第一人稱（我）的角度，寫 2-3 句簡潔的反思："""

    def callback(response: str) -> str:
        response = response.strip()
        # 確保以第一人稱
        if not any(response.startswith(word) for word in ["我", "最近", "這段時間"]):
            response = f"我注意到{response}"
        return response

    failsafe = f"我發現最近的情緒主要是{emotion_summary}，我會繼續調整我的互動方式。"

    return {
        "prompt": prompt,
        "callback": callback,
        "failsafe": failsafe,
        "retry": 2
    }
