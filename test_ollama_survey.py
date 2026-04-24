"""
測試Ollama LLM問卷填寫功能
"""

import os
import sys
from survey_system.models import SurveyManager
from survey_system.ai_filler import AIResidentSurveyFiller

def main():
    """測試Ollama問卷填寫"""

    # 設定路徑
    simulation_md_path = "results/compressed/test_0513/simulation.md"

    if not os.path.exists(simulation_md_path):
        print(f"❌ 找不到simulation.md文件: {simulation_md_path}")
        print("請確認文件路徑是否正確")
        return

    print("=" * 60)
    print("Ollama LLM 問卷填寫測試")
    print("=" * 60)
    print()

    # 初始化問卷管理器
    print("📋 初始化問卷管理器...")
    manager = SurveyManager()

    # 初始化AI填寫器（純 Ollama LLM 模式）
    print("🤖 初始化 Ollama AI 填寫器（純 LLM 模式）...")
    print("   - 從 agent.json 動態載入 AI 居民資料")
    print("   - 從 simulation.md 載入活動歷史")
    print("   - 從 data/prompts/ 載入 prompt 模板")
    print("   ⚠️  不使用任何硬編碼規則，所有回答由 LLM 生成")
    print()

    filler = AIResidentSurveyFiller(
        survey_manager=manager,
        simulation_md_path=simulation_md_path
    )

    print(f"✓ 已載入 {len(filler.residents_info)} 位AI居民")
    print()

    # 列出所有問卷
    surveys = manager.list_surveys()

    if not surveys:
        print("❌ 目前沒有可用的問卷")
        print("請先創建一個問卷，然後再運行此測試")
        return

    print(f"找到 {len(surveys)} 個問卷:")
    for i, survey in enumerate(surveys, 1):
        print(f"  {i}. {survey['title']} (ID: {survey['survey_id']})")
    print()

    # 選擇第一個問卷進行測試
    test_survey = surveys[0]
    survey_id = test_survey['survey_id']
    survey_title = test_survey['title']

    print(f"📝 使用問卷: {survey_title}")
    print()

    # 選擇一個居民進行測試
    test_residents = ["李昇峰", "游庭瑄", "蔡宗陞"]

    for resident_name in test_residents:
        if resident_name in filler.residents_info:
            print("-" * 60)
            print(f"🧑 測試居民: {resident_name}")
            print("-" * 60)

            resident_info = filler.residents_info[resident_name]
            print(f"年齡: {resident_info['age']}歲")
            print(f"性格: {', '.join(resident_info['personality'])}")
            print(f"當前活動: {resident_info['current_activity'][:50]}...")
            print()

            try:
                print("⏳ 正在生成問卷回答（使用Ollama LLM）...")
                response = filler.fill_survey_for_resident(survey_id, resident_name)

                if response:
                    print(f"✅ {resident_name} 已完成問卷填寫")
                    print()

                    # 顯示前3個問題的回答
                    survey = manager.load_survey(survey_id)
                    print("前3個問題的回答:")
                    for i, question in enumerate(survey.questions[:3], 1):
                        answer = response.responses.get(question['id'], '無回答')
                        print(f"\n問題{i}: {question['text']}")
                        print(f"回答: {answer}")

                    if len(survey.questions) > 3:
                        print(f"\n... 還有 {len(survey.questions) - 3} 個問題的回答")

                else:
                    print(f"❌ {resident_name} 問卷填寫失敗")

            except Exception as e:
                print(f"❌ 填寫過程發生錯誤: {e}")
                import traceback
                traceback.print_exc()

            print()

            # 只測試第一個居民
            break

    print("=" * 60)
    print("測試完成！")
    print()
    print("說明:")
    print("1. ✅ 系統使用 Ollama LLM (qwen2.5:7b) 生成所有問卷回答")
    print("2. ✅ LLM 會根據以下資料生成回答:")
    print("   - agent.json: 居民背景、家庭、經濟狀況")
    print("   - simulation.md: 近期活動歷史")
    print("   - data/prompts/: 問題類型對應的 prompt 模板")
    print("3. ✅ 完全移除硬編碼規則，無預設回答")
    print("4. ⚠️  如果 Ollama 失敗，回答將顯示為「無法回答」")
    print()
    print("提示：確保 Ollama 服務運行中 (ollama serve)")
    print("=" * 60)


if __name__ == "__main__":
    main()
