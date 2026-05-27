import os
import json
import argparse
import collections
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for

from compress import frames_per_step, file_movement
from start import load_personas_from_config

# 載入AI居民列表
personas = load_personas_from_config()

# CLI 預設模擬名稱（在 __main__ 設定）
DEFAULT_SIM_NAME = ""


def _resolve_active_simulation():
    """問卷填寫時綁定的 simulation = `python replay.py --name <X>` 的 <X>。

    沒指定 `--name` 時回傳 None（居民只用靜態人設回答）。
    """
    return DEFAULT_SIM_NAME or None

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
    static_url_path="/static",
)

# 模擬的角色基本資料
persona_profiles = {
    "盧品蓉": {
        "age": 20,
        "innate": "友善、樂於助人",
        "learned": "擅長溝通、組織能力強",
        "lifestyle": "喜歡社交活動，經常參與社團",
        "currently": "正在準備期末考試"
    },
    "鄭傑丞": {
        "age": 21,
        "innate": "認真負責、注重細節",
        "learned": "程式設計、數據分析",
        "lifestyle": "規律作息，喜歡閱讀和學習",
        "currently": "正在做實習專案"
    },
    "莊于萱": {
        "age": 19,
        "innate": "創意豐富、藝術天賦",
        "learned": "繪畫、設計、攝影",
        "lifestyle": "自由奔放，喜歡探索新事物",
        "currently": "正在準備藝術作品展覽"
    },
    "施宇鴻": {
        "age": 22,
        "innate": "邏輯思維強、冷靜理性",
        "learned": "工程技術、問題解決",
        "lifestyle": "喜歡思考和研究，偏向獨處",
        "currently": "正在進行研究項目"
    },
    "游庭瑄": {
        "age": 45,
        "innate": "博學多聞、循循善誘",
        "learned": "教學經驗豐富、學術研究",
        "lifestyle": "熱愛教育，關心學生成長",
        "currently": "正在指導學生論文"
    },
    "李昇峰": {
        "age": 50,
        "innate": "細心專業、服務至上",
        "learned": "藥學專業、健康諮詢",
        "lifestyle": "穩重可靠，重視家庭和工作平衡",
        "currently": "正在管理藥店業務"
    },
    "魏祺紘": {
        "age": 18,
        "innate": "活潑好動、求知慾強",
        "learned": "快速學習、適應能力強",
        "lifestyle": "青春活力，喜歡嘗試新事物",
        "currently": "正在適應大學生活"
    },
    "陳冠佑": {
        "age": 35,
        "innate": "健談幽默、善於交際",
        "learned": "調酒技藝、人際溝通",
        "lifestyle": "夜貓子，享受與人互動的樂趣",
        "currently": "正在經營酒吧生意"
    },
    "蔡宗陞": {
        "age": 28,
        "innate": "溫和親切、追求品質",
        "learned": "咖啡製作、店面管理",
        "lifestyle": "注重生活品味，熱愛咖啡文化",
        "currently": "正在研發新的咖啡配方"
    }
}


def extract_interaction_data(checkpoints_folder):
    """提取AI與物品/地區的交互數據"""
    object_interactions = collections.defaultdict(int)
    location_interactions = collections.defaultdict(int)
    
    print(f"正在提取交互數據，資料夾：{checkpoints_folder}")
    
    try:
        checkpoint_files = [f for f in os.listdir(checkpoints_folder) 
                           if f.startswith("simulate-") and f.endswith(".json")]
        checkpoint_files.sort()
        print(f"找到 {len(checkpoint_files)} 個檢查點檔案")
    except FileNotFoundError:
        print(f"錯誤：找不到資料夾 {checkpoints_folder}")
        return object_interactions, location_interactions
    
    if not checkpoint_files:
        print("警告：沒有找到任何檢查點檔案")
        return object_interactions, location_interactions
    
    for checkpoint_file in checkpoint_files:
        file_path = os.path.join(checkpoints_folder, checkpoint_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "agents" not in data:
                print(f"警告：檔案 {checkpoint_file} 中沒有 agents 數據")
                continue
                
            for agent_name, agent_data in data["agents"].items():
                if agent_name not in personas:
                    continue
                
                # 檢查 action 和 event 結構
                if "action" not in agent_data:
                    continue
                    
                action = agent_data["action"]
                if "event" not in action or not action["event"]:
                    continue
                    
                event = action["event"]
                if "address" not in event or not event["address"]:
                    continue
                
                address = event["address"]

                # 確保 address 是列表
                if not isinstance(address, list):
                    continue

                # 地址層次：['the Ville', '建築', '場所/區域', '物品/設施']
                # 例：['the Ville', '柳樹市場和藥店', '商店', '藥店櫃檯']

                # 處理「物品交互」= 最深層（address[-1]），具體設施
                if len(address) >= 3:
                    obj = address[-1]
                    if obj and obj.strip():
                        object_interactions[(agent_name, obj.strip())] += 1

                # 處理「地區交互」= 倒數第二層（address[-2]），場所/房間
                if len(address) >= 3:
                    loc = address[-2]
                    if loc and loc.strip():
                        location_interactions[(agent_name, loc.strip())] += 1
                elif len(address) == 2:
                    # 只有兩層時，倒數第二層其實是「建築」，當作地區
                    loc = address[-1]
                    if loc and loc.strip():
                        location_interactions[(agent_name, loc.strip())] += 1
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"錯誤：讀取檔案 {checkpoint_file} 時發生錯誤：{e}")
            continue
    
    print(f"提取完成：物品交互 {len(object_interactions)} 筆，地區交互 {len(location_interactions)} 筆")
    return object_interactions, location_interactions


def _render_simulation_picker():
    """`/` 沒給 name 時顯示可用 simulation 清單供使用者點選。"""
    compressed_root = "results/compressed"
    items = []
    if os.path.isdir(compressed_root):
        for entry in sorted(os.listdir(compressed_root)):
            folder = os.path.join(compressed_root, entry)
            if not os.path.isdir(folder):
                continue
            replay_ready = os.path.exists(os.path.join(folder, file_movement))
            md_ready = os.path.exists(os.path.join(folder, "simulation.md"))
            items.append({
                "name": entry,
                "replay_ready": replay_ready,
                "has_simulation_md": md_ready,
            })

    rows = []
    for it in items:
        if it["replay_ready"]:
            link = f'<a href="/?name={it["name"]}">▶ 開始回放</a>'
        else:
            link = '<span style="color:#888">尚未壓縮（請先 python compress.py --name ...）</span>'
        md_tag = ' · <span style="color:#0a7">md</span>' if it["has_simulation_md"] else ''
        rows.append(f'<li><b>{it["name"]}</b>{md_tag} — {link}</li>')

    body = (
        "<h2>請選擇要回放的模擬</h2>"
        "<p style='color:#666'>提示：可用 <code>python replay.py --name &lt;sim&gt;</code> "
        "啟動時就指定，下次直接訪問 / 即可。</p>"
        f"<ul>{''.join(rows) if rows else '<li>尚無可用模擬，請先執行 start.py 與 compress.py</li>'}</ul>"
        '<p><a href="/surveys">→ 進入問卷系統</a></p>'
    )
    return f"<!doctype html><meta charset='utf-8'><title>Replay</title>{body}"


@app.route("/", methods=['GET'])
def index():
    name = request.args.get("name", "")          # 紀錄名稱
    step = int(request.args.get("step", 0))      # 回放起始步數
    speed = int(request.args.get("speed", 2))    # 回放速度（0~5）
    zoom = float(request.args.get("zoom", 0.8))  # 畫面缩放比例

    # 沒帶 name 時：優先使用 CLI 預設；仍沒有則顯示可用模擬清單
    if not name and DEFAULT_SIM_NAME:
        return redirect(url_for("index", name=DEFAULT_SIM_NAME, step=step or None,
                                speed=speed, zoom=zoom))

    if not name:
        return _render_simulation_picker()

    compressed_folder = f"results/compressed/{name}"

    replay_file = f"{compressed_folder}/{file_movement}"
    if not os.path.exists(replay_file):
        return f"The data file doesn't exist: '{replay_file}'<br />Run compress.py to generate the data first."

    with open(replay_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    if step < 1:
        step = 1
    if step > 1:
        # 重新設置回放的起始時間
        t = datetime.fromisoformat(params["start_datetime"])
        dt = t + timedelta(minutes=params["stride"]*(step-1))
        params["start_datetime"] = dt.isoformat()
        step = (step-1) * frames_per_step + 1
        if step >= len(params["all_movement"]):
            step = len(params["all_movement"])-1

        # 重新設置Agent的初始位置
        for agent in params["persona_init_pos"].keys():
            persona_init_pos = params["persona_init_pos"]
            persona_step_pos = params["all_movement"][f"{step}"]
            persona_init_pos[agent] = persona_step_pos[agent]["movement"]

    if speed < 0:
        speed = 0
    elif speed > 5:
        speed = 5
    speed = 2 ** speed

    return render_template(
        "index.html",
        persona_names=personas,
        step=step,
        play_speed=speed,
        zoom=zoom,
        **params
    )


@app.route("/interaction-graph", methods=['GET'])
def interaction_graph():
    name = request.args.get("name", "")  # 紀錄名稱

    if len(name) > 0:
        checkpoint_folder = f"results/checkpoints/{name}"
    else:
        return f"Invalid name of the simulation: '{name}'"

    # 讀取對話數據
    conversation_file = f"{checkpoint_folder}/conversation.json"
    if not os.path.exists(conversation_file):
        return f"The conversation file doesn't exist: '{conversation_file}'"

    with open(conversation_file, "r", encoding="utf-8") as f:
        conversation_data = json.load(f)

    # 計算角色之間的互動次數和對話長度
    interaction_count = collections.defaultdict(int)  # 互動次數
    chat_length = collections.defaultdict(int)        # 對話總長度
    
    print(f"Processing conversation data for {len(conversation_data)} time periods...")
    
    for time_key, interactions_list in conversation_data.items():
        print(f"Processing time: {time_key}, interactions: {len(interactions_list)}")
        
        for interaction in interactions_list:
            # 每個互動記錄的結構: {"角色1 -> 角色2 @ 位置": [("角色1", "對話內容"), ("角色2", "對話內容"), ...]}
            for conversation_key, chat_history in interaction.items():
                try:
                    # 解析對話鍵值，格式: "角色1 -> 角色2 @ 位置"
                    if " -> " in conversation_key and " @ " in conversation_key:
                        # 提取角色名稱
                        participants_part = conversation_key.split(" @ ")[0]  # "角色1 -> 角色2"
                        parts = participants_part.split(" -> ")
                        
                        if len(parts) == 2:
                            person1 = parts[0].strip()
                            person2 = parts[1].strip()
                            
                            # 確保角色在已知角色列表中
                            if person1 in personas and person2 in personas:
                                # 創建統一的鍵值對（按字母順序排序，避免重複）
                                pair_key = tuple(sorted([person1, person2]))
                                
                                # 統計互動次數（每次對話算一次互動）
                                interaction_count[pair_key] += 1
                                
                                # 統計對話長度（所有對話內容的字符總數）
                                if isinstance(chat_history, list):
                                    for speaker, message in chat_history:
                                        if isinstance(message, str):
                                            chat_length[pair_key] += len(message)
                                            
                                print(f"  Found interaction: {person1} <-> {person2}, messages: {len(chat_history)}")
                            else:
                                print(f"  Skipping unknown personas: {person1}, {person2}")
                        else:
                            print(f"  Invalid conversation key format: {conversation_key}")
                    else:
                        print(f"  Skipping malformed key: {conversation_key}")
                        
                except Exception as e:
                    print(f"  Error processing conversation key '{conversation_key}': {e}")
                    continue

    # 獲取物品和地區交互數據
    object_interactions, location_interactions = extract_interaction_data(checkpoint_folder)

    print(f"Total unique interactions found: {len(interaction_count)}")
    print("Interaction summary:")
    for (p1, p2), count in interaction_count.items():
        length = chat_length[(p1, p2)]
        print(f"  {p1} <-> {p2}: {count} interactions, {length} characters")

    # 創建D3.js力導向圖所需的數據格式
    nodes = []
    for i, person in enumerate(personas):
        nodes.append({
            "id": person,
            "group": (i % 3) + 1  # 分成3個組，用於不同顏色
        })

    # 創建連接數據（基於互動次數）
    interaction_links = []
    for (person1, person2), count in interaction_count.items():
        if count > 0:  # 只添加有互動的連接
            interaction_links.append({
                "source": person1,
                "target": person2,
                "value": count  # 互動次數
            })

    # 創建連接數據（基於對話長度）
    chat_length_links = []
    for (person1, person2), length in chat_length.items():
        if length > 0:  # 只添加有對話的連接
            chat_length_links.append({
                "source": person1,
                "target": person2,
                "length": length  # 對話總長度
            })

    # 如果沒有真實數據，創建一些示例數據
    if not interaction_links and not chat_length_links:
        print("No interaction data found, creating sample data...")
        sample_interactions = [
            ("盧品蓉", "魏祈紘", 15, 450),
            ("莊于萱", "施宇鴻", 12, 320),
            ("游庭瑄", "李昇峰", 25, 680),
            ("鄭傑丞", "游庭瑄", 8, 190),
            ("陳冠佑", "蔡宗陞", 18, 520),
            ("盧品蓉", "莊于萱", 6, 180),
            ("魏祈紘", "鄭傑丞", 9, 240),
            ("李昇峰", "陳冠佑", 7, 200),
            ("施宇鴻", "蔡宗陞", 5, 150),
            ("游庭瑄", "盧品蓉", 4, 120),
        ]
        
        for person1, person2, count, length in sample_interactions:
            if person1 in personas and person2 in personas:
                interaction_links.append({
                    "source": person1,
                    "target": person2,
                    "value": count
                })
                chat_length_links.append({
                    "source": person1,
                    "target": person2,
                    "length": length
                })

    # 構建返回數據
    interaction_data = {
        "nodes": nodes,
        "links": interaction_links
    }
    
    chat_length_data = {
        "nodes": nodes,
        "links": chat_length_links
    }

    # 轉換物品和地區交互數據為前端需要的格式
    object_interaction_list = []
    for (agent, obj), count in object_interactions.items():
        object_interaction_list.append({
            "agent": agent,
            "object": obj,
            "count": count
        })

    location_interaction_list = []
    for (agent, location), count in location_interactions.items():
        location_interaction_list.append({
            "agent": agent,
            "location": location,
            "count": count
        })

    # 合併所有交互數據
    all_interaction_data = {
        "chat_interactions": interaction_data,
        "chat_lengths": chat_length_data,
        "object_interactions": object_interaction_list,
        "location_interactions": location_interaction_list,
        "persona_profiles": persona_profiles
    }

    print(f"Final data - Nodes: {len(nodes)}, Interaction links: {len(interaction_links)}, Chat length links: {len(chat_length_links)}")
    print(f"Object interactions: {len(object_interaction_list)}, Location interactions: {len(location_interaction_list)}")

    # 將數據作為JSON字符串傳遞給模板，避免在模板中進行復雜的數據處理
    return render_template(
        "interaction_graph.html",
        interaction_data=json.dumps(interaction_data, ensure_ascii=False),
        chat_length_data=json.dumps(chat_length_data, ensure_ascii=False),
        all_interaction_data=json.dumps(all_interaction_data, ensure_ascii=False),
        persona_names=personas
    )


@app.route("/object-interaction", methods=['GET'])
def object_interaction():
    """AI與物品/地區交互分析頁面"""
    name = request.args.get("name", "")

    if len(name) > 0:
        checkpoint_folder = f"results/checkpoints/{name}"
    else:
        return f"Invalid name of the simulation: '{name}'"

    print(f"正在分析交互數據，模擬名稱：{name}")
    print(f"檢查點資料夾：{checkpoint_folder}")

    # 獲取物品和地區交互數據
    object_interactions, location_interactions = extract_interaction_data(checkpoint_folder)

    # 轉換為前端需要的格式
    object_interaction_list = []
    for (agent, obj), count in object_interactions.items():
        object_interaction_list.append({
            "agent": agent,
            "object": obj,
            "count": count
        })

    location_interaction_list = []
    for (agent, location), count in location_interactions.items():
        location_interaction_list.append({
            "agent": agent,
            "location": location,
            "count": count
        })

    # 如果沒有真實數據，創建一些示例數據避免頁面錯誤
    if not object_interaction_list and not location_interaction_list:
        print("沒有找到交互數據，創建示例數據...")
        # 創建示例數據
        sample_objects = ["電腦", "咖啡機", "書桌", "椅子", "手機", "筆記本", "投影機", "白板"]
        sample_locations = ["辦公室", "咖啡廳", "圖書館", "會議室", "休息室", "教室", "實驗室", "宿舍"]
        
        import random
        for i, agent in enumerate(personas):
            # 物品交互
            for j in range(3):  # 每個 AI 與 3 個物品有交互
                obj = sample_objects[(i + j) % len(sample_objects)]
                object_interaction_list.append({
                    "agent": agent,
                    "object": obj,
                    "count": random.randint(5, 25)
                })
            
            # 地區交互
            for j in range(2):  # 每個 AI 與 2 個地區有交互
                loc = sample_locations[(i + j) % len(sample_locations)]
                location_interaction_list.append({
                    "agent": agent,
                    "location": loc,
                    "count": random.randint(8, 30)
                })

    # 按交互次數排序，優先顯示高頻交互
    object_interaction_list.sort(key=lambda x: x['count'], reverse=True)
    location_interaction_list.sort(key=lambda x: x['count'], reverse=True)

    interaction_data = {
        "object_interactions": object_interaction_list,
        "location_interactions": location_interaction_list
    }

    print(f"最終數據：物品交互 {len(object_interaction_list)} 筆，地區交互 {len(location_interaction_list)} 筆")
    
    # 打印前幾筆數據用於調試
    if object_interaction_list:
        print("物品交互示例：")
        for item in object_interaction_list[:3]:
            print(f"  {item['agent']} -> {item['object']}: {item['count']}")
    
    if location_interaction_list:
        print("地區交互示例：")
        for item in location_interaction_list[:3]:
            print(f"  {item['agent']} -> {item['location']}: {item['count']}")

    return render_template(
        "object_interaction.html",
        interaction_data=interaction_data,
        persona_names=personas
    )


# 問卷系統路由
@app.route("/surveys", methods=['GET'])
def surveys_list():
    """問卷列表頁面"""
    from survey_system import SurveyManager
    manager = SurveyManager()
    surveys = manager.get_all_surveys()
    
    return render_template(
        "surveys/list.html",
        surveys=surveys
    )


@app.route("/surveys/create", methods=['GET', 'POST'])
def survey_create():
    """創建問卷頁面"""
    if request.method == 'POST':
        from survey_system import SurveyManager, SurveyImportManager
        
        manager = SurveyManager()
        import_manager = SurveyImportManager()
        
        # 處理問卷創建
        import_type = request.form.get('import_type', 'manual')
        
        if import_type == 'manual':
            # 手動創建問卷
            title = request.form.get('title', '未命名問卷')
            description = request.form.get('description', '')
            
            from survey_system.models import Survey
            survey = Survey(title=title, description=description)
            
            # 從表單提取問題
            question_texts = {}
            question_types = {}
            question_required = {}
            question_options = {}
            
            # 解析表單數據
            for key, value in request.form.items():
                if key.startswith('question_text_'):
                    question_id = key.split('_')[-1]
                    question_texts[question_id] = value
                elif key.startswith('question_type_'):
                    question_id = key.split('_')[-1]
                    question_types[question_id] = value
                elif key.startswith('question_required_'):
                    question_id = key.split('_')[-1]
                    question_required[question_id] = True
                elif key.startswith('option_'):
                    # 格式: option_questionId_optionId
                    parts = key.split('_')
                    if len(parts) >= 3:
                        question_id = parts[1]
                        if question_id not in question_options:
                            question_options[question_id] = []
                        if value.strip():  # 只添加非空選項
                            question_options[question_id].append(value.strip())
            
            # 創建問題
            for question_id in sorted(question_texts.keys(), key=int):
                if question_texts[question_id].strip():  # 只添加有內容的問題
                    question_type = question_types.get(question_id, 'text')
                    options = question_options.get(question_id, []) if question_type in ['single_choice', 'multiple_choice'] else None
                    required = question_id in question_required
                    
                    survey.add_question(
                        question_type=question_type,
                        question_text=question_texts[question_id].strip(),
                        options=options,
                        required=required
                    )
            
            if len(survey.questions) > 0:
                manager.save_survey(survey)
            else:
                from flask import redirect, url_for, flash
                flash('問卷必須包含至少一個問題', 'error')
                return redirect(url_for('survey_create'))
            
        elif import_type == 'url':
            # 從URL匯入
            url = request.form.get('import_url', '')
            survey = import_manager.import_survey(url, 'url')
            if survey:
                manager.save_survey(survey)
            
        elif import_type == 'json':
            # 從JSON匯入
            json_data = request.form.get('json_data', '')
            survey = import_manager.import_survey(json_data, 'json')
            if survey:
                manager.save_survey(survey)
        
        from flask import redirect
        return redirect('/surveys')
    
    return render_template("surveys/create.html")


@app.route("/surveys/<survey_id>", methods=['GET'])
def survey_detail(survey_id):
    """問卷詳情頁面"""
    from survey_system import SurveyManager
    manager = SurveyManager()

    survey = manager.load_survey(survey_id)
    if not survey:
        return "問卷不存在", 404

    responses = manager.get_responses_by_survey(survey_id)
    stats = manager.get_survey_stats(survey_id)
    active_simulation = _resolve_active_simulation()

    return render_template(
        "surveys/detail.html",
        survey=survey,
        responses=responses,
        stats=stats,
        active_simulation=active_simulation
    )


@app.route("/surveys/<survey_id>/fill", methods=['POST'])
def survey_fill(survey_id):
    """讓AI居民填寫問卷（自動綁定當前 simulation）"""
    from survey_system import SurveyManager, AIResidentSurveyFiller

    simulation_name = _resolve_active_simulation()

    manager = SurveyManager()
    try:
        filler = AIResidentSurveyFiller(manager, simulation_name=simulation_name)
        responses = filler.fill_survey_for_all_residents(survey_id)
        return {
            "success": True,
            "message": f"成功生成 {len(responses)} 個AI居民的問卷回應",
            "response_count": len(responses),
            "simulation_name": simulation_name,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}, 400


@app.route("/surveys/simulations", methods=['GET'])
def survey_simulations():
    """列出可用的模擬上下文（給前端動態下拉選單使用）"""
    from survey_system import SimulationContext
    return {"simulations": SimulationContext.list_available()}


@app.route("/surveys/<survey_id>/clear-responses", methods=['POST'])
def survey_clear_responses(survey_id):
    """清空問卷回應。mode=all 全清；mode=fallback 只清明顯 fallback 的回應。"""
    import json as _json
    from survey_system import SurveyManager

    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "all")  # all | fallback

    manager = SurveyManager()
    survey = manager.load_survey(survey_id)
    if not survey:
        return {"success": False, "error": "問卷不存在"}, 404

    if mode == "all":
        deleted = manager.delete_survey_responses(survey_id)
        return {"success": True, "deleted": deleted, "mode": "all"}

    # mode == fallback：逐一檢查並刪除高比例 fallback 的回應
    fallback_signals = ["其他", "其他收入", "抱歉，目前無法提供", "無法回答"]
    responses_dir = os.path.join(manager.storage_path, "responses")
    deleted = 0
    if not os.path.isdir(responses_dir):
        return {"success": True, "deleted": 0, "mode": "fallback"}

    for filename in os.listdir(responses_dir):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(responses_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            continue
        if data.get("survey_id") != survey_id:
            continue
        answers = data.get("responses", {})
        total = len(answers)
        if total == 0:
            continue
        fb = 0
        for v in answers.values():
            if isinstance(v, str) and any(s in v for s in fallback_signals):
                fb += 1
            elif isinstance(v, list) and v and all(
                isinstance(x, str) and any(s in x for s in fallback_signals) for x in v
            ):
                fb += 1
        if fb / total >= 0.7:
            try:
                os.remove(path)
                deleted += 1
            except Exception:
                pass

    return {"success": True, "deleted": deleted, "mode": "fallback"}


@app.route("/surveys/<survey_id>/export", methods=['GET'])
def survey_export(survey_id):
    """匯出問卷結果"""
    from survey_system import SurveyManager, SurveyExportManager

    format_type = request.args.get('format', 'csv')

    manager = SurveyManager()
    exporter = SurveyExportManager(manager)

    try:
        output_path = exporter.export_survey(survey_id, format_type)

        # 返回文件下載
        from flask import send_file
        return send_file(
            output_path,
            as_attachment=True,
            download_name=os.path.basename(output_path)
        )
    except Exception as e:
        return f"匯出失敗: {str(e)}", 400


@app.route("/surveys/<survey_id>/analytics", methods=['GET'])
def survey_analytics(survey_id):
    """問卷數據視覺化分析頁面"""
    from survey_system import SurveyManager

    manager = SurveyManager()
    survey = manager.load_survey(survey_id)

    if not survey:
        return "問卷不存在", 404

    # 獲取詳細分析數據
    analytics = manager.get_response_analytics(survey_id)

    if not analytics:
        return "無法獲取分析數據", 400

    return render_template(
        "surveys/analytics.html",
        survey=survey,
        analytics=analytics
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay server for the simulated village")
    parser.add_argument("--name", type=str, default="",
                        help="預設要載入的模擬名稱（例：scafv10）。設定後 / 會自動帶入。")
    parser.add_argument("--port", type=int, default=5001, help="HTTP port（預設 5001）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="綁定主機（預設 127.0.0.1）")
    parser.add_argument("--no-debug", action="store_true", help="關閉 Flask debug 模式")
    args = parser.parse_args()

    DEFAULT_SIM_NAME = args.name
    if DEFAULT_SIM_NAME:
        print(f"[replay] 預設模擬: {DEFAULT_SIM_NAME} → 直接開 http://{args.host}:{args.port}/")
    else:
        print(f"[replay] 未指定預設模擬，開 http://{args.host}:{args.port}/ 會看到清單")

    app.run(host=args.host, port=args.port, debug=not args.no_debug)