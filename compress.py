import os
import json
import argparse
import collections
from datetime import datetime

from modules.maze import Maze
from start import load_personas_from_config

# 載入AI居民列表
personas = load_personas_from_config()

file_markdown = "simulation.md"
file_movement = "movement.json"

frames_per_step = 60  # 每個step包含的幀數


# 從存檔文件中讀取stride
def get_stride(json_files):
    if len(json_files) < 1:
        return 1

    with open(json_files[-1], "r", encoding="utf-8") as f:
        config = json.load(f)

    return config["stride"]


# 將address轉換為字符串
def get_location(address):
    # 僅為兼容原版
    # if address[0] == "<waiting>" or address[0] == "<persona>":
    #     return None

    # 不需要顯示address第一級（"the Ville"）
    location = "，".join(address[1:])

    return location


# 插入第0幀數據（Agent的初始狀態）
def insert_frame0(init_pos, movement, agent_name):
    key = "0"
    if key not in movement.keys():
        movement[key] = dict()

    json_path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        address = json_data["spatial"]["address"]["living_area"]
    location = get_location(address)
    coord = json_data["coord"]
    init_pos[agent_name] = coord
    movement[key][agent_name] = {
        "location": location,
        "movement": coord,
        "description": "正在睡覺",
    }
    movement["description"][agent_name] = {
        "currently": json_data["currently"],
        "scratch": json_data["scratch"],
    }
def _load_resident_buildings():
    """從 agent.json 載入 9 位居民各自的住宅建築（living_area[1]）。"""
    buildings = set()
    base = os.path.join("frontend", "static", "assets", "village", "agents")
    for name in personas:
        path = os.path.join(base, name, "agent.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            la = data.get("spatial", {}).get("address", {}).get("living_area", [])
            if len(la) > 1 and la[1]:
                buildings.add(la[1].strip())
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return buildings


_RESIDENT_BUILDINGS = _load_resident_buildings()


def _resolve_location(address):
    """取得地區標籤；只對「主人房」加建築前綴並過濾非本社區居民的。"""
    if not isinstance(address, list):
        return None
    if len(address) >= 3:
        room = address[-2]
    elif len(address) == 2:
        room = address[-1]
    else:
        return None
    room = room.strip() if room else ""
    if not room:
        return None
    # 移除「塔瑪拉的房間」
    if "塔瑪拉" in room:
        return None
    # 合併男/女衛生間為「衛生間」
    if room in ("男衛生間", "女衛生間"):
        return "衛生間"
    if room == "主人房":
        building = address[1].strip() if len(address) > 1 and address[1] else ""
        if building in _RESIDENT_BUILDINGS:
            return f"{building}·主人房"
        return None
    return room


def extract_interaction_data(checkpoints_folder):
    """提取AI與物品/地區的交互數據"""
    object_interactions = collections.defaultdict(int)
    location_interactions = collections.defaultdict(int)

    conversation_file = "conversation.json"
    files = sorted(os.listdir(checkpoints_folder))
    json_files = [f for f in files if f.endswith(".json") and f != conversation_file]

    for file_name in json_files:
        file_path = os.path.join(checkpoints_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for agent_name, agent_data in data.get("agents", {}).items():
            action = agent_data.get("action", {})

            if "event" not in action:
                continue
            event = action["event"]
            address = event.get("address", [])

            # 物品 = 最深層
            if isinstance(address, list) and len(address) >= 4:
                obj = address[-1]
                if obj and obj.strip():
                    object_interactions[(agent_name, obj.strip())] += 1

            loc = _resolve_location(address)
            if loc:
                location_interactions[(agent_name, loc)] += 1

    return object_interactions, location_interactions


# 從所有存檔文件中提取數據（用於回放）
def generate_movement(checkpoints_folder, compressed_folder, compressed_file):
    movement_file = os.path.join(compressed_folder, compressed_file)

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    files = sorted(os.listdir(checkpoints_folder))
    json_files = list()
    for file_name in files:
        if file_name.endswith(".json") and file_name != conversation_file:
            json_files.append(os.path.join(checkpoints_folder, file_name))

    persona_init_pos = dict()
    all_movement = dict()
    all_movement["description"] = dict()
    all_movement["conversation"] = dict()

    stride = get_stride(json_files)
    sec_per_step = stride

    result = {
        "start_datetime": "",  # 起始時間
        "stride": stride,  # 每個step對應的分鐘數（必須與生成時的參數一致）
        "sec_per_step": sec_per_step,  # 回放時每一幀對應的秒數
        "persona_init_pos": persona_init_pos,  # 每個Agent的初始位置
        "all_movement": all_movement,  # 所有Agent在每個setp中的位置變化
    }

    last_location = dict()

    # 加載地圖數據，用於計算Agent移動路徑
    json_path = "frontend/static/assets/village/maze.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        maze = Maze(json_data, None)

    for file_name in json_files:
        # 依次讀取所有存檔文件
        with open(file_name, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            step = json_data["step"]
            agents = json_data["agents"]

            # 保存回放的起始時間
            if len(result["start_datetime"]) < 1:
                t = datetime.strptime(json_data["time"], "%Y%m%d-%H:%M")
                result["start_datetime"] = t.isoformat()

            # 遍歷單個存檔文件中的所有Agent
            for agent_name, agent_data in agents.items():
                # 插入第0幀
                if step == 1:
                    insert_frame0(persona_init_pos, all_movement, agent_name)

                source_coord = last_location.get(agent_name, all_movement["0"][agent_name])["movement"]
                target_coord = agent_data["coord"]
                location = get_location(agent_data["action"]["event"]["address"])
                if location is None:
                    location = last_location.get(agent_name, all_movement["0"][agent_name])["location"]
                    path = [source_coord]
                else:
                    path = maze.find_path(source_coord, target_coord)

                had_conversation = False
                step_conversation = ""
                persons_in_conversation = []
                step_time = json_data["time"]
                if step_time in conversation.keys():
                    for chats in conversation[step_time]:
                        for persons, chat in chats.items():
                            persons_in_conversation.append(persons.split(" @ ")[0].split(" -> "))
                            step_conversation += f"\n地點：{persons.split(' @ ')[1]}\n\n"
                            for c in chat:
                                agent = c[0]
                                text = c[1]
                                step_conversation += f"{agent}：{text}\n"

                for i in range(frames_per_step):
                    moving = len(path) > 1
                    if len(path) > 0:
                        movement = list(path[0])
                        path = path[1:]
                        if agent_name not in last_location.keys():
                            last_location[agent_name] = dict()
                        last_location[agent_name]["movement"] = movement
                        last_location[agent_name]["location"] = location
                    else:
                        movement = None

                    if moving:
                        action = f"前往 {location}"
                    elif movement is not None:
                        action = agent_data["action"]["event"]["describe"]
                        if len(action) < 1:
                            action = f'{agent_data["action"]["event"]["predicate"]}{agent_data["action"]["event"]["object"]}'

                        # 判斷該存檔文件中當前Agent是否有新的對話（用於設置圖標）
                        for persons in persons_in_conversation:
                            if agent_name in persons:
                                had_conversation = True
                                break

                        # 針對睡覺和對話設置圖標
                        if "睡覺" in action:
                            action = "😴 " + action
                        elif had_conversation:
                            action = "💬 " + action

                    step_key = "%d" % ((step-1) * frames_per_step + 1 + i)
                    if step_key not in all_movement.keys():
                        all_movement[step_key] = dict()

                    if movement is not None:
                        all_movement[step_key][agent_name] = {
                            "location": location,
                            "movement": movement,
                            "action": action,
                        }
                all_movement["conversation"][step_time] = step_conversation
    object_interactions, location_interactions = extract_interaction_data(checkpoints_folder)
    
    # 轉換為前端需要的格式
    interaction_data = {
        "object_interactions": [
            {"agent": agent, "object": obj, "count": count}
            for (agent, obj), count in object_interactions.items()
        ],
        "location_interactions": [
            {"agent": agent, "location": location, "count": count}
            for (agent, location), count in location_interactions.items()
        ]
    }
    
    # 保存交互數據
    interaction_file = os.path.join(compressed_folder, "interactions.json")
    with open(interaction_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(interaction_data, indent=2, ensure_ascii=False))

    # 保存回放資料（提供 replay.py 載入）
    with open(movement_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))

    return result


# 生成Markdown文檔
def generate_report(checkpoints_folder, compressed_folder, compressed_file):
    last_state = dict()

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    def extract_description():
        markdown_content = "# 基礎人設\n\n"
        for agent_name in personas:
            json_path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                markdown_content += f"## {agent_name}\n\n"
                markdown_content += f"年齡：{json_data['scratch']['age']}歲  \n"
                markdown_content += f"先天：{json_data['scratch']['innate']}  \n"
                markdown_content += f"後天：{json_data['scratch']['learned']}  \n"
                markdown_content += f"生活習慣：{json_data['scratch']['lifestyle']}  \n"
                markdown_content += f"當前狀態：{json_data['currently']}\n\n"
        return markdown_content

    def extract_action(json_data):
        markdown_content = ""
        agents = json_data["agents"]
        for agent_name, agent_data in agents.items():
            if agent_name not in last_state.keys():
                last_state[agent_name] = {"currently": "", "location": "", "action": ""}

            location = "，".join(agent_data["action"]["event"]["address"])
            action = agent_data["action"]["event"]["describe"]

            if location == last_state[agent_name]["location"] and action == last_state[agent_name]["action"]:
                continue

            last_state[agent_name]["location"] = location
            last_state[agent_name]["action"] = action

            if len(markdown_content) < 1:
                markdown_content = f"# {json_data['time']}\n\n"
                markdown_content += "## 活動紀錄：\n\n"

            markdown_content += f"### {agent_name}\n"

            if len(action) < 1:
                action = "睡覺"

            markdown_content += f"位置：{location}  \n"
            markdown_content += f"活動：{action}  \n"

            markdown_content += f"\n"

        if json_data['time'] not in conversation.keys():
            return markdown_content

        markdown_content += "## 對話紀錄：\n\n"
        for chats in conversation[json_data['time']]:
            for agents, chat in chats.items():
                markdown_content += f"### {agents}\n\n"
                for item in chat:
                    markdown_content += f"`{item[0]}`\n> {item[1]}\n\n"
        return markdown_content

    all_markdown_content = extract_description()
    files = sorted(os.listdir(checkpoints_folder))
    for file_name in files:
        if (not file_name.endswith(".json")) or (file_name == conversation_file):
            continue

        file_path = os.path.join(checkpoints_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            content = extract_action(json_data)
            all_markdown_content += content + "\n\n"
    with open(f"{compressed_folder}/{compressed_file}", "w", encoding="utf-8") as compressed_file:
        compressed_file.write(all_markdown_content)


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="the name of the simulation")
args = parser.parse_args()


if __name__ == "__main__":
    name = args.name
    if len(name) < 1:
        name = input("Please enter a simulation name: ")

    while not os.path.exists(f"results/checkpoints/{name}"):
        name = input(f"'{name}' doesn't exists, please re-enter the simulation name: ")

    checkpoints_folder = f"results/checkpoints/{name}"
    compressed_folder = f"results/compressed/{name}"
    os.makedirs(compressed_folder, exist_ok=True)

    generate_report(checkpoints_folder, compressed_folder, file_markdown)
    generate_movement(checkpoints_folder, compressed_folder, file_movement)
