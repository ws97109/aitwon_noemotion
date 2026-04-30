"""generative_agents.prompt.scratch"""

import json
import random
import datetime
import re
from string import Template

from modules import utils
from modules.memory import Event
from modules.model import parse_llm_output
from modules.emotion import EmotionState


class Scratch:
    def __init__(self, name, currently, config):
        self.name = name
        self.currently = currently
        self.config = config
        self.template_path = "data/prompts"
        self.emotion = EmotionState()  # 當前情緒狀態，由 EmotionModel 更新

    def build_prompt(self, template, data):
        with open(f"{self.template_path}/{template}.txt", "r", encoding="utf-8") as file:
            file_content = file.read()

        tmpl = Template(file_content)
        filled_content = tmpl.safe_substitute(data)

        return filled_content

    def _base_desc(self):
        # 始終顯示情緒狀態（含平靜），讓 LLM 了解角色的完整情緒脈絡
        # 即使是平靜狀態，也讓 LLM 知道「目前很平靜」，
        # 這樣才能自然地判斷何時應該從平靜轉變為其他情緒
        emotion = getattr(self, "emotion", None) or EmotionState()
        if emotion.label == "平靜":
            emotion_line = f"目前情緒狀態：平靜（強度：{emotion.intensity}/10）— 情緒穩定，但可能因事件而改變"
        else:
            emotion_line = f"目前情緒狀態：{emotion.describe()}"
            if emotion.reason:
                emotion_line += f"（原因：{emotion.reason}）"
        return self.build_prompt(
            "base_desc",
            {
                "name": self.name,
                "age": self.config["age"],
                "innate": self.config["innate"],
                "learned": self.config["learned"],
                "lifestyle": self.config["lifestyle"],
                "daily_plan": self.config["daily_plan"],
                "date": utils.get_timer().daily_format_cn(),
                "currently": self.currently,
                "emotion_state": emotion_line,
            }
        )

    def prompt_poignancy_event(self, event):
        prompt = self.build_prompt(
            "poignancy_event",
            {
                "base_desc": self._base_desc(),
                "agent": self.name,
                "event": event.get_describe(),
            }
        )

        def _callback(response):
            pattern = [
                "評分[:： ]+(\d{1,2})",
                "(\d{1,2})",
            ]
            return int(parse_llm_output(response, pattern, "match_last"))

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": random.choice(list(range(10))) + 1,
        }

    def prompt_poignancy_chat(self, event):
        prompt = self.build_prompt(
            "poignancy_chat",
            {
                "base_desc": self._base_desc(),
                "agent": self.name,
                "event": event.get_describe(),
            }
        )

        def _callback(response):
            pattern = [
                "評分[:： ]+(\d{1,2})",
                "(\d{1,2})",
            ]
            return int(parse_llm_output(response, pattern, "match_last"))

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": random.choice(list(range(10))) + 1,
        }

    def prompt_wake_up(self):
        prompt = self.build_prompt(
            "wake_up",
            {
                "base_desc": self._base_desc(),
                "lifestyle": self.config["lifestyle"],
                "agent": self.name,
            }
        )

        def _callback(response):
            patterns = [
                "(\d{1,2}):00",
                "(\d{1,2})",
                "\d{1,2}",
            ]
            wake_up_time = int(parse_llm_output(response, patterns))
            if wake_up_time > 11:
                wake_up_time = 11
            return wake_up_time

        return {"prompt": prompt, "callback": _callback, "failsafe": 6}

    def prompt_schedule_init(self, wake_up):
        prompt = self.build_prompt(
            "schedule_init",
            {
                "base_desc": self._base_desc(),
                "lifestyle": self.config["lifestyle"],
                "agent": self.name,
                "wake_up": wake_up,
            }
        )

        def _callback(response):
            patterns = [
                "\d{1,2}\. (.*)。",
                "\d{1,2}\. (.*)",
                "\d{1,2}\) (.*)。",
                "\d{1,2}\) (.*)",
                "(.*)。",
                "(.*)",
            ]
            return parse_llm_output(response, patterns, mode="match_all")

        failsafe = [
            "早上6點起床並完成早餐的例行工作",
            "早上7點吃早餐",
            "早上8點看書",
            "中午12點吃午飯",
            "下午1點小睡一會兒",
            "晚上7點放松一下，看電視",
            "晚上11點睡覺",
        ]
        return {"prompt": prompt, "callback": _callback, "failsafe": failsafe}

    def prompt_schedule_daily(self, wake_up, daily_schedule):
        hourly_schedule = ""
        for i in range(wake_up):
            hourly_schedule += f"[{i}:00] 睡覺\n"
        for i in range(wake_up, 24):
            hourly_schedule += f"[{i}:00] <活動>\n"

        prompt = self.build_prompt(
            "schedule_daily",
            {
                "base_desc": self._base_desc(),
                "agent": self.name,
                "daily_schedule": "；".join(daily_schedule),
                "hourly_schedule": hourly_schedule,
            }
        )

        failsafe = {
            "6:00": "起床並完成早晨的例行工作",
            "7:00": "吃早餐",
            "8:00": "讀書",
            "9:00": "讀書",
            "10:00": "讀書",
            "11:00": "讀書",
            "12:00": "吃午飯",
            "13:00": "小睡一會兒",
            "14:00": "小睡一會兒",
            "15:00": "小睡一會兒",
            "16:00": "繼續工作",
            "17:00": "繼續工作",
            "18:00": "回家",
            "19:00": "放松，看電視",
            "20:00": "放松，看電視",
            "21:00": "睡前看書",
            "22:00": "準備睡覺",
            "23:00": "睡覺",
        }

        def _callback(response):
            patterns = [
                "\[(\d{1,2}:\d{2})\] " + self.name + "(.*)。",
                "\[(\d{1,2}:\d{2})\] " + self.name + "(.*)",
                "\[(\d{1,2}:\d{2})\] " + "(.*)。",
                "\[(\d{1,2}:\d{2})\] " + "(.*)",
            ]
            outputs = parse_llm_output(response, patterns, mode="match_all")
            assert len(outputs) >= 5, "less than 5 schedules"
            return {s[0]: s[1] for s in outputs}

        return {"prompt": prompt, "callback": _callback, "failsafe": failsafe}

    def prompt_schedule_decompose(self, plan, schedule):
        def _plan_des(plan):
            start, end = schedule.plan_stamps(plan, time_format="%H:%M")
            return f'{start} 至 {end}，{self.name} 計畫 {plan["describe"]}'

        indices = range(
            max(plan["idx"] - 1, 0), min(plan["idx"] + 2, len(schedule.daily_schedule))
        )

        start, end = schedule.plan_stamps(plan, time_format="%H:%M")
        increment = max(int(plan["duration"] / 100) * 5, 5)

        prompt = self.build_prompt(
            "schedule_decompose",
            {
                "base_desc": self._base_desc(),
                "agent": self.name,
                "plan": "；".join([_plan_des(schedule.daily_schedule[i]) for i in indices]),
                "increment": increment,
                "start": start,
                "end": end,
            }
        )

        def _callback(response):
            patterns = [
                "\d{1,2}\) .*\*計畫\* (.*)[\(（]+耗時[:： ]+(\d{1,2})[,， ]+剩餘[:： ]+\d*[\)）]",
            ]
            schedules = parse_llm_output(response, patterns, mode="match_all")
            schedules = [(s[0].strip("."), int(s[1])) for s in schedules]
            left = plan["duration"] - sum([s[1] for s in schedules])
            if left > 0:
                schedules.append((plan["describe"], left))
            return schedules

        failsafe = [(plan["describe"], 10) for _ in range(int(plan["duration"] / 10))]
        return {"prompt": prompt, "callback": _callback, "failsafe": failsafe}

    def prompt_schedule_revise(self, action, schedule):
        plan, _ = schedule.current_plan()
        start, end = schedule.plan_stamps(plan, time_format="%H:%M")
        act_start_minutes = utils.daily_duration(action.start)
        original_plan, new_plan = [], []

        def _plan_des(start, end, describe):
            if not isinstance(start, str):
                start = start.strftime("%H:%M")
            if not isinstance(end, str):
                end = end.strftime("%H:%M")
            return "[{} 至 {}] {}".format(start, end, describe)

        for de_plan in plan["decompose"]:
            de_start, de_end = schedule.plan_stamps(de_plan, time_format="%H:%M")
            original_plan.append(_plan_des(de_start, de_end, de_plan["describe"]))
            if de_plan["start"] + de_plan["duration"] <= act_start_minutes:
                new_plan.append(_plan_des(de_start, de_end, de_plan["describe"]))
            elif de_plan["start"] <= act_start_minutes:
                new_plan.extend(
                    [
                        _plan_des(de_start, action.start, de_plan["describe"]),
                        _plan_des(
                            action.start, action.end, action.event.get_describe(False)
                        ),
                    ]
                )

        original_plan, new_plan = "\n".join(original_plan), "\n".join(new_plan)

        prompt = self.build_prompt(
            "schedule_revise",
            {
                "agent": self.name,
                "start": start,
                "end": end,
                "original_plan": original_plan,
                "duration": action.duration,
                "event": action.event.get_describe(),
                "new_plan": new_plan,
            }
        )

        def _callback(response):
            patterns = [
                "^\[(\d{1,2}:\d{1,2}) ?- ?(\d{1,2}:\d{1,2})\] (.*)",
                "^\[(\d{1,2}:\d{1,2}) ?~ ?(\d{1,2}:\d{1,2})\] (.*)",
                "^\[(\d{1,2}:\d{1,2}) ?至 ?(\d{1,2}:\d{1,2})\] (.*)",
            ]
            schedules = parse_llm_output(response, patterns, mode="match_all")
            decompose = []
            for start, end, describe in schedules:
                m_start = utils.daily_duration(utils.to_date(start, "%H:%M"))
                m_end = utils.daily_duration(utils.to_date(end, "%H:%M"))
                decompose.append(
                    {
                        "idx": len(decompose),
                        "describe": describe,
                        "start": m_start,
                        "duration": m_end - m_start,
                    }
                )
            return decompose

        return {"prompt": prompt, "callback": _callback, "failsafe": plan["decompose"]}

    def prompt_determine_sector(self, describes, spatial, address, tile):
        live_address = spatial.find_address("living_area", as_list=True)[:-1]
        curr_address = tile.get_address("sector", as_list=True)

        # 情緒對地點選擇的自然影響
        emotion = getattr(self, "emotion", None) or EmotionState()
        if emotion.label != "平靜" and emotion.intensity >= 3:
            emotion_hint = (
                f"{self.name} 目前感到{emotion.describe()}。"
                f"情緒可能自然影響他/她選擇去哪裡（例如疲憊時想回家休息，"
                f"興奮時想去社交場所，焦慮時可能待在熟悉的地方）。"
            )
        else:
            emotion_hint = ""

        prompt = self.build_prompt(
            "determine_sector",
            {
                "agent": self.name,
                "live_sector": live_address[-1],
                "live_arenas": ", ".join(i for i in spatial.get_leaves(live_address)),
                "current_sector": curr_address[-1],
                "current_arenas": ", ".join(i for i in spatial.get_leaves(curr_address)),
                "daily_plan": self.config["daily_plan"],
                "emotion_hint": emotion_hint,
                "areas": ", ".join(i for i in spatial.get_leaves(address)),
                "complete_plan": describes[0],
                "decomposed_plan": describes[1],
            }
        )

        sectors = spatial.get_leaves(address)
        arenas = {}
        for sec in sectors:
            arenas.update(
                {a: sec for a in spatial.get_leaves(address + [sec]) if a not in arenas}
            )
        failsafe = random.choice(sectors)

        def _callback(response):
            patterns = [
                ".*應該去[:： ]*(.*)。",
                ".*應該去[:： ]*(.*)",
                "(.+)。",
                "(.+)",
            ]
            sector = parse_llm_output(response, patterns)
            if sector in sectors:
                return sector
            if sector in arenas:
                return arenas[sector]
            for s in sectors:
                if sector.startswith(s):
                    return s
            return failsafe

        return {"prompt": prompt, "callback": _callback, "failsafe": failsafe}

    def prompt_determine_arena(self, describes, spatial, address):
        prompt = self.build_prompt(
            "determine_arena",
            {
                "agent": self.name,
                "target_sector": address[-1],
                "target_arenas": ", ".join(i for i in spatial.get_leaves(address)),
                "daily_plan": self.config["daily_plan"],
                "complete_plan": describes[0],
                "decomposed_plan": describes[1],
            }
        )

        arenas = spatial.get_leaves(address)
        failsafe = random.choice(arenas)

        def _callback(response):
            patterns = [
                ".*應該去[:： ]*(.*)。",
                ".*應該去[:： ]*(.*)",
                "(.+)。",
                "(.+)",
            ]
            arena = parse_llm_output(response, patterns)
            return arena if arena in arenas else failsafe

        return {"prompt": prompt, "callback": _callback, "failsafe": failsafe}

    def prompt_determine_object(self, describes, spatial, address):
        objects = spatial.get_leaves(address)

        prompt = self.build_prompt(
            "determine_object",
            {
                "activity": describes[1],
                "objects": ", ".join(objects),
            }
        )

        failsafe = random.choice(objects)

        def _callback(response):
            # pattern = ["The most relevant object from the Objects is: <(.+?)>", "<(.+?)>"]
            patterns = [
                ".*是[:： ]*(.*)。",
                ".*是[:： ]*(.*)",
                "(.+)。",
                "(.+)",
            ]
            obj = parse_llm_output(response, patterns)
            return obj if obj in objects else failsafe

        return {"prompt": prompt, "callback": _callback, "failsafe": failsafe}

    def prompt_describe_emoji(self, describe):
        prompt = self.build_prompt(
            "describe_emoji",
            {
                "action": describe,
            }
        )

        def _callback(response):
            # 正則表达式：匹配大多數emoji
            emoji_pattern = u"([\U0001F600-\U0001F64F]|"   # 表情符號
            emoji_pattern += u"[\U0001F300-\U0001F5FF]|"   # 符號和圖標
            emoji_pattern += u"[\U0001F680-\U0001F6FF]|"   # 運輸和地圖符號
            emoji_pattern += u"[\U0001F700-\U0001F77F]|"   # 午夜符號
            emoji_pattern += u"[\U0001F780-\U0001F7FF]|"   # 英镑符號
            emoji_pattern += u"[\U0001F800-\U0001F8FF]|"   # 合成擴展
            emoji_pattern += u"[\U0001F900-\U0001F9FF]|"   # 補充符號和圖標
            emoji_pattern += u"[\U0001FA00-\U0001FA6F]|"   # 補充符號和圖標
            emoji_pattern += u"[\U0001FA70-\U0001FAFF]|"   # 補充符號和圖標
            emoji_pattern += u"[\U00002702-\U000027B0]+)"  # 雜項符號

            emoji = re.compile(emoji_pattern, flags=re.UNICODE).findall(response)
            if len(emoji) > 0:
                response = "Emoji: " + "".join(i for i in emoji)
            else:
                response = ""

            return parse_llm_output(response, ["Emoji: (.*)"])[:3]

        return {"prompt": prompt, "callback": _callback, "failsafe": "💭", "retry": 1}

    def prompt_describe_event(self, subject, describe, address, emoji=None):
        prompt = self.build_prompt(
            "describe_event",
            {
                "action": describe,
            }
        )

        e_describe = describe.replace("(", "").replace(")", "").replace("<", "").replace(">", "")
        if e_describe.startswith(subject + "此時"):
            e_describe = e_describe.replace(subject + "此時", "")
        failsafe = Event(
            subject, "此時", e_describe, describe=describe, address=address, emoji=emoji
        )

        def _callback(response):
            response_list = response.replace(")", ")\n").split("\n")
            for response in response_list:
                if len(response.strip()) < 7:
                    continue
                if response.count("(") > 1 or response.count(")") > 1 or response.count("（") > 1 or response.count("）") > 1:
                    continue

                patterns = [
                    "[\(（]<(.+?)>[,， ]+<(.+?)>[,， ]+<(.*)>[\)）]",
                    "[\(（](.+?)[,， ]+(.+?)[,， ]+(.*)[\)）]",
                ]
                outputs = parse_llm_output(response, patterns)
                if len(outputs) == 3:
                    return Event(*outputs, describe=describe, address=address, emoji=emoji)

            return None

        return {"prompt": prompt, "callback": _callback, "failsafe": failsafe}

    def prompt_describe_object(self, obj, describe):
        prompt = self.build_prompt(
            "describe_object",
            {
                "object": obj,
                "agent": self.name,
                "action": describe,
            }
        )

        def _callback(response):
            patterns = [
                "<" + obj + "> ?" + "(.*)。",
                "<" + obj + "> ?" + "(.*)",
            ]
            return parse_llm_output(response, patterns)

        return {"prompt": prompt, "callback": _callback, "failsafe": "空閒"}

    def prompt_decide_chat(self, agent, other, focus, chats):
        def _status_des(a):
            event = a.get_event()
            if a.path:
                return f"{a.name} 正去往 {event.get_describe(False)}"
            return event.get_describe()

        context = "。".join(
            [c.describe for c in focus["events"]]
        )
        context += "\n" + "。".join([c.describe for c in focus["thoughts"]])
        date_str = utils.get_timer().get_date("%Y-%m-%d %H:%M:%S")
        chat_history = ""
        if chats:
            chat_history = f" {agent.name} 和 {other.name} 上次在 {chats[0].create} 聊過關於 {chats[0].describe} 的話題"
        a_des, o_des = _status_des(agent), _status_des(other)

        prompt = self.build_prompt(
            "decide_chat",
            {
                "context": context,
                "date": date_str,
                "chat_history": chat_history,
                "agent_status": a_des,
                "another_status": o_des,
                "agent": agent.name,
                "another": other.name,
            }
        )

        def _callback(response):
            if "No" in response or "no" in response or "否" in response or "不" in response:
                return False
            return True

        return {"prompt": prompt, "callback": _callback, "failsafe": False}

    def prompt_decide_chat_terminate(self, agent, other, chats):
        conversation = "\n".join(["{}: {}".format(n, u) for n, u in chats])
        conversation = (
            conversation or "[對話尚未開始]"
        )

        prompt = self.build_prompt(
            "decide_chat_terminate",
            {
                "conversation": conversation,
                "agent": agent.name,
                "another": other.name,
            }
        )

        def _callback(response):
            if "No" in response or "no" in response or "否" in response or "不" in response:
                return False
            return True

        return {"prompt": prompt, "callback": _callback, "failsafe": False}

    def prompt_decide_wait(self, agent, other, focus):
        example1 = self.build_prompt(
            "decide_wait_example",
            {
                "context": "簡是麗茲的室友。2022-10-25 07:05，簡和麗茲互相問候了早上好。",
                "date": "2022-10-25 07:09",
                "agent": "簡",
                "another": "麗茲",
                "status": "簡 正要去浴室",
                "another_status": "麗茲 已經在 使用浴室",
                "action": "使用浴室",
                "another_action": "使用浴室",
                "reason": "推理：簡和麗茲都想用浴室。簡和麗茲同時使用浴室會很奇怪。所以，既然麗茲已經在用浴室了，對簡來說最好的選擇就是等著用浴室。\n",
                "answer": "答案：<選項A>",
            }
        )
        example2 = self.build_prompt(
            "decide_wait_example",
            {
                "context": "山姆是莎拉的朋友。2022-10-24 23:00，山姆和莎拉就最喜歡的電影進行了交談。",
                "date": "2022-10-25 12:40",
                "agent": "山姆",
                "another": "莎拉",
                "status": "山姆 正要去吃午飯",
                "another_status": "莎拉 已經在 洗衣服",
                "action": "吃午飯",
                "another_action": "洗衣服",
                "reason": "推理：山姆可能會在餐廳吃午飯。莎拉可能會去洗衣房洗衣服。由於山姆和莎拉需要使用不同的區域，他们的行為並不冲突。所以，由於山姆和莎拉將在不同的區域，山姆現在繼續吃午飯。\n",
                "answer": "答案：<選項B>",
            }
        )

        def _status_des(a):
            event, loc = a.get_event(), ""
            if event.address:
                loc = " 在 {} 的 {}".format(event.address[-2], event.address[-1])
            if not a.path:
                return f"{a.name} 已經在 {event.get_describe(False)}{loc}"
            return f"{a.name} 正要去 {event.get_describe(False)}{loc}"

        context = ". ".join(
            [c.describe for c in focus["events"]]
        )
        context += "\n" + ". ".join([c.describe for c in focus["thoughts"]])

        task = self.build_prompt(
            "decide_wait_example",
            {
                "context": context,
                "date": utils.get_timer().get_date("%Y-%m-%d %H:%M"),
                "agent": agent.name,
                "another": other.name,
                "status": _status_des(agent),
                "another_status": _status_des(other),
                "action": agent.get_event().get_describe(False),
                "another_action": other.get_event().get_describe(False),
                "reason": "",
                "answer": "",
            }
        )

        prompt = self.build_prompt(
            "decide_wait",
            {
                "examples_1": example1,
                "examples_2": example2,
                "task": task,
            }
        )

        def _callback(response):
            return "A" in response

        return {"prompt": prompt, "callback": _callback, "failsafe": False}

    def prompt_summarize_relation(self, agent, other_name):
        nodes = agent.associate.retrieve_focus([other_name], 50)

        prompt = self.build_prompt(
            "summarize_relation",
            {
                "context": "\n".join(["{}. {}".format(idx, n.describe) for idx, n in enumerate(nodes)]),
                "agent": agent.name,
                "another": other_name,
            }
        )

        def _callback(response):
            return response

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": agent.name + " 正在看著 " + other_name,
        }

    def prompt_generate_chat(self, agent, other, relation, chats):
        focus = [relation, other.get_event().get_describe()]
        if len(chats) > 4:
            focus.append("; ".join("{}: {}".format(n, t) for n, t in chats[-4:]))
        nodes = agent.associate.retrieve_focus(focus, 15)
        memory = "\n- " + "\n- ".join([n.describe for n in nodes])
        chat_nodes = agent.associate.retrieve_chats(other.name)
        pass_context = ""
        for n in chat_nodes:
            delta = utils.get_timer().get_delta(n.create)
            if delta > 480:
                continue
            pass_context += f"{delta} 分鐘前，{agent.name} 和 {other.name} 進行過對話。{n.describe}\n"

        address = agent.get_tile().get_address()
        if len(pass_context) > 0:
            prev_context = f'\n背景：\n"""\n{pass_context}"""\n\n'
        else:
            prev_context = ""
        curr_context = (
            f"{agent.name} {agent.get_event().get_describe(False)} 時，看到 {other.name} {other.get_event().get_describe(False)}。"
        )

        conversation = "\n".join(["{}: {}".format(n, u) for n, u in chats])
        conversation = (
            conversation or "[對話尚未開始]"
        )

        prompt = self.build_prompt(
            "generate_chat",
            {
                "agent": agent.name,
                "base_desc": self._base_desc(),
                "memory": memory,
                "address": f"{address[-2]}，{address[-1]}",
                "current_time": utils.get_timer().get_date("%H:%M"),
                "previous_context": prev_context,
                "current_context": curr_context,
                "another": other.name,
                "conversation": conversation,
            }
        )

        def _callback(response):
            assert "{" in response and "}" in response
            json_content = utils.load_dict(
                "{" + response.split("{")[1].split("}")[0] + "}"
            )
            text = json_content[agent.name].replace("\n\n", "\n").strip(" \n\"'“”‘’")
            return text

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": "嗯",
        }

    def prompt_generate_chat_check_repeat(self, agent, chats, content):
        conversation = "\n".join(["{}: {}".format(n, u) for n, u in chats])
        conversation = (
                conversation or "[對話尚未開始]"
        )

        prompt = self.build_prompt(
            "generate_chat_check_repeat",
            {
                "conversation": conversation,
                "content": f"{agent.name}: {content}",
                "agent": agent.name,
            }
        )

        def _callback(response):
            if "No" in response or "no" in response or "否" in response or "不" in response:
                return False
            return True

        return {"prompt": prompt, "callback": _callback, "failsafe": False}

    def prompt_summarize_chats(self, chats):
        conversation = "\n".join(["{}: {}".format(n, u) for n, u in chats])

        prompt = self.build_prompt(
            "summarize_chats",
            {
                "conversation": conversation,
            }
        )

        def _callback(response):
            return response.strip()

        if len(chats) > 1:
            failsafe = "{} 和 {} 之間的普通對話".format(chats[0][0], chats[1][0])
        else:
            failsafe = "{} 說的話没有得到回應".format(chats[0][0])

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": failsafe,
        }

    def prompt_reflect_focus(self, nodes, topk):
        prompt = self.build_prompt(
            "reflect_focus",
            {
                "reference": "\n".join(["{}. {}".format(idx, n.describe) for idx, n in enumerate(nodes)]),
                "number": topk,
            }
        )

        def _callback(response):
            pattern = ["^\d{1}\. (.*)", "^\d{1}\) (.*)", "^\d{1} (.*)"]
            return parse_llm_output(response, pattern, mode="match_all")

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": [
                "{} 是谁？".format(self.name),
                "{} 住在哪裡？".format(self.name),
                "{} 今天要做什麼？".format(self.name),
            ],
        }

    def prompt_reflect_insights(self, nodes, topk):
        prompt = self.build_prompt(
            "reflect_insights",
            {
                "reference": "\n".join(["{}. {}".format(idx, n.describe) for idx, n in enumerate(nodes)]),
                "number": topk,
            }
        )

        def _callback(response):
            patterns = [
                "^\d{1}[\. ]+(.*)[。 ]*[\(（]+.*序號[:： ]+([\d,， ]+)[\)）]",
                "^\d{1}[\. ]+(.*)[。 ]*[\(（]([\d,， ]+)[\)）]",
            ]
            insights, outputs = [], parse_llm_output(
                response, patterns, mode="match_all"
            )
            if outputs:
                for output in outputs:
                    if isinstance(output, str):
                        insight, node_ids = output, []
                    elif len(output) == 2:
                        insight, reason = output
                        indices = [int(e.strip()) for e in reason.split(",")]
                        node_ids = [nodes[i].node_id for i in indices if i < len(nodes)]
                    insights.append([insight.strip(), node_ids])
                return insights
            raise Exception("Can not find insights")

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": [
                [
                    "{} 在考慮下一步該做什麼".format(self.name),
                    [nodes[0].node_id],
                ]
            ],
        }

    def prompt_reflect_chat_planing(self, chats):
        all_chats = "\n".join(["{}: {}".format(n, c) for n, c in chats])

        prompt = self.build_prompt(
            "reflect_chat_planing",
            {
                "conversation": all_chats,
                "agent": self.name,
            }
        )

        def _callback(response):
            return response

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": f"{self.name} 進行了一次對話",
        }

    def prompt_reflect_chat_memory(self, chats):
        all_chats = "\n".join(["{}: {}".format(n, c) for n, c in chats])

        prompt = self.build_prompt(
            "reflect_chat_memory",
            {
                "conversation": all_chats,
                "agent": self.name,
            }
        )

        def _callback(response):
            return response

        return {
            "prompt": prompt,
            "callback": _callback,
            # "failsafe": f"{self.name} had a sonversation",
            "failsafe": f"{self.name} 進行了一次對話",
        }

    def prompt_retrieve_plan(self, nodes):
        statements = [
            n.create.strftime("%Y-%m-%d %H:%M") + ": " + n.describe for n in nodes
        ]

        prompt = self.build_prompt(
            "retrieve_plan",
            {
                "description": "\n".join(statements),
                "agent": self.name,
                "date": utils.get_timer().get_date("%Y-%m-%d"),
            }
        )

        def _callback(response):
            pattern = [
                "^\d{1,2}\. (.*)。",
                "^\d{1,2}\. (.*)",
                "^\d{1,2}\) (.*)。",
                "^\d{1,2}\) (.*)",
            ]
            return parse_llm_output(response, pattern, mode="match_all")

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": [r.describe for r in random.choices(nodes, k=5)],
        }

    def prompt_retrieve_thought(self, nodes):
        statements = [
            n.create.strftime("%Y-%m-%d %H:%M") + "：" + n.describe for n in nodes
        ]

        prompt = self.build_prompt(
            "retrieve_thought",
            {
                "description": "\n".join(statements),
                "agent": self.name,
            }
        )

        def _callback(response):
            return response

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": "{} 應該遵循昨天的日程".format(self.name),
        }

    def prompt_retrieve_currently(self, plan_note, thought_note):
        time_stamp = (
            utils.get_timer().get_date() - datetime.timedelta(days=1)
        ).strftime("%Y-%m-%d")

        prompt = self.build_prompt(
            "retrieve_currently",
            {
                "agent": self.name,
                "time": time_stamp,
                "currently": self.currently,
                "plan": ". ".join(plan_note),
                "thought": thought_note,
                "current_time": utils.get_timer().get_date("%Y-%m-%d"),
            }
        )

        def _callback(response):
            pattern = [
                "^狀態: (.*)。",
                "^狀態: (.*)",
            ]
            return parse_llm_output(response, pattern)

        return {
            "prompt": prompt,
            "callback": _callback,
            "failsafe": self.currently,
        }

    # ──────────────────────────────────────────────────────────────
    # 情緒相關提示詞
    # ──────────────────────────────────────────────────────────────

    def prompt_emotion_detect(self, text, other_name=None, relation=None, chat_memory=None):
        """
        分析行動或對話文字中的情緒狀態。
        回傳 dict（可直接傳入 EmotionState.from_dict()）。
        當 EmoLLM 專用後端未啟用時，由主要 LLM 執行此提示詞。

        Args:
            text:       要分析的文字
            other_name: 對話對象姓名（對話場景才傳入）
            relation:   自身對對方的關係印象描述
            chat_memory: 近期與對方的對話記憶摘要
        """
        # 組裝關係上下文（只有對話場景才會有）
        relation_context = ""
        if other_name:
            parts = [f"對話對象：{other_name}"]
            if relation:
                parts.append(f"{self.name} 對 {other_name} 的印象：{relation}")
            if chat_memory:
                parts.append(f"近期互動記憶：{chat_memory}")
            relation_context = "\n".join(parts) + "\n"
        else:
            # 非對話場景：加入 currently 作為行動背景脈絡
            if self.currently:
                relation_context = f"今日狀態與計畫：{self.currently[:200]}\n"

        # ── 情境因素：讓 LLM 自然推理出情緒變化的生理/社交因素 ──
        # 這些是客觀事實，不預設任何情緒結論，由 LLM 自由判斷
        situational_lines = []
        try:
            timer = utils.get_timer()
            current_hour = timer.daily_duration(mode="hour")

            # 清醒時長：從 lifestyle 中解析起床時間
            lifestyle = self.config.get("lifestyle", "")
            import re as _re
            wake_match = _re.search(r'(\d+)點.*(?:醒|起)', lifestyle)
            wake_hour = int(wake_match.group(1)) if wake_match else 7
            hours_awake = max(0, current_hour - wake_hour)
            if hours_awake > 0:
                situational_lines.append(f"已清醒約 {hours_awake} 小時")

            # 時段描述
            if current_hour < 6:
                situational_lines.append("現在是深夜/凌晨")
            elif current_hour < 12:
                situational_lines.append("現在是上午")
            elif current_hour < 18:
                situational_lines.append("現在是下午")
            else:
                situational_lines.append("現在是晚上")

            # 就寢時間提醒
            sleep_match = _re.search(r'(\d+)點.*(?:睡|床)', lifestyle)
            if sleep_match:
                sleep_hour = int(sleep_match.group(1))
                if current_hour >= sleep_hour - 1:
                    situational_lines.append("已接近或超過平時就寢時間")
        except Exception:
            pass

        # 前一個情緒狀態（讓 LLM 知道情緒轉變的脈絡）
        emotion = getattr(self, "emotion", None) or EmotionState()
        if emotion._history:
            recent_labels = [h["label"] for h in emotion._history[-3:]]
            if len(set(recent_labels)) == 1 and recent_labels[0] == emotion.label:
                situational_lines.append(
                    f"近幾次情緒偵測都是「{emotion.label}」，"
                    f"考慮是否有更細膩的情緒變化被忽略"
                )

        situational_context = ""
        if situational_lines:
            situational_context = "\n情境因素：" + "；".join(situational_lines)

        prompt = self.build_prompt(
            "emotion_detect",
            {
                "agent":            self.name,
                "base_desc":        self._base_desc(),
                "text":             str(text)[:600],
                "relation_context": relation_context + situational_context,
            }
        )

        def _callback(response):
            # 優先嘗試解析 JSON
            try:
                match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    label     = data.get("label",     EmotionState.DEFAULT)
                    intensity = max(1, min(10, int(data.get("intensity", 5))))
                    reason    = str(data.get("reason", ""))
                    return {"label": label, "intensity": intensity, "reason": reason}
            except Exception:
                pass
            # 備援：關鍵字掃描
            for label in EmotionState.LABELS:
                if label in response:
                    return {"label": label, "intensity": 5, "reason": ""}
            return None

        return {
            "prompt":   prompt,
            "callback": _callback,
            "failsafe": {"label": EmotionState.DEFAULT, "intensity": 3, "reason": ""},
        }

    def prompt_emotion_memory_reflect(self, emotion_nodes):
        """
        根據情緒記憶節點列表，反思整體情緒傾向。
        回傳描述字串，存入 thought 記憶中。
        """
        history_lines = []
        for n in emotion_nodes[:8]:
            time_str   = n.create.strftime("%m/%d %H:%M")
            label      = n.event.object      # 情緒標籤存於 event.object
            intensity  = n.poignancy
            short_desc = n.reason[:20] if hasattr(n, "reason") and n.reason else n.describe[:30]
            history_lines.append(
                f"- [{time_str}] {label}（強度 {intensity}/10）：{short_desc}"
            )

        prompt = self.build_prompt(
            "emotion_memory_reflect",
            {
                "agent":           self.name,
                "emotion_history": "\n".join(history_lines) or "（暫無情緒記錄）",
            }
        )

        def _callback(response):
            return response.strip()[:100]

        return {
            "prompt":   prompt,
            "callback": _callback,
            "failsafe": f"維持平靜的情緒狀態，以日常節奏生活",
        }
