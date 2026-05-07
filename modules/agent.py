"""generative_agents.agent"""

import os
import math
import random
import datetime

from modules import memory, prompt, utils
from modules.model.llm_model import create_llm_model
from modules.memory.associate import Concept
from modules.emotion import EmotionModel, EmotionState


class Agent:
    def __init__(self, config, maze, conversation, logger):
        self.name = config["name"]
        self.maze = maze
        self.conversation = conversation
        self._llm = None
        self._emotion_model = None
        self.logger = logger

        # agent config
        self.percept_config = config["percept"]
        self.think_config = config["think"]
        self.chat_iter = config["chat_iter"]

        # memory
        self.spatial = memory.Spatial(**config["spatial"])
        self.schedule = memory.Schedule(**config["schedule"])
        self.associate = memory.Associate(
            os.path.join(config["storage_root"], "associate"), **config["associate"]
        )
        self.concepts, self.chats = [], config.get("chats", [])

        # prompt
        self.scratch = prompt.Scratch(self.name, config["currently"], config["scratch"])

        # status
        status = {"poignancy": 0}
        self.status = utils.update_dict(status, config.get("status", {}))
        # emotion state (persisted in status dict)
        if "emotion" not in self.status:
            self.status["emotion"] = EmotionState().to_dict()
        self.plan = config.get("plan", {})

        # record
        self.last_record = utils.get_timer().daily_duration()

        # action and events
        if "action" in config:
            self.action = memory.Action.from_dict(config["action"])
            tiles = self.maze.get_address_tiles(self.get_event().address)
            config["coord"] = random.choice(list(tiles))
        else:
            tile = self.maze.tile_at(config["coord"])
            address = tile.get_address("game_object", as_list=True)
            self.action = memory.Action(
                memory.Event(self.name, address=address),
                memory.Event(address[-1], address=address),
            )

        # update maze
        self.coord, self.path = None, None
        self.move(config["coord"], config.get("path"))
        if self.coord is None:
            self.coord = config["coord"]

    def abstract(self):
        des = {
            "name": self.name,
            "currently": self.scratch.currently,
            "tile": self.maze.tile_at(self.coord).abstract(),
            "status": self.status,
            "concepts": {c.node_id: c.abstract() for c in self.concepts},
            "chats": self.chats,
            "action": self.action.abstract(),
            "associate": self.associate.abstract(),
        }
        if self.schedule.scheduled():
            des["schedule"] = self.schedule.abstract()
        if self.llm_available():
            des["llm"] = self._llm.get_summary()
        # if self.plan.get("path"):
        #     des["path"] = "-".join(
        #         ["{},{}".format(c[0], c[1]) for c in self.plan["path"]]
        #     )
        return des

    def __str__(self):
        return utils.dump_dict(self.abstract())

    def reset(self, keys):
        if self.think_config["mode"] == "llm" and not self._llm:
            self._llm = create_llm_model(**self.think_config["llm"], keys=keys)
        # 初始化專用情緒偵測後端（sacf / local / ollama 擇一）
        emotion_cfg = self.think_config.get("emotion", {})
        if emotion_cfg.get("enabled", True) and not self._emotion_model:
            self._emotion_model = EmotionModel(emotion_cfg)
            if self._emotion_model.is_available():
                self.logger.info("{} 使用專用情緒後端（{}）".format(
                    self.name, self._emotion_model._mode))
            else:
                self.logger.info("{} 使用主要 LLM 做情緒偵測".format(self.name))
                self._emotion_model = None  # 由 _update_emotion 的 completion() 路徑處理
        # 從 checkpoint 恢復情緒狀態至 scratch
        self.scratch.emotion = EmotionState.from_dict(self.status.get("emotion"))

    def _update_emotion(self, text, other_agent=None, relation=None):
        """
        偵測文字中的情緒並更新代理人的情緒狀態。

        雙路徑流程：
          1. SACF 模型提供情感極性信號（正面/負面/中性）
          2. LLM 作為主要��緒判斷（10 類情緒，情境感知）
          3. SACF 與 LLM 互相校準，避免單方面偏向
          4. 若情緒強度≥4 或標籤改變 → 存入 Associate 記憶

        Args:
            text:        行動描述或對話全文
            other_agent: 對話對象 Agent（對話場景才傳入）
            relation:    自身對 other_agent 的關係印象描述
        """
        if not text or not text.strip():
            return

        prev       = EmotionState.from_dict(self.status.get("emotion"))
        prev_label = prev.label   # 融合前的原始標籤，用於顯著性判斷

        # ── 組裝對話上下文 ────────────────────────────────────────
        # 有對話對象時，加入關係印象與近期互動記憶，讓情緒判斷與對象有關
        chat_memory = ""
        if other_agent:
            recent = self.associate.retrieve_chats(other_agent.name)
            for c in recent[:2]:
                delta = utils.get_timer().get_delta(c.create)
                chat_memory += f"{delta}分鐘前：{c.describe}。"

        # EmoLLM context：給專用後端用的純文字上下文
        if other_agent:
            emollm_context = f"正在與 {other_agent.name} 對話。"
            if relation:
                emollm_context += f"對{other_agent.name}的印象：{relation}。"
            if chat_memory:
                emollm_context += f" 過去互動：{chat_memory}"
        else:
            emollm_context = self.scratch.currently

        # ── 雙路徑情緒偵測：LLM 為主（10 類情緒 + 情境感知），SACF 為輔（情感強度校準）──
        #
        # 設計理念：讓 AI 自然產生多樣情緒，而非僅依賴 SACF 的正面偏向
        # LLM 能理解角色個性、情境脈絡、社交關係，因此是更好的情緒判斷者
        # SACF 提供客觀的情感極性信號，作為 LLM 判斷的參考校準

        state = None
        sacf_signal = None

        # 路徑 1：SACF 提供情感極性信號（不直接決定情緒標籤）
        if self._emotion_model and self._emotion_model.is_available():
            try:
                sacf_state = self._emotion_model.detect(
                    text, agent_name=self.name, context=emollm_context
                )
                sacf_signal = sacf_state  # 保存為參考信號
            except Exception:
                sacf_signal = None

        # 路徑 2：LLM 作為主要情緒判斷（10 類情緒，情境感知）
        if self.llm_available():
            result = self.completion(
                "emotion_detect", text,
                other_name  = other_agent.name if other_agent else None,
                relation    = relation,
                chat_memory = chat_memory or None,
            )
            state = EmotionState.from_dict(result) if isinstance(result, dict) else None
            # 防呆：若 LLM 輸出無效標籤（如模板佔位符），隨機選擇一個有效標籤
            if state and state.label == EmotionState.DEFAULT and isinstance(result, dict):
                raw_label = result.get("label", "")
                if raw_label not in EmotionState.LABELS:
                    import random
                    fallback_options = ["快樂", "焦慮", "興奮", "悲傷", "厭惡"]
                    state.label = random.choice(fallback_options)
                    state.intensity = max(1, min(3, state.intensity))

        # 路徑 3：若 LLM 不可用，退回 SACF 結果
        if state is None and sacf_signal is not None:
            state = sacf_signal
        elif state is None:
            return

        # ── SACF 校準：當 SACF 偵測到負面情感但 LLM 給了正面標籤時，
        #    適度降低 LLM 結果的強度，反之亦然。
        #    這讓兩個系統互相制衡，避免單方面偏向。
        if sacf_signal is not None and state is not sacf_signal:
            sacf_negative = sacf_signal.label in ("悲傷", "焦慮")
            llm_positive  = state.label in ("快樂", "興奮")
            sacf_positive = sacf_signal.label in ("快樂", "興奮")
            llm_negative  = state.label in ("悲傷", "焦慮", "憤怒", "恐懼", "厭惡", "疲憊")

            if sacf_negative and llm_positive:
                # SACF 感知到負面，但 LLM 給了正面 → 降低正面強度
                state.intensity = max(1, state.intensity - 2)
            elif sacf_positive and llm_negative:
                # SACF 感知到正面，但 LLM 給了負面 → 降低負面強度
                state.intensity = max(1, state.intensity - 1)

        # 慣性融合：以舊情緒為基礎，漸進融入新偵測結果
        prev.blend_update(state.label, state.intensity, state.reason)
        self.status["emotion"] = prev.to_dict()
        self.scratch.emotion   = prev
        self.logger.info(
            "{} 情緒偵測 → {}（原因：{}）".format(self.name, prev.describe(), prev.reason)
        )

        # ── 情緒傳染：對方情緒對自己有微弱影響 ─────────────────
        # 對話時，對方的顯著情緒（強度≥5）會以 20% 強度傳染給自己，
        # 模擬人類對話中的情緒感染現象。
        if other_agent is not None:
            other_emo = other_agent.scratch.emotion
            if other_emo.label != "平靜" and other_emo.intensity >= 5:
                contagion_intensity = max(1, min(2, round(other_emo.intensity * 0.2)))
                prev.blend_update(
                    other_emo.label,
                    contagion_intensity,
                    f"受{other_agent.name}感染",
                )
                self.status["emotion"] = prev.to_dict()
                self.scratch.emotion   = prev
                self.logger.info(
                    "{} 情緒傳染：受 {} 的{}（強度{}）微弱影響".format(
                        self.name, other_agent.name, other_emo.label, other_emo.intensity
                    )
                )

        # 情緒變化時存入長期記憶（強度≥4，或情緒標籤改變）
        # 降低門檻讓更多情緒記憶累積，豐富角色的情感經歷
        if prev.intensity >= 4 or state.label != prev_label:
            self._add_emotion_concept(prev)

    def _add_emotion_concept(self, state):
        """將顯著情緒存入 Associate 記憶（node_type='emotion'，保留 7 天）"""
        describe = (
            f"{self.name} 感受到{state.label}（強度：{state.intensity}/10）。"
            + (state.reason if state.reason else "")
        )
        event = memory.Event(
            self.name, "感受到", state.label,
            describe=describe,
            address=self.get_tile().get_address(),
        )
        expire = utils.get_timer().get_date() + datetime.timedelta(days=7)
        self.associate.add_node("emotion", event, state.intensity, expire=expire)
        self.logger.debug("{} 情緒記憶存入：{}".format(self.name, describe))

    def completion(self, func_hint, *args, **kwargs):
        assert hasattr(
            self.scratch, "prompt_" + func_hint
        ), "Can not find func prompt_{} from scratch".format(func_hint)
        func = getattr(self.scratch, "prompt_" + func_hint)
        prompt = func(*args, **kwargs)
        title, msg = "{}.{}".format(self.name, func_hint), {}
        if self.llm_available():
            self.logger.info("{} -> {}".format(self.name, func_hint))
            output = self._llm.completion(**prompt, caller=func_hint)
            responses = self._llm.meta_responses
            msg = {"<PROMPT>": "\n" + prompt["prompt"] + "\n"}
            msg.update(
                {
                    "<RESPONSE[{}/{}]>".format(idx+1, len(responses)): "\n" + r + "\n"
                    for idx, r in enumerate(responses)
                }
            )
        else:
            output = prompt.get("failsafe")
        msg["<OUTPUT>"] = "\n" + str(output) + "\n"
        self.logger.debug(utils.block_msg(title, msg))
        return output

    def think(self, status, agents):
        events = self.move(status["coord"], status.get("path"))
        plan, _ = self.make_schedule()

        if (plan["describe"] == "sleeping" or "睡" in plan["describe"]) and self.is_awake():
            self.logger.info("{} is going to sleep...".format(self.name))
            address = self.spatial.find_address("睡覺", as_list=True)
            tiles = self.maze.get_address_tiles(address)
            coord = random.choice(list(tiles))
            events = self.move(coord)
            self.action = memory.Action(
                memory.Event(self.name, "正在", "睡覺", address=address, emoji="😴"),
                memory.Event(
                    address[-1],
                    "被占用",
                    self.name,
                    address=address,
                    emoji="🛌",
                ),
                duration=plan["duration"],
                start=utils.get_timer().daily_time(plan["start"]),
            )
        if self.is_awake():
            self.percept()
            self.make_plan(agents)
            self.reflect()
        else:
            if self.action.finished():
                self.action = self._determine_action()

        emojis = {}
        if self.action:
            emojis[self.name] = {"emoji": self.get_event().emoji, "coord": self.coord}
        for eve, coord in events.items():
            if eve.subject in agents:
                continue
            emojis[":".join(eve.address)] = {"emoji": eve.emoji, "coord": coord}
        self.plan = {
            "name": self.name,
            "path": self.find_path(agents),
            "emojis": emojis,
        }
        return self.plan

    def move(self, coord, path=None):
        events = {}

        def _update_tile(coord):
            tile = self.maze.tile_at(coord)
            if not self.action:
                return {}
            if not tile.update_events(self.get_event()):
                tile.add_event(self.get_event())
            obj_event = self.get_event(False)
            if obj_event:
                self.maze.update_obj(coord, obj_event)
            return {e: coord for e in tile.get_events()}

        if self.coord and self.coord != coord:
            tile = self.get_tile()
            tile.remove_events(subject=self.name)
            if tile.has_address("game_object"):
                addr = tile.get_address("game_object")
                self.maze.update_obj(
                    self.coord, memory.Event(addr[-1], address=addr)
                )
            events.update({e: self.coord for e in tile.get_events()})
        if not path:
            events.update(_update_tile(coord))
        self.coord = coord
        self.path = path or []

        return events

    def make_schedule(self):
        if not self.schedule.scheduled():
            self.logger.info("{} is making schedule...".format(self.name))
            # update currently
            if self.associate.index.nodes_num > 0:
                self.associate.cleanup_index()
                focus = [
                    f"{self.name} 在 {utils.get_timer().daily_format_cn()} 的計畫。",
                    f"在 {self.name} 的生活中，重要的近期事件。",
                ]
                # 始終將情緒加入記憶檢索焦點（即使平靜也提供脈絡）
                # 讓 LLM 自然判斷情緒如何影響每日計畫
                current_emotion = EmotionState.from_dict(self.status.get("emotion"))
                if current_emotion.label != "平靜":
                    focus.append(
                        f"{self.name} 目前感到{current_emotion.describe()}，"
                        f"這可能影響今日活動的選擇與心態。"
                    )
                else:
                    focus.append(
                        f"{self.name} 目前情緒平靜，"
                        f"但日常生活中的各種事件可能帶來情緒起伏。"
                    )
                retrieved = self.associate.retrieve_focus(focus)
                self.logger.info(
                    "{} retrieved {} concepts".format(self.name, len(retrieved))
                )
                if retrieved:
                    plan = self.completion("retrieve_plan", retrieved)
                    thought = self.completion("retrieve_thought", retrieved)
                    self.scratch.currently = self.completion(
                        "retrieve_currently", plan, thought
                    )
            # make init schedule
            self.schedule.create = utils.get_timer().get_date()
            wake_up = self.completion("wake_up")
            init_schedule = self.completion("schedule_init", wake_up)
            # make daily schedule
            hours = [f"{i}:00" for i in range(24)]
            # seed = [(h, "sleeping") for h in hours[:wake_up]]
            seed = [(h, "睡覺") for h in hours[:wake_up]]
            seed += [(h, "") for h in hours[wake_up:]]
            schedule = {}
            for _ in range(self.schedule.max_try):
                schedule = {h: s for h, s in seed[:wake_up]}
                schedule.update(
                    self.completion("schedule_daily", wake_up, init_schedule)
                )
                if len(set(schedule.values())) >= self.schedule.diversity:
                    break

            def _to_duration(date_str):
                return utils.daily_duration(utils.to_date(date_str, "%H:%M"))

            schedule = {_to_duration(k): v for k, v in schedule.items()}
            starts = list(sorted(schedule.keys()))
            for idx, start in enumerate(starts):
                end = starts[idx + 1] if idx + 1 < len(starts) else 24 * 60
                self.schedule.add_plan(schedule[start], end - start)
            schedule_time = utils.get_timer().time_format_cn(self.schedule.create)
            thought = "這是 {} 在 {} 的計畫：{}".format(
                self.name, schedule_time, "；".join(init_schedule)
            )
            event = memory.Event(
                self.name,
                "計畫",
                schedule_time,
                describe=thought,
                address=self.get_tile().get_address(),
            )
            self._add_concept(
                "thought",
                event,
                expire=self.schedule.create + datetime.timedelta(days=30),
            )
        # decompose current plan
        plan, _ = self.schedule.current_plan()
        if self.schedule.decompose(plan):
            decompose_schedule = self.completion(
                "schedule_decompose", plan, self.schedule
            )
            decompose, start = [], plan["start"]
            for describe, duration in decompose_schedule:
                decompose.append(
                    {
                        "idx": len(decompose),
                        "describe": describe,
                        "start": start,
                        "duration": duration,
                    }
                )
                start += duration
            plan["decompose"] = decompose
        return self.schedule.current_plan()

    def revise_schedule(self, event, start, duration):
        self.action = memory.Action(event, start=start, duration=duration)
        plan, _ = self.schedule.current_plan()
        if len(plan["decompose"]) > 0:
            plan["decompose"] = self.completion(
                "schedule_revise", self.action, self.schedule
            )

    def percept(self):
        scope = self.maze.get_scope(self.coord, self.percept_config)
        # add spatial memory
        for tile in scope:
            if tile.has_address("game_object"):
                self.spatial.add_leaf(tile.address)
        events, arena = {}, self.get_tile().get_address("arena")
        # gather events in scope
        for tile in scope:
            if not tile.events or tile.get_address("arena") != arena:
                continue
            dist = math.dist(tile.coord, self.coord)
            for event in tile.get_events():
                if dist < events.get(event, float("inf")):
                    events[event] = dist
        events = list(sorted(events.keys(), key=lambda k: events[k]))
        # get concepts
        self.concepts, valid_num = [], 0
        for idx, event in enumerate(events[: self.percept_config["att_bandwidth"]]):
            recent_nodes = (
                self.associate.retrieve_events() + self.associate.retrieve_chats()
            )
            recent_nodes = set(n.describe for n in recent_nodes)
            if event.get_describe() not in recent_nodes:
                if event.object == "idle" or event.object == "空閒":
                    node = Concept.from_event(
                        "idle_" + str(idx), "event", event, poignancy=1
                    )
                else:
                    valid_num += 1
                    node_type = "chat" if event.fit(self.name, "對話") else "event"
                    node = self._add_concept(node_type, event)
                    self.status["poignancy"] += node.poignancy
                self.concepts.append(node)
        self.concepts = [c for c in self.concepts if c.event.subject != self.name]
        self.logger.info(
            "{} percept {}/{} concepts".format(self.name, valid_num, len(self.concepts))
        )

    def make_plan(self, agents):
        if self._reaction(agents):
            return
        if self.path:
            return
        if self.action.finished():
            self.action = self._determine_action()

    # create action && object events
    def make_event(self, subject, describe, address):
        # emoji = self.completion("describe_emoji", describe)
        # return self.completion(
        #     "describe_event", subject, subject + describe, address, emoji
        # )

        e_describe = describe.replace("(", "").replace(")", "").replace("<", "").replace(">", "")
        if e_describe.startswith(subject + "此時"):
            e_describe = e_describe[len(subject + "此時"):]
        if e_describe.startswith(subject):
            e_describe = e_describe[len(subject):]
        event = memory.Event(
            subject, "此時", e_describe, describe=describe, address=address
        )
        return event

    def reflect(self):
        def _add_thought(thought, evidence=None):
            # event = self.completion(
            #     "describe_event",
            #     self.name,
            #     thought,
            #     address=self.get_tile().get_address(),
            # )
            event = self.make_event(self.name, thought, self.get_tile().get_address())
            return self._add_concept("thought", event, filling=evidence)

        if self.status["poignancy"] < self.think_config["poignancy_max"]:
            return
        nodes = self.associate.retrieve_events() + self.associate.retrieve_thoughts()
        if not nodes:
            return
        self.logger.info(
            "{} reflect(P{}/{}) with {} concepts...".format(
                self.name,
                self.status["poignancy"],
                self.think_config["poignancy_max"],
                len(nodes),
            )
        )
        nodes = sorted(nodes, key=lambda n: n.access, reverse=True)[
            : self.associate.max_importance
        ]
        # summary thought
        focus = self.completion("reflect_focus", nodes, 3)
        retrieved = self.associate.retrieve_focus(focus, reduce_all=False)
        for r_nodes in retrieved.values():
            thoughts = self.completion("reflect_insights", r_nodes, 5)
            for thought, evidence in thoughts:
                _add_thought(thought, evidence)
        # summary chats
        if self.chats:
            recorded, evidence = set(), []
            for name, _ in self.chats:
                if name == self.name or name in recorded:
                    continue
                res = self.associate.retrieve_chats(name)
                if res and len(res) > 0:
                    node = res[-1]
                    evidence.append(node.node_id)
            thought = self.completion("reflect_chat_planing", self.chats)
            _add_thought(f"對於 {self.name} 的計畫：{thought}", evidence)
            thought = self.completion("reflect_chat_memory", self.chats)
            _add_thought(f"{self.name} {thought}", evidence)
        # 情緒記憶模式反思：當累積 3 條以上情緒記憶時，生成整體情緒洞察
        emotion_nodes = self.associate.retrieve_emotions()
        if len(emotion_nodes) >= 3:
            thought = self.completion("emotion_memory_reflect", emotion_nodes)
            _add_thought(f"{self.name} 的情緒狀態：{thought}")

        self.status["poignancy"] = 0
        self.chats = []

    def find_path(self, agents):
        address = self.get_event().address
        if self.path:
            return self.path
        if address == self.get_tile().get_address():
            return []
        if address[0] == "<waiting>":
            return []
        if address[0] == "<persona>":
            target_tiles = self.maze.get_around(agents[address[1]].coord)
        else:
            target_tiles = self.maze.get_address_tiles(address)
        if tuple(self.coord) in target_tiles:
            return []

        # filter tile with self event
        def _ignore_target(t_coord):
            if list(t_coord) == list(self.coord):
                return True
            events = self.maze.tile_at(t_coord).get_events()
            if any(e.subject in agents for e in events):
                return True
            return False

        target_tiles = [t for t in target_tiles if not _ignore_target(t)]
        if not target_tiles:
            return []
        if len(target_tiles) >= 4:
            target_tiles = random.sample(target_tiles, 4)
        pathes = {t: self.maze.find_path(self.coord, t) for t in target_tiles}
        target = min(pathes, key=lambda p: len(pathes[p]))
        return pathes[target][1:]

    def _determine_action(self):
        self.logger.info("{} is determining action...".format(self.name))
        plan, de_plan = self.schedule.current_plan()
        describes = [plan["describe"], de_plan["describe"]]
        address = self.spatial.find_address(describes[0], as_list=True)
        if not address:
            tile = self.get_tile()
            kwargs = {
                "describes": describes,
                "spatial": self.spatial,
                "address": tile.get_address("world", as_list=True),
            }
            kwargs["address"].append(
                self.completion("determine_sector", **kwargs, tile=tile)
            )
            arenas = self.spatial.get_leaves(kwargs["address"])
            if len(arenas) == 1:
                kwargs["address"].append(arenas[0])
            else:
                kwargs["address"].append(self.completion("determine_arena", **kwargs))
            objs = self.spatial.get_leaves(kwargs["address"])
            if len(objs) == 1:
                kwargs["address"].append(objs[0])
            elif len(objs) > 1:
                kwargs["address"].append(self.completion("determine_object", **kwargs))
            address = kwargs["address"]

        event = self.make_event(self.name, describes[-1], address)
        obj_describe = self.completion("describe_object", address[-1], describes[-1])
        obj_event = self.make_event(address[-1], obj_describe, address)

        event.emoji = f"{de_plan['describe']}"

        action = memory.Action(
            event,
            obj_event,
            duration=de_plan["duration"],
            start=utils.get_timer().daily_time(de_plan["start"]),
        )
        # 根據當前行動偵測情緒，影響後續對話生成
        # 將主計畫（describes[0]）與細分計畫（describes[-1]）合併傳入，
        # 提供更豐富的脈絡讓 LLM 判斷情緒
        emotion_text = describes[-1]
        if describes[0] and describes[0] != describes[-1]:
            emotion_text = f"{describes[0]}：{describes[-1]}"
        self._update_emotion(emotion_text)
        return action

    def _reaction(self, agents=None, ignore_words=None):
        focus = None
        ignore_words = ignore_words or ["空閒"]

        def _focus(concept):
            return concept.event.subject in agents

        def _ignore(concept):
            return any(i in concept.describe for i in ignore_words)

        if agents:
            priority = [i for i in self.concepts if _focus(i)]
            if priority:
                focus = random.choice(priority)
        if not focus:
            priority = [i for i in self.concepts if not _ignore(i)]
            if priority:
                focus = random.choice(priority)
        if not focus or focus.event.subject not in agents:
            return
        other, focus = agents[focus.event.subject], self.associate.get_relation(focus)

        if self._chat_with(other, focus):
            return True
        if self._wait_other(other, focus):
            return True
        return False

    def _skip_react(self, other):
        def _skip(event):
            if not event.address or "sleeping" in event.get_describe(False) or "睡覺" in event.get_describe(False):
                return True
            if event.predicate == "待開始":
                return True
            return False

        if utils.get_timer().daily_duration(mode="hour") >= 23:
            return True
        if _skip(self.get_event()) or _skip(other.get_event()):
            return True
        return False

    def _chat_with(self, other, focus):
        if len(self.schedule.daily_schedule) < 1 or len(other.schedule.daily_schedule) < 1:
            # initializing
            return False
        if self._skip_react(other):
            return False
        if other.path:
            return False
        if self.get_event().fit(predicate="對話") or other.get_event().fit(predicate="對話"):
            return False

        chats = self.associate.retrieve_chats(other.name)
        if chats:
            delta = utils.get_timer().get_delta(chats[0].create)
            self.logger.info(
                "retrieved chat between {} and {}({} min):\n{}".format(
                    self.name, other.name, delta, chats[0]
                )
            )
            if delta < 60:
                return False

        if not self.completion("decide_chat", self, other, focus, chats):
            return False

        self.logger.info("{} decides chat with {}".format(self.name, other.name))
        start, chats = utils.get_timer().get_date(), []
        relations = [
            self.completion("summarize_relation", self, other.name),
            other.completion("summarize_relation", other, self.name),
        ]

        for i in range(self.chat_iter):
            text = self.completion(
                "generate_chat", self, other, relations[0], chats
            )

            if i > 0:
                # 對於發起對話的Agent，從第2轮對話開始，检查是否出現“复讀”現象
                end = self.completion(
                    "generate_chat_check_repeat", self, chats, text
                )
                if end:
                    break

                # 對於發起對話的Agent，從第2轮對話開始，检查話題是否結束
                chats.append((self.name, text))
                end = self.completion(
                    "decide_chat_terminate", self, other, chats
                )
                if end:
                    break
            else :
                chats.append((self.name, text))

            text = other.completion(
                "generate_chat", other, self, relations[1], chats
            )
            if i > 0:
                # 對於響應對話的Agent，從第2轮開始，检查是否出現“复讀”現象
                end = self.completion(
                    "generate_chat_check_repeat", other, chats, text
                )
                if end:
                    break

            chats.append((other.name, text))

            # 對於響應對話的Agent，從第1轮開始，检查話題是否結束
            end = other.completion(
                "decide_chat_terminate", other, self, chats
            )
            if end:
                break

        key = utils.get_timer().get_date("%Y%m%d-%H:%M")
        if key not in self.conversation.keys():
            self.conversation[key] = []
        self.conversation[key].append({f"{self.name} -> {other.name} @ {'，'.join(self.get_event().address)}": chats})

        self.logger.info(
            "{} and {} has chats\n  {}".format(
                self.name,
                other.name,
                "\n  ".join(["{}: {}".format(n, c) for n, c in chats]),
            )
        )
        chat_summary = self.completion("summarize_chats", chats)
        # 對話結束後，根據各自的發言偵測情緒，作為下一輪互動的情緒底色
        # 傳入對方資訊與關係印象，讓同一句話對不同居民產生差異化情緒反應
        chat_text = "\n".join(f"{n}: {c}" for n, c in chats)
        self._update_emotion(chat_text, other_agent=other, relation=relations[0])
        other._update_emotion(chat_text, other_agent=self, relation=relations[1])
        duration = int(sum([len(c[1]) for c in chats]) / 240)
        self.schedule_chat(
            chats, chat_summary, start, duration, other
        )
        other.schedule_chat(chats, chat_summary, start, duration, self)
        return True

    def _wait_other(self, other, focus):
        if self._skip_react(other):
            return False
        if not self.path:
            return False
        if self.get_event().address != other.get_tile().get_address():
            return False
        if not self.completion("decide_wait", self, other, focus):
            return False
        self.logger.info("{} decides wait to {}".format(self.name, other.name))
        start = utils.get_timer().get_date()
        # duration = other.action.end - start
        t = other.action.end - start
        duration = int(t.total_seconds() / 60)
        event = memory.Event(
            self.name,
            "waiting to start",
            self.get_event().get_describe(False),
            # address=["<waiting>"] + self.get_event().address,
            address=self.get_event().address,
            emoji=f"⌛",
        )
        self.revise_schedule(event, start, duration)

    def schedule_chat(self, chats, chats_summary, start, duration, other, address=None):
        self.chats.extend(chats)
        event = memory.Event(
            self.name,
            "對話",
            other.name,
            describe=chats_summary,
            address=address or self.get_tile().get_address(),
            emoji=f"💬",
        )
        self.revise_schedule(event, start, duration)

    def _add_concept(
        self,
        e_type,
        event,
        create=None,
        expire=None,
        filling=None,
    ):
        if event.fit(None, "is", "idle"):
            poignancy = 1
        elif event.fit(None, "此時", "空閒"):
            poignancy = 1
        elif e_type == "chat":
            poignancy = self.completion("poignancy_chat", event)
        else:
            poignancy = self.completion("poignancy_event", event)
        self.logger.debug("{} add associate {}".format(self.name, event))
        return self.associate.add_node(
            e_type,
            event,
            poignancy,
            create=create,
            expire=expire,
            filling=filling,
        )

    def get_tile(self):
        return self.maze.tile_at(self.coord)

    def get_event(self, as_act=True):
        return self.action.event if as_act else self.action.obj_event

    def is_awake(self):
        if not self.action:
            return True
        if self.get_event().fit(self.name, "is", "sleeping"):
            return False
        if self.get_event().fit(self.name, "正在", "睡覺"):
            return False
        return True

    def llm_available(self):
        if not self._llm:
            return False
        return self._llm.is_available()

    def to_dict(self, with_action=True):
        info = {
            "status": self.status,
            "schedule": self.schedule.to_dict(),
            "associate": self.associate.to_dict(),
            "chats": self.chats,
            "currently": self.scratch.currently,
        }
        if with_action:
            info.update({"action": self.action.to_dict()})
        return info
