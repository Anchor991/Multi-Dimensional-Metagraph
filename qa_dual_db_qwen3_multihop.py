"""
qa_dual_db_qwen3_multihop.py

使用 Qwen3 (OpenAI 兼容 API) 把中文自然语言问题转换为「多跳」结构化 JSON 查询，
在「新库 graph.json」和「旧库 old_graph.json」上分别执行，并对比多跳准确率。

多跳语义：
- LLM 生成一个 JSON：
  {
    "target": "person" | "geo" | "year",
    "sub_queries": [
      {
        "target": "...",      # 必须和外层 target 一致
        "filters": {...}      # 单跳查询条件
      },
      ...
    ]
  }
- 程序对每个 sub_query 跑一次单跳查询，然后对结果做交集：
  final_result = ⋂_i result(sub_queries[i])

特性：
- 使用 Qwen3 非思考模式：extra_body={"enable_thinking": False}
- temperature = 0.0，保证输出稳定
- few-shot 提示，固定 JSON 结构
- LLM 输出缓存到 qwen3_multihop_cache.json，避免重复调用

***************填入你的 DashScope API Key 后即可运行。****************
"""

import json
from dataclasses import dataclass
import os
from typing import Any, Dict, List
from openai import OpenAI


# =========================== 固定配置（按需修改） ===========================

# 把这里改成你的真实 DashScope API Key
DASHSCOPE_API_KEY = "#"

# 中国地域（国内）兼容接口，如果你用新加坡，把域名改成 dashscope-intl.aliyuncs.com
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 使用的 Qwen3 模型名
DEFAULT_MODEL = "qwen3-8b"

# 图数据库文件路径
NEW_GRAPH_PATH = "graph.json"
OLD_GRAPH_PATH = "old_graph.json"

# LLM 缓存文件（和单跳的缓存区分开）
CACHE_PATH = "qwen3_multihop_cache.json"

# 修改系统提示或 few-shot 时，把版本号改一下，避免旧缓存干扰
PROMPT_VERSION = "qwen3-dual-multihop-v1"


# =========================== 基础数据结构 ===========================

@dataclass
class Question:
    qid: str
    text: str
    gold_answer: List[str]


class NewGraphDB:
    """
    新库：三维超立方体结构
    - dimensions: person / time / geo
    - facts: person + time + geo 的组合
    """

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dims = data["dimensions"]
        self.person_by_id = {p["id"]: p["name"] for p in dims["person"]}
        self.time_by_id = {t["id"]: t["value"] for t in dims["time"]}
        self.geo_by_id = {g["id"]: g["name"] for g in dims["geo"]}

        self.person_id_by_name = {v: k for k, v in self.person_by_id.items()}
        self.time_id_by_year = {v: k for k, v in self.time_by_id.items()}
        self.geo_id_by_name = {v: k for k, v in self.geo_by_id.items()}

        self.facts: List[Dict[str, Any]] = data["facts"]

    def run_single_query(self, query: Dict[str, Any]) -> List[str]:
        """
        单跳查询（和你原来的 run_query 完全一样）：
        query = {
          "target": "person" | "geo" | "year",
          "filters": {
            "person": [...],
            "year":   [...],
            "geo":    [...]
          }
        }
        """
        target = query.get("target")
        if target not in {"person", "geo", "year"}:
            raise ValueError(f"invalid target: {target!r}")

        filters = query.get("filters") or {}

        person_names = filters.get("person", [])
        years = filters.get("year", [])
        geo_names = filters.get("geo", [])

        person_ids = {
            self.person_id_by_name[n] for n in person_names
            if n in self.person_id_by_name
        }
        time_ids = {
            self.time_id_by_year[y] for y in years
            if y in self.time_id_by_year
        }
        geo_ids = {
            self.geo_id_by_name[g] for g in geo_names
            if g in self.geo_id_by_name
        }

        results: List[str] = []
        for fact in self.facts:
            if person_ids and fact["person"] not in person_ids:
                continue
            if time_ids and fact["time"] not in time_ids:
                continue
            if geo_ids and fact["geo"] not in geo_ids:
                continue

            if target == "person":
                results.append(self.person_by_id[fact["person"]])
            elif target == "geo":
                results.append(self.geo_by_id[fact["geo"]])
            else:  # "year"
                results.append(self.time_by_id[fact["time"]])

        return sorted(set(results))

    def run_multihop_query(self, query: Dict[str, Any]) -> List[str]:
        """
        多跳查询：对 sub_queries 结果求交集。
        query = {
          "target": "...",
          "sub_queries": [
            { "target": "...", "filters": {...} },
            ...
          ]
        }
        """
        target = query.get("target")
        sub_queries = query.get("sub_queries") or []
        if not sub_queries:
            return []

        final_set = None
        for sq in sub_queries:
            sq_target = sq.get("target", target)
            if sq_target != target:
                raise ValueError(f"sub_query target {sq_target!r} != outer target {target!r}")
            single_res = set(self.run_single_query({
                "target": target,
                "filters": sq.get("filters", {}),
            }))
            if final_set is None:
                final_set = single_res
            else:
                final_set &= single_res

        return sorted(final_set) if final_set is not None else []


class OldGraphDB:
    """
    旧库：节点 + 边结构
    - nodes: Person / Geo
    - edges: Person -> Geo，带 year
    """

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.person_by_id: Dict[str, str] = {}
        self.geo_by_id: Dict[str, str] = {}

        for node in data["nodes"]:
            if node["type"] == "Person":
                self.person_by_id[node["id"]] = node["name"]
            elif node["type"] == "Geo":
                self.geo_by_id[node["id"]] = node["name"]

        self.person_id_by_name = {v: k for k, v in self.person_by_id.items()}
        self.geo_id_by_name = {v: k for k, v in self.geo_by_id.items()}

        # edges: from person_id -> geo_id with year
        self.edges = data["edges"]

    def run_single_query(self, query: Dict[str, Any]) -> List[str]:
        target = query.get("target")
        if target not in {"person", "geo", "year"}:
            raise ValueError(f"invalid target: {target!r}")

        filters = query.get("filters") or {}

        person_names = filters.get("person", [])
        years = filters.get("year", [])
        geo_names = filters.get("geo", [])

        person_ids = {
            self.person_id_by_name[n] for n in person_names
            if n in self.person_id_by_name
        }
        geo_ids = {
            self.geo_id_by_name[g] for g in geo_names
            if g in self.geo_id_by_name
        }
        year_set = set(years)

        results: List[str] = []
        for e in self.edges:
            if person_ids and e["from"] not in person_ids:
                continue
            if geo_ids and e["to"] not in geo_ids:
                continue
            if year_set and e["year"] not in year_set:
                continue

            if target == "person":
                results.append(self.person_by_id[e["from"]])
            elif target == "geo":
                results.append(self.geo_by_id[e["to"]])
            else:  # "year"
                results.append(e["year"])

        return sorted(set(results))

    def run_multihop_query(self, query: Dict[str, Any]) -> List[str]:
        target = query.get("target")
        sub_queries = query.get("sub_queries") or []
        if not sub_queries:
            return []

        final_set = None
        for sq in sub_queries:
            sq_target = sq.get("target", target)
            if sq_target != target:
                raise ValueError(f"sub_query target {sq_target!r} != outer target {target!r}")
            single_res = set(self.run_single_query({
                "target": target,
                "filters": sq.get("filters", {}),
            }))
            if final_set is None:
                final_set = single_res
            else:
                final_set &= single_res

        return sorted(final_set) if final_set is not None else []


# =========================== LLM 客户端 & 缓存 ===========================

def make_qwen_client() -> OpenAI:
    """
    创建 OpenAI 兼容的 Qwen3 客户端。
    所有配置（API Key / base_url / model）都在文件顶部写死。
    """
    api_key = DASHSCOPE_API_KEY
    if not api_key or api_key.startswith("<YOUR_") or api_key == "#":
        raise RuntimeError("请在文件顶部 DASHSCOPE_API_KEY 中填入你的真实 DashScope API Key。")

    return OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)


def load_cache(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: Dict[str, Any], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# =========================== Prompt & few-shot ===========================

def build_messages_for_question(question_text: str) -> List[Dict[str, str]]:
    """
    构造 system prompt + few-shot + 当前 user 问题。
    多跳 JSON 结构：

    {
      "target": "person" | "geo" | "year",
      "sub_queries": [
        {
          "target": "person" | "geo" | "year",
          "filters": {
            "person": [...],
            "year":   [...],
            "geo":    [...]
          }
        },
        ...
      ]
    }

    语义：对所有 sub_queries 的结果求交集，得到最终答案。
    """

    system_prompt = f"""
你是一个只负责“生成多跳查询 JSON”的助手，用于在旅游行程数据库上检索答案。

数据库有三类实体：
- person：人名（如“小红、小张、小明”等）；
- year：年份字符串（如 "2020"、"2021"）；
- geo：城市名（如“苏州、深圳、南京、北京”等）。

给定一个中文问题，你只需要生成一个 JSON 对象，不要直接回答问题。
这个 JSON 将在两套数据库实现上执行，因此格式必须非常严格。

多跳 JSON 结构（不要输出注释）：

{{
  "target": "person" | "geo" | "year",
  "sub_queries": [
    {{
      "target": "person" | "geo" | "year",
      "filters": {{
        "person": ["小明"],
        "year": ["2020"],
        "geo": ["苏州"]
      }}
    }}
  ]
}}

含义说明：
- 外层 target 表示最终想要的实体类别：
  - 问“谁”的问题，target = "person"；
  - 问“哪些地方 / 哪些城市”的问题，target = "geo"；
  - 问“哪几年”的问题，target = "year"。
- sub_queries 是若干个单跳查询，每个 sub_query 的 target 必须和外层 target 一致。
- filters 含义与单跳相同：
  - "person"：限定涉及的人员（字符串数组）；
  - "year"：限定年份（字符串数组，例如 ["2020"]）；
  - "geo"：限定城市名（字符串数组）。
- 所有 sub_queries 的结果会被取交集，得到最终答案：
  - 例如两个 sub_queries 都是 target="person"：
    第一个查询出“2020年去过苏州的人”，第二个查询出“2021年去过深圳的人”，
    最终答案是两者交集，即“两个条件都满足的人”。

严格要求：
1. 只输出一个 JSON 对象，不要输出任何解释或额外文字。
2. JSON 中所有键和值都使用半角双引号，必须是合法 JSON。
3. 不要生成 target 和 sub_queries 之外的其他字段。
4. 所有名字和城市名都保持问题中的中文原文，不要翻译或改写。
5. 你的输出将直接被 json.loads() 解析，请避免多余符号。

当前提示版本：{PROMPT_VERSION}
""".strip()

    few_shot: List[Dict[str, str]] = []

    # 示例 1：多跳，target=person，两跳：2020苏州 & 2021深圳
    few_shot += [
        {
            "role": "user",
            "content": "谁在2020年去过苏州，并且在2021年也去过深圳？",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "target": "person",
                    "sub_queries": [
                        {
                            "target": "person",
                            "filters": {
                                "geo": ["苏州"],
                                "year": ["2020"],
                            },
                        },
                        {
                            "target": "person",
                            "filters": {
                                "geo": ["深圳"],
                                "year": ["2021"],
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]

    # 示例 2：多跳，target=geo，两跳：小张2020 & 小张2021
    few_shot += [
        {
            "role": "user",
            "content": "哪些城市既是小张在2020年去过的城市，又是他在2021年去过的城市？",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "target": "geo",
                    "sub_queries": [
                        {
                            "target": "geo",
                            "filters": {
                                "person": ["小张"],
                                "year": ["2020"],
                            },
                        },
                        {
                            "target": "geo",
                            "filters": {
                                "person": ["小张"],
                                "year": ["2021"],
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]

    # 示例 3：多跳，target=year，两跳：小李 + 南京 / 苏州
    few_shot += [
        {
            "role": "user",
            "content": "小李在哪些年份同时去过南京和苏州？",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "target": "year",
                    "sub_queries": [
                        {
                            "target": "year",
                            "filters": {
                                "person": ["小李"],
                                "geo": ["南京"],
                            },
                        },
                        {
                            "target": "year",
                            "filters": {
                                "person": ["小李"],
                                "geo": ["苏州"],
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        *few_shot,
        {"role": "user", "content": question_text},
    ]
    return messages


def parse_llm_json(content: str) -> Dict[str, Any]:
    """
    从 LLM 输出中解析 JSON。
    优先整体解析；失败时尝试截取首尾花括号之间内容。
    """
    content = content.strip()
    try:
        return json.loads(content)
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(content[start: end + 1])
        except Exception:
            pass

    raise ValueError(f"无法解析 LLM 输出为 JSON：{content!r}")


def generate_query_for_question(
    client: OpenAI,
    question: Question,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    为单个问题生成多跳查询 JSON，带缓存。
    缓存 key = PROMPT_VERSION + '||' + question.text
    """
    cache_key = f"{PROMPT_VERSION}||{question.text}"
    if cache_key in cache:
        return cache[cache_key]["query"]

    messages = build_messages_for_question(question.text)

    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.0,
        extra_body={"enable_thinking": False},  # 非思考模式
    )

    content = completion.choices[0].message.content
    query = parse_llm_json(content)

    cache[cache_key] = {"query": query, "raw_content": content}
    save_cache(cache, CACHE_PATH)
    return query


# =========================== 多跳问题与评测 ===========================

def build_multihop_questions() -> List[Question]:
    """
    多跳问题 + gold answer。
    gold answer 已根据 graph.json / old_graph.json 计算好。
    """
    return [
        # MH1：target=person，2 跳：2020苏州 & 2021深圳
        Question(
            qid="MH1",
            text="谁在2020年去过苏州，并且在2021年也去过深圳？",
            gold_answer=["小张", "小红"],
        ),
        # MH2：target=person，2 跳：2020北京 & 2022上海
        Question(
            qid="MH2",
            text="谁在2020年去过北京，而且在2022年也去过上海？",
            gold_answer=["小明", "小王"],
        ),
        # MH3：target=person，2 跳：2020武汉 & 2021重庆
        Question(
            qid="MH3",
            text="谁在2020年去过武汉，并且在2021年也去过重庆？",
            gold_answer=["小王"],
        ),
        # MH4：target=geo，2 跳：小张2020 & 小张2021
        Question(
            qid="MH4",
            text="哪些城市既是小张在2020年去过的城市，又是他在2021年去过的城市？",
            gold_answer=["深圳"],
        ),
        # MH5：target=year，2 跳：小李 + 南京 / 苏州
        Question(
            qid="MH5",
            text="小李在哪些年份同时去过南京和苏州？",
            gold_answer=["2021"],
        ),
        # MH6：target=year，2 跳：小张 + 深圳 / 重庆
        Question(
            qid="MH6",
            text="小张在哪些年份既去过深圳又去过重庆？",
            gold_answer=["2021"],
        ),
        # MH7：target=person，3 跳：2020北京 & 2021南京 & 2022深圳
        Question(
            qid="MH7",
            text="谁在2020年去过北京、2021年去过南京，并且在2022年还去过深圳？",
            gold_answer=["小刘", "小王", "小赵"],
        ),
        # MH8：target=geo，2 跳：小红2020 & 小赵2022
        Question(
            qid="MH8",
            text="哪些城市既在2020年被小红去过，也在2022年被小赵去过？",
            gold_answer=["青岛"],
        ),
        # MH9：target=person，2 跳：2020南京 & 2021南京
        Question(
            qid="MH9",
            text="谁在2020年和2021年都去过南京？",
            gold_answer=["小刘", "小李", "小杨", "小红"],
        ),
        # MH10：target=person，2 跳：2020北京 & 2021北京
        Question(
            qid="MH10",
            text="谁在2020年和2021年都去过北京？",
            gold_answer=["小杨", "小王"],
        ),
        # MH11：target=person，2 跳：2021上海 & 2022上海
        Question(
            qid="MH11",
            text="谁在2021年和2022年都去过上海？",
            gold_answer=["小明"],
        ),
        # MH12：target=person，2 跳：2022上海 & 2022深圳
        Question(
            qid="MH12",
            text="谁在2022年既去过上海又去过深圳？",
            gold_answer=["小周", "小明", "小王"],
        ),
        # MH13：target=year，2 跳：小刘 + 北京 / 南京
        Question(
            qid="MH13",
            text="小刘在哪些年份同时去过北京和南京？",
            gold_answer=["2020", "2022", "2024"],
        ),
        # MH14：target=year，2 跳：小王 + 北京 / 武汉
        Question(
            qid="MH14",
            text="小王在哪些年份既去过北京又去过武汉？",
            gold_answer=["2020", "2021"],
        ),
        # MH15：target=geo，2 跳：小刘2020 & 小刘2022
        Question(
            qid="MH15",
            text="哪些城市既是小刘在2020年去过的城市，又是他在2022年去过的城市？",
            gold_answer=["北京", "南京", "西安", "重庆"],
        ),
        # MH16：target=geo，2 跳：小周2022 & 小王2024
        Question(
            qid="MH16",
            text="哪些城市既在2022年被小周去过，也在2024年被小王去过？",
            gold_answer=["上海", "深圳"],
        ),
        # MH17：target=person，3 跳：2020上海 & 2021北京 & 2022广州
        Question(
            qid="MH17",
            text="谁在2020年去过上海、2021年去过北京，并且在2022年还去过广州？",
            gold_answer=["小杨"],
        ),
        # MH18：target=person，3 跳：2021南京 & 2022深圳 & 2023深圳
        Question(
            qid="MH18",
            text="谁在2021年去过南京、2022年去过深圳，并且在2023年还去过深圳？",
            gold_answer=[],
        ),
        # MH19：target=year，2 跳：小张 + 海口 / 青岛
        Question(
            qid="MH19",
            text="小张在哪些年份同时去过海口和青岛？",
            gold_answer=["2024"],
        ),
        # MH20：target=year，2 跳：小赵 + 上海 / 西安
        Question(
            qid="MH20",
            text="小赵在哪些年份既去过上海又去过西安？",
            gold_answer=[],
        ),
        # MH21：target=geo，2 跳：2020小明 & 2020小王
        Question(
            qid="MH21",
            text="哪些城市在2020年既被小明去过，也被小王去过？",
            gold_answer=["北京"],
        ),
        # MH22：target=geo，2 跳：2022小红 & 2022小刘
        Question(
            qid="MH22",
            text="哪些城市在2022年既被小红去过，也被小刘去过？",
            gold_answer=["西安"],
        ),
        # MH23：target=person，2 跳：2022成都 & 2022武汉
        Question(
            qid="MH23",
            text="谁在2022年既去过成都又去过武汉？",
            gold_answer=[],
        ),
        # MH24：target=person，2 跳：2020成都 & 2020重庆
        Question(
            qid="MH24",
            text="谁在2020年既去过成都又去过重庆？",
            gold_answer=["小周"],
        ),
        # MH25：target=year，2 跳：小明 + 北京 / 西安
        Question(
            qid="MH25",
            text="小明在哪些年份既去过北京又去过西安？",
            gold_answer=[],
        ),
        # MH26：target=year，2 跳：小陈 + 苏州 / 西安
        Question(
            qid="MH26",
            text="小陈在哪些年份同时去过苏州和西安？",
            gold_answer=["2024"],
        ),
        # MH27：target=geo，2 跳：小周2022 & 小周2023
        Question(
            qid="MH27",
            text="哪些城市既是小周在2022年去过的城市，又是他在2023年去过的城市？",
            gold_answer=["海口", "深圳"],
        ),
        # MH28：target=person，3 跳：2020北京 & 2021南京 & 2024上海
        Question(
            qid="MH28",
            text="谁在2020年去过北京、2021年去过南京，并且在2024年还去过上海？",
            gold_answer=["小王"],
        ),
        # MH29：target=person，3 跳：2020/2022/2023 广州
        Question(
            qid="MH29",
            text="谁在2020年、2022年和2023年都去过广州？",
            gold_answer=[],
        ),
        # MH30：target=year，2 跳：小刘 & 小王 都去过深圳
        Question(
            qid="MH30",
            text="在哪些年份小刘和小王都去过深圳？",
            gold_answer=["2022", "2024"],
        ),
    ]


def run_experiment() -> None:
    new_db = NewGraphDB(NEW_GRAPH_PATH)
    old_db = OldGraphDB(OLD_GRAPH_PATH)
    client = make_qwen_client()
    cache = load_cache(CACHE_PATH)
    questions = build_multihop_questions()

    total = len(questions)
    correct_new = 0
    correct_old = 0

    for q in questions:
        print("=" * 80)
        print(f"[{q.qid}] {q.text}")

        try:
            query = generate_query_for_question(client, q, cache)
        except Exception as e:
            print(f"生成查询 JSON 失败：{e}")
            continue

        print("Generated Multi-hop Query JSON:")
        print(json.dumps(query, ensure_ascii=False, indent=2))

        # 新库多跳
        try:
            ans_new = new_db.run_multihop_query(query)
        except Exception as e:
            print(f"新库执行失败：{e}")
            ans_new = []

        # 旧库多跳
        try:
            ans_old = old_db.run_multihop_query(query)
        except Exception as e:
            print(f"旧库执行失败：{e}")
            ans_old = []

        print("MMDB Answer   :", ans_new)
        print("PGDB Answer   :", ans_old)
        print("Gold Answer   :", q.gold_answer)

        if sorted(ans_new) == sorted(q.gold_answer):
            correct_new += 1
            print("MMDB：Correct")
        else:
            print("MMDB：False")

        if sorted(ans_old) == sorted(q.gold_answer):
            correct_old += 1
            print("PGDB：Correct")
        else:
            print("PGDB：False")

    print("=" * 80)
    print(f"MMDB Multi-hop Accuracy = {correct_new}/{total} = {correct_new/total*100:.1f}%")
    print(f"PGDB Multi-hop Accuracy = {correct_old}/{total} = {correct_old/total*100:.1f}%")


if __name__ == "__main__":
    run_experiment()
