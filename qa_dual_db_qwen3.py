"""
qa_dual_db_qwen3.py

使用 Qwen3 (OpenAI 兼容 API) 把中文自然语言问题转换为结构化 JSON 查询，
在「新库 graph.json」和「旧库 old_graph.json」上分别执行，并对比准确率。

特性：
- 使用 Qwen3 非思考模式：extra_body={"enable_thinking": False}
- temperature = 0.0，保证输出稳定
- few-shot 提示，固定 JSON 结构
- LLM 输出缓存到 qwen3_cache.json，避免重复调用
"""

import json
from dataclasses import dataclass
import os
from typing import Any, Dict, List
from openai import OpenAI


# =========================== 固定配置（按需修改） ===========================

# 把这里改成你的真实 DashScope 国内版 API Key
DASHSCOPE_API_KEY = "#"

# 中国的 OpenAI 兼容接口
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 使用的 Qwen3 模型名
DEFAULT_MODEL = "qwen3-8b"

# 图数据库文件路径
NEW_GRAPH_PATH = "graph.json"
OLD_GRAPH_PATH = "old_graph.json"

# LLM 缓存文件
CACHE_PATH = "qwen3_cache.json"

# 修改系统提示或 few-shot 时，把版本号改一下，避免旧缓存干扰
PROMPT_VERSION = "qwen3-dual-v1"


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

    def run_query(self, query: Dict[str, Any]) -> List[str]:
        """
        执行受限查询 JSON：

        {
          "target": "person" | "geo" | "year",
          "filters": {
            "person": ["小明"],
            "year": ["2020"],
            "geo": ["苏州"]
          }
        }

        filters 为空则不限制；多个条件为交集过滤。
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

    def run_query(self, query: Dict[str, Any]) -> List[str]:
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


# =========================== LLM 客户端 & 缓存 ===========================

def make_qwen_client() -> OpenAI:
    """
    创建 OpenAI 兼容的 Qwen3 客户端。
    所有配置（API Key / base_url / model）都在文件顶部写死。
    """
    api_key = DASHSCOPE_API_KEY
    if not api_key or api_key.startswith("<YOUR_"):
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
    import os
    os.replace(tmp, path)


# =========================== Prompt & few-shot ===========================

def build_messages_for_question(question_text: str) -> List[Dict[str, str]]:
    """
    构造 system prompt + few-shot + 当前 user 问题。
    """

    system_prompt = f"""
你是一个只负责“生成查询 JSON”的助手，用于在旅游行程数据库上检索答案。

数据库有三类实体：
- person：人名（如“小红、小张、小明”等）；
- year：年份字符串（如 "2020"、"2021"）；
- geo：城市名（如“苏州、深圳、南京、北京”等）。

给定一个中文问题，你只需要生成一个 JSON 对象，不要直接回答问题。
这个 JSON 将在两套数据库实现上执行，因此格式必须非常严格。

JSON 结构（不要输出注释）：

{{
  "target": "person" | "geo" | "year",
  "filters": {{
    "person": ["小明"],
    "year": ["2020"],
    "geo": ["苏州"]
  }}
}}

含义说明：
- target 表示最终想要的实体类别：
  - 问“谁”的问题，target = "person"；
  - 问“哪些地方 / 哪些城市”的问题，target = "geo"；
  - 问“哪几年”的问题，target = "year"。
- filters 含义：
  - "person"：限定涉及的人员（字符串数组）；
  - "year"：限定年份（字符串数组，例如 ["2020"]）；
  - "geo"：限定城市名（字符串数组）。
- filters 中，每个字段都可以省略；存在多个字段时，表示交集过滤。

严格要求：
1. 只输出一个 JSON 对象，不要输出任何解释或额外文字。
2. JSON 中所有键和值都使用半角双引号，必须是合法 JSON。
3. 不要生成 target 和 filters 之外的其他字段。
4. 所有名字和城市名都保持问题中的中文原文，不要翻译或改写。
5. 你的输出将直接被 json.loads() 解析，请避免多余符号。

当前提示版本：{PROMPT_VERSION}
""".strip()

    few_shot: List[Dict[str, str]] = []

    # 示例 1：问城市
    few_shot += [
        {
            "role": "user",
            "content": "小明在2020年去了哪些地方？",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "target": "geo",
                    "filters": {
                        "person": ["小明"],
                        "year": ["2020"],
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]

    # 示例 2：问人
    few_shot += [
        {
            "role": "user",
            "content": "2020年去了苏州的是哪些人？",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "target": "person",
                    "filters": {
                        "geo": ["苏州"],
                        "year": ["2020"],
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]

    # 示例 3：问人（不同城市年份）
    few_shot += [
        {
            "role": "user",
            "content": "2021年谁去过深圳？",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "target": "person",
                    "filters": {
                        "geo": ["深圳"],
                        "year": ["2021"],
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]

    # 示例 4：问年份
    few_shot += [
        {
            "role": "user",
            "content": "小张在哪几年去过苏州？",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "target": "year",
                    "filters": {
                        "person": ["小张"],
                        "geo": ["苏州"],
                    },
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
            return json.loads(content[start : end + 1])
        except Exception:
            pass

    raise ValueError(f"无法解析 LLM 输出为 JSON：{content!r}")


def generate_query_for_question(
    client: OpenAI,
    question: Question,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    为单个问题生成查询 JSON，带缓存。
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


# =========================== 问题与评测 ===========================

def build_default_questions() -> List[Question]:
    """
    示例题目 + gold answer。
    gold answer 基于 graph.json / old_graph.json 中的事实预先计算好。
    """
    return [
    Question(
        qid="Q1",
        text="小红在2020年去了哪些城市？",
        gold_answer=["南京", "成都", "苏州", "青岛"],
    ),
    Question(
        qid="Q2",
        text="小红在2021年都去过哪里？",
        gold_answer=["南京", "杭州", "深圳", "重庆"],
    ),
    Question(
        qid="Q3",
        text="小张在2023年去了哪些地方？",
        gold_answer=["南京", "杭州", "武汉", "深圳"],
    ),
    Question(
        qid="Q4",
        text="小李在2024年去过哪些城市？",
        gold_answer=["上海", "北京", "天津", "成都", "海口", "苏州"],
    ),
    Question(
        qid="Q5",
        text="小杨在2020年都去了哪里旅行？",
        gold_answer=["上海", "北京", "南京", "杭州", "海口", "青岛"],
    ),
    Question(
        qid="Q6",
        text="小刘在2022年去了哪些城市？",
        gold_answer=["北京", "南京", "天津", "杭州", "深圳", "西安", "重庆"],
    ),
    Question(
        qid="Q7",
        text="小陈在2021年去了哪些地方？",
        gold_answer=["南京", "天津", "杭州", "深圳", "西安"],
    ),
    Question(
        qid="Q8",
        text="小明在2022年去过哪些城市？",
        gold_answer=["上海", "天津", "武汉", "深圳", "西安", "重庆"],
    ),
    Question(
        qid="Q9",
        text="小赵在2021年去过哪些城市？",
        gold_answer=["南京", "杭州", "武汉"],
    ),
    Question(
        qid="Q10",
        text="小周在2024年去了哪些城市？",
        gold_answer=["上海", "北京", "南京", "西安"],
    ),
    Question(
        qid="Q11",
        text="小王在2020年都去了哪些地方？",
        gold_answer=["北京", "武汉", "重庆"],
    ),
    Question(
        qid="Q12",
        text="小王在2021年去过哪些城市？",
        gold_answer=["北京", "南京", "成都", "武汉", "重庆"],
    ),
    Question(
        qid="Q13",
        text="小周在2022年去了哪些城市？",
        gold_answer=["上海", "北京", "海口", "深圳", "重庆"],
    ),
    Question(
        qid="Q14",
        text="小张在2020年去过哪些城市？",
        gold_answer=["上海", "海口", "深圳", "苏州"],
    ),
    Question(
        qid="Q15",
        text="小明在2020年去了哪些地方？",
        gold_answer=["北京", "南京", "天津", "苏州"],
    ),
    Question(
        qid="Q16",
        text="2020年去了苏州的是哪些人？",
        gold_answer=["小张", "小明", "小红"],
    ),
    Question(
        qid="Q17",
        text="2021年谁去过深圳？",
        gold_answer=["小张", "小红", "小陈"],
    ),
    Question(
        qid="Q18",
        text="2022年去过杭州的人都有谁？",
        gold_answer=["小刘"],
    ),
    Question(
        qid="Q19",
        text="2023年哪些人去了南京？",
        gold_answer=["小张", "小杨", "小陈"],
    ),
    Question(
        qid="Q20",
        text="2024年去过北京的是谁？",
        gold_answer=["小刘", "小周", "小李", "小杨"],
    ),
    Question(
        qid="Q21",
        text="2020年哪些人去了上海？",
        gold_answer=["小张", "小杨", "小赵"],
    ),
    Question(
        qid="Q22",
        text="2021年是谁去了天津？",
        gold_answer=["小李", "小陈"],
    ),
    Question(
        qid="Q23",
        text="2023年去过深圳的人有哪些？",
        gold_answer=["小周", "小张", "小明", "小杨", "小陈"],
    ),
    Question(
        qid="Q24",
        text="2022年有哪些人去了重庆？",
        gold_answer=["小刘", "小周", "小明", "小赵"],
    ),
    Question(
        qid="Q25",
        text="2021年去过武汉的是哪些人？",
        gold_answer=["小杨", "小王", "小赵"],
    ),
        
    ]


def run_experiment() -> None:
    new_db = NewGraphDB(NEW_GRAPH_PATH)
    old_db = OldGraphDB(OLD_GRAPH_PATH)
    client = make_qwen_client()
    cache = load_cache(CACHE_PATH)
    questions = build_default_questions()

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

        print("Generated Query JSON：")
        print(json.dumps(query, ensure_ascii=False, indent=2))

        # 新库
        try:
            ans_new = new_db.run_query(query)
        except Exception as e:
            print(f"新库执行失败：{e}")
            ans_new = []

        # 旧库
        try:
            ans_old = old_db.run_query(query)
        except Exception as e:
            print(f"旧库执行失败：{e}")
            ans_old = []

        print("MMDB Answer   :", ans_new)
        print("PGDB Answer   :", ans_old)
        print("Gold Answer:", q.gold_answer)

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
    print(f"MMDB Accuracy = {correct_new}/{total} = {correct_new/total*100:.1f}%")
    print(f"PGDB Accuracy = {correct_old}/{total} = {correct_old/total*100:.1f}%")


if __name__ == "__main__":
    run_experiment()
