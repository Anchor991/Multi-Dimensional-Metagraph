"""
qa_pipeline_newdb_deepseek.py

流程：
  自然语言问题 -> DeepSeek LLM 生成查询 JSON (op + 参数)
            -> 在 graph.json (hypercube) 上执行
            -> 得到答案，与 gold 对比，计算 accuracy
"""

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI  # deepseek 使用 openai 兼容 SDK


# ========== 0. 配置区域 ==========

DEEPSEEK_MODEL = "deepseek-chat"          # 指定的模型
GRAPH_JSON_PATH = "graph.json"

# 创建 DeepSeek 客户端（注意 base_url）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),   # 建议用环境变量保存 key
    base_url="https://api.deepseek.com",
)


# ========== 1. 数据结构 & 载入 ==========

@dataclass
class Question:
    qid: str
    text: str
    gold_answer: List[str]  # 标准答案


def load_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


graph = load_graph(GRAPH_JSON_PATH)

# 索引
person_id2name = {p["id"]: p["name"] for p in graph["dimensions"]["person"]}
person_name2id = {v: k for k, v in person_id2name.items()}

time_id2value = {t["id"]: t["value"] for t in graph["dimensions"]["time"]}
time_value2id = {v: k for k, v in time_id2value.items()}

geo_id2name = {g["id"]: g["name"] for g in graph["dimensions"]["geo"]}
geo_name2id = {v: k for k, v in geo_id2name.items()}


# ========== 2. 定义一批测试问题 ==========

QUESTIONS: List[Question] = [
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
    Question(
        qid="Q26",
        text="小红在2020到2022年之间去过哪些城市？",
        gold_answer=["南京", "广州", "成都", "杭州", "武汉", "深圳", "苏州", "西安", "重庆", "青岛"],
    ),
    Question(
        qid="Q27",
        text="小张在2021到2023年去过哪些地方？",
        gold_answer=["北京", "南京", "杭州", "武汉", "深圳", "重庆"],
    ),
    Question(
        qid="Q28",
        text="小李在2020到2024年间去过哪些城市？",
        gold_answer=["上海", "北京", "南京", "天津", "广州", "成都", "海口", "苏州", "重庆", "青岛"],
    ),
    Question(
        qid="Q29",
        text="小杨在2020到2022年都去过哪里？",
        gold_answer=["上海", "北京", "南京", "天津", "广州", "杭州", "武汉", "海口", "苏州", "青岛"],
    ),
    Question(
        qid="Q30",
        text="小刘在2020到2023年之间去过哪些城市？",
        gold_answer=["北京", "南京", "天津", "广州", "成都", "杭州", "武汉", "海口", "深圳", "西安", "重庆", "青岛"],
    ),
    Question(
        qid="Q31",
        text="小陈在2020到2024年去过哪些地方？",
        gold_answer=["南京", "天津", "广州", "杭州", "深圳", "苏州", "西安"],
    ),
    Question(
        qid="Q32",
        text="小明在2020到2023年期间去过哪些城市？",
        gold_answer=["上海", "北京", "南京", "天津", "成都", "武汉", "深圳", "苏州", "西安", "重庆"],
    ),
    Question(
        qid="Q33",
        text="小赵在2020到2024年都去过哪些城市？",
        gold_answer=["上海", "北京", "南京", "天津", "广州", "成都", "杭州", "武汉", "深圳", "西安", "重庆", "青岛"],
    ),
    Question(
        qid="Q34",
        text="在2020到2022年之间，去过北京的人有哪些？",
        gold_answer=["小刘", "小周", "小张", "小明", "小杨", "小王", "小赵"],
    ),
    Question(
        qid="Q35",
        text="从2021到2024年，哪些人去过上海？",
        gold_answer=["小周", "小明", "小李", "小王", "小红"],
    ),
    Question(
        qid="Q36",
        text="在2020到2023年间，去过苏州的是谁？",
        gold_answer=["小张", "小明", "小李", "小杨", "小红"],
    ),
    Question(
        qid="Q37",
        text="从2020到2024年，哪些人去过深圳？",
        gold_answer=["小刘", "小周", "小张", "小明", "小杨", "小王", "小红", "小赵", "小陈"],
    ),
    Question(
        qid="Q38",
        text="在2020到2024年之间，去过西安的人都有谁？",
        gold_answer=["小刘", "小周", "小明", "小红", "小赵", "小陈"],
    ),
    Question(
        qid="Q39",
        text="2020到2023年去过重庆的人有哪些？",
        gold_answer=["小刘", "小周", "小张", "小明", "小李", "小王", "小红", "小赵"],
    ),
    Question(
        qid="Q40",
        text="在2020到2024年，去过南京的是哪些人？",
        gold_answer=["小刘", "小周", "小张", "小明", "小李", "小杨", "小王", "小红", "小赵", "小陈"],
    ),
    Question(
        qid="Q41",
        text="小明是在哪些年份去过上海？",
        gold_answer=["2021", "2022"],
    ),
    Question(
        qid="Q42",
        text="小红是在哪些年份去了青岛？",
        gold_answer=["2020", "2022", "2023"],
    ),
    Question(
        qid="Q43",
        text="小张在哪些年份去过海口？",
        gold_answer=["2020", "2024"],
    ),
    Question(
        qid="Q44",
        text="小刘是什么时候去过北京？",
        gold_answer=["2020", "2022", "2024"],
    ),
    Question(
        qid="Q45",
        text="小李在哪些年份去过广州？",
        gold_answer=["2022", "2023"],
    ),
    Question(
        qid="Q46",
        text="小红一共去了多少次旅行？",
        gold_answer=["26"],
    ),
    Question(
        qid="Q47",
        text="小张总共出行了多少次？",
        gold_answer=["21"],
    ),
    Question(
        qid="Q48",
        text="小李一共记录了多少次出行？",
        gold_answer=["18"],
    ),
    Question(
        qid="Q49",
        text="小明总共旅行了多少次？",
        gold_answer=["18"],
    ),
    Question(
        qid="Q50",
        text="小王一共有多少条出行记录？",
        gold_answer=["16"],
    ),
    
    # TODO: 继续添加更多问题……
]


# ========== 3. DeepSeek: 问题 -> 查询 JSON ==========

def build_system_prompt() -> str:
    """
    告诉 LLM：我们有一个三维超立方体数据库，
    只允许输出特定格式的 JSON 查询。
    """
    return (
        "你是一个查询生成器。我们有一个三维超立方体数据库，"
        "维度为 Person(人物)、Time(年份)、Geo(城市)。"
        "每条事实 fact 有 person, time, geo 三个字段。\n\n"
        "请根据用户的问题，生成一个 JSON 对象，不要输出其他文字。"
        "JSON 格式必须满足：\n"
        "{\n"
        '  \"op\": string,                # 操作类型之一：\n'
        '                                #   \"cities_by_person_year\" 或\n'
        '                                #   \"persons_by_city_year\"\n'
        '  ...  其他所需参数 ...\n'
        "}\n\n"
        "各操作含义：\n"
        "- op = \"cities_by_person_year\": 参数为 person_name(人物名, 字符串), year(整数年份)，\n"
        "  表示查询“某人在某年去过的所有城市名”。\n"
        "- op = \"persons_by_city_year\": 参数为 city_name(城市名, 字符串), year(整数年份)，\n"
        "  表示查询“某年去某城市的人名列表”。\n\n"
        "严格要求：\n"
        "1. 只能输出一个 JSON 对象。\n"
        "2. 不要加注释、不要加多余文字。\n"
        "3. JSON 键名必须是上面指定的英文。\n"
    )


def llm_generate_query(question: str) -> Dict[str, Any]:
    """
    调用 DeepSeek，让模型根据自然语言问题生成查询 JSON。
    返回一个 Python dict。
    """
    system_prompt = build_system_prompt()

    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        # 要求模型输出合法 JSON
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    query = json.loads(content)
    return query


# ========== 4. 在 hypercube JSON 上执行查询 ==========

def execute_query_on_graph(query: Dict[str, Any]) -> List[str]:
    """
    根据我们定义好的 op，在 graph.json 上执行查询，
    返回字符串列表作为“答案”。

    为简单起见，这里返回的答案类型统一用 List[str]：
      - cities_by_person_year：返回城市名列表
      - persons_by_city_year：返回人物名列表
    """
    op = query.get("op")

    if op == "cities_by_person_year":
        person_name = query.get("person_name")
        year = str(query.get("year"))
        return cities_by_person_year(person_name, year)

    elif op == "persons_by_city_year":
        city_name = query.get("city_name")
        year = str(query.get("year"))
        return persons_by_city_year(city_name, year)

    else:
        # 未知 op，返回空
        return []


def cities_by_person_year(person_name: str, year: str) -> List[str]:
    """给定人物名 + 年份，返回去过的城市名列表（去重）。"""
    if person_name not in person_name2id:
        return []
    pid = person_name2id[person_name]
    if year not in time_value2id:
        return []
    tid = time_value2id[year]

    results = set()
    for fact in graph["facts"]:
        if fact["person"] == pid and fact["time"] == tid:
            gid = fact["geo"]
            results.add(geo_id2name[gid])

    return sorted(results)


def persons_by_city_year(city_name: str, year: str) -> List[str]:
    """给定城市名 + 年份，返回去过的人的名字列表（去重）。"""
    if city_name not in geo_name2id:
        return []
    gid = geo_name2id[city_name]
    if year not in time_value2id:
        return []
    tid = time_value2id[year]

    results = set()
    for fact in graph["facts"]:
        if fact["geo"] == gid and fact["time"] == tid:
            pid = fact["person"]
            results.add(person_id2name[pid])

    return sorted(results)


# ========== 5. 计算 accuracy ==========

def is_correct(pred: List[str], gold: List[str]) -> bool:
    """集合是否一致（忽略顺序）"""
    return set(pred) == set(gold)


def run_experiment():
    if not client.api_key:
        print("请先在环境变量中设置 DEEPSEEK_API_KEY!")
        return

    correct = 0
    total = len(QUESTIONS)

    for q in QUESTIONS:
        print("=" * 60)
        print(f"{q.qid}: {q.text}")
        print("Gold:", q.gold_answer)

        # 1) DeepSeek 生成查询 JSON
        query = llm_generate_query(q.text)
        print("LLM 生成的查询 JSON:", query)

        # 2) 在 hypercube JSON 上执行
        pred = execute_query_on_graph(query)
        print("系统回答:", pred)

        # 3) 判定是否正确
        ok = is_correct(pred, q.gold_answer)
        print("Correct?", ok)
        if ok:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print("=" * 60)
    print(
        f"Accuracy (new DB / hypercube JSON + DeepSeek) = "
        f"{acc * 100:.1f}% ({correct} / {total})"
    )


if __name__ == "__main__":
    run_experiment()
