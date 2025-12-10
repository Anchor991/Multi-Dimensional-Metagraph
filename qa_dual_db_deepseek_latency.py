"""
qa_dual_db_deepseek_latency.py

For the same list of natural-language questions:
  1) DeepSeek generates ONE query JSON (op + params)
  2) Execute on MMDB (graph.json, 3D hypercube)
  3) Execute on PGDB (old_graph.json, property graph baseline)
  4) Compare to gold_answer -> accuracy
  5) Measure latency:
       - LLM latency (ms)
       - MMDB execution latency (ms)
       - PGDB execution latency (ms)
     and report per-question + summary (mean / p50 / p95)

Notes:
- This script keeps the QUESTIONS + gold_answer as defined in qa_dual_db_deepseek.py.
- To run: python qa_dual_db_deepseek_latency.py
"""

import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from openai import OpenAI  # DeepSeek uses OpenAI-compatible SDK


# ========== 0. Config ==========

DEEPSEEK_MODEL = "deepseek-chat"
NEWDB_PATH = "graph.json"
OLDDB_PATH = "old_graph.json"

# If you want to reduce variance, you can run each DB query multiple times and take median.
# Keep 1 for simplest reporting.
REPEAT_DB_EXEC = 1

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ========== 1. Load MMDB and PGDB ==========

def load_newdb(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_olddb(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


newdb = load_newdb(NEWDB_PATH)
olddb = load_olddb(OLDDB_PATH)

# MMDB indices
person_id2name_new = {p["id"]: p["name"] for p in newdb["dimensions"]["person"]}
person_name2id_new = {v: k for k, v in person_id2name_new.items()}

time_id2value_new = {t["id"]: t["value"] for t in newdb["dimensions"]["time"]}
time_value2id_new = {v: k for k, v in time_id2value_new.items()}

geo_id2name_new = {g["id"]: g["name"] for g in newdb["dimensions"]["geo"]}
geo_name2id_new = {v: k for k, v in geo_id2name_new.items()}

facts_new = newdb["facts"]

# PGDB indices
person_nodes_old = {}
geo_nodes_old = {}

for n in olddb["nodes"]:
    if n.get("type") == "Person":
        person_nodes_old[n["name"]] = n["id"]
    elif n.get("type") == "Geo":
        geo_nodes_old[n["name"]] = n["id"]

edges_old = olddb["edges"]  # list of {from, to, year, fact_id}

# For faster lookup on PGDB, pre-index nodes by id
old_node_by_id = {n["id"]: n for n in olddb["nodes"]}


# ========== 2. Question schema ==========
@dataclass
class Question:
    qid: str
    text: str
    gold_answer: List[str]


# IMPORTANT:
# Keep the same QUESTIONS as in qa_dual_db_deepseek.py.
# If you update that file, copy/paste the QUESTIONS block here as well.
QUESTIONS: List[Question] = [
    # --- Paste your questions here (already filled in your current file) ---
    Question(qid="Q1", text="小红在2020年去了哪些城市？", gold_answer=["南京", "成都", "苏州", "青岛"]),
    Question(qid="Q2", text="小红在2021年都去过哪里？", gold_answer=["南京", "杭州", "深圳", "重庆"]),
    Question(qid="Q3", text="小张在2023年去了哪些地方？", gold_answer=["南京", "杭州", "武汉", "深圳"]),
    Question(qid="Q4", text="小李在2024年去过哪些城市？", gold_answer=["上海", "北京", "天津", "成都", "海口", "苏州"]),
    Question(qid="Q5", text="小杨在2020年都去了哪里旅行？", gold_answer=["上海", "北京", "南京", "杭州", "海口", "青岛"]),
    Question(qid="Q6", text="小刘在2022年去了哪些城市？", gold_answer=["北京", "南京", "天津", "杭州", "深圳", "西安", "重庆"]),
    Question(qid="Q7", text="小陈在2021年去了哪些地方？", gold_answer=["南京", "天津", "杭州", "深圳", "西安"]),
    Question(qid="Q8", text="小明在2022年去过哪些城市？", gold_answer=["上海", "天津", "武汉", "深圳", "西安", "重庆"]),
    Question(qid="Q9", text="小赵在2021年去过哪些城市？", gold_answer=["南京", "杭州", "武汉"]),
    Question(qid="Q10", text="小周在2024年去了哪些城市？", gold_answer=["上海", "北京", "南京", "西安"]),
    Question(qid="Q11", text="小王在2020年都去了哪些地方？", gold_answer=["北京", "武汉", "重庆"]),
    Question(qid="Q12", text="小王在2021年去过哪些城市？", gold_answer=["北京", "南京", "成都", "武汉", "重庆"]),
    Question(qid="Q13", text="小周在2022年去了哪些城市？", gold_answer=["上海", "北京", "海口", "深圳", "重庆"]),
    Question(qid="Q14", text="小张在2020年去过哪些城市？", gold_answer=["上海", "海口", "深圳", "苏州"]),
    Question(qid="Q15", text="小明在2020年去了哪些地方？", gold_answer=["北京", "南京", "天津", "苏州"]),
    Question(qid="Q16", text="2020年去了苏州的是哪些人？", gold_answer=["小张", "小明", "小红"]),
    Question(qid="Q17", text="2021年谁去过深圳？", gold_answer=["小张", "小红", "小陈"]),
    Question(qid="Q18", text="2022年去过杭州的人都有谁？", gold_answer=["小刘"]),
    Question(qid="Q19", text="2023年哪些人去了南京？", gold_answer=["小张", "小杨", "小陈"]),
    Question(qid="Q20", text="2024年去过北京的是谁？", gold_answer=["小刘", "小周", "小李", "小杨"]),
    Question(qid="Q21", text="2020年哪些人去了上海？", gold_answer=["小张", "小杨", "小赵"]),
    Question(qid="Q22", text="2021年是谁去了天津？", gold_answer=["小李", "小陈"]),
    Question(qid="Q23", text="2023年去过深圳的人有哪些？", gold_answer=["小周", "小张", "小明", "小杨", "小陈"]),
    Question(qid="Q24", text="2022年有哪些人去了重庆？", gold_answer=["小刘", "小周", "小明", "小赵"]),
    Question(qid="Q25", text="2021年去过武汉的是哪些人？", gold_answer=["小杨", "小王", "小赵"]),
]


# ========== 3. DeepSeek: NL -> query JSON ==========
def build_system_prompt() -> str:
    return (
        "你是一个查询生成器。我们有一个三维超立方体数据库，"
        "维度为 Person(人物)、Time(年份)、Geo(城市)。"
        "每条事实 fact 有 person, time, geo 三个字段。\n\n"
        "请根据用户的问题，生成一个 JSON 对象，不要输出其他文字。"
        "JSON 格式必须满足：\n"
        "{\n"
        "  \"op\": string,  # 操作类型之一：\n"
        "                  #   \"cities_by_person_year\"\n"
        "                  #   \"persons_by_city_year\"\n"
        "                  #   \"cities_by_person_year_range\"\n"
        "                  #   \"persons_by_city_year_range\"\n"
        "                  #   \"years_by_person_city\"\n"
        "                  #   \"count_trips_by_person\"\n"
        "  ...  其他所需参数 ...\n"
        "}\n\n"
        "各操作含义和参数：\n"
        "- op = \"cities_by_person_year\":\n"
        "    参数: person_name(str), year(int)\n"
        "    含义: 查询“某人在某年去过的所有城市名”。\n"
        "- op = \"persons_by_city_year\":\n"
        "    参数: city_name(str), year(int)\n"
        "    含义: 查询“某年去某城市的人名列表”。\n"
        "- op = \"cities_by_person_year_range\":\n"
        "    参数: person_name(str), year_start(int), year_end(int)\n"
        "    含义: 查询“某人在某个年份区间内去过的所有城市名”。\n"
        "- op = \"persons_by_city_year_range\":\n"
        "    参数: city_name(str), year_start(int), year_end(int)\n"
        "    含义: 查询“在某城市、某年份区间内去过的人名列表”。\n"
        "- op = \"years_by_person_city\":\n"
        "    参数: person_name(str), city_name(str)\n"
        "    含义: 查询“某人在哪些年份去过某城市”（返回年份列表）。\n"
        "- op = \"count_trips_by_person\":\n"
        "    参数: person_name(str)\n"
        "    含义: 查询“某人一共旅行多少次”（返回次数字符串，如 \"3\"）。\n\n"
        "严格要求：\n"
        "1. 只能输出一个 JSON 对象。\n"
        "2. 不要加注释、不要加多余文字。\n"
        "3. JSON 键名必须是上面指定的英文。\n"
    )


def llm_generate_query(question: str) -> Tuple[Dict[str, Any], float]:
    """Return (query_json, llm_latency_ms)."""
    system_prompt = build_system_prompt()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )
    llm_ms = (time.perf_counter() - t0) * 1000.0
    content = resp.choices[0].message.content
    return json.loads(content), llm_ms


# ========== 4. MMDB query implementations ==========
def cities_by_person_year_new(person_name: str, year: str) -> List[str]:
    if person_name not in person_name2id_new:
        return []
    pid = person_name2id_new[person_name]
    if year not in time_value2id_new:
        return []
    tid = time_value2id_new[year]

    results = set()
    for fact in facts_new:
        if fact["person"] == pid and fact["time"] == tid:
            results.add(geo_id2name_new[fact["geo"]])
    return sorted(results)


def persons_by_city_year_new(city_name: str, year: str) -> List[str]:
    if city_name not in geo_name2id_new:
        return []
    gid = geo_name2id_new[city_name]
    if year not in time_value2id_new:
        return []
    tid = time_value2id_new[year]

    results = set()
    for fact in facts_new:
        if fact["geo"] == gid and fact["time"] == tid:
            results.add(person_id2name_new[fact["person"]])
    return sorted(results)


def cities_by_person_year_range_new(person_name: str, ys: int, ye: int) -> List[str]:
    if person_name not in person_name2id_new:
        return []
    pid = person_name2id_new[person_name]
    years = [str(y) for y in range(ys, ye + 1)]
    tids = {time_value2id_new[y] for y in years if y in time_value2id_new}

    results = set()
    for fact in facts_new:
        if fact["person"] == pid and fact["time"] in tids:
            results.add(geo_id2name_new[fact["geo"]])
    return sorted(results)


def persons_by_city_year_range_new(city_name: str, ys: int, ye: int) -> List[str]:
    if city_name not in geo_name2id_new:
        return []
    gid = geo_name2id_new[city_name]
    years = [str(y) for y in range(ys, ye + 1)]
    tids = {time_value2id_new[y] for y in years if y in time_value2id_new}

    results = set()
    for fact in facts_new:
        if fact["geo"] == gid and fact["time"] in tids:
            results.add(person_id2name_new[fact["person"]])
    return sorted(results)


def years_by_person_city_new(person_name: str, city_name: str) -> List[str]:
    if person_name not in person_name2id_new or city_name not in geo_name2id_new:
        return []
    pid = person_name2id_new[person_name]
    gid = geo_name2id_new[city_name]

    tids = set()
    for fact in facts_new:
        if fact["person"] == pid and fact["geo"] == gid:
            tids.add(fact["time"])
    years = {time_id2value_new[tid] for tid in tids}
    return sorted(years)


def count_trips_by_person_new(person_name: str) -> List[str]:
    if person_name not in person_name2id_new:
        return ["0"]
    pid = person_name2id_new[person_name]
    cnt = sum(1 for f in facts_new if f["person"] == pid)
    return [str(cnt)]


def execute_query_newdb(query: Dict[str, Any]) -> List[str]:
    op = query.get("op")
    if op == "cities_by_person_year":
        return cities_by_person_year_new(query.get("person_name"), str(query.get("year")))
    if op == "persons_by_city_year":
        return persons_by_city_year_new(query.get("city_name"), str(query.get("year")))
    if op == "cities_by_person_year_range":
        return cities_by_person_year_range_new(query.get("person_name"),
                                               int(query.get("year_start")),
                                               int(query.get("year_end")))
    if op == "persons_by_city_year_range":
        return persons_by_city_year_range_new(query.get("city_name"),
                                              int(query.get("year_start")),
                                              int(query.get("year_end")))
    if op == "years_by_person_city":
        return years_by_person_city_new(query.get("person_name"), query.get("city_name"))
    if op == "count_trips_by_person":
        return count_trips_by_person_new(query.get("person_name"))
    return []


# ========== 5. PGDB query implementations ==========
def cities_by_person_year_old(person_name: str, year: str) -> List[str]:
    if person_name not in person_nodes_old:
        return []
    pid = person_nodes_old[person_name]
    results = set()
    for e in edges_old:
        if e["from"] == pid and e["year"] == year:
            n = old_node_by_id.get(e["to"])
            if n and n.get("type") == "Geo":
                results.add(n["name"])
    return sorted(results)


def persons_by_city_year_old(city_name: str, year: str) -> List[str]:
    if city_name not in geo_nodes_old:
        return []
    gid = geo_nodes_old[city_name]
    results = set()
    for e in edges_old:
        if e["to"] == gid and e["year"] == year:
            n = old_node_by_id.get(e["from"])
            if n and n.get("type") == "Person":
                results.add(n["name"])
    return sorted(results)


def cities_by_person_year_range_old(person_name: str, ys: int, ye: int) -> List[str]:
    if person_name not in person_nodes_old:
        return []
    pid = person_nodes_old[person_name]
    years = {str(y) for y in range(ys, ye + 1)}
    results = set()
    for e in edges_old:
        if e["from"] == pid and e["year"] in years:
            n = old_node_by_id.get(e["to"])
            if n and n.get("type") == "Geo":
                results.add(n["name"])
    return sorted(results)


def persons_by_city_year_range_old(city_name: str, ys: int, ye: int) -> List[str]:
    if city_name not in geo_nodes_old:
        return []
    gid = geo_nodes_old[city_name]
    years = {str(y) for y in range(ys, ye + 1)}
    results = set()
    for e in edges_old:
        if e["to"] == gid and e["year"] in years:
            n = old_node_by_id.get(e["from"])
            if n and n.get("type") == "Person":
                results.add(n["name"])
    return sorted(results)


def years_by_person_city_old(person_name: str, city_name: str) -> List[str]:
    if person_name not in person_nodes_old or city_name not in geo_nodes_old:
        return []
    pid = person_nodes_old[person_name]
    gid = geo_nodes_old[city_name]
    years = set()
    for e in edges_old:
        if e["from"] == pid and e["to"] == gid:
            years.add(e["year"])
    return sorted(years)


def count_trips_by_person_old(person_name: str) -> List[str]:
    if person_name not in person_nodes_old:
        return ["0"]
    pid = person_nodes_old[person_name]
    cnt = sum(1 for e in edges_old if e["from"] == pid)
    return [str(cnt)]


def execute_query_olddb(query: Dict[str, Any]) -> List[str]:
    op = query.get("op")
    if op == "cities_by_person_year":
        return cities_by_person_year_old(query.get("person_name"), str(query.get("year")))
    if op == "persons_by_city_year":
        return persons_by_city_year_old(query.get("city_name"), str(query.get("year")))
    if op == "cities_by_person_year_range":
        return cities_by_person_year_range_old(query.get("person_name"),
                                               int(query.get("year_start")),
                                               int(query.get("year_end")))
    if op == "persons_by_city_year_range":
        return persons_by_city_year_range_old(query.get("city_name"),
                                              int(query.get("year_start")),
                                              int(query.get("year_end")))
    if op == "years_by_person_city":
        return years_by_person_city_old(query.get("person_name"), query.get("city_name"))
    if op == "count_trips_by_person":
        return count_trips_by_person_old(query.get("person_name"))
    return []


# ========== 6. Metrics helpers ==========
def is_correct(pred: List[str], gold: List[str]) -> bool:
    return set(pred) == set(gold)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return v[f]
    return v[f] + (v[c] - v[f]) * (k - f)


def timed_call(fn, *args, repeat: int = 1, **kwargs) -> Tuple[Any, float]:
    """Run fn(*args) 'repeat' times and return (last_result, median_ms)."""
    times = []
    res = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return res, times[len(times) // 2]


# ========== 7. Run dual experiment with latency ==========
def run_experiment_dual_latency():
    if not client.api_key:
        print("Please set DEEPSEEK_API_KEY in your environment variables.")
        return

    correct_new = 0
    correct_old = 0
    total = len(QUESTIONS)

    llm_ms_list: List[float] = []
    new_ms_list: List[float] = []
    old_ms_list: List[float] = []
    end2end_ms_list: List[float] = []

    for q in QUESTIONS:
        print("=" * 90)
        print(f"{q.qid}: {q.text}")
        print("Gold:", q.gold_answer)

        t_all0 = time.perf_counter()

        # 1) LLM generates ONE query JSON
        try:
            query, llm_ms = llm_generate_query(q.text)
        except Exception as e:
            print("LLM error:", repr(e))
            query, llm_ms = {}, 0.0

        print("LLM query JSON:", query)
        print(f"LLM latency: {llm_ms:.2f} ms")

        # 2) MMDB exec
        pred_new, new_ms = timed_call(execute_query_newdb, query, repeat=REPEAT_DB_EXEC)
        print("MMDB answer:", pred_new)
        print(f"MMDB exec latency (median of {REPEAT_DB_EXEC}): {new_ms:.4f} ms")

        # 3) PGDB exec
        pred_old, old_ms = timed_call(execute_query_olddb, query, repeat=REPEAT_DB_EXEC)
        print("PGDB answer:", pred_old)
        print(f"PGDB exec latency (median of {REPEAT_DB_EXEC}): {old_ms:.4f} ms")

        # 4) Accuracy
        ok_new = is_correct(pred_new, q.gold_answer)
        ok_old = is_correct(pred_old, q.gold_answer)
        print("MMDB Correct?", ok_new)
        print("PGDB Correct?", ok_old)

        if ok_new:
            correct_new += 1
        if ok_old:
            correct_old += 1

        end2end_ms = (time.perf_counter() - t_all0) * 1000.0

        # record metrics
        llm_ms_list.append(llm_ms)
        new_ms_list.append(new_ms)
        old_ms_list.append(old_ms)
        end2end_ms_list.append(end2end_ms)

        print(f"End-to-end latency: {end2end_ms:.2f} ms")

    acc_new = correct_new / total if total else 0.0
    acc_old = correct_old / total if total else 0.0

    def summarize(name: str, xs: List[float]):
        mean = sum(xs) / len(xs) if xs else 0.0
        p50 = percentile(xs, 50)
        p95 = percentile(xs, 95)
        print(f"{name}: mean={mean:.2f} ms, p50={p50:.2f} ms, p95={p95:.2f} ms (n={len(xs)})")

    print("=" * 90)
    print(f"MMDB Accuracy = {acc_new * 100:.1f}% ({correct_new} / {total})")
    print(f"PGDB Accuracy = {acc_old * 100:.1f}% ({correct_old} / {total})")
    print("-" * 90)
    summarize("LLM latency", llm_ms_list)
    summarize("MMDB exec latency", new_ms_list)
    summarize("PGDB exec latency", old_ms_list)
    summarize("End-to-end latency", end2end_ms_list)


if __name__ == "__main__":
    run_experiment_dual_latency()
