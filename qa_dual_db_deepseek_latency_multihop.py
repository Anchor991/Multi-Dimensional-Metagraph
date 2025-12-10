"""qa_dual_db_deepseek_latency_multihop30.py

Runs 30 multi-hop questions on:
  - MMDB: graph.json (3D hypercube metagraph)
  - PGDB: old_graph.json (baseline property graph)

Pipeline:
  NL question -> DeepSeek generates query plan JSON -> execute on both DBs -> compare to gold_answer -> accuracy
Also measures latency:
  - LLM latency (ms)
  - MMDB exec latency (ms)
  - PGDB exec latency (ms)
  - End-to-end latency (ms)

Requirements:
  - Set DEEPSEEK_API_KEY in environment
  - graph.json and old_graph.json in the same directory as this script (or edit paths below)

Run:
  python qa_dual_db_deepseek_latency_multihop30.py
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from openai import OpenAI

DEEPSEEK_MODEL = "deepseek-chat"
NEWDB_PATH = "graph.json"
OLDDB_PATH = "old_graph.json"
REPEAT_DB_EXEC = 1  # set >1 to reduce noise (median is reported)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

QUESTIONS = [
    {"qid": "Q1", "text": "小刘在2020年和2021年都去过哪些城市？", "gold_answer": ["南京", "广州"]},
    {"qid": "Q2", "text": "小刘在2020年和2022年都去过哪些城市？", "gold_answer": ["北京", "南京", "西安", "重庆"]},
    {"qid": "Q3", "text": "小刘在2020年和2023年都去过哪些城市？", "gold_answer": ["重庆"]},
    {"qid": "Q4", "text": "小刘在2020年和2024年都去过哪些城市？", "gold_answer": ["北京", "南京"]},
    {"qid": "Q5", "text": "小刘在2021年和2022年都去过哪些城市？", "gold_answer": ["南京"]},
    {"qid": "Q6", "text": "小刘在2021年和2024年都去过哪些城市？", "gold_answer": ["南京"]},
    {"qid": "Q7", "text": "小刘在2022年和2023年都去过哪些城市？", "gold_answer": ["重庆"]},
    {"qid": "Q8", "text": "小刘在2022年和2024年都去过哪些城市？", "gold_answer": ["北京", "南京", "深圳"]},
    {"qid": "Q9", "text": "小周在2020年和2022年都去过哪些城市？", "gold_answer": ["重庆"]},
    {"qid": "Q10", "text": "小周在2022年和2023年都去过哪些城市？", "gold_answer": ["海口", "深圳"]},
    {"qid": "Q11", "text": "在2020年同时去过上海和北京的人有哪些？", "gold_answer": ["小杨", "小赵"]},
    {"qid": "Q12", "text": "在2022年同时去过上海和北京的人有哪些？", "gold_answer": ["小周"]},
    {"qid": "Q13", "text": "在2024年同时去过上海和北京的人有哪些？", "gold_answer": ["小周", "小李"]},
    {"qid": "Q14", "text": "在2020年同时去过上海和南京的人有哪些？", "gold_answer": ["小杨"]},
    {"qid": "Q15", "text": "在2024年同时去过上海和南京的人有哪些？", "gold_answer": ["小周"]},
    {"qid": "Q16", "text": "在2022年同时去过上海和天津的人有哪些？", "gold_answer": ["小明"]},
    {"qid": "Q17", "text": "在2023年同时去过上海和天津的人有哪些？", "gold_answer": ["小红"]},
    {"qid": "Q18", "text": "在2024年同时去过上海和天津的人有哪些？", "gold_answer": ["小李"]},
    {"qid": "Q19", "text": "在2022年同时去过上海和广州的人有哪些？", "gold_answer": ["小王"]},
    {"qid": "Q20", "text": "在2023年同时去过上海和广州的人有哪些？", "gold_answer": ["小红"]},
    {"qid": "Q21", "text": "在2020年去过上海的人，在2021年去过哪些城市（合并去重）？", "gold_answer": ["北京", "南京", "杭州", "武汉", "深圳", "苏州", "重庆", "青岛"]},
    {"qid": "Q22", "text": "在2021年去过上海的人，在2020年去过哪些城市（合并去重）？", "gold_answer": ["北京", "南京", "天津", "苏州"]},
    {"qid": "Q23", "text": "在2022年去过上海的人，在2020年去过哪些城市（合并去重）？", "gold_answer": ["北京", "南京", "天津", "成都", "武汉", "苏州", "重庆"]},
    {"qid": "Q24", "text": "在2023年去过上海的人，在2020年去过哪些城市（合并去重）？", "gold_answer": ["南京", "成都", "苏州", "青岛"]},
    {"qid": "Q25", "text": "在2024年去过上海的人，在2020年去过哪些城市（合并去重）？", "gold_answer": ["北京", "南京", "成都", "武汉", "海口", "重庆"]},
    {"qid": "Q26", "text": "在2020年去过北京的人，在2024年去过哪些城市（合并去重）？", "gold_answer": ["上海", "北京", "南京", "成都", "杭州", "武汉", "深圳", "西安"]},
    {"qid": "Q27", "text": "在2021年去过北京的人，在2022年去过哪些城市（合并去重）？", "gold_answer": ["上海", "北京", "南京", "天津", "广州", "深圳", "青岛"]},
    {"qid": "Q28", "text": "在2021年与小刘同去过南京的人里，哪些人在2020年也去过北京？", "gold_answer": ["小杨", "小王", "小赵"]},
    {"qid": "Q29", "text": "在2023年与小刘同去过重庆的人里，哪些人在2021年也去过南京？", "gold_answer": ["小红"]},
    {"qid": "Q30", "text": "在2024年与小刘同去过北京的人里，哪些人在2020年也去过北京？", "gold_answer": ["小杨"]},
]

# -------------------- Load DBs --------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

newdb = load_json(NEWDB_PATH)
olddb = load_json(OLDDB_PATH)

# MMDB indices
person_id2name_new = {p["id"]: p["name"] for p in newdb["dimensions"]["person"]}
person_name2id_new = {v: k for k, v in person_id2name_new.items()}

time_id2value_new = {t["id"]: t["value"] for t in newdb["dimensions"]["time"]}
time_value2id_new = {v: k for k, v in time_id2value_new.items()}

geo_id2name_new = {g["id"]: g["name"] for g in newdb["dimensions"]["geo"]}
geo_name2id_new = {v: k for k, v in geo_id2name_new.items()}

facts_new = newdb["facts"]

# PGDB indices
old_node_by_id = {n["id"]: n for n in olddb["nodes"]}
person_nodes_old = {}
geo_nodes_old = {}
for n in olddb["nodes"]:
    if n.get("type") == "Person":
        person_nodes_old[n["name"]] = n["id"]
    elif n.get("type") == "Geo":
        geo_nodes_old[n["name"]] = n["id"]
edges_old = olddb["edges"]

# -------------------- LLM prompt --------------------
def build_system_prompt() -> str:
    return (
        "你是一个查询生成器。我们有一个三维数据库，维度为 Person(人物)、Time(年份)、Geo(城市)。"
        "每条事实 fact 有 person, time, geo 三个字段。\n\n"
        "请根据用户的问题，生成一个 JSON 对象，不要输出其他文字。\n\n"
        "你只能使用以下原子操作（op）来分解多跳问题：\n"
        "1) cities_by_person_year: person_name, year -> 城市列表\n"
        "2) persons_by_city_year: city_name, year -> 人物列表\n"
        "3) cities_by_person_year_range: person_name, year_start, year_end -> 城市列表\n"
        "4) persons_by_city_year_range: city_name, year_start, year_end -> 人物列表\n"
        "5) years_by_person_city: person_name, city_name -> 年份列表\n"
        "6) count_trips_by_person: person_name -> 次数(字符串)\n\n"
        "多跳问题必须输出一个执行计划 plan（多个原子步骤）以及合并方式 combine。\n"
        "输出 JSON 格式：\n"
        "{\n"
        "  \"plan\": [\n"
        "    { \"op\": \"...\", ... },\n"
        "    ...\n"
        "  ],\n"
        "  \"combine\": \"intersection\" | \"union\" | \"difference\"\n"
        "}\n\n"
        "combine 说明：\n"
        "- intersection: 对多个步骤结果做交集\n"
        "- union: 对多个步骤结果做并集去重\n"
        "- difference: step1 - step2\n\n"
        "严格要求：只能输出 JSON 对象，不要输出任何额外文字。"
    )

def llm_generate_plan(question: str) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )
    llm_ms = (time.perf_counter() - t0) * 1000.0
    return json.loads(resp.choices[0].message.content), llm_ms

# -------------------- DB primitive ops (MMDB) --------------------
def cities_by_person_year_new(person_name: str, year: int) -> List[str]:
    pid = person_name2id_new.get(person_name)
    tid = time_value2id_new.get(str(year))
    if pid is None or tid is None:
        return []
    out = set()
    for fact in facts_new:
        if fact["person"] == pid and fact["time"] == tid:
            out.add(geo_id2name_new[fact["geo"]])
    return sorted(out)

def persons_by_city_year_new(city_name: str, year: int) -> List[str]:
    gid = geo_name2id_new.get(city_name)
    tid = time_value2id_new.get(str(year))
    if gid is None or tid is None:
        return []
    out = set()
    for fact in facts_new:
        if fact["geo"] == gid and fact["time"] == tid:
            out.add(person_id2name_new[fact["person"]])
    return sorted(out)

def cities_by_person_year_range_new(person_name: str, ys: int, ye: int) -> List[str]:
    pid = person_name2id_new.get(person_name)
    if pid is None:
        return []
    tids = set()
    for y in range(ys, ye + 1):
        tid = time_value2id_new.get(str(y))
        if tid:
            tids.add(tid)
    out = set()
    for fact in facts_new:
        if fact["person"] == pid and fact["time"] in tids:
            out.add(geo_id2name_new[fact["geo"]])
    return sorted(out)

def persons_by_city_year_range_new(city_name: str, ys: int, ye: int) -> List[str]:
    gid = geo_name2id_new.get(city_name)
    if gid is None:
        return []
    tids = set()
    for y in range(ys, ye + 1):
        tid = time_value2id_new.get(str(y))
        if tid:
            tids.add(tid)
    out = set()
    for fact in facts_new:
        if fact["geo"] == gid and fact["time"] in tids:
            out.add(person_id2name_new[fact["person"]])
    return sorted(out)

def years_by_person_city_new(person_name: str, city_name: str) -> List[str]:
    pid = person_name2id_new.get(person_name)
    gid = geo_name2id_new.get(city_name)
    if pid is None or gid is None:
        return []
    tids = set()
    for fact in facts_new:
        if fact["person"] == pid and fact["geo"] == gid:
            tids.add(fact["time"])
    out = set()
    for tid in tids:
        out.add(time_id2value_new[tid])
    return sorted(out)

def count_trips_by_person_new(person_name: str) -> List[str]:
    pid = person_name2id_new.get(person_name)
    if pid is None:
        return ["0"]
    cnt = sum(1 for f in facts_new if f["person"] == pid)
    return [str(cnt)]

def exec_primitive_new(step: Dict[str, Any]) -> List[str]:
    op = step.get("op")
    if op == "cities_by_person_year":
        return cities_by_person_year_new(step.get("person_name"), int(step.get("year")))
    if op == "persons_by_city_year":
        return persons_by_city_year_new(step.get("city_name"), int(step.get("year")))
    if op == "cities_by_person_year_range":
        return cities_by_person_year_range_new(step.get("person_name"), int(step.get("year_start")), int(step.get("year_end")))
    if op == "persons_by_city_year_range":
        return persons_by_city_year_range_new(step.get("city_name"), int(step.get("year_start")), int(step.get("year_end")))
    if op == "years_by_person_city":
        return years_by_person_city_new(step.get("person_name"), step.get("city_name"))
    if op == "count_trips_by_person":
        return count_trips_by_person_new(step.get("person_name"))
    return []

# -------------------- DB primitive ops (PGDB) --------------------
def cities_by_person_year_old(person_name: str, year: int) -> List[str]:
    pid = person_nodes_old.get(person_name)
    if pid is None:
        return []
    y = str(year)
    out = set()
    for e in edges_old:
        if e["from"] == pid and e["year"] == y:
            n = old_node_by_id.get(e["to"])
            if n and n.get("type") == "Geo":
                out.add(n["name"])
    return sorted(out)

def persons_by_city_year_old(city_name: str, year: int) -> List[str]:
    gid = geo_nodes_old.get(city_name)
    if gid is None:
        return []
    y = str(year)
    out = set()
    for e in edges_old:
        if e["to"] == gid and e["year"] == y:
            n = old_node_by_id.get(e["from"])
            if n and n.get("type") == "Person":
                out.add(n["name"])
    return sorted(out)

def cities_by_person_year_range_old(person_name: str, ys: int, ye: int) -> List[str]:
    pid = person_nodes_old.get(person_name)
    if pid is None:
        return []
    years = {str(y) for y in range(ys, ye + 1)}
    out = set()
    for e in edges_old:
        if e["from"] == pid and e["year"] in years:
            n = old_node_by_id.get(e["to"])
            if n and n.get("type") == "Geo":
                out.add(n["name"])
    return sorted(out)

def persons_by_city_year_range_old(city_name: str, ys: int, ye: int) -> List[str]:
    gid = geo_nodes_old.get(city_name)
    if gid is None:
        return []
    years = {str(y) for y in range(ys, ye + 1)}
    out = set()
    for e in edges_old:
        if e["to"] == gid and e["year"] in years:
            n = old_node_by_id.get(e["from"])
            if n and n.get("type") == "Person":
                out.add(n["name"])
    return sorted(out)

def years_by_person_city_old(person_name: str, city_name: str) -> List[str]:
    pid = person_nodes_old.get(person_name)
    gid = geo_nodes_old.get(city_name)
    if pid is None or gid is None:
        return []
    out = set()
    for e in edges_old:
        if e["from"] == pid and e["to"] == gid:
            out.add(e["year"])
    return sorted(out)

def count_trips_by_person_old(person_name: str) -> List[str]:
    pid = person_nodes_old.get(person_name)
    if pid is None:
        return ["0"]
    cnt = sum(1 for e in edges_old if e["from"] == pid)
    return [str(cnt)]

def exec_primitive_old(step: Dict[str, Any]) -> List[str]:
    op = step.get("op")
    if op == "cities_by_person_year":
        return cities_by_person_year_old(step.get("person_name"), int(step.get("year")))
    if op == "persons_by_city_year":
        return persons_by_city_year_old(step.get("city_name"), int(step.get("year")))
    if op == "cities_by_person_year_range":
        return cities_by_person_year_range_old(step.get("person_name"), int(step.get("year_start")), int(step.get("year_end")))
    if op == "persons_by_city_year_range":
        return persons_by_city_year_range_old(step.get("city_name"), int(step.get("year_start")), int(step.get("year_end")))
    if op == "years_by_person_city":
        return years_by_person_city_old(step.get("person_name"), step.get("city_name"))
    if op == "count_trips_by_person":
        return count_trips_by_person_old(step.get("person_name"))
    return []

# -------------------- Plan execution (multi-hop) --------------------
def combine_results(results: List[List[str]], combine: str) -> List[str]:
    sets = [set(r) for r in results]
    if not sets:
        return []
    if combine == "intersection":
        s = sets[0].copy()
        for t in sets[1:]:
            s &= t
        return sorted(s)
    if combine == "union":
        s = set()
        for t in sets:
            s |= t
        return sorted(s)
    if combine == "difference":
        s = sets[0].copy()
        if len(sets) > 1:
            s -= sets[1]
        return sorted(s)
    # default
    s = sets[0].copy()
    for t in sets[1:]:
        s &= t
    return sorted(s)

def execute_plan(exec_primitive, plan_json: Dict[str, Any]) -> List[str]:
    plan = plan_json.get("plan", [])
    combine = plan_json.get("combine", "intersection")
    results = [exec_primitive(step) for step in plan]
    return combine_results(results, combine)

# -------------------- Metrics --------------------
def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def timed(fn, *args, repeat: int = 1, **kwargs) -> Tuple[Any, float]:
    ts = []
    res = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return res, ts[len(ts)//2]

def correct(pred: List[str], gold: List[str]) -> bool:
    return set(pred) == set(gold)

# -------------------- Main --------------------
def main():
    if not client.api_key:
        print("Please set DEEPSEEK_API_KEY first.")
        return

    llm_ms_list, new_ms_list, old_ms_list, e2e_ms_list = [], [], [], []
    ok_new = ok_old = 0

    for q in QUESTIONS:
        print("=" * 96)
        print(f'{q["qid"]}: {q["text"]}')
        print("Gold:", q["gold_answer"])

        t_all0 = time.perf_counter()

        plan_json, llm_ms = llm_generate_plan(q["text"])
        print("LLM plan JSON:", plan_json)
        print(f"LLM latency: {llm_ms:.2f} ms")

        pred_new, new_ms = timed(lambda pj: execute_plan(exec_primitive_new, pj), plan_json, repeat=REPEAT_DB_EXEC)
        pred_old, old_ms = timed(lambda pj: execute_plan(exec_primitive_old, pj), plan_json, repeat=REPEAT_DB_EXEC)

        print("MMDB answer:", pred_new)
        print("PGDB answer:", pred_old)
        print(f"MMDB exec latency (median of {REPEAT_DB_EXEC}): {new_ms:.4f} ms")
        print(f"PGDB exec latency (median of {REPEAT_DB_EXEC}): {old_ms:.4f} ms")

        c_new = correct(pred_new, q["gold_answer"])
        c_old = correct(pred_old, q["gold_answer"])
        print("MMDB Correct?", c_new)
        print("PGDB Correct?", c_old)

        ok_new += int(c_new)
        ok_old += int(c_old)

        e2e_ms = (time.perf_counter() - t_all0) * 1000.0
        print(f"End-to-end latency: {e2e_ms:.2f} ms")

        llm_ms_list.append(llm_ms)
        new_ms_list.append(new_ms)
        old_ms_list.append(old_ms)
        e2e_ms_list.append(e2e_ms)

    total = len(QUESTIONS)
    print("=" * 96)
    print(f"MMDB Accuracy = {ok_new/total*100:.1f}% ({ok_new} / {total})")
    print(f"PGDB Accuracy = {ok_old/total*100:.1f}% ({ok_old} / {total})")
    print("-" * 96)

    def summarize(name: str, xs: List[float]):
        mean = sum(xs) / len(xs) if xs else 0.0
        print(f"{name}: mean={mean:.2f} ms, p50={percentile(xs, 50):.2f} ms, p95={percentile(xs, 95):.2f} ms (n={len(xs)})")

    summarize("LLM latency", llm_ms_list)
    summarize("MMDB exec latency", new_ms_list)
    summarize("PGDB exec latency", old_ms_list)
    summarize("End-to-end latency", e2e_ms_list)

if __name__ == "__main__":
    main()
