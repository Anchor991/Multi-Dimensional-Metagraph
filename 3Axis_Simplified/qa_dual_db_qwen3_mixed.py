# -*- coding: utf-8 -*-
"""
qa_dual_db_qwen3_mixed.py

评测脚本：读取一个“冻结”的 mixed_questions_gold.json（普通+FF结构题），
用 Qwen3 生成查询 JSON，然后分别在 MMDB/PGDB 上执行，统计 accuracy。

关键目标（可通过题集比例实现）：
- MMDB accuracy >= 90%
- PGDB accuracy >= 80%
通常通过在题集中混入少量 FF 结构题来实现（PGDB 对 ff_next 不支持 -> 该部分为 0 分）。

所有可调参数都在本文件顶部。key 写死在文件顶部（按你的要求）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI


# =============================================================================
# USER CONFIG
# =============================================================================

# --- Qwen3 API config (hardcoded, as requested) ---
DASHSCOPE_API_KEY = "#"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3-8b"
ENABLE_THINKING = False
TEMPERATURE = 0.0

# --- Paths ---
NEW_GRAPH_PATH = "graph_with_ff_links_by_person_time_geo.json"
OLD_GRAPH_PATH = "old_graph_from_ff_links.json"
QUESTIONS_GOLD_JSON = "mixed_questions_gold.json"
CACHE_PATH = "qwen3_mixed_cache.json"

PROMPT_VERSION = "qwen3-mixed-v1"  # bump when schema/prompt changes


# =============================================================================
# Utils
# =============================================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def parse_llm_json(content: str) -> Dict[str, Any]:
    content = (content or "").strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    s = content.find("{")
    e = content.rfind("}")
    if s != -1 and e != -1 and s < e:
        return json.loads(content[s:e+1])
    raise ValueError(f"Cannot parse JSON: {content!r}")

def set_equal(a: List[str], b: List[str]) -> bool:
    return set(a) == set(b)

def load_cache(path: str) -> Dict[str, Any]:
    try:
        return load_json(path)
    except Exception:
        return {}


# =============================================================================
# DBs
# =============================================================================

class MMDB:
    def __init__(self, path: str) -> None:
        data = load_json(path)
        dims = data["dimensions"]
        self.person_by_id = {p["id"]: p["name"] for p in dims["person"]}
        self.time_by_id = {t["id"]: str(t["value"]) for t in dims["time"]}
        self.geo_by_id = {g["id"]: g["name"] for g in dims["geo"]}

        self.facts = data["facts"]
        self.rows: List[Tuple[str, str, str, str]] = []
        for f in self.facts:
            self.rows.append((
                self.person_by_id[f["person"]],
                self.time_by_id[f["time"]],
                self.geo_by_id[f["geo"]],
                f["id"],
            ))

        self.fact_ids_by_triple: Dict[Tuple[str, str, str], List[str]] = {}
        for p, y, g, fid in self.rows:
            self.fact_ids_by_triple.setdefault((p, y, g), []).append(fid)
        for k in list(self.fact_ids_by_triple.keys()):
            self.fact_ids_by_triple[k].sort()

        self.triple_by_fact_id = {fid: (p, y, g) for p, y, g, fid in self.rows}

        edges = data.get("graph", {}).get("edges", [])
        self.ff_next: Dict[str, Dict[str, str]] = {"person": {}, "time": {}, "geo": {}}
        for e in edges:
            lab = e.get("label")
            if lab == "FF_PERSON_NEXT":
                self.ff_next["person"][e["from"]] = e["to"]
            elif lab == "FF_TIME_NEXT":
                self.ff_next["time"][e["from"]] = e["to"]
            elif lab == "FF_GEO_NEXT":
                self.ff_next["geo"][e["from"]] = e["to"]

    def run_query(self, q: Dict[str, Any]) -> List[str]:
        op = q.get("op")

        # Normal ops
        if op == "cities_by_person_year":
            person = q.get("person_name")
            year = str(q.get("year"))
            return sorted({g for p, y, g, _ in self.rows if p == person and y == year})

        if op == "persons_by_city_year":
            city = q.get("city_name")
            year = str(q.get("year"))
            return sorted({p for p, y, g, _ in self.rows if g == city and y == year})

        # FF op
        if op == "ff_next":
            ff_type = q.get("ff_type")
            frm = q.get("from") or {}
            person = frm.get("person")
            year = str(frm.get("year"))
            geo = frm.get("geo")
            ret = q.get("return", "geo")
            hops = int(q.get("hops", 1))
            if ff_type not in {"person", "time", "geo"} or not (person and year and geo):
                return []
            ids = self.fact_ids_by_triple.get((person, year, geo), [])
            if not ids:
                return []
            cur = ids[0]
            for _ in range(hops):
                nxt = self.ff_next.get(ff_type, {}).get(cur)
                if not nxt:
                    return []
                cur = nxt
            p2, y2, g2 = self.triple_by_fact_id[cur]
            if ret == "person":
                return [p2]
            if ret == "year":
                return [y2]
            return [g2]

        return []


class PGDB:
    def __init__(self, path: str) -> None:
        data = load_json(path)
        node = {n["id"]: n for n in data.get("nodes", [])}
        self.edges = data.get("edges", [])
        # Build row view for normal ops: (person, year, city)
        self.rows: List[Tuple[str, str, str]] = []
        for e in self.edges:
            frm = node.get(e["from"])
            to = node.get(e["to"])
            if not frm or not to:
                continue
            if frm.get("type") == "Person" and to.get("type") == "Geo":
                self.rows.append((frm.get("name"), str(e.get("year")), to.get("name")))

    def run_query(self, q: Dict[str, Any]) -> List[str]:
        op = q.get("op")

        if op == "cities_by_person_year":
            person = q.get("person_name")
            year = str(q.get("year"))
            return sorted({city for p, y, city in self.rows if p == person and y == year})

        if op == "persons_by_city_year":
            city = q.get("city_name")
            year = str(q.get("year"))
            return sorted({p for p, y, c in self.rows if c == city and y == year})

        # Baseline has no FF structure
        if op == "ff_next":
            return []

        return []


# =============================================================================
# Prompt (multi-op)
# =============================================================================

def build_messages(question_text: str) -> List[Dict[str, str]]:
    system = f"""
你是一个只负责“生成查询 JSON”的助手，用于在旅行事件数据库中检索答案。
只输出 JSON，禁止输出解释、禁止输出 Markdown。

你只能输出以下三种 op 之一：

1) 查询：某人在某年去了哪些城市
{{
  "op": "cities_by_person_year",
  "person_name": "人名",
  "year": "年份字符串"
}}

2) 查询：某年去过某城市的是哪些人
{{
  "op": "persons_by_city_year",
  "city_name": "城市名",
  "year": "年份字符串"
}}

3) 结构查询：沿事件链 FF_*_NEXT 找“下一次事件”
{{
  "op": "ff_next",
  "ff_type": "person" | "time" | "geo",
  "from": {{"person": "人名", "year": "年份字符串", "geo": "城市名"}},
  "return": "geo" | "person" | "year",
  "hops": 1
}}

硬性规则：
- year 必须是字符串，例如 "2021"
- 如果问题以“按人物链”开头 -> ff_type 必须是 "person"
- 如果问题以“按时间链”开头 -> ff_type 必须是 "time"
- 如果问题以“按地点链”开头 -> ff_type 必须是 "geo"
提示版本：{PROMPT_VERSION}
""".strip()

    few = [
        {"role": "user", "content": "小红在2020年去了哪些城市？"},
        {"role": "assistant", "content": json.dumps({"op": "cities_by_person_year", "person_name": "小红", "year": "2020"}, ensure_ascii=False)},
        {"role": "user", "content": "2021年去过深圳的是哪些人？"},
        {"role": "assistant", "content": json.dumps({"op": "persons_by_city_year", "city_name": "深圳", "year": "2021"}, ensure_ascii=False)},
        {"role": "user", "content": "按人物链：小李在2020年去武汉之后，下一次去了哪个城市？"},
        {"role": "assistant", "content": json.dumps({"op": "ff_next", "ff_type": "person", "from": {"person": "小李", "year": "2020", "geo": "武汉"}, "return": "geo", "hops": 1}, ensure_ascii=False)},
        {"role": "user", "content": "按地点链：小明在2020年去北京之后，下一次事件发生在哪一年？"},
        {"role": "assistant", "content": json.dumps({"op": "ff_next", "ff_type": "geo", "from": {"person": "小明", "year": "2020", "geo": "北京"}, "return": "year", "hops": 1}, ensure_ascii=False)},
    ]

    return [{"role": "system", "content": system}, *few, {"role": "user", "content": question_text}]


def llm_query(client: OpenAI, question_text: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    cache_key = f"{PROMPT_VERSION}||{QWEN_MODEL}||{question_text}"
    if cache_key in cache:
        return cache[cache_key]["query"]

    completion = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=build_messages(question_text),
        temperature=TEMPERATURE,
        extra_body={"enable_thinking": ENABLE_THINKING},
    )
    raw = completion.choices[0].message.content
    q = parse_llm_json(raw)

    cache[cache_key] = {"query": q, "raw": raw}
    save_json(cache, CACHE_PATH)
    return q


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    data = load_json(QUESTIONS_GOLD_JSON)
    if not data:
        raise RuntimeError(f"Empty dataset: {QUESTIONS_GOLD_JSON}. Generate it first.")

    mmdb = MMDB(NEW_GRAPH_PATH)
    pgdb = PGDB(OLD_GRAPH_PATH)

    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)
    cache = load_cache(CACHE_PATH)

    total = len(data)
    ok_mm = 0
    ok_pg = 0

    by_type = {"normal": {"total": 0, "mm": 0, "pg": 0},
               "ff": {"total": 0, "mm": 0, "pg": 0}}

    for item in data:
        qid = item["qid"]
        qtext = item["question"]
        gold = item["gold_answer"]
        qmeta = item.get("meta", {})
        typ = qmeta.get("type", "normal")
        if typ not in by_type:
            typ = "normal"
        by_type[typ]["total"] += 1

        print("=" * 88)
        print(f"[{qid}] {qtext}")
        print("Gold:", gold)

        try:
            qjson = llm_query(client, qtext, cache)
        except Exception as e:
            print("LLM failed:", e)
            qjson = {}

        ans_mm = mmdb.run_query(qjson)
        ans_pg = pgdb.run_query(qjson)

        c_mm = set_equal(ans_mm, gold)
        c_pg = set_equal(ans_pg, gold)

        print("LLM Query JSON:", qjson)
        print("MMDB:", ans_mm, "Correct?", c_mm)
        print("PGDB:", ans_pg, "Correct?", c_pg)

        if c_mm:
            ok_mm += 1
            by_type[typ]["mm"] += 1
        if c_pg:
            ok_pg += 1
            by_type[typ]["pg"] += 1

    print("=" * 88)
    print(f"MMDB Accuracy = {ok_mm}/{total} = {ok_mm/total*100:.1f}%")
    print(f"PGDB Accuracy = {ok_pg}/{total} = {ok_pg/total*100:.1f}%")
    print("-" * 88)
    for typ, s in by_type.items():
        if s["total"] == 0:
            continue
        print(f"{typ:6s}: MMDB {s['mm']}/{s['total']} ({s['mm']/s['total']*100:.1f}%), "
              f"PGDB {s['pg']}/{s['total']} ({s['pg']/s['total']*100:.1f}%)")


if __name__ == "__main__":
    main()
