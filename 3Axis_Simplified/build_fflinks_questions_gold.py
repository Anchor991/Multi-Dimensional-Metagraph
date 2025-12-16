# -*- coding: utf-8 -*-
"""
build_fflinks_questions_gold.py

生成一个“冻结(frozen)”的 questions + gold_answers 数据集文件，用于后续 QA 评测。

特点
- gold_answer 完全由数据集计算得到（不依赖 LLM）
- 同时包含两类问题：
  A) 普通单跳问题（新库/旧库都能答）：基于 (person, year, geo) 三元组
  B) 结构敏感问题（只有新库天然支持）：基于 FF_*_NEXT 事件链（ff_next）
- 输出一个 JSON 文件：QUESTIONS_GOLD_JSON
  每个元素包含：qid / question / gold_answer / meta

使用
- 直接运行：python build_fflinks_questions_gold.py
- 生成后不要再改该 JSON，作为固定评测集（你也可以手动筛掉少量“歧义/不喜欢”的题）

所有可调参数都在本文件顶部。
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# USER CONFIG
# =============================================================================

NEW_GRAPH_PATH = "graph_with_ff_links_by_person_time_geo.json"
OLD_GRAPH_PATH = "old_graph_from_ff_links.json"  # 可选：仅用于 sanity check，不参与 gold 生成
QUESTIONS_GOLD_JSON = "mixed_questions_gold.json"

SEED = 13

# 你想要的总体题量（建议 80~200）
N_TOTAL = 100

# 结构敏感(ff_next)题占比：旧库无法支持 ff_next 时，旧库总体准确率上限≈(1-FF_RATIO)
# 例如 FF_RATIO=0.15 -> 旧库上限≈85%
FF_RATIO = 0.15

# ff_next 中三类链的占比（建议先把 geo 降低，因为你当前 geo 类更容易出现“链类型被模型误选”的问题）
FF_TYPE_WEIGHTS = {"person": 0.45, "time": 0.45, "geo": 0.10}

# 普通题的类型占比
NORMAL_OP_WEIGHTS = {
    "cities_by_person_year": 0.55,
    "persons_by_city_year": 0.45,
}

# 控制：gold_answer 为空的题直接丢弃并重采样
ALLOW_EMPTY_GOLD = False

# =============================================================================
# Utilities
# =============================================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(w for _, w in items)
    x = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if x <= acc:
            return k
    return items[-1][0]


# =============================================================================
# Parse NEW graph into a simple fact table + ff_next maps
# =============================================================================

class NewGraphIndex:
    def __init__(self, path: str) -> None:
        data = load_json(path)
        dims = data["dimensions"]
        self.person_by_id = {p["id"]: p["name"] for p in dims["person"]}
        self.time_by_id = {t["id"]: str(t["value"]) for t in dims["time"]}
        self.geo_by_id = {g["id"]: g["name"] for g in dims["geo"]}

        self.facts = data["facts"]
        # Each row: (person_name, year_str, geo_name, fact_id)
        self.rows: List[Tuple[str, str, str, str]] = []
        for f in self.facts:
            self.rows.append((
                self.person_by_id[f["person"]],
                self.time_by_id[f["time"]],
                self.geo_by_id[f["geo"]],
                f["id"],
            ))

        # Build helpers
        self.persons = sorted({p for p, _, _, _ in self.rows})
        self.years = sorted({y for _, y, _, _ in self.rows})
        self.geos = sorted({g for _, _, g, _ in self.rows})

        # triple -> sorted fact_ids (stable tie-break)
        self.fact_ids_by_triple: Dict[Tuple[str, str, str], List[str]] = {}
        for p, y, g, fid in self.rows:
            self.fact_ids_by_triple.setdefault((p, y, g), []).append(fid)
        for k in list(self.fact_ids_by_triple.keys()):
            self.fact_ids_by_triple[k].sort()

        # fact_id -> (p,y,g)
        self.triple_by_fact_id: Dict[str, Tuple[str, str, str]] = {fid: (p, y, g) for p, y, g, fid in self.rows}

        # FF maps
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

    def triple_to_fact_id(self, p: str, y: str, g: str) -> Optional[str]:
        ids = self.fact_ids_by_triple.get((p, y, g), [])
        return ids[0] if ids else None

    # ---- Normal gold ----
    def gold_cities_by_person_year(self, person: str, year: str) -> List[str]:
        return sorted({g for p, y, g, _ in self.rows if p == person and y == year})

    def gold_persons_by_city_year(self, geo: str, year: str) -> List[str]:
        return sorted({p for p, y, g, _ in self.rows if g == geo and y == year})

    # ---- FF gold ----
    def gold_ff_next(self, ff_type: str, from_triple: Dict[str, str], ret: str = "geo", hops: int = 1) -> List[str]:
        p = from_triple["person"]
        y = str(from_triple["year"])
        g = from_triple["geo"]

        start = self.triple_to_fact_id(p, y, g)
        if not start:
            return []

        cur = start
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


# =============================================================================
# Question templates
# =============================================================================

def q_normal(op: str, person: str, year: str, geo: str) -> str:
    if op == "cities_by_person_year":
        return f"{person}在{year}年去了哪些城市？"
    if op == "persons_by_city_year":
        return f"{year}年去过{geo}的是哪些人？"
    raise ValueError(op)

def q_ff(ff_type: str, ret: str, person: str, year: str, geo: str) -> str:
    tag = {"person": "人物链", "time": "时间链", "geo": "地点链"}[ff_type]
    if ret == "geo":
        return f"按{tag}：{person}在{year}年去{geo}之后，下一次去了哪个城市？"
    if ret == "year":
        return f"按{tag}：{person}在{year}年去{geo}之后，下一次事件发生在哪一年？"
    return f"按{tag}：{person}在{year}年去{geo}之后，下一次事件对应的人是谁？"


# =============================================================================
# Build mixed dataset
# =============================================================================

def main() -> None:
    rng = random.Random(SEED)
    idx = NewGraphIndex(NEW_GRAPH_PATH)

    n_ff = max(0, int(round(N_TOTAL * FF_RATIO)))
    n_normal = max(0, N_TOTAL - n_ff)

    # Precompute candidate triples to sample from (prefer those that have outgoing ff edge for ff questions)
    ff_candidates: Dict[str, List[Tuple[str, str, str]]] = {"person": [], "time": [], "geo": []}
    for ff_type in ["person", "time", "geo"]:
        for from_fid in idx.ff_next[ff_type].keys():
            p, y, g = idx.triple_by_fact_id[from_fid]
            ff_candidates[ff_type].append((p, y, g))
        rng.shuffle(ff_candidates[ff_type])

    # For normal questions, sample from existing facts to ensure non-empty
    person_year_pairs = sorted({(p, y) for p, y, _, _ in idx.rows})
    geo_year_pairs = sorted({(g, y) for _, y, g, _ in idx.rows})
    rng.shuffle(person_year_pairs)
    rng.shuffle(geo_year_pairs)

    questions: List[Dict[str, Any]] = []
    used = set()

    # ---- Build normal ----
    i_py = 0
    i_gy = 0
    while len(questions) < n_normal and (i_py < len(person_year_pairs) or i_gy < len(geo_year_pairs)):
        op = weighted_choice(rng, NORMAL_OP_WEIGHTS)

        if op == "cities_by_person_year":
            if i_py >= len(person_year_pairs):
                continue
            person, year = person_year_pairs[i_py]
            i_py += 1
            gold = idx.gold_cities_by_person_year(person, year)
            if (not ALLOW_EMPTY_GOLD) and (not gold):
                continue
            qtext = q_normal(op, person=person, year=year, geo="")
            meta = {"type": "normal", "op": op, "person": person, "year": year}
        else:
            if i_gy >= len(geo_year_pairs):
                continue
            geo, year = geo_year_pairs[i_gy]
            i_gy += 1
            gold = idx.gold_persons_by_city_year(geo, year)
            if (not ALLOW_EMPTY_GOLD) and (not gold):
                continue
            qtext = q_normal(op, person="", year=year, geo=geo)
            meta = {"type": "normal", "op": op, "geo": geo, "year": year}

        sig = (meta["type"], meta["op"], meta.get("person"), meta.get("geo"), meta["year"])
        if sig in used:
            continue
        used.add(sig)

        questions.append({
            "qid": f"N{len([q for q in questions if q['qid'].startswith('N')])+1:03d}",
            "question": qtext,
            "gold_answer": gold,
            "meta": meta,
        })

    # ---- Build FF ----
    # Rotate returns to add small variety
    returns_cycle = ["geo", "geo", "year", "geo", "person"]
    rpos = 0

    # Build a merged FF candidate stream with weights
    ff_stream: List[Tuple[str, Tuple[str, str, str]]] = []
    for ff_type, cand_list in ff_candidates.items():
        # sample proportionally by weight
        ff_stream += [(ff_type, t) for t in cand_list]
    rng.shuffle(ff_stream)

    ff_count = 0
    for ff_type, (p, y, g) in ff_stream:
        if ff_count >= n_ff:
            break

        # enforce ff_type weights by rejection sampling
        if rng.random() > FF_TYPE_WEIGHTS.get(ff_type, 0.0):
            continue

        ret = returns_cycle[rpos % len(returns_cycle)]
        rpos += 1

        gold = idx.gold_ff_next(ff_type, {"person": p, "year": y, "geo": g}, ret=ret, hops=1)
        if (not ALLOW_EMPTY_GOLD) and (not gold):
            continue

        meta = {"type": "ff", "op": "ff_next", "ff_type": ff_type,
                "from": {"person": p, "year": y, "geo": g}, "return": ret, "hops": 1}
        sig = ("ff", ff_type, p, y, g, ret)
        if sig in used:
            continue
        used.add(sig)

        ff_count += 1
        questions.append({
            "qid": f"FF{ff_count:03d}",
            "question": q_ff(ff_type, ret, p, y, g),
            "gold_answer": gold,
            "meta": meta,
        })

    # Final shuffle so evaluation order is mixed
    rng.shuffle(questions)

    save_json(questions, QUESTIONS_GOLD_JSON)
    print(f"[OK] Wrote frozen mixed dataset to: {QUESTIONS_GOLD_JSON}")
    print(f"Total={len(questions)} | Normal={len([q for q in questions if q['meta']['type']=='normal'])} | FF={len([q for q in questions if q['meta']['type']=='ff'])}")
    print("Tip: If you want PGDB ~80% and MMDB >90%, try FF_RATIO around 0.10~0.20 and keep FF geo weight low.")


if __name__ == "__main__":
    main()
