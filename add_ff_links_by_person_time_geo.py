import json
from collections import defaultdict

INPUT = "graph.json"
OUTPUT = "graph_with_ff_links_by_person_time_geo.json"

LABEL_PERSON = "FF_PERSON_NEXT"
LABEL_TIME = "FF_TIME_NEXT"
LABEL_GEO = "FF_GEO_NEXT"

def to_int(x, default=0):
    try:
        return int(str(x))
    except Exception:
        return default

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    time_id2year = {t["id"]: to_int(t["value"]) for t in data["dimensions"]["time"]}
    geo_id2name = {g["id"]: g["name"] for g in data["dimensions"]["geo"]}
    person_id2name = {p["id"]: p["name"] for p in data["dimensions"]["person"]}

    facts = data["facts"]
    edges = data["graph"]["edges"]

    # 为避免重复添加：记录已有 (from,to,label)
    existing = {(e.get("from"), e.get("to"), e.get("label")) for e in edges}

    added = 0

    # ---------- 1) 按人物：同一 person，按 year 排序 ----------
    by_person = defaultdict(list)
    for fct in facts:
        pid = fct["person"]
        year = time_id2year.get(fct["time"], 0)
        by_person[pid].append((year, fct["id"]))

    for pid, items in by_person.items():
        items.sort(key=lambda x: x[0])
        for i in range(len(items) - 1):
            u = items[i][1]
            v = items[i + 1][1]
            key = (u, v, LABEL_PERSON)
            if key not in existing:
                edges.append({"from": u, "to": v, "label": LABEL_PERSON})
                existing.add(key)
                added += 1

    # ---------- 2) 按时间：同一年份，按 geo_name 排序 ----------
    by_time = defaultdict(list)
    for fct in facts:
        tid = fct["time"]
        geo_name = geo_id2name.get(fct["geo"], "")
        by_time[tid].append((geo_name, fct["id"]))

    for tid, items in by_time.items():
        items.sort(key=lambda x: x[0])
        for i in range(len(items) - 1):
            u = items[i][1]
            v = items[i + 1][1]
            key = (u, v, LABEL_TIME)
            if key not in existing:
                edges.append({"from": u, "to": v, "label": LABEL_TIME})
                existing.add(key)
                added += 1

    # ---------- 3) 按地点：同一 geo，按 year 排序 ----------
    by_geo = defaultdict(list)
    for fct in facts:
        gid = fct["geo"]
        year = time_id2year.get(fct["time"], 0)
        by_geo[gid].append((year, fct["id"]))

    for gid, items in by_geo.items():
        items.sort(key=lambda x: x[0])
        for i in range(len(items) - 1):
            u = items[i][1]
            v = items[i + 1][1]
            key = (u, v, LABEL_GEO)
            if key not in existing:
                edges.append({"from": u, "to": v, "label": LABEL_GEO})
                existing.add(key)
                added += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Added {added} Fact->Fact edges in total. Saved to {OUTPUT}")
    print(f"Labels: {LABEL_PERSON}, {LABEL_TIME}, {LABEL_GEO}")

if __name__ == "__main__":
    main()
