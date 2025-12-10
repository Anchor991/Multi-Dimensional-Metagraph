# build_olddb_from_json.py
# 从三维 hypercube 的 graph.json 生成一个“传统图数据库” old_graph.json

import json


INPUT_PATH = "graph.json"
OUTPUT_PATH = "old_graph.json"


def build_old_graph_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    persons = {p["id"]: p["name"] for p in data["dimensions"]["person"]}
    times = {t["id"]: t["value"] for t in data["dimensions"]["time"]}
    geos = {g["id"]: g["name"] for g in data["dimensions"]["geo"]}

    # 旧图数据库结构
    old_graph = {
        "nodes": [],
        "edges": [],
    }

    # Person / Geo 节点
    for pid, name in persons.items():
        old_graph["nodes"].append(
            {"id": pid, "type": "Person", "name": name}
        )

    for gid, name in geos.items():
        old_graph["nodes"].append(
            {"id": gid, "type": "Geo", "name": name}
        )

    # 每条 fact -> 一条 Person -> Geo 的边，year 作为属性
    for fact in data["facts"]:
        pid = fact["person"]
        tid = fact["time"]
        gid = fact["geo"]
        year = times[tid]

        old_graph["edges"].append(
            {
                "from": pid,
                "to": gid,
                "year": year,
                "fact_id": fact["id"],
            }
        )

    return old_graph


if __name__ == "__main__":
    old_graph = build_old_graph_from_json(INPUT_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(old_graph, f, ensure_ascii=False, indent=2)
    print(f"旧图数据库已写入: {OUTPUT_PATH}")
