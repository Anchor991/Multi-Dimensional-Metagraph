import re
import json
from typing import Dict, List

# ========= 1. 你要处理的句子：在这里改即可 =========
sentences = [
    "小明在2020年去了北京。",
    "小明在2021年去了上海。",
    "小明在2023年又去了北京。",
    "小红在2020年去了上海。",
    "小红在2021年去了深圳。",
    "小李在2020年去了武汉。",
    "小李在2021年去了北京。",
    "小王在2020年去了成都。",
    "小王在2021年去了西安。",
    "小张在2021年去了广州。",
    "小张在2022年去了北京。",
]

# ========= 2. 简单句式的正则 =========
# 支持：
#   小明在2020年去了北京。
#   小明2020年去北京。
#   小明在2020年又去了北京。
pattern = re.compile(
    r'(?P<person>[\u4e00-\u9fa5]+?)(在)?\s*'  # 人名（非贪婪）+ 可选“在”
    r'(?P<time>\d{4})年'                     # 4 位年份
    r'.*?'                                   # 中间可以有“又”等
    r'去(了)?'                               # “去”或“去了”
    r'(?P<location>[\u4e00-\u9fa5]+)'        # 地点
)

def get_or_create_id(mapping: Dict[str, str], key: str, prefix: str) -> str:
    """
    给某个实体值（如 '小明'）分配一个唯一 ID。
    同一个值多次出现会复用同一个 ID。
    """
    if key in mapping:
        return mapping[key]
    new_id = f"{prefix}_{len(mapping) + 1:04d}"
    mapping[key] = new_id
    return new_id

def build_metagraph(sentences: List[str]) -> Dict:
    # 三个维度：人物 / 时间 / 地理
    person_map: Dict[str, str] = {}
    time_map: Dict[str, str] = {}
    geo_map: Dict[str, str] = {}

    # 图结构
    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    # 超立方体中的事实（Object1, Object2 ...）
    facts: List[Dict] = []
    fact_counter = 0

    for s in sentences:
        m = pattern.search(s)
        if not m:
            print(f"无法解析句子，已跳过：{s}")
            continue

        person = m.group("person")
        year = m.group("time")
        loc = m.group("location")

        # 为三个维度元素分配 GUID
        pid = get_or_create_id(person_map, person, "P")
        tid = get_or_create_id(time_map, year, "T")
        gid = get_or_create_id(geo_map, loc, "G")

        # 维度节点（只创建一次）
        if pid not in nodes:
            nodes[pid] = {"id": pid, "type": "Person", "label": person}
        if tid not in nodes:
            nodes[tid] = {"id": tid, "type": "Time", "label": year}
        if gid not in nodes:
            nodes[gid] = {"id": gid, "type": "Geo", "label": loc}

        # 创建一个新的 fact（Object1）
        fact_counter += 1
        fid = f"F_{fact_counter:04d}"

        # fact 在超立方体里的表示：指向三个维度元素
        facts.append({
            "id": fid,
            "person": pid,
            "time": tid,
            "geo": gid,
            "text": s,     # 原始句子，作为“measure”的简化版本
        })

        # 在图里，也把 fact 当成一个节点
        nodes[fid] = {"id": fid, "type": "Fact"}

        # 有向边：Fact → Person/Time/Geo
        edges.extend([
            {"from": fid, "to": pid, "label": "has_person"},
            {"from": fid, "to": tid, "label": "has_time"},
            {"from": fid, "to": gid, "label": "has_geo"},
        ])

    # 组织成最终的 JSON 结构
    graph_json = {
        "dimensions": {
            "person": [
                {"id": pid, "name": name}
                for name, pid in person_map.items()
            ],
            "time": [
                {"id": tid, "value": year}
                for year, tid in time_map.items()
            ],
            "geo": [
                {"id": gid, "name": loc}
                for loc, gid in geo_map.items()
            ],
        },
        "facts": facts,
        "graph": {
            "nodes": list(nodes.values()),
            "edges": edges,
        }
    }
    return graph_json

if __name__ == "__main__":
    graph = build_metagraph(sentences)

    # 写入 JSON 文件
    output_file = "graph.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"知识图谱 / 超立方体数据已写入：{output_file}")
