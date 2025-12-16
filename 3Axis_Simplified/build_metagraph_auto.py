import re
import json
import random
from typing import Dict, List


# ========= 1. 自动生成语料 =========

def generate_sentences(num_sentences: int = 200) -> List[str]:
    """
    随机生成若干句子，形式为：
      某某在YYYY年去了某地。
    保证符合原本的正则解析格式。
    """
    random.seed(42)  # 固定随机种子，便于复现实验

    persons = ["小明", "小红", "小李", "小王", "小张",
               "小赵", "小刘", "小陈", "小杨", "小周"]

    years = [2020, 2021, 2022, 2023, 2024]

    cities = ["北京", "上海", "广州", "深圳", "杭州",
              "武汉", "重庆", "成都", "西安", "海口",
              "南京", "苏州", "天津", "青岛"]

    sentences: List[str] = []
    for _ in range(num_sentences):
        p = random.choice(persons)
        y = random.choice(years)
        c = random.choice(cities)
        s = f"{p}在{y}年去了{c}。"
        sentences.append(s)

    return sentences


# ========= 2. 抽取用的正则（保持不变） =========

pattern = re.compile(
    r'(?P<person>[\u4e00-\u9fa5]+?)(在)?\s*'  # 人名（非贪婪）+ 可选“在”
    r'(?P<time>\d{4})年'                     # 4 位年份
    r'.*?'                                   # 中间可以有“又”等
    r'去(了)?'                               # “去”或“去了”
    r'(?P<location>[\u4e00-\u9fa5]+)'        # 地点
)


def get_or_create_id(mapping: Dict[str, str], key: str, prefix: str) -> str:
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
            "text": s,
        })

        # 在图里，也把 fact 当成一个节点
        nodes[fid] = {"id": fid, "type": "Fact"}

        # 有向边：Fact → Person/Time/Geo
        edges.extend([
            {"from": fid, "to": pid, "label": "has_person"},
            {"from": fid, "to": tid, "label": "has_time"},
            {"from": fid, "to": gid, "label": "has_geo"},
        ])

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
    # 生成 200 条句子（可以改成 500 / 1000）
    sentences = generate_sentences(num_sentences=200)
    graph = build_metagraph(sentences)

    output_file = "graph.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"生成句子数量: {len(sentences)}")
    print(f"知识图谱 / 超立方体数据已写入：{output_file}")
