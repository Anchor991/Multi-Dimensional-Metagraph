import json
from collections import defaultdict

INPUT = "graph.json"
OUTPUT = "graph_with_ff_links_by_person.json"

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    # time id -> year(string) -> int
    time_id2year = {t["id"]: int(t["value"]) for t in data["dimensions"]["time"]}

    # 按 person 分组收集 facts
    facts_by_person = defaultdict(list)
    for fact in data["facts"]:
        pid = fact["person"]
        year = time_id2year.get(fact["time"], 0)
        facts_by_person[pid].append((year, fact["id"]))

    # 在 graph.edges 里新增 Fact->Fact 的边
    edges = data["graph"]["edges"]
    added = 0

    for pid, items in facts_by_person.items():
        items.sort(key=lambda x: x[0])  # 按年份排序
        for i in range(len(items) - 1):
            f1 = items[i][1]
            f2 = items[i + 1][1]
            edges.append({"from": f1, "to": f2, "label": "next_event"})
            added += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Added {added} Fact->Fact edges. Saved to {OUTPUT}")

if __name__ == "__main__":
    main()
