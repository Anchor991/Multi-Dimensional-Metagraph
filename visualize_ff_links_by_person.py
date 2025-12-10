import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

INPUT_JSON = "graph_with_ff_links_by_person.json"  # 改成你的文件名
FACT_EDGE_LABELS = {"next_event"}           # 事件间边 label
OUTPUT_PNG = "visualize_ff_links_by_person.png"

MAX_FACT_LABELS = 40
LABEL_ONLY_ON_EDGES = True
LABEL_OFFSET = (0.06, 0.06, 0.06)


def pick_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def safe_int_year(y):
    try:
        return int(str(y))
    except Exception:
        return 0


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    cn_font = pick_chinese_font()
    if cn_font:
        plt.rcParams["font.sans-serif"] = [cn_font]
        plt.rcParams["axes.unicode_minus"] = False

    persons = data["dimensions"]["person"]
    times = data["dimensions"]["time"]
    geos = data["dimensions"]["geo"]
    facts = data["facts"]
    edges = data["graph"]["edges"]

    # sort dims for stable axis order
    persons_sorted = sorted(persons, key=lambda x: x["name"])
    geos_sorted = sorted(geos, key=lambda x: x["name"])
    times_sorted = sorted(times, key=lambda x: safe_int_year(x["value"]))

    person_id2idx = {p["id"]: i for i, p in enumerate(persons_sorted)}
    geo_id2idx = {g["id"]: i for i, g in enumerate(geos_sorted)}
    time_id2year = {t["id"]: safe_int_year(t["value"]) for t in times_sorted}

    person_idx2name = {i: p["name"] for i, p in enumerate(persons_sorted)}
    geo_idx2name = {i: g["name"] for i, g in enumerate(geos_sorted)}
    years = sorted(list(dict.fromkeys([safe_int_year(t["value"]) for t in times_sorted])))

    # Fact positions: (person_idx, year, geo_idx) + jitter
    random.seed(42)
    np.random.seed(42)

    jitter_x = 0.12
    jitter_y = 0.10
    jitter_z = 0.12

    pos_fact = {}
    for fct in facts:
        fid = fct["id"]
        px = person_id2idx.get(fct["person"], 0)
        py = time_id2year.get(fct["time"], years[0] if years else 0)
        pz = geo_id2idx.get(fct["geo"], 0)

        jx = (random.random() - 0.5) * 2 * jitter_x
        jy = (random.random() - 0.5) * 2 * jitter_y
        jz = (random.random() - 0.5) * 2 * jitter_z

        pos_fact[fid] = (px + jx, py + jy, pz + jz)

    # Filter Fact->Fact edges only
    fact_ids = set(pos_fact.keys())
    ff_edges = []
    for e in edges:
        u, v = e.get("from"), e.get("to")
        lab = e.get("label", "")
        if lab in FACT_EDGE_LABELS and u in fact_ids and v in fact_ids:
            ff_edges.append((u, v, lab))

    print(f"Fact nodes: {len(fact_ids)}")
    print(f"Fact->Fact edges: {len(ff_edges)}")
    if not ff_edges:
        print("未找到 Fact->Fact 边。请检查 label 或 from/to 是否为 F_****。")
        return

    # Decide which facts to label
    facts_on_edges = set()
    for u, v, _ in ff_edges:
        facts_on_edges.add(u)
        facts_on_edges.add(v)

    if LABEL_ONLY_ON_EDGES:
        label_candidates = sorted(list(facts_on_edges))
    else:
        label_candidates = sorted(list(fact_ids))

    label_candidates = label_candidates[:MAX_FACT_LABELS]

    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Only draw Fact nodes
    ax.scatter(
        [pos_fact[f["id"]][0] for f in facts],
        [pos_fact[f["id"]][1] for f in facts],
        [pos_fact[f["id"]][2] for f in facts],
        marker="x", s=90, label="Fact（事件）"
    )

    # Fact labels
    ox, oy, oz = LABEL_OFFSET
    for fid in label_candidates:
        x, y, z = pos_fact[fid]
        ax.text(x + ox, y + oy, z + oz, fid, fontsize=9)

    # Axes & ticks (keep the same semantic axes)
    ax.set_xlabel("Person")
    ax.set_ylabel("Time")
    ax.set_zlabel("Geo")

    ax.set_xticks(list(person_idx2name.keys()))
    ax.set_xticklabels([person_idx2name[i] for i in person_idx2name.keys()], rotation=45, ha="right")

    ax.set_yticks(years)
    ax.set_yticklabels([str(y) for y in years])

    ax.set_zticks(list(geo_idx2name.keys()))
    ax.set_zticklabels([geo_idx2name[i] for i in geo_idx2name.keys()])

    # Draw only Fact->Fact directed edges
    for (u, v, lab) in ff_edges:
        x0, y0, z0 = pos_fact[u]
        x1, y1, z1 = pos_fact[v]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

        ax.quiver(
            x0, y0, z0,
            dx, dy, dz,
            arrow_length_ratio=0.15,
            linewidth=1.2
        )

    # Legend: add proxy for edges
    edge_proxy = Line2D([0], [0], linewidth=2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(edge_proxy)
    labels.append("事件间有向边 (Fact→Fact)")
    ax.legend(handles, labels, loc="upper right")

    ax.set_title("3D 多维轴（无维度节点）— 事件节点 + 事件间指向")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220)
    plt.show()
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
