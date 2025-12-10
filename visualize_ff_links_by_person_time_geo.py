import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

INPUT_JSON = "graph_with_ff_links_by_person_time_geo.json"
OUTPUT_PNG = "visualize_ff_links_by_person_time_geo.png"

# 三类事件边 label
LABEL_PERSON = "FF_PERSON_NEXT"
LABEL_TIME = "FF_TIME_NEXT"
LABEL_GEO = "FF_GEO_NEXT"

# 颜色映射（你可改）
EDGE_STYLE = {
    LABEL_PERSON: {"color": "red",   "legend": "事件边-按人物 (Person chain)"},
    LABEL_TIME:   {"color": "blue",  "legend": "事件边-按时间 (Time chain)"},
    LABEL_GEO:    {"color": "green", "legend": "事件边-按地点 (Geo chain)"},
}

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

    # axis orders
    persons_sorted = sorted(persons, key=lambda x: x["name"])
    geos_sorted = sorted(geos, key=lambda x: x["name"])
    times_sorted = sorted(times, key=lambda x: safe_int_year(x["value"]))

    person_id2idx = {p["id"]: i for i, p in enumerate(persons_sorted)}
    geo_id2idx = {g["id"]: i for i, g in enumerate(geos_sorted)}
    time_id2year = {t["id"]: safe_int_year(t["value"]) for t in times_sorted}

    person_idx2name = {i: p["name"] for i, p in enumerate(persons_sorted)}
    geo_idx2name = {i: g["name"] for i, g in enumerate(geos_sorted)}
    years = sorted(list(dict.fromkeys([safe_int_year(t["value"]) for t in times_sorted])))

    # Fact positions (semantic) + jitter
    random.seed(42)
    np.random.seed(42)
    jitter_x, jitter_y, jitter_z = 0.12, 0.10, 0.12

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

    fact_ids = set(pos_fact.keys())

    # Collect Fact->Fact edges by type
    ff_edges_by_label = {LABEL_PERSON: [], LABEL_TIME: [], LABEL_GEO: []}
    for e in edges:
        u, v, lab = e.get("from"), e.get("to"), e.get("label", "")
        if u in fact_ids and v in fact_ids and lab in ff_edges_by_label:
            ff_edges_by_label[lab].append((u, v))

    total_ff = sum(len(v) for v in ff_edges_by_label.values())
    print("Fact nodes:", len(fact_ids))
    print("Fact->Fact edges (person/time/geo):",
          {k: len(v) for k, v in ff_edges_by_label.items()},
          "total=", total_ff)

    if total_ff == 0:
        print("未找到三类 Fact->Fact 边。请确认已运行 add_ff_links_by_person_time_geo.py")
        return

    # Facts to label
    facts_on_edges = set()
    for lab, pairs in ff_edges_by_label.items():
        for u, v in pairs:
            facts_on_edges.add(u)
            facts_on_edges.add(v)

    label_candidates = sorted(list(facts_on_edges if LABEL_ONLY_ON_EDGES else fact_ids))[:MAX_FACT_LABELS]

    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Fact nodes only
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

    # Axes & ticks
    ax.set_xlabel("Person")
    ax.set_ylabel("Time")
    ax.set_zlabel("Geo")

    ax.set_xticks(list(person_idx2name.keys()))
    ax.set_xticklabels([person_idx2name[i] for i in person_idx2name.keys()], rotation=45, ha="right")

    ax.set_yticks(years)
    ax.set_yticklabels([str(y) for y in years])

    ax.set_zticks(list(geo_idx2name.keys()))
    ax.set_zticklabels([geo_idx2name[i] for i in geo_idx2name.keys()])

    # Draw colored edges:
    # Use ax.plot for colored line, and a tiny quiver at the end for direction.
    for lab, pairs in ff_edges_by_label.items():
        style = EDGE_STYLE[lab]
        color = style["color"]
        for u, v in pairs:
            x0, y0, z0 = pos_fact[u]
            x1, y1, z1 = pos_fact[v]

            # line segment
            ax.plot([x0, x1], [y0, y1], [z0, z1], color=color, linewidth=1.4)

            # direction arrow head (short quiver near end)
            # Put arrow starting at 85% of the segment, pointing to end
            sx = x0 + 0.85 * (x1 - x0)
            sy = y0 + 0.85 * (y1 - y0)
            sz = z0 + 0.85 * (z1 - z0)
            dx = 0.15 * (x1 - x0)
            dy = 0.15 * (y1 - y0)
            dz = 0.15 * (z1 - z0)

            ax.quiver(
                sx, sy, sz,
                dx, dy, dz,
                arrow_length_ratio=0.6,
                linewidth=1.2,
                color=color,
            )

    # Legend: proxies (because 3D plot/quiver don't always legend well)
    legend_handles = [Line2D([0], [0], marker="x", linestyle="None", markersize=9, label="Fact（事件）")]
    for lab, style in EDGE_STYLE.items():
        legend_handles.append(Line2D([0], [0], color=style["color"], linewidth=2.5, label=style["legend"]))

    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title("彩色事件间有向边：按人物/按时间/按地点）")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220)
    plt.show()
    print("Saved:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
