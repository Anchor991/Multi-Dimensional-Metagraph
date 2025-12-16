# visualize_3d.py
# 读取 build_metagraph.py 生成的 graph.json，画 3D 立方体 + 有向边 + 图例

import json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  只为注册 3D 投影
from matplotlib.lines import Line2D


# ====== 1. 字体设置：解决中文不显示 / 方块问题 ======
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",  # Windows 常见
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False


# ====== 2. 读取 JSON 数据 ======
def load_all(path: str = "graph.json"):
    """
    从 build_metagraph.py 生成的 json 中读取：
    - dimensions: person / time / geo
    - facts: 每一句话对应一个 fact
    - graph: 节点和有向边
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dims = data["dimensions"]
    facts = data["facts"]
    graph = data["graph"]
    return dims, facts, graph


# ====== 3. 为每个节点分配三维坐标 ======
def build_coordinates(dims, facts):
    """
    为 Person / Time / Geo / Fact 分配 3D 坐标：
      - Person 节点：在 X 轴上 (i, 0, 0)
      - Time 节点：  在 Y 轴上 (0, j, 0)
      - Geo 节点：   在 Z 轴上 (0, 0, k)
      - Fact 节点：  在立方体内部 (i, j, k)
    """
    persons = sorted(dims["person"], key=lambda x: x["id"])
    times = sorted(dims["time"], key=lambda x: int(x["value"]))
    geos = sorted(dims["geo"], key=lambda x: x["id"])

    person_index = {p["id"]: idx for idx, p in enumerate(persons)}
    time_index = {t["id"]: idx for idx, t in enumerate(times)}
    geo_index = {g["id"]: idx for idx, g in enumerate(geos)}

    pos = {}

    # 三个维度节点放在三条轴上
    for p in persons:
        i = person_index[p["id"]]
        pos[p["id"]] = (i, 0, 0)

    for t in times:
        j = time_index[t["id"]]
        pos[t["id"]] = (0, j, 0)

    for g in geos:
        k = geo_index[g["id"]]
        pos[g["id"]] = (0, 0, k)

    # Fact 节点放在立方体内部
    for fact in facts:
        pid = fact["person"]
        tid = fact["time"]
        gid = fact["geo"]
        x = person_index[pid]
        y = time_index[tid]
        z = geo_index[gid]
        pos[fact["id"]] = (x, y, z)

    return pos, persons, times, geos


# ====== 4. 主可视化函数 ======
def visualize_3d(
    path: str = "graph.json",
    save_path: str | None = "graph_3d_edges.png",
):
    dims, facts, graph = load_all(path)
    pos, persons, times, geos = build_coordinates(dims, facts)

    node_attrs = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    # 画布稍微大一点，给右侧和底部留空间放图例
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 视角稍微调一下，让三根轴都清楚
    ax.view_init(elev=25, azim=-60)

    # ------- 4.1 画节点（不同类型不同 marker），同时保留句柄用于图例 -------
    type_to_nodes = {"Person": [], "Time": [], "Geo": [], "Fact": [], "Other": []}
    for nid, attrs in node_attrs.items():
        if nid not in pos:
            continue
        t = attrs.get("type", "Other")
        type_to_nodes.setdefault(t, []).append(nid)

    def scatter_nodes(node_ids, marker: str, size: float):
        xs, ys, zs = [], [], []
        for nid in node_ids:
            if nid not in pos:
                continue
            x, y, z = pos[nid]
            xs.append(x)
            ys.append(y)
            zs.append(z)
        if xs:
            # 不指定颜色，使用 matplotlib 默认颜色
            sc = ax.scatter(xs, ys, zs, marker=marker, s=size)
            return sc
        return None

    h_person = scatter_nodes(type_to_nodes["Person"], marker="o", size=80)  # 人物
    h_time = scatter_nodes(type_to_nodes["Time"], marker="^", size=80)      # 时间
    h_geo = scatter_nodes(type_to_nodes["Geo"], marker="s", size=80)        # 地点
    h_fact = scatter_nodes(type_to_nodes["Fact"], marker="x", size=40)      # 事实

    # ------- 4.2 节点标签 -------
    # 只给 Fact 打标签，避免和坐标轴刻度文字堆在一起
    for nid in type_to_nodes["Fact"]:
        if nid not in pos:
            continue
        x, y, z = pos[nid]
        # 做一点小偏移，避免文本压在点正中心
        ax.text(x + 0.05, y + 0.05, z + 0.05, nid, fontsize=8)

    # ------- 4.3 画有向边：Fact -> Person/Time/Geo -------
    for e in edges:
        u = e["from"]
        v = e["to"]
        if u not in pos or v not in pos:
            continue
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        ax.quiver(
            x0,
            y0,
            z0,
            dx,
            dy,
            dz,
            arrow_length_ratio=0.15,
            linewidth=0.8,
        )

    # ------- 4.4 画三条坐标轴 + 刻度标签 -------
    max_x = len(persons) - 1 if persons else 0
    max_y = len(times) - 1 if times else 0
    max_z = len(geos) - 1 if geos else 0

    # 手动画三条“轴线”，强调这是三维坐标
    ax.plot([0, max_x], [0, 0], [0, 0])  # X: Person
    ax.plot([0, 0], [0, max_y], [0, 0])  # Y: Time
    ax.plot([0, 0], [0, 0], [0, max_z])  # Z: Geo

    ax.set_xlabel("Person")
    ax.set_ylabel("Time")
    ax.set_zlabel("Geo")

    # 坐标轴刻度 & 中文标签
    ax.set_xticks(range(len(persons)))
    ax.set_yticks(range(len(times)))
    ax.set_zticks(range(len(geos)))

    ax.set_xticklabels([p["name"] for p in persons], rotation=45, ha="right")
    ax.set_yticklabels([t["value"] for t in times])
    ax.set_zticklabels([g["name"] for g in geos])

    # 给立方体留一点边距
    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_ylim(-0.5, max_y + 0.5)
    ax.set_zlim(-0.5, max_z + 0.5)

    # ------- 4.5 图例（Legend） -------
    legend_handles = []
    legend_labels = []

    if h_person is not None:
        legend_handles.append(h_person)
        legend_labels.append("Person 节点")
    if h_time is not None:
        legend_handles.append(h_time)
        legend_labels.append("Time 节点")
    if h_geo is not None:
        legend_handles.append(h_geo)
        legend_labels.append("Geo 节点")
    if h_fact is not None:
        legend_handles.append(h_fact)
        legend_labels.append("Fact（事件）")

    # 边的图例：用一条简单的线表示
    edge_handle = Line2D([0], [0], linestyle="-", linewidth=1)
    legend_handles.append(edge_handle)
    legend_labels.append("有向边")

    # 图例放在右侧空白处
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    # 手动调整边距，给右侧图例留空间
    plt.subplots_adjust(left=0.15, right=0.78, bottom=0.25, top=0.93)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"3D 图谱已保存到：{save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    visualize_3d("graph.json", save_path="visualize_graph.png")
