#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_olddb_from_graph_with_ff_links.py

Build a baseline "traditional" property-graph dataset (PGDB) from the enriched
MMDB dataset that already contains fact-fact links (FF_PERSON_NEXT / FF_TIME_NEXT / FF_GEO_NEXT).

IMPORTANT:
- This builder intentionally ignores ALL fact-fact links.
- It only uses the canonical facts (person, time, geo) triples.
- Output format matches old_graph.json used in your project:
    {
      "nodes": [{"id": "...", "type": "Person|Geo", "name": "..."}, ...],
      "edges": [{"from": "P_xxxx", "to": "G_xxxx", "year": "2020", "fact_id": "F_xxxx"}, ...]
    }

Usage:
  python build_olddb_from_graph_with_ff_links.py ^
      --in graph_with_ff_links_by_person_time_geo.json ^
      --out old_graph_from_ff_links.json
"""

import argparse
import json
from typing import Dict, Any, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_pgdb(mmdb: Dict[str, Any]) -> Dict[str, Any]:
    dims = mmdb.get("dimensions", {})
    facts = mmdb.get("facts", [])

    # Build ID->name/value maps from dimensions (source of truth)
    person_id2name = {p["id"]: p["name"] for p in dims.get("person", [])}
    geo_id2name = {g["id"]: g["name"] for g in dims.get("geo", [])}
    time_id2value = {t["id"]: str(t["value"]) for t in dims.get("time", [])}

    # Nodes: keep Person and Geo only (baseline PGDB)
    nodes: List[Dict[str, str]] = []
    for pid, pname in person_id2name.items():
        nodes.append({"id": pid, "type": "Person", "name": pname})
    for gid, gname in geo_id2name.items():
        nodes.append({"id": gid, "type": "Geo", "name": gname})

    # Edges: one edge per fact (preserve multiplicity via fact_id)
    edges: List[Dict[str, str]] = []
    for fact in facts:
        fid = fact.get("id")
        pid = fact.get("person")
        tid = fact.get("time")
        gid = fact.get("geo")

        if not (fid and pid and tid and gid):
            # Skip malformed records
            continue

        year = time_id2value.get(tid)
        if year is None:
            # Fallback: if time dimension missing, keep raw tid
            year = str(tid)

        # Preserve duplicates by keeping every fact as its own edge.
        edges.append({"from": pid, "to": gid, "year": str(year), "fact_id": fid})

    return {"nodes": nodes, "edges": edges}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default="graph_with_ff_links_by_person_time_geo.json",
        help="Input MMDB file (with FF_* links).",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="old_graph_from_ff_links.json",
        help="Output PGDB JSON file.",
    )
    args = ap.parse_args()

    mmdb = load_json(args.in_path)
    pgdb = build_pgdb(mmdb)
    save_json(pgdb, args.out_path)

    print(f"[OK] Wrote PGDB to: {args.out_path}")
    print(f"     Nodes: {len(pgdb['nodes'])}  Edges: {len(pgdb['edges'])}")


if __name__ == "__main__":
    main()
