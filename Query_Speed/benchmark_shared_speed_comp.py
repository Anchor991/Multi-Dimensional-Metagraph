# -*- coding: utf-8 -*-
"""
benchmark_shared_speed_comp.py

simplified to compare ONLY:
  - MMDB(axis + composite)
  - PGDB(indexed + composite)

Output:
  - Prints len(inputs) and calls/repeat for A/B/C.
  - Runs workloads A/B/C with batch-timing + optional baseline subtraction.
  - Prints a 2-column summary per workload plus a final speedup table.

Cell meaning:
  - per-call mean latency (µs) and QPS (M/s)
  - Speedup = PGDB_mean_ms / MMDB_mean_ms. >1 means MMDB faster.

"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


# =============================================================================
# USER CONFIG (edit here)
# =============================================================================

MMDB_PATH = "graph.json"
PGDB_PATH = "old_graph.json"

SEED = 13

# Workloads
RUN_A = True   # single-hop shared queries
RUN_B = True   # intersection person ∩ year ∩ city
RUN_C = True   # batch queries

# Timing controls
WARMUP_ROUNDS = 50
MEASURE_ROUNDS = 200
REPEATS = 10

# Pool sizes (upper bounds; actual len(inputs) may be smaller)
POOL_SINGLE = 1000
POOL_INTERS = 2000
POOL_BATCH = 500

# Baseline subtraction (empty-loop overhead)
SUBTRACT_BASELINE = True

# Verbose per-repeat raw timing
PRINT_PROGRESS = True

# Avoid duplicates in intersection sampling
UNIQUE_TRIPLES_ONLY = True


# =============================================================================
# Helpers
# =============================================================================

def now_ns() -> int:
    return time.perf_counter_ns()

def percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def time_whole_loop(func, inputs: List[Any], rounds: int) -> int:
    t0 = now_ns()
    for _ in range(rounds):
        for x in inputs:
            func(x)
    t1 = now_ns()
    return t1 - t0

def measure_per_call(func, inputs: List[Any], warmup_rounds: int, measure_rounds: int, repeats: int, tag: str) -> Tuple[List[float], List[float]]:
    assert inputs, "empty inputs"
    calls = len(inputs) * measure_rounds

    # Warmup
    _ = time_whole_loop(func, inputs, warmup_rounds)

    # Baseline (loop overhead)
    def baseline(_: Any) -> None:
        return None

    _ = time_whole_loop(baseline, inputs, warmup_rounds)

    per_call_ms: List[float] = []
    qps: List[float] = []

    for r in range(repeats):
        base_ns = time_whole_loop(baseline, inputs, measure_rounds)
        work_ns = time_whole_loop(func, inputs, measure_rounds)
        net_ns = work_ns
        if SUBTRACT_BASELINE:
            net_ns = max(0, work_ns - base_ns)

        ms = (net_ns / calls) / 1e6
        per_call_ms.append(ms)

        secs = net_ns / 1e9
        qps.append((calls / secs) if secs > 0 else float("inf"))

        if PRINT_PROGRESS:
            print(f"  [{tag}] repeat {r+1}/{repeats}: base={base_ns/1e6:.2f} ms, work={work_ns/1e6:.2f} ms, net={net_ns/1e6:.2f} ms")

    return per_call_ms, qps

def summarize(per_call_ms: List[float], qps: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": statistics.mean(per_call_ms),
        "p50_ms": percentile(per_call_ms, 0.50),
        "p95_ms": percentile(per_call_ms, 0.95),
        "mean_qps": statistics.mean(qps),
        "p50_qps": percentile(qps, 0.50),
        "p95_qps": percentile(qps, 0.95),
    }

def fmt(stat: Dict[str, float]) -> str:
    mean_us = stat["mean_ms"] * 1000.0
    qps_m = stat["mean_qps"] / 1e6
    return f"mean={mean_us:.3f} µs | QPS={qps_m:.3f} M/s"

def print_inputs_calls(label: str, n_inputs: int) -> None:
    calls = n_inputs * MEASURE_ROUNDS
    print(f"[Inputs] {label}: len(inputs)={n_inputs}, MEASURE_ROUNDS={MEASURE_ROUNDS}, calls/repeat={calls}")


# =============================================================================
# Backends
# =============================================================================

class MMDB:
    """
    axis indices + composite:
      - axis inverted indices for completeness
      - 2D materializations for single-hop shared queries
      - 3D composite (person,year,city)->exists/count
    """
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dims = data["dimensions"]
        person_by_id = {p["id"]: p["name"] for p in dims["person"]}
        year_by_id = {t["id"]: str(t["value"]) for t in dims["time"]}
        city_by_id = {g["id"]: g["name"] for g in dims["geo"]}

        self.fact_triple: Dict[str, Tuple[str, str, str]] = {}
        for fact in data["facts"]:
            fid = fact["id"]
            p = person_by_id[fact["person"]]
            y = year_by_id[fact["time"]]
            c = city_by_id[fact["geo"]]
            self.fact_triple[fid] = (p, y, c)

        tmp_cities: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        tmp_persons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        self.facts_by_person: Dict[str, Set[str]] = defaultdict(set)
        self.facts_by_year: Dict[str, Set[str]] = defaultdict(set)
        self.facts_by_city: Dict[str, Set[str]] = defaultdict(set)

        self.comp_exists: Set[Tuple[str, str, str]] = set()
        self.comp_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

        for fid, (p, y, c) in self.fact_triple.items():
            tmp_cities[(p, y)].add(c)
            tmp_persons[(c, y)].add(p)

            self.facts_by_person[p].add(fid)
            self.facts_by_year[y].add(fid)
            self.facts_by_city[c].add(fid)

            self.comp_exists.add((p, y, c))
            self.comp_count[(p, y, c)] += 1

        self.cities_by_person_year = {k: tuple(sorted(v)) for k, v in tmp_cities.items()}
        self.persons_by_city_year = {k: tuple(sorted(v)) for k, v in tmp_persons.items()}

        self.all_people = sorted(self.facts_by_person.keys())
        self.all_years = sorted(self.facts_by_year.keys())
        self.all_cities = sorted(self.facts_by_city.keys())
        self.unique_triples = sorted({v for v in self.fact_triple.values()})

    def cities_by_person_year_q(self, person: str, year: str) -> Tuple[str, ...]:
        return self.cities_by_person_year.get((person, year), ())

    def persons_by_city_year_q(self, city: str, year: str) -> Tuple[str, ...]:
        return self.persons_by_city_year.get((city, year), ())

    def exists_q(self, person: str, year: str, city: str) -> bool:
        return (person, year, city) in self.comp_exists

    def count_q(self, person: str, year: str, city: str) -> int:
        return int(self.comp_count.get((person, year, city), 0))


class PGDB:
    """
    indexed + composite:
      - 2D indices (person,year)->cities ; (city,year)->persons
      - 3D composite (person,year,city)->exists/count
    """
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        node = {n["id"]: n for n in data.get("nodes", [])}
        edges = data.get("edges", [])

        rows: List[Tuple[str, str, str]] = []
        for e in edges:
            fr = node.get(e["from"])
            to = node.get(e["to"])
            if not fr or not to:
                continue
            if fr.get("type") == "Person" and to.get("type") == "Geo":
                rows.append((fr.get("name"), str(e.get("year")), to.get("name")))

        tmp_cities: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        tmp_persons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        self.comp_exists: Set[Tuple[str, str, str]] = set()
        self.comp_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

        for p, y, c in rows:
            tmp_cities[(p, y)].add(c)
            tmp_persons[(c, y)].add(p)
            self.comp_exists.add((p, y, c))
            self.comp_count[(p, y, c)] += 1

        self.idx_cities_by_person_year = {k: tuple(sorted(v)) for k, v in tmp_cities.items()}
        self.idx_persons_by_city_year = {k: tuple(sorted(v)) for k, v in tmp_persons.items()}

    def cities_by_person_year_q(self, person: str, year: str) -> Tuple[str, ...]:
        return self.idx_cities_by_person_year.get((person, year), ())

    def persons_by_city_year_q(self, city: str, year: str) -> Tuple[str, ...]:
        return self.idx_persons_by_city_year.get((city, year), ())

    def exists_q(self, person: str, year: str, city: str) -> bool:
        return (person, year, city) in self.comp_exists

    def count_q(self, person: str, year: str, city: str) -> int:
        return int(self.comp_count.get((person, year, city), 0))


# =============================================================================
# Sampling
# =============================================================================

def sample_single(mmdb: MMDB, rng: random.Random, n: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    py = list(mmdb.cities_by_person_year.keys())
    cy = list(mmdb.persons_by_city_year.keys())
    rng.shuffle(py)
    rng.shuffle(cy)
    return py[:n], cy[:n]

def sample_intersection(mmdb: MMDB, rng: random.Random, n: int, unique_only: bool) -> List[Tuple[str, str, str]]:
    triples = mmdb.unique_triples[:] if unique_only else list(mmdb.fact_triple.values())
    rng.shuffle(triples)

    out: List[Tuple[str, str, str]] = []
    pos_n = n // 2
    out.extend(triples[:pos_n])

    seen = set(mmdb.unique_triples)
    people, years, cities = mmdb.all_people, mmdb.all_years, mmdb.all_cities

    while len(out) < n:
        p, y, c = rng.choice(triples[:pos_n])
        r = rng.random()
        if r < 0.34:
            p = rng.choice(people)
        elif r < 0.67:
            y = rng.choice(years)
        else:
            c = rng.choice(cities)
        if (p, y, c) in seen:
            continue
        out.append((p, y, c))

    rng.shuffle(out)
    return out

def sample_batch(mmdb: MMDB, rng: random.Random, n: int) -> List[Tuple[str, Tuple[str, ...]]]:
    people = mmdb.all_people
    years = mmdb.all_years
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for _ in range(n):
        p = rng.choice(people)
        k = rng.randint(3, min(10, max(3, len(years))))
        ys = tuple(rng.sample(years, k=k))
        out.append((p, ys))
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    rng = random.Random(SEED)

    mm = MMDB(MMDB_PATH)
    pg = PGDB(PGDB_PATH)

    print("=" * 96)
    print("Shared-speed benchmark (comp: MMDB(axis+composite) vs PGDB(indexed+composite))")
    print(f"MMDB = {MMDB_PATH}")
    print(f"PGDB = {PGDB_PATH}")
    print(f"SUBTRACT_BASELINE={SUBTRACT_BASELINE}, PRINT_PROGRESS={PRINT_PROGRESS}")
    print(f"WARMUP_ROUNDS={WARMUP_ROUNDS}, MEASURE_ROUNDS={MEASURE_ROUNDS}, REPEATS={REPEATS}")
    print(f"POOL_SINGLE={POOL_SINGLE}, POOL_INTERS={POOL_INTERS}, POOL_BATCH={POOL_BATCH}, UNIQUE_TRIPLES_ONLY={UNIQUE_TRIPLES_ONLY}")

    py, cy = sample_single(mm, rng, POOL_SINGLE)
    triples = sample_intersection(mm, rng, POOL_INTERS, UNIQUE_TRIPLES_ONLY)
    batches = sample_batch(mm, rng, POOL_BATCH)

    print("\n" + "-" * 96)
    print_inputs_calls("A/cities_by_person_year", len(py))
    print_inputs_calls("A/persons_by_city_year", len(cy))
    print_inputs_calls("B/exists&count intersection", len(triples))
    print_inputs_calls("C/batch queries", len(batches))
    print("-" * 96)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    def run_pair(label: str, mm_func, pg_func, inputs: List[Any]) -> None:
        mm_ms, mm_qps = measure_per_call(mm_func, inputs, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, f"{label}/mm")
        pg_ms, pg_qps = measure_per_call(pg_func, inputs, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, f"{label}/pg")
        results[label] = {
            "MMDB(axis+comp)": summarize(mm_ms, mm_qps),
            "PGDB(indexed+comp)": summarize(pg_ms, pg_qps),
        }
        print(f"[{label}] MMDB: {fmt(results[label]['MMDB(axis+comp)'])}")
        print(f"[{label}] PGDB: {fmt(results[label]['PGDB(indexed+comp)'])}")

    if RUN_A:
        print("\n" + "=" * 96)
        print("Workload A: Single-hop shared queries")
        run_pair(
            "A cities_by_person_year",
            lambda x: mm.cities_by_person_year_q(x[0], x[1]),
            lambda x: pg.cities_by_person_year_q(x[0], x[1]),
            py,
        )
        run_pair(
            "A persons_by_city_year",
            lambda x: mm.persons_by_city_year_q(x[0], x[1]),
            lambda x: pg.persons_by_city_year_q(x[0], x[1]),
            cy,
        )

    if RUN_B:
        print("\n" + "=" * 96)
        print("Workload B: Intersection person ∩ year ∩ city")
        run_pair(
            "B exists(person,year,city)",
            lambda t: mm.exists_q(t[0], t[1], t[2]),
            lambda t: pg.exists_q(t[0], t[1], t[2]),
            triples,
        )
        run_pair(
            "B count(person,year,city)",
            lambda t: mm.count_q(t[0], t[1], t[2]),
            lambda t: pg.count_q(t[0], t[1], t[2]),
            triples,
        )

    if RUN_C:
        print("\n" + "=" * 96)
        print("Workload C: Batch queries (one person, many years)")

        def mm_batch(one: Tuple[str, Tuple[str, ...]]) -> int:
            p, years = one
            s = 0
            for y in years:
                s += len(mm.cities_by_person_year_q(p, y))
            return s

        def pg_batch(one: Tuple[str, Tuple[str, ...]]) -> int:
            p, years = one
            s = 0
            for y in years:
                s += len(pg.cities_by_person_year_q(p, y))
            return s

        run_pair("C batch cities_by_person_year", mm_batch, pg_batch, batches)

    # Final speedup table
    print("\n" + "=" * 96)
    print("Speedup table (PGDB mean_ms / MMDB mean_ms). >1 means MMDB faster.")
    print("-" * 96)
    for label, d in results.items():
        mm_mean = d["MMDB(axis+comp)"]["mean_ms"]
        pg_mean = d["PGDB(indexed+comp)"]["mean_ms"]
        speedup = (pg_mean / mm_mean) if mm_mean > 0 else float("inf")
        print(f"{label:<30} | MMDB mean={mm_mean*1000:.3f} µs | PGDB mean={pg_mean*1000:.3f} µs | speedup={speedup:.2f}x")
    print("-" * 96)
    print("Done.")


if __name__ == "__main__":
    main()
