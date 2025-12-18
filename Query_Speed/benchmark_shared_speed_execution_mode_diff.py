# -*- coding: utf-8 -*-
"""
Goal:
- Benchmark 4 backends in ONE run and print a 4-column comparison table for screenshots:
    1) MMDB(axis)
    2) MMDB(axis + composite)
    3) PGDB(indexed)                 (2D indices only; no 3D composite)
    4) PGDB(indexed + composite)     (adds (person,year,city)->exists/count)

Notes:
- This is an in-memory Python operator microbenchmark. Results reflect data-structure choices and
  Python runtime constants, not a full database engine.
- "axis" here means the 3 axis inverted indices (person/year/city -> fact_ids) plus common 2D materializations
  for shared single-hop queries:
      (person,year)->cities and (city,year)->persons.

Paper-friendly output:
- Prints len(inputs) and calls/repeat for each workload.
- Prints one 4-column table with mean latency (µs) and QPS (M/s).

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

# Run which workloads?
RUN_A = True   # single-hop shared queries
RUN_B = True   # intersection person ∩ year ∩ city
RUN_C = True   # batch queries

# Timing controls (safe defaults; increase gradually)
WARMUP_ROUNDS = 50
MEASURE_ROUNDS = 200
REPEATS = 10

# Query pool sizes (upper bounds; actual len(inputs) may be smaller due to dataset size)
POOL_SINGLE = 1000
POOL_INTERS = 2000
POOL_BATCH = 500

# Baseline subtraction (empty-loop overhead)
SUBTRACT_BASELINE = True

# Print per-repeat raw timing (set False for clean screenshot output)
PRINT_PROGRESS = False

# Avoid duplicates in sampling intersection queries
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
    """Returns per_call_ms_list, qps_list."""
    assert inputs, "empty inputs"
    calls = len(inputs) * measure_rounds

    # Warmup
    _ = time_whole_loop(func, inputs, warmup_rounds)

    # Baseline: loop overhead
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
            print(f"  [{tag}] repeat {r+1}/{repeats}: base={base_ns/1e6:.2f} ms, work={work_ns/1e6:.2f} ms, net={(net_ns)/1e6:.2f} ms")

    return per_call_ms, qps

def summarize(per_call_ms: List[float], qps: List[float]) -> Dict[str, float]:
    mean_ms = statistics.mean(per_call_ms)
    p50_ms = percentile(per_call_ms, 0.50)
    p95_ms = percentile(per_call_ms, 0.95)
    mean_qps = statistics.mean(qps)
    p50_qps = percentile(qps, 0.50)
    p95_qps = percentile(qps, 0.95)
    return {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "mean_qps": mean_qps,
        "p50_qps": p50_qps,
        "p95_qps": p95_qps,
    }

def fmt_cell(stat: Dict[str, float]) -> str:
    mean_us = stat["mean_ms"] * 1000.0
    qps_m = stat["mean_qps"] / 1e6
    return f"{mean_us:7.3f} µs | {qps_m:7.3f} M/s"

def print_inputs_calls(label: str, n_inputs: int, measure_rounds: int) -> None:
    calls = n_inputs * measure_rounds
    print(f"[Inputs] {label}: len(inputs)={n_inputs}, MEASURE_ROUNDS={measure_rounds}, calls/repeat={calls}")


# =============================================================================
# Backends
# =============================================================================

class MMDB:
    """
    axis indices:
      - facts_by_person/person->set(fact_ids)
      - facts_by_year/year->set(fact_ids)
      - facts_by_city/city->set(fact_ids)
    plus common 2D materializations for shared queries:
      - cities_by_person_year[(p,y)]
      - persons_by_city_year[(c,y)]
    optional composite:
      - comp_exists[(p,y,c)]
      - comp_count[(p,y,c)]
    """

    def __init__(self, path: str, composite: bool) -> None:
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

        self.use_composite = composite
        self.comp_exists: Set[Tuple[str, str, str]] = set()
        self.comp_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

        for fid, (p, y, c) in self.fact_triple.items():
            tmp_cities[(p, y)].add(c)
            tmp_persons[(c, y)].add(p)

            self.facts_by_person[p].add(fid)
            self.facts_by_year[y].add(fid)
            self.facts_by_city[c].add(fid)

            if self.use_composite:
                self.comp_exists.add((p, y, c))
                self.comp_count[(p, y, c)] += 1

        self.cities_by_person_year = {k: tuple(sorted(v)) for k, v in tmp_cities.items()}
        self.persons_by_city_year = {k: tuple(sorted(v)) for k, v in tmp_persons.items()}

        self.all_people = sorted(self.facts_by_person.keys())
        self.all_years = sorted(self.facts_by_year.keys())
        self.all_cities = sorted(self.facts_by_city.keys())
        self.unique_triples = sorted({v for v in self.fact_triple.values()})

    def q_cities_by_person_year(self, person: str, year: str) -> Tuple[str, ...]:
        return self.cities_by_person_year.get((person, year), ())

    def q_persons_by_city_year(self, city: str, year: str) -> Tuple[str, ...]:
        return self.persons_by_city_year.get((city, year), ())

    def _exists_axis(self, person: str, year: str, city: str) -> bool:
        s1 = self.facts_by_person.get(person)
        s2 = self.facts_by_year.get(year)
        s3 = self.facts_by_city.get(city)
        if not s1 or not s2 or not s3:
            return False
        a, b, c = sorted([s1, s2, s3], key=len)
        for fid in a:
            if fid in b and fid in c:
                return True
        return False

    def _count_axis(self, person: str, year: str, city: str) -> int:
        s1 = self.facts_by_person.get(person)
        s2 = self.facts_by_year.get(year)
        s3 = self.facts_by_city.get(city)
        if not s1 or not s2 or not s3:
            return 0
        a, b, c = sorted([s1, s2, s3], key=len)
        cnt = 0
        for fid in a:
            if fid in b and fid in c:
                cnt += 1
        return cnt

    def q_exists_person_year_city(self, person: str, year: str, city: str) -> bool:
        if self.use_composite:
            return (person, year, city) in self.comp_exists
        return self._exists_axis(person, year, city)

    def q_count_person_year_city(self, person: str, year: str, city: str) -> int:
        if self.use_composite:
            return int(self.comp_count.get((person, year, city), 0))
        return self._count_axis(person, year, city)


class PGDB:
    """
    indexed means:
      - (person,year)->cities
      - (city,year)->persons
      - (person,year)->city_count dict for count without needing 3D composite
    optional composite means:
      - (person,year,city)->exists/count
    """

    def __init__(self, path: str, indexed: bool, composite: bool) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.indexed = indexed
        self.use_composite = composite

        node = {n["id"]: n for n in data.get("nodes", [])}
        edges = data.get("edges", [])

        self.rows: List[Tuple[str, str, str]] = []
        for e in edges:
            fr = node.get(e["from"])
            to = node.get(e["to"])
            if not fr or not to:
                continue
            if fr.get("type") == "Person" and to.get("type") == "Geo":
                self.rows.append((fr.get("name"), str(e.get("year")), to.get("name")))

        self.idx_cities_by_person_year: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self.idx_persons_by_city_year: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self.idx_city_count_by_person_year: Dict[Tuple[str, str], Dict[str, int]] = {}

        self.idx_exists: Set[Tuple[str, str, str]] = set()
        self.idx_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

        if self.indexed:
            tmp_cities: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            tmp_persons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            tmp_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for p, y, c in self.rows:
                tmp_cities[(p, y)].add(c)
                tmp_persons[(c, y)].add(p)
                tmp_counts[(p, y)][c] += 1

                if self.use_composite:
                    self.idx_exists.add((p, y, c))
                    self.idx_count[(p, y, c)] += 1

            self.idx_cities_by_person_year = {k: tuple(sorted(v)) for k, v in tmp_cities.items()}
            self.idx_persons_by_city_year = {k: tuple(sorted(v)) for k, v in tmp_persons.items()}
            self.idx_city_count_by_person_year = {k: dict(v) for k, v in tmp_counts.items()}

    def q_cities_by_person_year(self, person: str, year: str) -> Tuple[str, ...]:
        if self.indexed:
            return self.idx_cities_by_person_year.get((person, year), ())
        out = set()
        for p, y, c in self.rows:
            if p == person and y == year:
                out.add(c)
        return tuple(sorted(out))

    def q_persons_by_city_year(self, city: str, year: str) -> Tuple[str, ...]:
        if self.indexed:
            return self.idx_persons_by_city_year.get((city, year), ())
        out = set()
        for p, y, c in self.rows:
            if c == city and y == year:
                out.add(p)
        return tuple(sorted(out))

    def q_exists_person_year_city(self, person: str, year: str, city: str) -> bool:
        if self.indexed and self.use_composite:
            return (person, year, city) in self.idx_exists
        if self.indexed:
            return city in self.idx_city_count_by_person_year.get((person, year), {})
        for p, y, c in self.rows:
            if p == person and y == year and c == city:
                return True
        return False

    def q_count_person_year_city(self, person: str, year: str, city: str) -> int:
        if self.indexed and self.use_composite:
            return int(self.idx_count.get((person, year, city), 0))
        if self.indexed:
            return int(self.idx_city_count_by_person_year.get((person, year), {}).get(city, 0))
        cnt = 0
        for p, y, c in self.rows:
            if p == person and y == year and c == city:
                cnt += 1
        return cnt


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

    mm_axis = MMDB(MMDB_PATH, composite=False)
    mm_comp = MMDB(MMDB_PATH, composite=True)

    pg_idx = PGDB(PGDB_PATH, indexed=True, composite=False)
    pg_comp = PGDB(PGDB_PATH, indexed=True, composite=True)

    backends = [
        ("MM(axis)", mm_axis),
        ("MM(axis+comp)", mm_comp),
        ("PG(indexed)", pg_idx),
        ("PG(indexed+comp)", pg_comp),
    ]

    print("=" * 96)
    print("Shared-speed benchmark (v5: 4 backends + 4-column table)")
    print(f"MMDB = {MMDB_PATH}")
    print(f"PGDB = {PGDB_PATH}")
    print(f"SUBTRACT_BASELINE={SUBTRACT_BASELINE}, PRINT_PROGRESS={PRINT_PROGRESS}")
    print(f"WARMUP_ROUNDS={WARMUP_ROUNDS}, MEASURE_ROUNDS={MEASURE_ROUNDS}, REPEATS={REPEATS}")
    print(f"POOL_SINGLE={POOL_SINGLE}, POOL_INTERS={POOL_INTERS}, POOL_BATCH={POOL_BATCH}, UNIQUE_TRIPLES_ONLY={UNIQUE_TRIPLES_ONLY}")

    py, cy = sample_single(mm_axis, rng, POOL_SINGLE)
    triples = sample_intersection(mm_axis, rng, POOL_INTERS, UNIQUE_TRIPLES_ONLY)
    batches = sample_batch(mm_axis, rng, POOL_BATCH)

    print("\n" + "-" * 96)
    print_inputs_calls("A/cities_by_person_year", len(py), MEASURE_ROUNDS)
    print_inputs_calls("A/persons_by_city_year", len(cy), MEASURE_ROUNDS)
    print_inputs_calls("B/exists&count intersection", len(triples), MEASURE_ROUNDS)
    print_inputs_calls("C/batch queries", len(batches), MEASURE_ROUNDS)
    print("-" * 96)

    stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    def bench_row(row: str, make_func, inputs: List[Any]) -> None:
        for name, be in backends:
            func = make_func(be)
            per_ms, qps = measure_per_call(func, inputs, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, f"{row}/{name}")
            stats[row][name] = summarize(per_ms, qps)

    if RUN_A:
        print("\n" + "=" * 96)
        print("Workload A: Single-hop shared queries")
        bench_row("A cities_by_person_year", lambda be: (lambda x: be.q_cities_by_person_year(x[0], x[1])), py)
        bench_row("A persons_by_city_year", lambda be: (lambda x: be.q_persons_by_city_year(x[0], x[1])), cy)

    if RUN_B:
        print("\n" + "=" * 96)
        print("Workload B: Intersection person ∩ year ∩ city")
        bench_row("B exists(person,year,city)", lambda be: (lambda t: be.q_exists_person_year_city(t[0], t[1], t[2])), triples)
        bench_row("B count(person,year,city)", lambda be: (lambda t: be.q_count_person_year_city(t[0], t[1], t[2])), triples)

    if RUN_C:
        print("\n" + "=" * 96)
        print("Workload C: Batch queries (one person, many years)")

        def batch_fn(be, one: Tuple[str, Tuple[str, ...]]) -> int:
            p, years = one
            s = 0
            for y in years:
                s += len(be.q_cities_by_person_year(p, y))
            return s

        bench_row("C batch cities_by_person_year", lambda be: (lambda one: batch_fn(be, one)), batches)

    print("\n" + "=" * 96)
    print("4-column comparison table (mean latency + QPS).")
    print("Cell format: <mean_µs> | <QPS_M/s>")
    print("-" * 96)

    cols = [name for name, _ in backends]
    colw = 24

    header = f"{'Workload':<32}" + "".join([f"{c:<{colw}}" for c in cols])
    print(header)
    print("-" * 96)

    ordered_rows: List[str] = []
    if RUN_A:
        ordered_rows += ["A cities_by_person_year", "A persons_by_city_year"]
    if RUN_B:
        ordered_rows += ["B exists(person,year,city)", "B count(person,year,city)"]
    if RUN_C:
        ordered_rows += ["C batch cities_by_person_year"]

    for row in ordered_rows:
        line = f"{row:<32}"
        for c in cols:
            line += f"{fmt_cell(stats[row][c]):<{colw}}"
        print(line)

    print("-" * 96)
    print("Done.")


if __name__ == "__main__":
    main()
