# -*- coding: utf-8 -*-
"""
benchmark_shared_speed_v4.py

Changes vs v1:
1) Prints len(inputs) and calls for each workload.
2) Prints a compact speedup table at the end (PGDB / MMDB, >1 means MMDB faster).
3) Applies "Route A" optimization for MMDB exists/count:
   - Avoids set intersection allocation; scans the smallest set and membership-checks in the other two.
4) Optional progress printing toggle.

This is an in-memory Python microbenchmark (operator-level), intended for relative comparison.
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

# Print per-repeat raw timing
PRINT_PROGRESS = True

# PGDB execution mode:
# - "naive": scan rows per query (traversal-like baseline)
# - "indexed": dict-of-sets indices per query (strong baseline)
#PGDB_MODE = "naive"
PGDB_MODE = "indexed"

# Avoid duplicates in sampling intersection queries
UNIQUE_TRIPLES_ONLY = True


# =============================================================================
# Helpers
# =============================================================================

def now_ns() -> int:
    return time.perf_counter_ns()

def percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
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

def measure_per_call(func, inputs: List[Any], warmup_rounds: int, measure_rounds: int, repeats: int, tag: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Returns:
      per_call_ms_list, qps_list, net_total_ms_list (per repeat)
    """
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
    net_total_ms: List[float] = []

    for r in range(repeats):
        base_ns = time_whole_loop(baseline, inputs, measure_rounds)
        work_ns = time_whole_loop(func, inputs, measure_rounds)

        net_ns = work_ns
        if SUBTRACT_BASELINE:
            net_ns = max(0, work_ns - base_ns)

        net_ms = net_ns / 1e6
        net_total_ms.append(net_ms)

        ms = (net_ns / calls) / 1e6
        per_call_ms.append(ms)

        secs = net_ns / 1e9
        qps.append((calls / secs) if secs > 0 else float("inf"))

        if PRINT_PROGRESS:
            print(f"  [{tag}] repeat {r+1}/{repeats}: base={base_ns/1e6:.2f} ms, work={work_ns/1e6:.2f} ms, net={net_ms:.2f} ms")

    return per_call_ms, qps, net_total_ms

def summarize(name: str, per_call_ms: List[float], qps: List[float]) -> Dict[str, float]:
    mean_ms = statistics.mean(per_call_ms)
    p50_ms = percentile(per_call_ms, 0.50)
    p95_ms = percentile(per_call_ms, 0.95)

    mean_qps = statistics.mean(qps)
    p50_qps = percentile(qps, 0.50)
    p95_qps = percentile(qps, 0.95)

    print(
        f"{name}: per-call mean={mean_ms:.4f} ms, p50={p50_ms:.4f} ms, p95={p95_ms:.4f} ms | "
        f"QPS mean={mean_qps:.1f}, p50={p50_qps:.1f}, p95={p95_qps:.1f} (n={len(per_call_ms)})"
    )

    return {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "mean_qps": mean_qps,
        "p50_qps": p50_qps,
        "p95_qps": p95_qps,
    }

def print_inputs_calls(label: str, n_inputs: int, measure_rounds: int) -> None:
    calls = n_inputs * measure_rounds
    print(f"[Inputs] {label}: len(inputs)={n_inputs}, MEASURE_ROUNDS={measure_rounds}, calls/repeat={calls}")

def speedup(pg_mean_ms: float, mm_mean_ms: float) -> float:
    # >1 means MMDB faster (since lower latency is better)
    if mm_mean_ms <= 0:
        return float("inf")
    return pg_mean_ms / mm_mean_ms


# =============================================================================
# Backends
# =============================================================================

class MMDB:
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

        for fid, (p, y, c) in self.fact_triple.items():
            tmp_cities[(p, y)].add(c)
            tmp_persons[(c, y)].add(p)
            self.facts_by_person[p].add(fid)
            self.facts_by_year[y].add(fid)
            self.facts_by_city[c].add(fid)

        # store tuples to avoid sorting in hot path
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

    # Route A optimization: scan smallest set + membership checks (no intersection allocation)
    def q_exists_person_year_city(self, person: str, year: str, city: str) -> bool:
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

    def q_count_person_year_city(self, person: str, year: str, city: str) -> int:
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


class PGDB:
    def __init__(self, path: str, mode: str = "naive") -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.mode = mode
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
        self.idx_exists: Set[Tuple[str, str, str]] = set()
        self.idx_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

        if self.mode == "indexed":
            tmp_cities: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            tmp_persons: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            for p, y, c in self.rows:
                tmp_cities[(p, y)].add(c)
                tmp_persons[(c, y)].add(p)
                self.idx_exists.add((p, y, c))
                self.idx_count[(p, y, c)] += 1
            self.idx_cities_by_person_year = {k: tuple(sorted(v)) for k, v in tmp_cities.items()}
            self.idx_persons_by_city_year = {k: tuple(sorted(v)) for k, v in tmp_persons.items()}

    def q_cities_by_person_year(self, person: str, year: str) -> Tuple[str, ...]:
        if self.mode == "indexed":
            return self.idx_cities_by_person_year.get((person, year), ())
        out = set()
        for p, y, c in self.rows:
            if p == person and y == year:
                out.add(c)
        return tuple(sorted(out))

    def q_persons_by_city_year(self, city: str, year: str) -> Tuple[str, ...]:
        if self.mode == "indexed":
            return self.idx_persons_by_city_year.get((city, year), ())
        out = set()
        for p, y, c in self.rows:
            if c == city and y == year:
                out.add(p)
        return tuple(sorted(out))

    def q_exists_person_year_city(self, person: str, year: str, city: str) -> bool:
        if self.mode == "indexed":
            return (person, year, city) in self.idx_exists
        for p, y, c in self.rows:
            if p == person and y == year and c == city:
                return True
        return False

    def q_count_person_year_city(self, person: str, year: str, city: str) -> int:
        if self.mode == "indexed":
            return int(self.idx_count.get((person, year, city), 0))
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
    mm = MMDB(MMDB_PATH)
    pg = PGDB(PGDB_PATH, mode=PGDB_MODE)

    print("=" * 96)
    print("Shared-speed benchmark (v4: len(inputs), calls, speedup table, MMDB Route-A exists/count)")
    print(f"MMDB = {MMDB_PATH}")
    print(f"PGDB = {PGDB_PATH} (mode={PGDB_MODE})")
    print(f"SUBTRACT_BASELINE={SUBTRACT_BASELINE}, PRINT_PROGRESS={PRINT_PROGRESS}")
    print(f"WARMUP_ROUNDS={WARMUP_ROUNDS}, MEASURE_ROUNDS={MEASURE_ROUNDS}, REPEATS={REPEATS}")
    print(f"POOL_SINGLE={POOL_SINGLE}, POOL_INTERS={POOL_INTERS}, POOL_BATCH={POOL_BATCH}, UNIQUE_TRIPLES_ONLY={UNIQUE_TRIPLES_ONLY}")

    py, cy = sample_single(mm, rng, POOL_SINGLE)
    triples = sample_intersection(mm, rng, POOL_INTERS, UNIQUE_TRIPLES_ONLY)
    batches = sample_batch(mm, rng, POOL_BATCH)

    # Print len(inputs) and calls/repeat for each list
    print("\n" + "-" * 96)
    print_inputs_calls("A/cities_by_person_year", len(py), MEASURE_ROUNDS)
    print_inputs_calls("A/persons_by_city_year", len(cy), MEASURE_ROUNDS)
    print_inputs_calls("B/exists&count intersection queries", len(triples), MEASURE_ROUNDS)
    print_inputs_calls("C/batch queries", len(batches), MEASURE_ROUNDS)
    print("-" * 96)

    results: Dict[str, Dict[str, float]] = {}

    if RUN_A:
        print("\n" + "=" * 96)
        print("Workload A: Single-hop shared queries (batch-timed)")

        mm_ms, mm_qps, _ = measure_per_call(lambda x: mm.q_cities_by_person_year(x[0], x[1]), py, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "A/mm/cities")
        pg_ms, pg_qps, _ = measure_per_call(lambda x: pg.q_cities_by_person_year(x[0], x[1]), py, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "A/pg/cities")
        results["A cities MMDB"] = summarize("[cities_by_person_year] MMDB", mm_ms, mm_qps)
        results["A cities PGDB"] = summarize(f"[cities_by_person_year] PGDB({pg.mode})", pg_ms, pg_qps)

        mm_ms, mm_qps, _ = measure_per_call(lambda x: mm.q_persons_by_city_year(x[0], x[1]), cy, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "A/mm/persons")
        pg_ms, pg_qps, _ = measure_per_call(lambda x: pg.q_persons_by_city_year(x[0], x[1]), cy, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "A/pg/persons")
        results["A persons MMDB"] = summarize("[persons_by_city_year] MMDB", mm_ms, mm_qps)
        results["A persons PGDB"] = summarize(f"[persons_by_city_year] PGDB({pg.mode})", pg_ms, pg_qps)

    if RUN_B:
        print("\n" + "=" * 96)
        print("Workload B: Intersection person ∩ year ∩ city (batch-timed)")

        mm_ms, mm_qps, _ = measure_per_call(lambda t: mm.q_exists_person_year_city(t[0], t[1], t[2]), triples, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "B/mm/exists")
        pg_ms, pg_qps, _ = measure_per_call(lambda t: pg.q_exists_person_year_city(t[0], t[1], t[2]), triples, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "B/pg/exists")
        results["B exists MMDB"] = summarize("[exists(person,year,city)] MMDB", mm_ms, mm_qps)
        results["B exists PGDB"] = summarize(f"[exists(person,year,city)] PGDB({pg.mode})", pg_ms, pg_qps)

        mm_ms, mm_qps, _ = measure_per_call(lambda t: mm.q_count_person_year_city(t[0], t[1], t[2]), triples, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "B/mm/count")
        pg_ms, pg_qps, _ = measure_per_call(lambda t: pg.q_count_person_year_city(t[0], t[1], t[2]), triples, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "B/pg/count")
        results["B count MMDB"] = summarize("[count(person,year,city)] MMDB", mm_ms, mm_qps)
        results["B count PGDB"] = summarize(f"[count(person,year,city)] PGDB({pg.mode})", pg_ms, pg_qps)

    if RUN_C:
        print("\n" + "=" * 96)
        print("Workload C: Batch queries (one person, many years) (batch-timed)")

        def mm_batch(one: Tuple[str, Tuple[str, ...]]) -> int:
            p, years = one
            s = 0
            for y in years:
                s += len(mm.q_cities_by_person_year(p, y))
            return s

        def pg_batch(one: Tuple[str, Tuple[str, ...]]) -> int:
            p, years = one
            s = 0
            for y in years:
                s += len(pg.q_cities_by_person_year(p, y))
            return s

        mm_ms, mm_qps, _ = measure_per_call(mm_batch, batches, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "C/mm/batch")
        pg_ms, pg_qps, _ = measure_per_call(pg_batch, batches, WARMUP_ROUNDS, MEASURE_ROUNDS, REPEATS, "C/pg/batch")
        results["C batch MMDB"] = summarize("[batch cities_by_person_year] MMDB", mm_ms, mm_qps)
        results["C batch PGDB"] = summarize(f"[batch cities_by_person_year] PGDB({pg.mode})", pg_ms, pg_qps)

    # Speedup table (paper-friendly)
    print("\n" + "=" * 96)
    print("Speedup table (PGDB mean_ms / MMDB mean_ms). >1 means MMDB faster.")
    print("-" * 96)

    def row(label: str, mm_key: str, pg_key: str) -> None:
        if mm_key not in results or pg_key not in results:
            return
        mm_mean = results[mm_key]["mean_ms"]
        pg_mean = results[pg_key]["mean_ms"]
        sp = speedup(pg_mean, mm_mean)
        print(f"{label:<34} | MMDB mean={mm_mean:.6f} ms | PGDB mean={pg_mean:.6f} ms | speedup={sp:.2f}x")

    row("A: cities_by_person_year", "A cities MMDB", "A cities PGDB")
    row("A: persons_by_city_year", "A persons MMDB", "A persons PGDB")
    row("B: exists(person,year,city)", "B exists MMDB", "B exists PGDB")
    row("B: count(person,year,city)", "B count MMDB", "B count PGDB")
    row("C: batch cities_by_person_year", "C batch MMDB", "C batch PGDB")

    print("-" * 96)
    print("Done.")


if __name__ == "__main__":
    main()
