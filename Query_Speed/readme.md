# SPEED — Shared-query Speed Benchmark

This folder contains **speed benchmarks for “shared information” queries** that exist in both structures:

- **MMDB**: multi-dimensional metagraph (`graph.json`)
- **PGDB**: projected/traditional graph representation (`old_graph.json`)

The goal is to verify that **even on the shared triples (Person–Year–Geo)**—where QA accuracy is usually identical—MMDB can still show advantages in **query latency / throughput**, especially under:
1) multi-dimensional filtering (intersection-style lookups), and  
2) batch / repeated queries where axis indexes can be reused.

---

## 1. Files in this folder

### Data
- `graph.json`  
  MMDB dataset (multi-dimensional structure).

- `old_graph.json`  
  PGDB dataset generated from `graph.json` (projection).

### Builders
- `build_olddb_from_json.py`  
  Generates `old_graph.json` from `graph.json`.

### Benchmarks
- `benchmark_shared_speed_v2.py`  
  Benchmark runner that compares **MMDB vs PGDB** on three workloads (A/B/C) and prints a **speedup table**.
  - Outputs per-workload latency (mean/p50/p95) and QPS.
  - Prints `len(inputs)` and `calls/repeat` for transparency.
  - Includes “safe defaults + progress + raw timing” to avoid the “too fast to measure” pitfall.

- `benchmark_shared_speed_comp.py`  
  “Fair comparison” version focusing on:
  - **MMDB(axis + composite)** vs **PGDB(indexed + composite)**
  - Used when writing paper/report, because both sides have comparable indexing capability.

- `benchmark_shared_speed_execution_mode_diff.py`  
  It benchmarks **four backends** in the same run and prints a **4-column comparison table**:
  - `MM(axis)`
  - `MM(axis+comp)`
  - `PG(indexed)`
  - `PG(indexed+comp)`

  Table cell format:
  - `<mean_µs> | <QPS_M/s>`

  This script is used to explain **where the gap comes from** (indexing strategy and query shape).

---

## 2. Workloads (A / B / C)

All benchmark scripts target the same shared-query workload set:

### Workload A — Single-hop shared queries (2D lookups)
These are “shared-triple” style queries that exist in both MMDB and PGDB:
- `cities_by_person_year(person, year) -> [cities]`
- `persons_by_city_year(city, year) -> [persons]`

This workload measures “classic” KG queries and is where PGDB can be very competitive once indexed.

### Workload B — 3D intersection (person ∩ year ∩ city)
These are “multi-dimensional filtering” checks:
- `exists(person, year, city) -> bool`
- `count(person, year, city) -> int` (typically 0/1 here, depending on duplicates)

This workload is important because it highlights that:
- axis-only intersection may cost more than expected, and
- a **3D composite structure** can dominate the performance story.

### Workload C — Batch queries (one person, many years)
A repeated/batched variant of Workload A:
- `batch_cities_by_person_year(person, [year1..yearN]) -> {year -> [cities]}`

This workload stresses:
- overhead amortization
- cache friendliness
- index reuse under repeated queries

---

## 3. Execution modes / backends

Different scripts compare different backend setups. Terminology used in output:

### MMDB backends
- `MM(axis)`  
  Uses only axis indexes such as:
  - `person -> events`, `year -> events`, `geo -> events`
  - and 2D maps derived from them (implementation-dependent)

- `MM(axis+comp)`  
  Adds a **composite 3D structure**, typically:
  - `(person, year, geo) -> count / existence`
  - or a fast 3D key lookup table

### PGDB backends
- `PG(indexed)`  
  Adds 2D indexes in the projected graph, typically:
  - `(person, year) -> [cities]`
  - `(city, year) -> [persons]`

- `PG(indexed+comp)`  
  Adds the same idea of composite index:
  - `(person, year, geo) -> count / existence`

The point of the 4-backend comparison is to show that **“speedup” is driven by index shape**, and the query workload determines which index matters.

---

## 4. How to run

### Step 0 — Ensure `old_graph.json` exists
From inside this folder:

```bash
python build_olddb_from_json.py
```

Expected output:
- `old_graph.json`

### Step 1 — Run v2 baseline benchmark (MMDB vs PGDB)
```bash
python benchmark_shared_speed_v4.py
```

You will see:
- inputs size (`len(inputs)`)
- calls per repeat
- A/B/C workload results
- speedup table (PGDB mean / MMDB mean)

### Step 2 — Run execution-mode comparison (4 backends)
```bash
python benchmark_shared_speed_execution_mode_diff.py
```

You will see a 4-column table like:

- `MM(axis)` / `MM(axis+comp)` / `PG(indexed)` / `PG(indexed+comp)`
- mean latency in **µs** and QPS in **M/s**

### Step 3 — Run “paper-ready fair comparison”
```bash
python benchmark_shared_speed_comp.py
```

Focuses on:
- `MM(axis+comp)` vs `PG(indexed+comp)`
- recommended for reporting because it compares “similar index power”.

---

## 5. Parameters you can tune (in scripts)

All benchmark scripts include tunable constants near the top (names may vary slightly):

- `WARMUP_ROUNDS`  
  Warm-up runs to stabilize caches and Python overhead.

- `MEASURE_ROUNDS`  
  Number of timing rounds.

- `REPEATS`  
  How many independent repeats to report (for p50/p95 stability).

- `POOL_SINGLE`, `POOL_INTERS`, `POOL_BATCH`  
  Size of the randomly sampled input pools for A/B/C.

- `SUBTRACT_BASELINE`  
  Whether to subtract a measured “empty loop baseline” from timing.
  Useful for microsecond-level timing, but can misbehave if loop bodies get optimized differently.

- `UNIQUE_TRIPLES_ONLY`  
  If enabled, input sampling avoids duplicate triples to reduce skew.

---

## 6. Notes on microsecond-level benchmarking

These scripts are intentionally “batch-timed” because a single Python function call often costs more than the query itself.

Recommendations for stable results:
- Close heavy background apps (browser tabs, IDE indexing).
- Use consistent CPU mode (avoid extreme power saving).
- Run multiple times and report medians (p50) rather than a single run.
- If results look unrealistically high, increase:
  - `MEASURE_ROUNDS`
  - `REPEATS`
  - pool sizes (e.g., `POOL_INTERS`)

---

## 7. What to report (suggested narrative)

A clean reporting structure is:

1) **MMDB vs PGDB naive**  
   Shows the “no-index baseline” gap (mostly engineering).

2) **MMDB vs PGDB indexed**  
   Shows the gap shrinks on Workload A/C when PGDB has good 2D indexes.

3) **4-backend comparison (execution_mode_diff)**  
   Shows “where the gap comes from”:
   - Workload B benefits massively from composite 3D index.
   - Workload A/C benefit mainly from good 2D mapping + low constant factors.

4) **Fair comparison (comp)**  
   `MM(axis+comp)` vs `PG(indexed+comp)`:
   - usually yields small but stable advantages for MMDB on A/C
   - B tends to converge (both have composite), demonstrating index parity.

---

## 8. Expected outputs

The benchmark scripts are designed to produce console tables that can be directly screenshot into a report/paper:
- per-workload latency summaries
- QPS summaries
- speedup table (v4)
- 4-column backend comparison table (execution_mode_diff / v5)

---

## 9. Troubleshooting

- “It runs forever”  
  Reduce `MEASURE_ROUNDS`, `REPEATS`, and pool sizes.
  Start with smaller values, verify correctness, then scale up.

- “Numbers look too good to be true”  
  Increase repeats, disable baseline subtraction (`SUBTRACT_BASELINE=False`), and compare medians.
  Ensure you are running the intended backend mode (naive/indexed/composite).

- “Speedup flips sign on Workload B”  
  This is expected if one backend has composite and the other does not.
  Workload B is specifically designed to expose composite index effects.
