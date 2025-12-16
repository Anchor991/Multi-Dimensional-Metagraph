# 3AXIS_SIMPLIFIED

This folder contains a **simplified 3-axis multi-dimensional knowledge graph (MMDB)** experiment pipeline and a **baseline graph database (PGDB)** projection.  
It is designed to **demonstrate when the new 3-axis structure becomes useful** by mixing:

- **Normal 1-hop QA** (both MMDB and PGDB can answer)
- **Structure-sensitive QA** based on **event→event directed links** (FF\_\*\_NEXT), which **MMDB supports naturally**, while **PGDB does not** (unless you explicitly extend it)

---

## What is MMDB vs. PGDB in this repo?

### MMDB (New structure)
A 3-axis “metagraph” where each travel event is represented as an **Event/Fact node** that links to three axes:

- **Person axis**
- **Time axis**
- **Geo axis**

In addition, MMDB includes **directed Fact→Fact edges** (FF links) that encode different “next-event” semantics:

- `FF_PERSON_NEXT` (events chained by person)
- `FF_TIME_NEXT` (events chained by time)
- `FF_GEO_NEXT` (events chained by geo)

### PGDB (Baseline / traditional graph)
A projection of the dataset into a more conventional graph form, storing only the **(person, year, city)** information (Person→Geo edges with a `year` attribute).  
It **does not** keep Event/Fact nodes or FF event chains.

---

## Folder contents (files)

### Data generation
- `build_metagraph_auto.py`  
  Generates a synthetic MMDB dataset and saves it as `graph.json`.  
  (This file typically contains `dimensions`, `facts`, and `graph.edges`.)

### Add FF (Fact→Fact) links
- `add_ff_links_by_person_time_geo.py`  
  Reads `graph.json` and outputs:

  - `graph_with_ff_links_by_person_time_geo.json`

  This is the **recommended MMDB dataset** for structure-sensitive evaluation because it includes `FF_PERSON_NEXT / FF_TIME_NEXT / FF_GEO_NEXT`.

### Build the baseline PGDB from MMDB
- `build_olddb_from_graph_with_ff_links.py`  
  Reads `graph_with_ff_links_by_person_time_geo.json` and outputs:

  - `old_graph_from_ff_links.json`

  This projection keeps only **triple-level** knowledge (Person→Geo edges with year), intentionally discarding FF chains.

### Frozen QA dataset builder (questions + gold answers)
- `build_fflinks_questions_gold.py`  
  Generates a **frozen evaluation set** (a JSON list) and outputs:

  - `mixed_questions_gold.json`

  The frozen set is **mixed**:
  - Normal questions: e.g. *“小红在2020年去了哪些城市？”*
  - FF-structure questions: e.g. *“按人物链：…之后，下一次去了哪个城市？”*

  `gold_answer` is computed **deterministically from the dataset** (not from LLM).

### QA evaluation (accuracy)
- `qa_dual_db_qwen3_mixed.py`  
  Reads:
  - `graph_with_ff_links_by_person_time_geo.json` (MMDB)
  - `old_graph_from_ff_links.json` (PGDB)
  - `mixed_questions_gold.json` (frozen questions + gold)

  Then uses **Qwen3** to convert questions → query JSON, executes the query on both DBs, and reports accuracy.

- `qa_dual_db_qwen3_multihop.py`  
  An older multi-hop QA script that focuses on **filter/intersection** style queries.  
  Those queries often remain **equivalent** between MMDB and PGDB if you only test triple-level reasoning.

### Visualization
- `visualize_graph.py` → `visualize_graph.png`  
  Visualizes the 3-axis MMDB dataset.

- `visualize_ff_links_by_person_time_geo.py` → `visualize_ff_links_by_person_time_geo.png`  
  Visualizes the dataset including FF event links.

### Cache files
- `qwen3_fflinks_cache.json`, `qwen3_mixed_cache.json`  
  Caches LLM outputs (question → query JSON) to improve reproducibility and reduce repeated API calls.

---

## Quickstart

### 1) Create / activate environment
Recommended: **Python 3.10**.

Install dependencies:
```bash
pip install numpy matplotlib networkx openai
```

> Note: Some scripts use the OpenAI-compatible client. You may need a specific version depending on your environment.
> If you already have a working environment, keep it consistent for reproducibility.

### 2) Generate the base dataset
```bash
python build_metagraph_auto.py
```
Output:
- `graph.json`

### 3) Add FF links (Fact→Fact edges)
```bash
python add_ff_links_by_person_time_geo.py
```
Output:
- `graph_with_ff_links_by_person_time_geo.json`

### 4) Build baseline PGDB projection
```bash
python build_olddb_from_graph_with_ff_links.py
```
Output:
- `old_graph_from_ff_links.json`

### 5) Build a frozen QA dataset (questions + gold)
```bash
python build_fflinks_questions_gold.py
```
Output:
- `mixed_questions_gold.json`

### 6) Run the mixed QA benchmark (accuracy)
Edit `qa_dual_db_qwen3_mixed.py` and set your Qwen3 API key at the top:

```python
DASHSCOPE_API_KEY = "REPLACE_WITH_YOUR_DASHSCOPE_KEY"
```

Run:
```bash
python qa_dual_db_qwen3_mixed.py
```

Expected behavior (typical):
- **MMDB** should achieve **high accuracy** on both normal and FF questions (depending on LLM query generation quality).
- **PGDB** should achieve **high accuracy on normal questions**, but perform **worse on FF questions** (because baseline does not have FF structure).

---

## Controlling “MMDB 90+ / PGDB 80+”

The easiest knob is the **mix ratio** in `build_fflinks_questions_gold.py`:

- `FF_RATIO`: fraction of FF-structure questions in the frozen dataset.

Because baseline PGDB does not support `ff_next`, its maximum accuracy is approximately:
- **PGDB_max ≈ 1 − FF_RATIO**

Examples:
- `FF_RATIO = 0.15` → PGDB upper bound ≈ 85%
- `FF_RATIO = 0.20` → PGDB upper bound ≈ 80%

MMDB accuracy depends mostly on whether the LLM correctly generates:
- the correct `op`, and
- the correct `ff_type` (person/time/geo) when the question starts with “按人物链 / 按时间链 / 按地点链”.

If you observe many MMDB errors on geo-chain questions, reduce geo-chain sampling weight in:
- `FF_TYPE_WEIGHTS` (e.g., keep geo at 0.05–0.10)

---

## Notes on reproducibility

- **Freeze your evaluation set**: generate `mixed_questions_gold.json` once and keep it unchanged.
- Keep cache files (`qwen3_mixed_cache.json`) to avoid variance from repeated LLM outputs.
- If you change prompt/schema, bump the script’s `PROMPT_VERSION` and optionally delete cache.

---

## Security note

Some scripts support **hardcoding API keys** for convenience.  
Do **not** commit your real keys to public repositories.

---

## Troubleshooting

### MMDB errors but gold is correct
Common cause: LLM generated a wrong `ff_type` for FF questions.

Fix:
- Strengthen the system prompt constraints in `qa_dual_db_qwen3_mixed.py`
- Keep `TEMPERATURE = 0.0`
- Reduce geo-chain question weight (`FF_TYPE_WEIGHTS["geo"]`)

### PGDB always returns empty on FF questions
This is expected: baseline projection does not store event→event links.

If you want PGDB to also attempt FF-style reasoning, you must implement a heuristic `next` operator in PGDB
(e.g., sort by year for a person and pick the next record). That would be a **different baseline**.

---

## Typical workflow diagram

```text
build_metagraph_auto.py
  -> graph.json
     -> add_ff_links_by_person_time_geo.py
        -> graph_with_ff_links_by_person_time_geo.json   (MMDB)
           -> build_olddb_from_graph_with_ff_links.py
              -> old_graph_from_ff_links.json            (PGDB baseline)

graph_with_ff_links_by_person_time_geo.json
  -> build_fflinks_questions_gold.py
     -> mixed_questions_gold.json                        (frozen eval set)

qa_dual_db_qwen3_mixed.py
  -> accuracy on MMDB vs PGDB
```
