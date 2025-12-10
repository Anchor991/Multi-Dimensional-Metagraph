# Multi-Dimensional Metagraph (3D Hypercube) + QA Evaluation (DeepSeek)

## 1. Overview

This project builds a simplified **multi-dimensional metagraph** (a 3D hypercube-style knowledge representation) from sentences such as:

> 小明在2020年去了北京。  
> Person: 小明, Time: 2020, Geo: 北京

Each sentence is represented as a **Fact** node (event). A Fact connects to three dimension nodes (**Person / Time / Geo**) via **directed edges**, forming a compact 3D structure that supports slice-style querying.

To evaluate the proposed structure, the same dataset is also converted into a **baseline property graph** and a DeepSeek-based QA pipeline is used to query both backends and compute **accuracy** against `gold_answer`.


## 2. Repository Files

### 2.1 Data generation (MMDB)

- `build_metagraph.py`  
  Generates `graph.json` from manually provided sentences (you edit the sentence list inside the script).

- `build_metagraph_auto.py`  
  Automatically generates a configurable number of sentences using random seeds, then outputs `graph.json`.

Output:
- `graph.json` — multi-dimensional metagraph (MMDB)


### 2.2 Baseline database (PGDB)

- `build_olddb_from_json.py`  
  Converts `graph.json` (MMDB) into a traditional **property graph baseline**.

Output:
- `old_graph.json` — baseline property graph (PGDB)


### 2.3 Derived Fact→Fact links (optional)

These scripts add **event-to-event** links (Fact → Fact). They are optional and mainly used for trajectory-style analysis and visualization.

- `add_ff_links_by_person.py`  
  Adds Fact → Fact links based on **person timeline** (events of the same person sorted by time).  
  Output: `graph_with_ff_links_by_person.json`

- `add_ff_links_by_person_time_geo.py`  
  Adds multiple types of Fact → Fact links:
  - by Person chain
  - by Time chain
  - by Geo chain  
  Output: `graph_with_ff_links_by_person_time_geo.json`


### 2.4 Visualization

- `visualize_graph.py`  
  Visualizes `graph.json` (Fact → Person/Time/Geo edges).  
  Output: `visualize_graph.png`

- `visualize_ff_links_by_person.py`  
  Visualizes Fact → Fact links from `graph_with_ff_links_by_person.json`.  
  Output: `visualize_ff_links_by_person.png`

- `visualize_ff_links_by_person_time_geo.py`  
  Visualizes multiple Fact → Fact link types (colored edges + legend) from `graph_with_ff_links_by_person_time_geo.json`.  
  Output: `visualize_ff_links_by_person_time_geo.png`


### 2.5 QA pipelines (DeepSeek)

- `qa_pipeline_newdb_deepseek.py`  
  QA pipeline for **MMDB only** (`graph.json`).

- `qa_dual_db_deepseek.py`  
  QA pipeline for both:
  - MMDB: `graph.json`
  - PGDB: `old_graph.json`

Accuracy is computed by comparing predicted answers with `gold_answer` in the `QUESTIONS` list.

Important:
- If you regenerate the dataset (especially with `build_metagraph_auto.py`), you must update `QUESTIONS` and `gold_answer` accordingly.


## 3. Workflow

### 3.1 Generate MMDB (`graph.json`)

Manual input:
```bash
python build_metagraph.py
```

Auto generation:
```bash
python build_metagraph_auto.py
```

### 3.2 Visualize MMDB

```bash
python visualize_graph.py
```
Output: `visualize_graph.png`

### 3.3 (Optional) Add and visualize Fact→Fact links

Person-only links:
```bash
python add_ff_links_by_person.py
python visualize_ff_links_by_person.py
```
Outputs:
- `graph_with_ff_links_by_person.json`
- `visualize_ff_links_by_person.png`

Person/Time/Geo links:
```bash
python add_ff_links_by_person_time_geo.py
python visualize_ff_links_by_person_time_geo.py
```
Outputs:
- `graph_with_ff_links_by_person_time_geo.json`
- `visualize_ff_links_by_person_time_geo.png`

### 3.4 Build baseline PGDB (`old_graph.json`)

```bash
python build_olddb_from_json.py
```

### 3.5 Run QA evaluation

MMDB-only:
```bash
python qa_pipeline_newdb_deepseek.py
```

Dual-backend (recommended):
```bash
python qa_dual_db_deepseek.py
```


## 4. Data Formats

### 4.1 `graph.json` (MMDB)

Contains:
- `dimensions.person`: Person nodes (id, name)
- `dimensions.time`: Time nodes (id, value)
- `dimensions.geo`: Geo nodes (id, name)
- `facts`: Fact records (id, person, time, geo, text)
- `graph.nodes`: nodes for visualization
- `graph.edges`: directed edges (typically):
  - Fact → Person (`has_person`)
  - Fact → Time (`has_time`)
  - Fact → Geo (`has_geo`)

### 4.2 `old_graph.json` (PGDB baseline)

Contains:
- `nodes`: Person and Geo nodes
- `edges`: Person → Geo edges with properties:
  - `year`
  - `fact_id`

### 4.3 `graph_with_ff_links_by_person*.json` (optional)

Adds Fact → Fact edges on top of MMDB:
- by-person timeline links (person chain)
- multi-type links for person/time/geo chains (combined version)


## 5. QA Query Types

The QA pipeline asks DeepSeek to output a single JSON object with:
- `op`: operation name
- required parameters for that operation

Supported operations:
- `cities_by_person_year(person_name, year)`
- `persons_by_city_year(city_name, year)`
- `cities_by_person_year_range(person_name, year_start, year_end)`
- `persons_by_city_year_range(city_name, year_start, year_end)`
- `years_by_person_city(person_name, city_name)`
- `count_trips_by_person(person_name)`


## 6. Environment Setup

Tested on Windows + Anaconda.

### 6.1 Python and packages

- Python: `3.10`
- matplotlib
- networkx
- numpy
- openai (OpenAI-compatible SDK used to call DeepSeek)

Install:
```bash
pip install matplotlib networkx numpy openai
```

### 6.2 DeepSeek configuration

Set the environment variable:

Windows (cmd):
```cmd
set DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

Windows (PowerShell):
```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

The code uses:
- Base URL: `https://api.deepseek.com`
- Model: `deepseek-chat`


## 7. Quick Start

```bash
python build_metagraph_auto.py
python visualize_graph.py
python build_olddb_from_json.py
python qa_dual_db_deepseek.py
```


## Update 2025/12/10 20:25

* Added two QA evaluation scripts:

  * `qa_dual_db_deepseek_latency.py`: dual-backend QA (MMDB vs PGDB) with latency metrics (LLM / MMDB / PGDB / end-to-end).
  * `qa_dual_db_deepseek_latency_multihop.py`: multi-hop QA with latency metrics; DeepSeek outputs a multi-step `plan` (atomic ops) plus a `combine` strategy (intersection/union/difference).

* Current observation:

  * Under the current setup (baseline `old_graph.json` is derived from `graph.json` without information loss), **MMDB and PGDB return identical answers** for both single-hop and multi-hop questions using the same DeepSeek-generated plan.
  * Single-hop questions generally achieve **high accuracy**.
  * Therefore, accuracy alone is unlikely to differentiate the two backends in this controlled, lossless setting.

