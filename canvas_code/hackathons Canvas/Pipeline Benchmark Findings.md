# Pipeline Benchmark Findings

## Output Variables
| Variable | Type | Shape | Description |
|---|---|---|---|
| `benchmark_report` | `dict` | 18 keys | Structured profiling summary exported for use by the Optimized Pipeline block |
| `user_retention` | `DataFrame` | 409,287 × 107 | Raw event log retained in memory (unmodified) |

## Confirmed Numeric Results

### Memory Profile
- **In-memory footprint (deep)**: 577.5 MB — an **11.5× bloat factor** over the 50.2 MB parquet file
- **Top 3 heaviest columns**: `person_id`, `uuid`, `distinct_id` — each consuming 28.1 MB (4.9% each), totalling 84.3 MB across just three UUID columns

### Column Type Distribution
| Type | Count |
|---|---|
| `str` / `object` | 77 columns |
| `float64` | 23 columns |
| `datetime64[us]` | 6 columns |
| `int32` | 1 column |

### Null Rate Analysis
| Tier | Count |
|---|---|
| 100% null (useless) | 0 columns |
| 50–99% null (high null) | **89 columns** |
| 20–49% null (moderate null) | 2 columns (`prop_$python_version`, `prop_$python_runtime`) |
| 1–19% null (low null) | 5 columns |
| 0% null (complete) | 11 columns |

### Cardinality Analysis
- **High-cardinality string columns** (> 1,000 unique or > 5% unique ratio): **22 found**, including `uuid` (409,287 unique — 100% ratio), `prop_$insert_id` (83,916), `prop_$session_id` (13,520)
- **Low-cardinality string columns** (candidates for `category` dtype): Identified for conversion in the Optimized Pipeline step

### Redundancy Detection
| Issue | Count |
|---|---|
| `prop_$set.*` mirror columns | 16 |
| `prop_$set_once.*` mirror columns | 16 |
| `Unnamed: 0` index artifact | 1 |
| **Total redundant columns** | **33** |

### Pipeline Stage Timings
| Stage | Time |
|---|---|
| 1. Load (`pd.read_parquet`) | 0.265 s |
| 2. Memory profiling | 0.688 s |
| 3. Column type audit | 0.003 s |
| 4. Null rate analysis | 0.145 s |
| 5. Cardinality analysis | 0.695 s |
| 6. Redundancy detection | 0.013 s |
| 7. Numeric skew / distribution | 0.082 s |
| 8. Memory optimisation estimate | 0.001 s |
| **TOTAL** | **1.892 s** |

### Memory Optimisation Potential
The `benchmark_report` dict quantifies the following projected savings: 89 high-null columns and 33 redundant mirror columns account for the majority of the 577.5 MB footprint, with `float64 → float32` and `str → category` conversions representing the two highest-yield transformation pathways.
