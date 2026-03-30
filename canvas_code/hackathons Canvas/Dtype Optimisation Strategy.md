# Dtype Optimisation Strategy

## Purpose
Apply all dtype optimisations identified in the **Pipeline Benchmark Report** to the raw 409,287 × 107 `user_retention` DataFrame, then measure and visualise the before/after performance improvement. This block is the memory and speed gate that makes downstream Feature Engineering feasible in a serverless compute environment.

## Input Data
| Source | Variable | Shape |
|---|---|---|
| Upstream raw parquet | `user_retention` | 409,287 rows × 107 columns, 577.5 MB in-memory |
| Upstream profiling results | `benchmark_report` | dict (18 keys) with `low_card_str_cols`, `constant_cols`, `redundant_mirror_cols` |

## Optimisations Applied (Five Steps)

1. **Drop redundant columns** — Removes all 33 identified drop candidates: the `Unnamed: 0` int32 index artefact, 16 `prop_$set.*` mirror columns, and 16 `prop_$set_once.*` mirror columns. These duplicate state already captured in their non-prefixed counterparts and provide zero additional signal for modelling. Result: 107 → **74 columns**.

2. **`float64 → float32` downcast** — Converts all 13 remaining `float64` columns to `float32`, trading 64-bit precision (unnecessary for sensor/telemetry data bounded to 0–100k ranges) for a **50% per-column memory reduction**. The `benchmark_report["low_card_str_cols"]` list drives column selection.

3. **Integer downcast** — Checks each `int64`/`int32` column's value range against `np.iinfo` bounds; safely downcasts to `int16` where the range permits (`min ≥ −32,768`, `max ≤ 32,767`) or `int32` otherwise.

4. **`str → category` conversion** — Converts 37 low-cardinality string columns (≤ 1,000 unique values, ≤ 5% unique ratio) to pandas `category` dtype. Category encoding replaces repeated full string storage with an integer code lookup table, yielding approximately 70% memory saving per column for high-repetition values such as `event`, `prop_$lib`, `prop_$os`, `prop_$timezone`.

5. **Memory layout defragmentation** — Applies a `.copy()` pass to consolidate all mutations into a contiguous memory block, preventing pandas internal fragmentation warnings in downstream operations.

## Visualisations Rendered
- `optimised_comparison.png` — Side-by-side 3-panel bar chart: Memory (MB), Column Count, and Stage Timings (Baseline vs. Optimised), with annotated percentage-reduction arrows
- `dtype_comparison.png` — Grouped bar chart showing column dtype distribution before and after all transformations

## Analytical Note
The 577.5 MB baseline footprint is the primary scaling bottleneck in this pipeline. Without this optimisation, the groupby aggregations in the Feature Engineering block would process nearly 600 MB of object-heavy data. The transformed `optimised_retention` DataFrame is the sole input to the Feature Engineering block.
