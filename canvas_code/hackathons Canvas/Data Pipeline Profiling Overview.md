# Pipeline Benchmark Report — Data Profiling

## Purpose
Profile the raw `user_retention.parquet` telemetry file across eight sequential pipeline stages to quantify its memory footprint, structural quality, and dtype optimisation potential. This step provides the evidence base for all transformations applied in the subsequent **Optimized Pipeline** block.

## Input Data
| Attribute | Value |
|---|---|
| **Source file** | `user_retention.parquet` |
| **File size on disk** | 50.2 MB (52,646,892 bytes) |
| **Rows** | 409,287 events |
| **Columns** | 107 |
| **Loaded via** | `pd.read_parquet()` (pyarrow backend) |

The raw DataFrame contains one row per PostHog event and covers user activity on the Zerve platform from **September 1 – December 8, 2025** (99 days). Columns span six semantic groups: 77 string/object columns, 23 float64 columns, 1 int32 column, and 6 datetime columns.

## Computations Performed
The block runs **eight instrumented pipeline stages**, each independently timed to the millisecond:

1. **Load** — `pd.read_parquet()` cold load from disk to in-memory DataFrame
2. **Memory profiling** — `df.memory_usage(deep=True)` on all 107 columns, producing a per-column byte breakdown and ranking the 15 heaviest columns
3. **Column type audit** — `df.dtypes.value_counts()` categorised into string, float, int, datetime, and boolean groups
4. **Null rate analysis** — `df.isnull().sum()` classified into four tiers: 100% null (drop candidates), 50–99% null (high null), 20–49% null (moderate null), and 1–19% null (low null)
5. **Cardinality analysis** — `df[col].nunique()` per column; string columns are classified as high-cardinality (> 1,000 unique values or > 5% unique ratio) or low-cardinality (≤ 1,000 unique / ≤ 5% ratio — candidates for `category` dtype conversion)
6. **Redundancy detection** — Identifies `prop_$set.*` and `prop_$set_once.*` mirror columns, zero-variance constant columns, and the `Unnamed: 0` index artifact
7. **Numeric skew analysis** — `Series.skew()` on all numeric columns; flags columns where `|skew| > 3` for potential log-transform prior to modelling
8. **Memory saving estimate** — Calculates projected savings from `float64 → float32` (50% per column), `int64 → int16/int32`, and `str → category` conversions

## Why This Step Matters
The raw parquet file expands to approximately **11.5×** its on-disk size in memory. Without profiling and optimisation, the full 409,287-row DataFrame would consume **~577.5 MB** of working memory — an unnecessary burden for all downstream groupby aggregations in the Feature Engineering step. With 89 of 107 columns carrying 50–99% null rates and 33 confirmed redundant mirror columns, the optimisation potential is substantial before any modelling begins.
