
import pandas as pd
import numpy as np
import time
import os
import sys
import io

# ─────────────────────────────────────────────
# STAGE 0 — FILE METADATA
# ─────────────────────────────────────────────
_parquet_path = "user_retention.parquet"
_file_size_bytes = os.path.getsize(_parquet_path)
_file_size_mb = _file_size_bytes / (1024 ** 2)

# ─────────────────────────────────────────────
# STAGE 1 — LOAD TIMING
# ─────────────────────────────────────────────
_t_load_start = time.perf_counter()
_df = pd.read_parquet(_parquet_path)
_t_load_end = time.perf_counter()
_load_time_s = _t_load_end - _t_load_start

_n_rows, _n_cols = _df.shape

# ─────────────────────────────────────────────
# STAGE 2 — MEMORY PROFILING
# ─────────────────────────────────────────────
_t_mem_start = time.perf_counter()
_mem_per_col = _df.memory_usage(deep=True)  # Series: col -> bytes
_total_mem_bytes = _mem_per_col.sum()
_total_mem_mb = _total_mem_bytes / (1024 ** 2)
_t_mem_end = time.perf_counter()
_mem_time_s = _t_mem_end - _t_mem_start

_mem_ratio = _total_mem_mb / _file_size_mb  # in-memory bloat factor

# ─────────────────────────────────────────────
# STAGE 3 — COLUMN TYPE AUDIT
# ─────────────────────────────────────────────
_t_types_start = time.perf_counter()
_dtype_counts = _df.dtypes.value_counts().to_dict()
_dtype_map = {col: str(dt) for col, dt in _df.dtypes.items()}
_t_types_end = time.perf_counter()
_types_time_s = _t_types_end - _t_types_start

# Categorize columns
_str_cols   = [c for c, d in _dtype_map.items() if d == "object" or d == "str"]
_float_cols = [c for c, d in _dtype_map.items() if "float" in d]
_int_cols   = [c for c, d in _dtype_map.items() if "int" in d]
_dt_cols    = [c for c, d in _dtype_map.items() if "datetime" in d]
_bool_cols  = [c for c, d in _dtype_map.items() if "bool" in d]

# ─────────────────────────────────────────────
# STAGE 4 — NULL RATE ANALYSIS
# ─────────────────────────────────────────────
_t_null_start = time.perf_counter()
_null_counts  = _df.isnull().sum()
_null_rates   = (_null_counts / _n_rows * 100).round(2)
_t_null_end   = time.perf_counter()
_null_time_s  = _t_null_end - _t_null_start

_fully_null_cols   = _null_rates[_null_rates == 100.0].index.tolist()
_high_null_cols    = _null_rates[((_null_rates >= 50) & (_null_rates < 100))].index.tolist()
_moderate_null_cols= _null_rates[((_null_rates >= 20) & (_null_rates < 50))].index.tolist()
_low_null_cols     = _null_rates[(_null_rates > 0) & (_null_rates < 20)].index.tolist()
_no_null_cols      = _null_rates[_null_rates == 0].index.tolist()

# ─────────────────────────────────────────────
# STAGE 5 — CARDINALITY ANALYSIS
# ─────────────────────────────────────────────
_t_card_start = time.perf_counter()
_cardinality  = {}
for _c in _df.columns:
    _cardinality[_c] = _df[_c].nunique(dropna=False)
_card_series  = pd.Series(_cardinality)
_t_card_end   = time.perf_counter()
_card_time_s  = _t_card_end - _t_card_start

# High-cardinality string columns (>= 5% unique ratio or >1000 unique values)
_high_card_str_cols = []
for _c in _str_cols:
    _u = _cardinality[_c]
    _ratio = _u / _n_rows if _n_rows > 0 else 0
    if _u > 1000 or _ratio > 0.05:
        _high_card_str_cols.append((_c, _u, round(_ratio * 100, 1)))

# Low-cardinality string columns (good candidates for category dtype)
_low_card_str_cols = []
for _c in _str_cols:
    _u = _cardinality[_c]
    _ratio = _u / _n_rows if _n_rows > 0 else 0
    if _u <= 1000 and _ratio <= 0.05:
        _low_card_str_cols.append((_c, _u, round(_ratio * 100, 1)))

# ─────────────────────────────────────────────
# STAGE 6 — REDUNDANCY DETECTION
# ─────────────────────────────────────────────
_t_red_start = time.perf_counter()

# Duplicate column names
_dup_col_names = [c for c in _df.columns if list(_df.columns).count(c) > 1]

# Columns that are entirely one value (zero variance)
_constant_cols = [c for c in _df.columns if _cardinality[c] <= 1]

# Columns with prop_$set / prop_$set_once prefix — potential redundant mirrors
_set_cols      = [c for c in _df.columns if c.startswith("prop_$set.")]
_set_once_cols = [c for c in _df.columns if c.startswith("prop_$set_once.")]

# Unnamed index column
_has_unnamed_index = "Unnamed: 0" in _df.columns

_t_red_end = time.perf_counter()
_red_time_s = _t_red_end - _t_red_start

# ─────────────────────────────────────────────
# STAGE 7 — NUMERIC DISTRIBUTION / SKEW
# ─────────────────────────────────────────────
_t_dist_start = time.perf_counter()
_numeric_cols = _float_cols + _int_cols
_skew_results = {}
for _c in _numeric_cols:
    _s = _df[_c].dropna()
    if len(_s) > 1:
        _sk = float(_s.skew())
        _skew_results[_c] = round(_sk, 2)
_skew_series = pd.Series(_skew_results)
_highly_skewed = _skew_series[_skew_series.abs() > 3].sort_values(key=lambda x: x.abs(), ascending=False)
_t_dist_end = time.perf_counter()
_dist_time_s = _t_dist_end - _t_dist_start

# ─────────────────────────────────────────────
# STAGE 8 — MEMORY SAVING ESTIMATE
# ─────────────────────────────────────────────
_t_opt_start = time.perf_counter()

# Float64 → float32 savings
_f64_cols = [c for c, d in _dtype_map.items() if d == "float64"]
_f64_mem_mb = sum(_mem_per_col.get(c, 0) for c in _f64_cols) / (1024**2)
_f64_saving_mb = _f64_mem_mb * 0.5  # 50% saving

# int64 / int32 → int16 savings (if max values allow)
_i64_cols = [c for c, d in _dtype_map.items() if d in ("int64",)]
_i64_mem_mb = sum(_mem_per_col.get(c, 0) for c in _i64_cols) / (1024**2)
_i64_saving_mb = _i64_mem_mb * 0.75  # 75% saving (int16 = 2 bytes vs 8 bytes)

# String → category savings for low-cardinality strings
_cat_saving_mb = 0.0
for _c, _u, _ratio_pct in _low_card_str_cols:
    _col_mem = _mem_per_col.get(_c, 0) / (1024**2)
    _cat_saving_mb += _col_mem * 0.7  # rough 70% saving for categoricals

_total_saving_mb = _f64_saving_mb + _i64_saving_mb + _cat_saving_mb
_t_opt_end = time.perf_counter()
_opt_time_s = _t_opt_end - _t_opt_start

# ─────────────────────────────────────────────
# BENCHMARK REPORT — PRINT
# ─────────────────────────────────────────────

_SEP  = "=" * 70
_sep2 = "─" * 70

print(_SEP)
print("  USER RETENTION DATASET — END-TO-END PIPELINE BENCHMARK REPORT")
print(_SEP)

print("\n📁  FILE METADATA")
print(_sep2)
print(f"  Path              : {_parquet_path}")
print(f"  File size         : {_file_size_mb:.1f} MB  ({_file_size_bytes:,} bytes)")
print(f"  Rows              : {_n_rows:,}")
print(f"  Columns           : {_n_cols}")

print("\n⏱️  PIPELINE STAGE TIMINGS")
print(_sep2)
print(f"  {'Stage':<40} {'Time (s)':>10}")
print(f"  {'─'*40} {'─'*10}")
_stage_times = [
    ("1. Load (pd.read_parquet)",           _load_time_s),
    ("2. Memory profiling (deep)",           _mem_time_s),
    ("3. Column type audit",                _types_time_s),
    ("4. Null rate analysis",               _null_time_s),
    ("5. Cardinality analysis",             _card_time_s),
    ("6. Redundancy detection",             _red_time_s),
    ("7. Numeric skew / distribution",      _dist_time_s),
    ("8. Memory optimisation estimate",     _opt_time_s),
]
_total_pipeline_time = sum(t for _, t in _stage_times)
for _name, _t in _stage_times:
    print(f"  {_name:<40} {_t:>10.3f}s")
print(f"  {'─'*40} {'─'*10}")
print(f"  {'TOTAL':<40} {_total_pipeline_time:>10.3f}s")

print("\n💾  MEMORY USAGE")
print(_sep2)
print(f"  In-memory size (deep)  : {_total_mem_mb:.1f} MB")
print(f"  Parquet file size      : {_file_size_mb:.1f} MB")
print(f"  In-memory bloat factor : {_mem_ratio:.1f}x")
print(f"\n  Top 15 memory-heaviest columns:")
_top15_mem = (_mem_per_col.drop("Index", errors="ignore")
               .sort_values(ascending=False).head(15))
for _col, _b in _top15_mem.items():
    _mb = _b / (1024**2)
    _pct = _b / _total_mem_bytes * 100
    print(f"    {_col:<55} {_mb:>6.1f} MB  ({_pct:.1f}%)")

print("\n🔡  COLUMN TYPE DISTRIBUTION")
print(_sep2)
for _dt, _cnt in sorted(_dtype_counts.items(), key=lambda x: -x[1]):
    print(f"  {str(_dt):<20} : {_cnt} columns")
print(f"\n  Breakdown by semantic group:")
print(f"    String/object   : {len(_str_cols)} cols")
print(f"    Float           : {len(_float_cols)} cols")
print(f"    Integer         : {len(_int_cols)} cols")
print(f"    Datetime        : {len(_dt_cols)} cols")
print(f"    Boolean/object  : {len(_bool_cols)} cols")

print("\n🕳️  NULL RATE ANALYSIS")
print(_sep2)
print(f"  {'Tier':<35} {'Count':>6}  {'Columns'}")
print(f"  {'─'*35} {'─'*6}  {'─'*30}")
print(f"  {'100% null (useless cols)':<35} {len(_fully_null_cols):>6}  {_fully_null_cols[:5]}{'...' if len(_fully_null_cols)>5 else ''}")
print(f"  {'50–99% null (high null)':<35} {len(_high_null_cols):>6}  {_high_null_cols[:3]}{'...' if len(_high_null_cols)>3 else ''}")
print(f"  {'20–49% null (moderate null)':<35} {len(_moderate_null_cols):>6}  {_moderate_null_cols[:3]}{'...' if len(_moderate_null_cols)>3 else ''}")
print(f"  {'1–19% null (low null)':<35} {len(_low_null_cols):>6}")
print(f"  {'0% null (complete)':<35} {len(_no_null_cols):>6}")

print("\n🎲  CARDINALITY ANALYSIS")
print(_sep2)
print(f"  High-cardinality string columns (>1000 unique or >5% unique ratio) — {len(_high_card_str_cols)} found:")
print(f"  {'Column':<55} {'Unique':>8}  {'Ratio%':>7}")
print(f"  {'─'*55} {'─'*8}  {'─'*7}")
for _c, _u, _r in sorted(_high_card_str_cols, key=lambda x: -x[1])[:20]:
    print(f"  {_c:<55} {_u:>8,}  {_r:>6.1f}%")
if len(_high_card_str_cols) > 20:
    print(f"  ... and {len(_high_card_str_cols)-20} more")

print(f"\n  Low-cardinality string columns (good for category dtype) — {len(_low_card_str_cols)} found:")
print(f"  {'Column':<55} {'Unique':>8}  {'Ratio%':>7}")
print(f"  {'─'*55} {'─'*8}  {'─'*7}")
for _c, _u, _r in sorted(_low_card_str_cols, key=lambda x: x[1])[:20]:
    print(f"  {_c:<55} {_u:>8,}  {_r:>6.1f}%")
if len(_low_card_str_cols) > 20:
    print(f"  ... and {len(_low_card_str_cols)-20} more")

print("\n🔁  REDUNDANCY & STRUCTURAL ISSUES")
print(_sep2)
print(f"  Unnamed index column (drop candidate)   : {'YES → Unnamed: 0' if _has_unnamed_index else 'No'}")
print(f"  Constant / zero-variance columns        : {len(_constant_cols)}  {_constant_cols[:5]}{'...' if len(_constant_cols)>5 else ''}")
print(f"  Duplicate column names                  : {len(_dup_col_names)}  {_dup_col_names[:5]}")
print(f"  prop_$set.* mirror columns              : {len(_set_cols)}  (duplicate current-session state)")
print(f"  prop_$set_once.* mirror columns         : {len(_set_once_cols)}  (duplicate initial acquisition state)")
print(f"  TOTAL mirror/redundant columns          : {len(_set_cols)+len(_set_once_cols)}")

print("\n📈  SKEW / DISTRIBUTION ANALYSIS (numeric cols)")
print(_sep2)
print(f"  Numeric columns analysed : {len(_numeric_cols)}")
print(f"  Highly skewed (|skew|>3) : {len(_highly_skewed)}")
if len(_highly_skewed) > 0:
    print(f"  {'Column':<55} {'Skew':>8}")
    print(f"  {'─'*55} {'─'*8}")
    for _col, _sk in _highly_skewed.head(15).items():
        print(f"  {_col:<55} {_sk:>8.2f}")

print("\n💡  INEFFICIENCY SUMMARY & RECOMMENDATIONS")
print(_sep2)
_issues = []

if _has_unnamed_index:
    _issues.append(("CRITICAL", "Unnamed: 0",
                    "Row index stored as int32 column. Drop with df.drop(columns=['Unnamed: 0']) — saves ~1.6 MB."))

if _fully_null_cols:
    _issues.append(("CRITICAL", f"{len(_fully_null_cols)} 100%-null columns",
                    f"Completely empty columns waste memory. Drop them. Estimated saving: "
                    f"{sum(_mem_per_col.get(c,0) for c in _fully_null_cols)/(1024**2):.1f} MB"))

if _set_cols or _set_once_cols:
    _mirror_mb = sum(_mem_per_col.get(c, 0) for c in _set_cols + _set_once_cols) / (1024**2)
    _issues.append(("HIGH", f"{len(_set_cols)+len(_set_once_cols)} prop_$set / prop_$set_once columns",
                    f"These mirror columns duplicate data already in non-prefixed columns. "
                    f"Estimated saving: {_mirror_mb:.1f} MB"))

if _high_card_str_cols:
    _hc_mb = sum(_mem_per_col.get(c, 0) for c, _, _ in _high_card_str_cols) / (1024**2)
    _issues.append(("HIGH", f"{len(_high_card_str_cols)} high-cardinality string columns",
                    f"UUIDs/URLs stored as Python object strings. Use hash/integer IDs or "
                    f"pyarrow string types. Estimated col memory: {_hc_mb:.1f} MB"))

if _low_card_str_cols:
    _issues.append(("MEDIUM", f"{len(_low_card_str_cols)} low-cardinality string cols",
                    f"Convert to pandas Categorical. Estimated saving: {_cat_saving_mb:.1f} MB"))

if _f64_cols:
    _issues.append(("MEDIUM", f"{len(_f64_cols)} float64 columns",
                    f"Downcast to float32 where precision allows. Estimated saving: {_f64_saving_mb:.1f} MB"))

if len(_highly_skewed) > 0:
    _issues.append(("MEDIUM", f"{len(_highly_skewed)} highly skewed numeric cols",
                    "Consider log1p transform before modelling. Skewed data bloats float representation."))

if _high_null_cols:
    _issues.append(("LOW", f"{len(_high_null_cols)} columns with 50–99% nulls",
                    "Evaluate whether these sparse columns are needed at query time. "
                    "Consider sparse storage or dropping if unused."))

_severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_issues.sort(key=lambda x: _severity_order.get(x[0], 9))

for _sev, _title, _desc in _issues:
    print(f"\n  [{_sev}] {_title}")
    print(f"  → {_desc}")

print("\n📊  MEMORY OPTIMISATION POTENTIAL")
print(_sep2)
print(f"  Current in-memory footprint  : {_total_mem_mb:>8.1f} MB")
print(f"  Float64 → float32 saving     : {_f64_saving_mb:>8.1f} MB")
print(f"  Int64 → int16/int32 saving   : {_i64_saving_mb:>8.1f} MB")
print(f"  String → category saving     : {_cat_saving_mb:>8.1f} MB")
print(f"  ─────────────────────────────────────────")
print(f"  Estimated optimised footprint: {(_total_mem_mb - _total_saving_mb):>8.1f} MB")
print(f"  Total potential saving       : {_total_saving_mb:>8.1f} MB  ({_total_saving_mb/_total_mem_mb*100:.1f}%)")

print(f"\n{_SEP}")
print("  END OF BENCHMARK REPORT")
print(_SEP)

# Export structured summary as a variable for downstream use
benchmark_report = {
    "file_size_mb": round(_file_size_mb, 2),
    "rows": _n_rows,
    "columns": _n_cols,
    "load_time_s": round(_load_time_s, 3),
    "total_mem_mb": round(_total_mem_mb, 2),
    "mem_bloat_factor": round(_mem_ratio, 2),
    "total_pipeline_time_s": round(_total_pipeline_time, 3),
    "null_rates": _null_rates.to_dict(),
    "cardinality": _cardinality,
    "fully_null_cols": _fully_null_cols,
    "high_null_cols": _high_null_cols,
    "high_card_str_cols": [c for c, _, _ in _high_card_str_cols],
    "low_card_str_cols": [c for c, _, _ in _low_card_str_cols],
    "redundant_mirror_cols": _set_cols + _set_once_cols,
    "constant_cols": _constant_cols,
    "highly_skewed_cols": _highly_skewed.to_dict(),
    "total_saving_mb": round(_total_saving_mb, 2),
    "issues": [{"severity": s, "title": t, "description": d} for s, t, d in _issues],
}
print(f"\n✅  benchmark_report variable created with {len(benchmark_report)} keys.")
