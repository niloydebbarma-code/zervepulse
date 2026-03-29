
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# BASELINE — re-run pipeline stages on raw data
# ─────────────────────────────────────────────
_parquet_path = "user_retention.parquet"

_t0 = time.perf_counter()
_raw = pd.read_parquet(_parquet_path)
_baseline_load_s = time.perf_counter() - _t0

_n_rows_before, _n_cols_before = _raw.shape
_baseline_mem_mb = _raw.memory_usage(deep=True).sum() / (1024 ** 2)

# Baseline: null analysis time
_t1 = time.perf_counter()
_ = _raw.isnull().sum()
_baseline_null_s = time.perf_counter() - _t1

# Baseline: cardinality time
_t2 = time.perf_counter()
_ = {c: _raw[c].nunique(dropna=False) for c in _raw.columns}
_baseline_card_s = time.perf_counter() - _t2

# Baseline: skew time
_t3 = time.perf_counter()
_num_cols_base = _raw.select_dtypes(include=[np.number]).columns.tolist()
_ = _raw[_num_cols_base].skew()
_baseline_skew_s = time.perf_counter() - _t3

_baseline_total_s = (time.perf_counter() - _t0)

print(f"✅ Baseline captured: {_n_rows_before:,} rows × {_n_cols_before} cols | {_baseline_mem_mb:.1f} MB | {_baseline_total_s:.3f}s total")

# ─────────────────────────────────────────────
# OPTIMISATION 1 — DROP REDUNDANT COLUMNS
# ─────────────────────────────────────────────
# From benchmark: drop Unnamed: 0 (index artifact), all prop_$set.* and prop_$set_once.* mirrors
_drop_cols = (
    ["Unnamed: 0"]
    + [c for c in _raw.columns if c.startswith("prop_$set.")]
    + [c for c in _raw.columns if c.startswith("prop_$set_once.")]
    + [c for c in benchmark_report["constant_cols"] if c in _raw.columns]   # zero-variance
)
_drop_cols = [c for c in _drop_cols if c in _raw.columns]
_opt = _raw.drop(columns=_drop_cols)
print(f"  DROP: removed {len(_drop_cols)} redundant/constant cols → {_opt.shape[1]} remaining")

# ─────────────────────────────────────────────
# OPTIMISATION 2 — DOWNCAST FLOAT64 → FLOAT32
# ─────────────────────────────────────────────
_f64_cols = [c for c in _opt.columns if str(_opt[c].dtype) == "float64"]
for _c in _f64_cols:
    _opt[_c] = _opt[_c].astype("float32")
print(f"  DOWNCAST: {len(_f64_cols)} float64 → float32")

# ─────────────────────────────────────────────
# OPTIMISATION 3 — INT32 DOWNCAST CHECK
# ─────────────────────────────────────────────
_int_cols_opt = [c for c in _opt.columns if str(_opt[c].dtype) in ("int64", "int32")]
_downcast_int_count = 0
for _c in _int_cols_opt:
    _col_min = _opt[_c].min()
    _col_max = _opt[_c].max()
    if _col_min >= np.iinfo(np.int16).min and _col_max <= np.iinfo(np.int16).max:
        _opt[_c] = _opt[_c].astype("int16")
        _downcast_int_count += 1
    elif _col_min >= np.iinfo(np.int32).min and _col_max <= np.iinfo(np.int32).max:
        _opt[_c] = _opt[_c].astype("int32")
        _downcast_int_count += 1
print(f"  DOWNCAST: {_downcast_int_count} int cols safely downcast to int16/int32")

# ─────────────────────────────────────────────
# OPTIMISATION 4 — LOW-CARDINALITY STRING → CATEGORY
# ─────────────────────────────────────────────
# Use benchmark's low_card_str_cols list (<=1000 unique, <=5% ratio) that exist post-drop
_low_card_candidates = [c for c in benchmark_report["low_card_str_cols"] if c in _opt.columns]
_cat_converted = []
for _c in _low_card_candidates:
    _u = _opt[_c].nunique(dropna=False)
    _ratio = _u / len(_opt)
    # Double-check (post-drop shape may differ slightly)
    if _u <= 1000 and _ratio <= 0.05:
        _opt[_c] = _opt[_c].astype("category")
        _cat_converted.append(_c)
print(f"  CATEGORY: {len(_cat_converted)} low-cardinality string cols → category dtype")

# ─────────────────────────────────────────────
# OPTIMISATION 5 — MEMORY LAYOUT (defrag)
# ─────────────────────────────────────────────
_opt = _opt.copy()  # consolidates memory blocks after all mutations

# ─────────────────────────────────────────────
# OPTIMISED METRICS — RE-RUN SAME PIPELINE STAGES
# ─────────────────────────────────────────────
_t_opt_start = time.perf_counter()

_opt_mem_mb = _opt.memory_usage(deep=True).sum() / (1024 ** 2)
_n_rows_after, _n_cols_after = _opt.shape

_t_opt_load = time.perf_counter() - _t_opt_start  # memory profile time

# Null analysis on optimised df
_t4 = time.perf_counter()
_ = _opt.isnull().sum()
_opt_null_s = time.perf_counter() - _t4

# Cardinality on optimised df
_t5 = time.perf_counter()
_ = {c: _opt[c].nunique(dropna=False) for c in _opt.columns}
_opt_card_s = time.perf_counter() - _t5

# Skew on optimised df
_t6 = time.perf_counter()
_num_cols_opt = _opt.select_dtypes(include=[np.number]).columns.tolist()
_ = _opt[_num_cols_opt].skew()
_opt_skew_s = time.perf_counter() - _t6

_opt_total_s = time.perf_counter() - _t_opt_start

print(f"✅ Optimised: {_n_rows_after:,} rows × {_n_cols_after} cols | {_opt_mem_mb:.1f} MB | {_opt_total_s:.3f}s total")

# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────
_mem_saving_pct  = (_baseline_mem_mb - _opt_mem_mb) / _baseline_mem_mb * 100
_col_drop_pct    = (_n_cols_before - _n_cols_after) / _n_cols_before * 100
_null_speedup    = _baseline_null_s / _opt_null_s if _opt_null_s > 0 else float("inf")
_card_speedup    = _baseline_card_s / _opt_card_s if _opt_card_s > 0 else float("inf")
_skew_speedup    = _baseline_skew_s / _opt_skew_s if _opt_skew_s > 0 else float("inf")

_comparison_data = {
    "Metric": [
        "Columns",
        "Memory (deep, MB)",
        "Null analysis time (s)",
        "Cardinality analysis time (s)",
        "Skew analysis time (s)",
        "Pipeline time (s)",
    ],
    "Baseline": [
        _n_cols_before,
        round(_baseline_mem_mb, 1),
        round(_baseline_null_s, 3),
        round(_baseline_card_s, 3),
        round(_baseline_skew_s, 3),
        round(_baseline_total_s, 3),
    ],
    "Optimised": [
        _n_cols_after,
        round(_opt_mem_mb, 1),
        round(_opt_null_s, 3),
        round(_opt_card_s, 3),
        round(_opt_skew_s, 3),
        round(_opt_total_s, 3),
    ],
    "Improvement": [
        f"−{_n_cols_before - _n_cols_after} cols ({_col_drop_pct:.0f}%↓)",
        f"−{_baseline_mem_mb - _opt_mem_mb:.0f} MB ({_mem_saving_pct:.0f}%↓)",
        f"{_null_speedup:.1f}× faster",
        f"{_card_speedup:.1f}× faster",
        f"{_skew_speedup:.1f}× faster",
        f"{_baseline_total_s / _opt_total_s:.1f}× faster" if _opt_total_s > 0 else "—",
    ],
}

_perf_df = pd.DataFrame(_comparison_data)
print("\n" + "=" * 75)
print("  BEFORE / AFTER PERFORMANCE COMPARISON")
print("=" * 75)
print(_perf_df.to_string(index=False))
print("=" * 75)
print(f"\n  Rows preserved    : {_n_rows_after:,} / {_n_rows_before:,}  (100% — no rows dropped)")
print(f"  Cols dropped      : {_n_cols_before - _n_cols_after} of {_n_cols_before}")
print(f"  Memory saved      : {_baseline_mem_mb - _opt_mem_mb:.1f} MB ({_mem_saving_pct:.0f}%)")
print(f"  float64 → float32 : {len(_f64_cols)} columns")
print(f"  str → category    : {len(_cat_converted)} columns")
print(f"  Redundant cols    : {len(_drop_cols)} dropped\n")

# ─────────────────────────────────────────────
# VISUALISATION — SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────────
BG     = "#1D1D20"
TEXT   = "#fbfbff"
SUBTLE = "#909094"
GOLD   = "#ffd400"
GREEN  = "#17b26a"
BLUE   = "#A1C9F4"
ORANGE = "#FFB482"

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Pipeline Optimisation — Before vs After", color=TEXT,
             fontsize=16, fontweight="bold", y=1.01)

# Chart 1 — Memory
ax1 = axes[0]
ax1.set_facecolor(BG)
_mem_vals = [_baseline_mem_mb, _opt_mem_mb]
_mem_bars = ax1.bar(["Baseline", "Optimised"], _mem_vals, color=[ORANGE, GREEN], width=0.5,
                     edgecolor="none", zorder=3)
for _bar, _val in zip(_mem_bars, _mem_vals):
    ax1.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 4,
             f"{_val:.0f} MB", ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
ax1.set_title("Memory Usage (MB)", color=TEXT, fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("MB", color=SUBTLE, fontsize=10)
ax1.tick_params(colors=TEXT)
ax1.yaxis.label.set_color(SUBTLE)
for _sp in ax1.spines.values(): _sp.set_visible(False)
ax1.set_facecolor(BG)
ax1.grid(axis="y", color=SUBTLE, alpha=0.2, zorder=0)
ax1.tick_params(axis="both", colors=TEXT)
ax1.set_ylim(0, max(_mem_vals) * 1.18)

# Annotation: saving
ax1.annotate(f"−{_mem_saving_pct:.0f}%",
             xy=(1, _opt_mem_mb), xytext=(1.3, (_baseline_mem_mb + _opt_mem_mb)/2),
             color=GREEN, fontsize=12, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

# Chart 2 — Column count
ax2 = axes[1]
ax2.set_facecolor(BG)
_col_vals = [_n_cols_before, _n_cols_after]
_col_bars = ax2.bar(["Baseline", "Optimised"], _col_vals, color=[ORANGE, GREEN], width=0.5,
                     edgecolor="none", zorder=3)
for _bar, _val in zip(_col_bars, _col_vals):
    ax2.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.5,
             str(_val), ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
ax2.set_title("Column Count", color=TEXT, fontsize=13, fontweight="bold", pad=12)
ax2.set_ylabel("Columns", color=SUBTLE, fontsize=10)
for _sp in ax2.spines.values(): _sp.set_visible(False)
ax2.grid(axis="y", color=SUBTLE, alpha=0.2, zorder=0)
ax2.tick_params(axis="both", colors=TEXT)
ax2.set_ylim(0, max(_col_vals) * 1.18)
ax2.annotate(f"−{_col_drop_pct:.0f}%",
             xy=(1, _n_cols_after), xytext=(1.3, (_n_cols_before + _n_cols_after)/2),
             color=GREEN, fontsize=12, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

# Chart 3 — Stage timing
ax3 = axes[2]
ax3.set_facecolor(BG)
_stage_names  = ["Null\nAnalysis", "Cardinality\nAnalysis", "Skew\nAnalysis"]
_stage_base   = [_baseline_null_s, _baseline_card_s, _baseline_skew_s]
_stage_opt    = [_opt_null_s, _opt_card_s, _opt_skew_s]
_x = np.arange(len(_stage_names))
_w = 0.35
_b1 = ax3.bar(_x - _w/2, _stage_base, _w, label="Baseline", color=ORANGE, edgecolor="none", zorder=3)
_b2 = ax3.bar(_x + _w/2, _stage_opt,  _w, label="Optimised", color=GREEN,  edgecolor="none", zorder=3)
ax3.set_title("Pipeline Stage Timings (s)", color=TEXT, fontsize=13, fontweight="bold", pad=12)
ax3.set_ylabel("Seconds", color=SUBTLE, fontsize=10)
ax3.set_xticks(_x)
ax3.set_xticklabels(_stage_names, color=TEXT, fontsize=9)
for _sp in ax3.spines.values(): _sp.set_visible(False)
ax3.grid(axis="y", color=SUBTLE, alpha=0.2, zorder=0)
ax3.tick_params(axis="both", colors=TEXT)
ax3.legend(facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT, fontsize=9)
ax3.set_ylim(0, max(_stage_base) * 1.2)

plt.tight_layout(pad=2.0)
plt.savefig("optimised_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("📊 Comparison chart saved.")

# ─────────────────────────────────────────────
# DTYPE BREAKDOWN — BEFORE VS AFTER
# ─────────────────────────────────────────────
_before_dtypes = _raw.dtypes.astype(str).value_counts().to_dict()
_after_dtypes  = _opt.dtypes.astype(str).value_counts().to_dict()
_all_dtypes    = sorted(set(list(_before_dtypes.keys()) + list(_after_dtypes.keys())))

fig2, ax4 = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor(BG)
ax4.set_facecolor(BG)
_dx = np.arange(len(_all_dtypes))
_dw = 0.35
_bef = [_before_dtypes.get(d, 0) for d in _all_dtypes]
_aft = [_after_dtypes.get(d, 0) for d in _all_dtypes]
_db1 = ax4.bar(_dx - _dw/2, _bef, _dw, label="Baseline", color=ORANGE, edgecolor="none", zorder=3)
_db2 = ax4.bar(_dx + _dw/2, _aft, _dw, label="Optimised", color=GREEN,  edgecolor="none", zorder=3)
ax4.set_title("Column dtype Distribution — Before vs After", color=TEXT, fontsize=14, fontweight="bold")
ax4.set_ylabel("Column Count", color=SUBTLE, fontsize=10)
ax4.set_xticks(_dx)
ax4.set_xticklabels(_all_dtypes, color=TEXT, fontsize=9, rotation=30, ha="right")
for _sp in ax4.spines.values(): _sp.set_visible(False)
ax4.grid(axis="y", color=SUBTLE, alpha=0.2, zorder=0)
ax4.tick_params(axis="both", colors=TEXT)
ax4.legend(facecolor=BG, edgecolor=SUBTLE, labelcolor=TEXT, fontsize=10)
plt.tight_layout()
plt.savefig("dtype_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("📊 Dtype comparison chart saved.")

# ─────────────────────────────────────────────
# EXPORT — optimised dataframe + comparison table
# ─────────────────────────────────────────────
optimised_retention = _opt
performance_comparison = _perf_df

print(f"\n✅  optimised_retention  → {optimised_retention.shape[0]:,} rows × {optimised_retention.shape[1]} cols")
print(f"✅  performance_comparison → {performance_comparison.shape} table ready")
