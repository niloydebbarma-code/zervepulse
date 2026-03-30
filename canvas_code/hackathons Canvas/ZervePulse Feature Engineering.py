
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# SETUP — work on the dtype-optimised events dataframe
# ─────────────────────────────────────────────────────────────────
_df = optimised_retention.copy()

# Convert category/object columns to plain strings safely
# event is a category dtype in optimised_retention
_event = _df["event"].astype(str)          # categories → strings
_ts    = pd.to_datetime(_df["timestamp"])
_df["_event_str"] = _event
_df["_ts"]        = _ts

# ─────────────────────────────────────────────────────────────────
# DATASET-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────
_max_date = _ts.max()
_14_days   = pd.Timedelta(days=14)
_7_days    = pd.Timedelta(days=7)

# ─────────────────────────────────────────────────────────────────
# PRE-COMPUTE PER-ROW FLAGS
# ─────────────────────────────────────────────────────────────────

# agent_ events
_df["_is_agent"]   = _event.str.startswith("agent_", na=False).astype(np.int8)

# deploy events
_df["_is_deploy"]  = _event.str.contains("deploy", case=False, na=False).astype(np.int8)

# onboarding
_df["_is_onboard"] = (_event == "canvas_onboarding_tour_finished").astype(np.int8)

# posthog-python lib — prop_$lib is category in optimised_retention
_lib_str = _df["prop_$lib"].astype(str).fillna("")
_df["_is_py_lib"] = (_lib_str == "posthog-python").astype(np.int8)

# Credits used (stored as str)
_df["_credits_used_num"] = pd.to_numeric(_df["prop_credits_used"], errors="coerce").fillna(0.0)

# Credit amount (stored as str)
_df["_credit_amount_num"] = pd.to_numeric(_df["prop_credit_amount"], errors="coerce")

# Canvas UUIDs per row — fillna with "" to avoid float NaN issue
_pathname_str = _df["prop_$pathname"].fillna("").astype(str)
_uuid_pattern = re.compile(r"/canvas/([a-f0-9\-]{36})")
_df["_canvas_uuid"] = _pathname_str.apply(lambda x: _uuid_pattern.findall(x))

# Date only (for days_active)
_df["_date"] = _ts.dt.normalize()

# ─────────────────────────────────────────────────────────────────
# GROUPBY AGGREGATIONS
# ─────────────────────────────────────────────────────────────────
print("⏳ Computing groupby aggregations …")

_grp = _df.groupby("distinct_id", sort=False)

# Core aggregations
_agg = _grp.agg(
    total_events         = ("_event_str", "count"),
    agent_usage_count    = ("_is_agent", "sum"),
    days_active          = ("_date", "nunique"),
    unique_event_types   = ("_event_str", "nunique"),
    onboarding_complete  = ("_is_onboard", "max"),
    has_deployed         = ("_is_deploy", "max"),
    is_python_user       = ("_is_py_lib", "max"),
    total_credits_used   = ("_credits_used_num", "sum"),
    first_event_date     = ("_ts", "min"),
    last_event_date      = ("_ts", "max"),
).reset_index()
print(f"  ✅ Core agg done: {_agg.shape}")

# ── session_count ─────────────────────────────────────────────────
# prop_$session_id is str dtype in optimised_retention; NaN as "nan"
# Convert to proper nulls before nunique
_sess_clean = _df["prop_$session_id"].astype(str).replace({"nan": None, "": None, "None": None})
_df["_sess_clean"] = _sess_clean

_sess_agg = _df.groupby("distinct_id", sort=False).agg(
    _sess_count_raw = ("_sess_clean", lambda x: x.dropna().nunique()),
    _sess_all_null  = ("_sess_clean", lambda x: x.isna().all()),
).reset_index()

# Build days_active lookup for fallback
_days_lookup = _agg.set_index("distinct_id")["days_active"]
_sess_agg["session_count"] = np.where(
    _sess_agg["_sess_all_null"],
    _sess_agg["distinct_id"].map(_days_lookup).values,
    _sess_agg["_sess_count_raw"].clip(lower=1)
)
print(f"  ✅ session_count done")

# ── unique_canvases ───────────────────────────────────────────────
_canvas_agg = _df.groupby("distinct_id", sort=False)["_canvas_uuid"].apply(
    lambda lists: len(set(uuid for sublist in lists for uuid in sublist))
).reset_index()
_canvas_agg.columns = ["distinct_id", "unique_canvases"]
print(f"  ✅ unique_canvases done")

# ── tool_diversity_count ──────────────────────────────────────────
# prop_tool_name is category in optimised_retention
_tool_str = _df["prop_tool_name"].astype(str).replace({"nan": None, "": None, "None": None})
_df["_tool_clean"] = _tool_str
_tool_agg = _df.groupby("distinct_id", sort=False).agg(
    tool_diversity_count = ("_tool_clean", lambda x: x.dropna().nunique())
).reset_index()
print(f"  ✅ tool_diversity_count done")

# ── avg_credit_per_transaction ────────────────────────────────────
_credit_agg = _df.groupby("distinct_id", sort=False).agg(
    _credit_sum   = ("_credit_amount_num", "sum"),
    _credit_count = ("_credit_amount_num", lambda x: x.dropna().count()),
).reset_index()
_credit_agg["avg_credit_per_transaction"] = np.where(
    _credit_agg["_credit_count"] > 0,
    _credit_agg["_credit_sum"] / _credit_agg["_credit_count"],
    0.0
)
print(f"  ✅ avg_credit_per_transaction done")

# ── python_sdk_share ──────────────────────────────────────────────
# prop_$python_version is category dtype
_pyver_str = _df["prop_$python_version"].astype(str).replace({"nan": None, "": None, "None": None})
_df["_py_ver"] = _pyver_str
_py_agg = _df.groupby("distinct_id", sort=False).agg(
    _py_count      = ("_py_ver", lambda x: x.dropna().count()),
    _total_events2 = ("_event_str", "count"),
).reset_index()
_py_agg["python_sdk_share"] = np.where(
    _py_agg["_total_events2"] > 0,
    (_py_agg["_py_count"] / _py_agg["_total_events2"]).round(4),
    0.0
)
print(f"  ✅ python_sdk_share done")

# ── time_to_first_agent_minutes ───────────────────────────────────
_agent_ts_df = _df[_df["_is_agent"] == 1].groupby("distinct_id", sort=False)["_ts"].min().reset_index()
_agent_ts_df.columns = ["distinct_id", "_first_agent_ts"]
print(f"  ✅ first agent ts for {len(_agent_ts_df):,} users")

# ── churn_velocity ────────────────────────────────────────────────
print("  ⏳ Computing churn_velocity (7-day windows) …")
_first_evt_map = _agg.set_index("distinct_id")["first_event_date"].to_dict()
_last_evt_map  = _agg.set_index("distinct_id")["last_event_date"].to_dict()

_df["_first_evt"] = _df["distinct_id"].map(_first_evt_map)
_df["_last_evt"]  = _df["distinct_id"].map(_last_evt_map)

_df["_in_first7"] = (_df["_ts"] < (_df["_first_evt"] + _7_days)).astype(np.int8)
_df["_in_last7"]  = (_df["_ts"] >= (_df["_last_evt"] - _7_days)).astype(np.int8)

_churn_vel = _df.groupby("distinct_id", sort=False).agg(
    _first7 = ("_in_first7", "sum"),
    _last7  = ("_in_last7", "sum"),
).reset_index()
_churn_vel["churn_velocity"] = np.where(
    _churn_vel["_first7"] > 0,
    ((_churn_vel["_first7"] - _churn_vel["_last7"]) / _churn_vel["_first7"]).round(4),
    0.0
)
print(f"  ✅ churn_velocity done")

# ─────────────────────────────────────────────────────────────────
# MERGE ALL FEATURES
# ─────────────────────────────────────────────────────────────────
print("⏳ Merging feature tables …")

user_features = _agg.copy()
user_features = user_features.merge(_sess_agg[["distinct_id", "session_count"]], on="distinct_id", how="left")
user_features = user_features.merge(_canvas_agg, on="distinct_id", how="left")
user_features = user_features.merge(_tool_agg, on="distinct_id", how="left")
user_features = user_features.merge(_credit_agg[["distinct_id", "avg_credit_per_transaction"]], on="distinct_id", how="left")
user_features = user_features.merge(_py_agg[["distinct_id", "python_sdk_share"]], on="distinct_id", how="left")
user_features = user_features.merge(_agent_ts_df, on="distinct_id", how="left")
user_features = user_features.merge(_churn_vel[["distinct_id", "churn_velocity"]], on="distinct_id", how="left")

# ─────────────────────────────────────────────────────────────────
# DERIVED FEATURES
# ─────────────────────────────────────────────────────────────────
print("⏳ Computing derived features …")

user_features["ai_adoption_index"] = np.where(
    user_features["total_events"] > 0,
    (user_features["agent_usage_count"] / user_features["total_events"]).round(4),
    0.0
)

user_features["session_count"] = user_features["session_count"].fillna(1).clip(lower=1)
user_features["session_depth_score"] = (
    user_features["total_events"] / user_features["session_count"]
).round(2)

user_features["has_used_agent"] = (user_features["agent_usage_count"] > 0).astype(np.int8)

user_features["observation_days"] = (
    (user_features["last_event_date"] - user_features["first_event_date"])
    .dt.total_seconds() / 86400
).clip(lower=1).round(2)

user_features["time_to_first_agent_minutes"] = np.where(
    user_features["_first_agent_ts"].notna(),
    ((user_features["_first_agent_ts"] - user_features["first_event_date"])
     .dt.total_seconds() / 60).round(2),
    -1
)

user_features["feature_breadth_score"] = (
    user_features["unique_event_types"] / 141
).round(4)

user_features["churn_label"] = (
    user_features["last_event_date"] < (_max_date - _14_days)
).astype(np.int8)

# Fill NaNs
user_features["unique_canvases"]            = user_features["unique_canvases"].fillna(0).astype(int)
user_features["tool_diversity_count"]       = user_features["tool_diversity_count"].fillna(0).astype(int)
user_features["avg_credit_per_transaction"] = user_features["avg_credit_per_transaction"].fillna(0.0)
user_features["python_sdk_share"]           = user_features["python_sdk_share"].fillna(0.0)
user_features["churn_velocity"]             = user_features["churn_velocity"].fillna(0.0)

# Drop internal helper columns
user_features.drop(columns=["_first_agent_ts"], errors="ignore", inplace=True)

# ─────────────────────────────────────────────────────────────────
# FINAL COLUMN ORDER
# ─────────────────────────────────────────────────────────────────
_final_cols = [
    "distinct_id",
    "total_events",
    "agent_usage_count",
    "ai_adoption_index",
    "days_active",
    "session_count",
    "session_depth_score",
    "unique_event_types",
    "onboarding_complete",
    "has_used_agent",
    "has_deployed",
    "is_python_user",
    "total_credits_used",
    "unique_canvases",
    "first_event_date",
    "last_event_date",
    "observation_days",
    "time_to_first_agent_minutes",
    "feature_breadth_score",
    "churn_velocity",
    "tool_diversity_count",
    "avg_credit_per_transaction",
    "python_sdk_share",
    "churn_label",
]

user_features = user_features[_final_cols].copy()

# ─────────────────────────────────────────────────────────────────
# ANALYSIS A — SHAPE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  A. SHAPE OF user_features")
print("=" * 65)
print(f"  Rows   : {user_features.shape[0]:,}")
print(f"  Columns: {user_features.shape[1]}  (distinct_id + 23 features)")

# ─────────────────────────────────────────────────────────────────
# ANALYSIS B — FIRST 5 ROWS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  B. FIRST 5 ROWS (23 feature columns, excl. distinct_id)")
print("=" * 65)
_feat_cols = [c for c in _final_cols if c != "distinct_id"]
print(user_features[_feat_cols].head(5).to_string())

# ─────────────────────────────────────────────────────────────────
# ANALYSIS C — DESCRIBE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  C. DESCRIBE() — numeric feature columns")
print("=" * 65)
_num_cols = user_features[_feat_cols].select_dtypes(include=[np.number]).columns.tolist()
print(user_features[_num_cols].describe().round(4).to_string())

# ─────────────────────────────────────────────────────────────────
# ANALYSIS D — CHURN LABEL VALUE COUNTS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  D. churn_label VALUE COUNTS")
print("=" * 65)
_vc    = user_features["churn_label"].value_counts()
_vpct  = user_features["churn_label"].value_counts(normalize=True) * 100
_churn_tbl = pd.DataFrame({"count": _vc, "pct_%": _vpct.round(2)})
_churn_tbl.index.name = "churn_label"
print(_churn_tbl.to_string())

# ─────────────────────────────────────────────────────────────────
# ANALYSIS E — CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  E. CORRELATION HEATMAP")
print("=" * 65)

BG     = "#1D1D20"
TEXT   = "#fbfbff"
SUBTLE = "#909094"

_corr = user_features[_num_cols].corr()

fig_corr, ax_corr = plt.subplots(figsize=(16, 13))
fig_corr.patch.set_facecolor(BG)
ax_corr.set_facecolor(BG)

_im = ax_corr.imshow(_corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")

_cbar = fig_corr.colorbar(_im, ax=ax_corr, fraction=0.03, pad=0.02)
_cbar.ax.tick_params(colors=TEXT, labelsize=9)
_cbar.ax.set_ylabel("Correlation", color=SUBTLE, fontsize=10)
_cbar.outline.set_edgecolor(SUBTLE)

ax_corr.set_xticks(range(len(_num_cols)))
ax_corr.set_yticks(range(len(_num_cols)))
ax_corr.set_xticklabels(_num_cols, rotation=45, ha="right", color=TEXT, fontsize=7.5)
ax_corr.set_yticklabels(_num_cols, color=TEXT, fontsize=7.5)

for _i in range(len(_num_cols)):
    for _j in range(len(_num_cols)):
        _v = _corr.values[_i, _j]
        ax_corr.text(_j, _i, f"{_v:.2f}",
                     ha="center", va="center",
                     fontsize=5.5,
                     color=BG if abs(_v) >= 0.6 else TEXT,
                     fontweight="bold")

ax_corr.set_title("ZervePulse — Feature Correlation with Churn",
                  color=TEXT, fontsize=15, fontweight="bold", pad=16)
for _sp in ax_corr.spines.values():
    _sp.set_visible(False)

plt.tight_layout(pad=1.5)
plt.show()
print("  ✅ Heatmap displayed.")

# ─────────────────────────────────────────────────────────────────
# ANALYSIS F — TOP CORRELATIONS WITH churn_label
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  F. TOP CORRELATIONS WITH churn_label")
print("=" * 65)
_churn_corr = _corr["churn_label"].drop("churn_label").sort_values(ascending=False)

print("\n  ▲ Top 5 POSITIVELY correlated (→ higher churn risk):")
for _f, _r in _churn_corr.head(5).items():
    print(f"    {_f:<38}  r = {_r:+.4f}")

print("\n  ▼ Top 5 NEGATIVELY correlated (→ lower churn risk / retained):")
for _f, _r in _churn_corr.tail(5).items():
    print(f"    {_f:<38}  r = {_r:+.4f}")

# ─────────────────────────────────────────────────────────────────
# ANALYSIS G — SAVE CSV
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  G. SAVE user_features.csv")
print("=" * 65)
_csv_path = "user_features.csv"
user_features.to_csv(_csv_path, index=False)
_sz = os.path.getsize(_csv_path)
print(f"  ✅ Saved: {_csv_path}")
print(f"     File size  : {_sz / 1024:.1f} KB  ({_sz:,} bytes)")
print(f"     Rows saved : {len(user_features):,}")
print(f"     Columns    : {user_features.shape[1]}  (distinct_id + 23 features)")

print("\n" + "=" * 65)
print("  ✅ ZervePulse COMPLETE")
print(f"     user_features: {user_features.shape}")
print("=" * 65)
