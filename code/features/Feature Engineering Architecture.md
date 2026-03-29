# Feature Engineering Architecture

## Purpose
Aggregate the 409,287-row optimised event log (`optimised_retention`) by `distinct_id` to produce a 5,410-row user-level feature matrix. This block engineers all 23 behavioural features — including five custom ZervePulse metrics — and applies the binary 14-day churn label that serves as the classification target for all subsequent modelling.

## Input Data
| Variable | Type | Shape | Description |
|---|---|---|---|
| `optimised_retention` | `DataFrame` | 409,287 × 74 | Dtype-optimised event log (228.1 MB, 37 category + 13 float32 + 6 datetime cols) |

**Key source columns used**: `distinct_id`, `event` (category), `timestamp` (datetime64), `prop_$session_id` (str), `prop_$pathname` (str — canvas UUID extraction via regex), `prop_$lib` (category — Python SDK detection), `prop_$python_version` (category), `prop_tool_name` (category), `prop_credit_amount` (str → numeric), `prop_credits_used` (str → numeric)

## Features Being Computed (23 total)

### Core Aggregations (10 via `groupby.agg`)
`total_events` · `agent_usage_count` (events starting with `agent_`) · `days_active` (`dt.normalize().nunique()`) · `unique_event_types` · `onboarding_complete` (binary, `canvas_onboarding_tour_finished`) · `has_deployed` (binary, `deploy*` events) · `is_python_user` (binary, `prop_$lib == "posthog-python"`) · `total_credits_used` (numeric sum of `prop_credits_used`) · `first_event_date` · `last_event_date`

### Derived Aggregations (additional groupby passes)
- **`session_count`** — distinct non-null `prop_$session_id` values per user; falls back to `days_active` for users with all-null session IDs
- **`unique_canvases`** — canvas UUIDs extracted from `prop_$pathname` via regex `/canvas/([a-f0-9\-]{36})`, then `set()` deduplicated per user
- **`tool_diversity_count`** — distinct non-null `prop_tool_name` values per user
- **`avg_credit_per_transaction`** — `sum(prop_credit_amount) / count(prop_credit_amount)` per user (0 if no transactions)
- **`python_sdk_share`** — `count(prop_$python_version not null) / total_events` per user
- **`time_to_first_agent_minutes`** — `(first agent_* timestamp − first_event_date).total_seconds() / 60`; sentinel value `−1` for users who never used the agent

### Five Custom ZervePulse Metrics ✦
| Feature | Formula |
|---|---|
| `ai_adoption_index` | `agent_usage_count / total_events` |
| `session_depth_score` | `total_events / session_count` |
| `feature_breadth_score` | `unique_event_types / 141` (proportion of full event vocabulary) |
| `churn_velocity` | `(events_in_first_7d − events_in_last_7d) / events_in_first_7d` |

### Churn Label
`churn_label = 1` if `last_event_date < (max_timestamp − 14 days)` else `0`. The observation ceiling is the dataset maximum timestamp (December 8, 2025). This binary target encodes whether a user has been absent for more than two weeks relative to that ceiling.

## Computations Performed
1. Pre-computes per-row binary flags (`_is_agent`, `_is_deploy`, `_is_onboard`, `_is_py_lib`) as `int8` columns before groupby — avoiding repeated string operations inside aggregation
2. Resolves category → string for `event` and `prop_$lib` before aggregation (pyarrow category dtypes require `.astype(str)` first)
3. Applies 7-day window event counts (`_in_first7`, `_in_last7`) using vectorised `pd.Timedelta` comparisons for `churn_velocity`
4. Merges all partial aggregation tables via sequential left-joins on `distinct_id`
5. Saves output to `user_features.csv` and renders a 16×13 in. Pearson correlation heatmap with annotated coefficients

## Analytical Note
The schema transformation from 409,287 raw events to 5,410 user-level rows — a 75.7× compression ratio — is the pivotal step that enables churn modelling. The five custom metrics capture platform-specific behaviour (AI adoption rate, session complexity, feature breadth, engagement velocity) not available in standard PostHog exports. The correlation heatmap confirms feature-target relationships before any model is trained.
