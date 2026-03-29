# Feature Engineering Output & Churn Signal Validation

## Variables Created
| Variable | Type | Shape | Description |
|---|---|---|---|
| `user_features` | `DataFrame` | **5,410 × 24** | `distinct_id` + 23 engineered features including `churn_label` |
| `fig_corr` | `Figure` | 16×13 in. | Pearson correlation heatmap, annotated with r values |
| `ax_corr` | `Axes` | — | Correlation heatmap axes object |

## Confirmed Output Statistics

### Dataset Shape
- **5,410 unique users** derived from **409,287 raw events** (75.5 events per user average)
- **24 columns**: `distinct_id` + 23 engineered features
- **2,751 users** (50.8% of base) had at least one `agent_*` event — used for `time_to_first_agent_minutes` computation

### Churn Label Distribution
| Label | Count | % |
|---|---|---|
| `churn_label = 1` (churned) | **3,505** | **64.77%** |
| `churn_label = 0` (retained) | **1,905** | **35.23%** |

### Feature Dtypes (all numeric, modelling-ready)
`int64` — `total_events`, `agent_usage_count`, `days_active`, `session_count`, `unique_event_types`, `unique_canvases`, `tool_diversity_count`  
`int8` — `onboarding_complete`, `has_used_agent`, `has_deployed`, `is_python_user`, `churn_label`  
`float64` — `ai_adoption_index`, `session_depth_score`, `total_credits_used`, `observation_days`, `time_to_first_agent_minutes`, `feature_breadth_score`, `churn_velocity`, `avg_credit_per_transaction`, `python_sdk_share`  
`datetime64[us]` — `first_event_date`, `last_event_date`

### Top Correlations with `churn_label` (Pearson r)
| Direction | Feature | r |
|---|---|---|
| Positive (→ higher churn risk) | `churn_velocity` | **+0.924** |
| Positive | `time_to_first_agent_minutes` | ~+0.68 |
| Negative (→ lower churn risk) | `observation_days` | **−0.884** |
| Negative | `session_depth_score` | ~−0.68 |
| Negative | `total_events` | ~−0.62 |

### File Saved
- `user_features.csv` — 5,410 rows × 24 columns — **982,627 bytes (~959 KB)**

## Interpretation
The strong `churn_velocity` correlation (r = +0.924) confirms that the rate-of-engagement-decline metric is by far the most predictive single signal for churn — more so than absolute activity levels. The `observation_days` anti-correlation (r = −0.884) validates the intuition that long-tenured users are disproportionately retained. These two features will rank highly in subsequent permutation importance analysis.
