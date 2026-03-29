# Churn Model Performance & Feature Importance

## Variables Created (key exports)
| Variable | Type | Shape | Description |
|---|---|---|---|
| `churn_df` | `DataFrame` | 5,410 × 26 | Feature matrix + `churn_probability` + `churn_predicted` columns added |
| `best_model` | `HistGradientBoostingClassifier` | — | Winning model, serialised to `best_model.pkl` |
| `best_model_name` | `str` | — | `"HistGradientBoosting"` |
| `best_metrics` | `dict` | 9 keys | All evaluation metrics for the winning model |
| `metrics_df` | `DataFrame` | 9 × 4 | Full comparison table: all models × all metrics |
| `fi_df` | `DataFrame` | 20 × 4 | Feature / Importance / Rank / Business Meaning |
| `X_train_sm` | `DataFrame` | 5,606 × 20 | SMOTE-balanced training set |
| `fig_cm`, `fig_roc`, `fig_pr`, `fig_fi` | matplotlib Figures | — | 4 charts rendered and saved |

## 9-Metric Model Comparison (test set, n=1,082)

| Metric | GradientBoosting | **HistGradBoosting ✓** | RandomForest | LogisticReg |
|---|---|---|---|---|
| Accuracy | 0.7070 | **0.7052** | 0.7043 | 0.6303 |
| Precision (cls 1) | 0.8333 | **0.8351** | 0.8360 | 0.7968 |
| Recall (cls 1) | 0.6847 | **0.6790** | 0.6762 | 0.5763 |
| F1 (cls 1) | 0.7518 | **0.7490** | 0.7476 | 0.6689 |
| **ROC-AUC** | 0.7786 | **0.7814** | 0.7796 | 0.7128 |
| **PR-AUC** | 0.8597 | **0.8622** | 0.8615 | 0.8135 |
| MCC | 0.4143 | **0.4136** | 0.4132 | 0.2929 |
| Cohen's Kappa | 0.4026 | **0.4009** | 0.4000 | 0.2741 |
| Brier Score (↓ better) | 0.1904 | **0.1865** | 0.1874 | 0.2164 |

**🏆 Winner: HistGradientBoosting** — highest combined F1 + ROC-AUC = **1.5305** (GradientBoosting = 1.5304 — margin of 0.0001)

## Feature Importance (Top 10 — Permutation on ROC-AUC, 10 repeats)
| Rank | Feature | Importance | Business Meaning |
|---|---|---|---|
| 1 | `unique_canvases` | **0.1037** | Canvas creation volume — primary activation north star |
| 2 | `ai_adoption_index` | 0.0958 | Composite AI feature adoption rate |
| 3 | `unique_event_types` | 0.0821 | Breadth of platform feature usage |
| 4 | `is_python_user` | 0.0671 | Python SDK users show markedly lower churn |
| 5 | `days_active` | 0.0219 | Calendar engagement span |
| 6 | `python_sdk_share` | 0.0195 | Proportion of SDK interactions |
| 7 | `session_depth_score` | 0.0191 | Average events per session |
| 8 | `total_events` | 0.0164 | Total platform interactions |
| 9 | `observation_days` | 0.0148 | Tenure length on platform |
| 10 | `agent_usage_count` | 0.0123 | AI agent invocation count |

## Top-10% Decile Lift (test set, n=1,082)
- **Top 10% bucket**: 109 users, sorted by predicted churn probability
- **Churn rate in top 10%**: 99.08% (108/109 users)
- **Baseline churn rate**: 64.8%
- **Decile lift ratio**: **1.529×**

## Files Saved
| File | Size | Description |
|---|---|---|
| `best_model.pkl` | **499,040 bytes** (~487 KB) | Serialised HistGradientBoostingClassifier |
| `user_features_final.csv` | **1,094,262 bytes** (~1.04 MB) | 5,410 × 26 — features + `churn_probability` + `churn_predicted` |
| `confusion_matrices.png` | 84,129 bytes | 2×2 grid for all four models |
| `roc_curves.png` | 145,754 bytes | ROC curves with per-model AUC annotations |
| `pr_curves.png` | 158,469 bytes | Precision-recall curves (PR-AUC baseline = 0.648) |
| `feature_importance.png` | 145,499 bytes | Horizontal bar chart, top predictor highlighted in gold |

## Interpretation
`unique_canvases` (importance = 0.1037) displaced all the correlation-based churn velocity metrics as the strongest permutation-importance predictor — because while `churn_velocity` is highly correlated with the label, it shares information with `observation_days`. The model identifies that creating ≥2 canvases is the most independently informative activation signal for retention.
