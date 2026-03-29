# Business Intelligence Evaluation Framework

## Purpose
Apply the winning `HistGradientBoostingClassifier` to all 5,410 users to compute eight business-oriented evaluation metrics beyond standard ML scores. This block moves from model validation to operational intelligence — quantifying where the model delivers the most value, which features drive individual predictions, and what the revenue and retention implications are.

## Input Data
| Source | Variable | Shape | Description |
|---|---|---|---|
| Churn Pipeline output | `churn_df` | 5,410 × 26 | Full user dataset with `churn_probability` and `churn_predicted` appended |
| Disk | `best_model.pkl` | 499,040 bytes | Serialised winning HistGradientBoostingClassifier |

The block also inherits `best_metrics` (dict, 9 keys) from the upstream Churn Prediction Pipeline for embedding into the final summary report.

## Metrics Being Computed (8 + Summary)

### Metric 1 — Top-10% Decile Lift (full population)
Sort all 5,410 users by descending `churn_probability`. Select the top 10% (541 users). Compute the churn rate in that bucket versus the 64.8% baseline. This measures how efficiently the model concentrates churners at the top of the ranked list — the primary metric for outreach campaign targeting.

### Metric 2 — Cumulative Gain Chart
Compute cumulative % of churners captured for each 10% population decile (10 deciles). Annotate the first decile value. Visualise the lift curve against a random baseline diagonal.

### Metric 3 — SHAP Feature Impact (Permutation Importance proxy)
`sklearn.inspection.permutation_importance` on ROC-AUC, 10 repeats, `n_jobs=−1`, applied to all 5,410 rows. Ranks all 20 features by mean |SHAP| contribution with error bars (±std). The top 3 features by this metric provide the canonical feature ranking for the BI report.

### Metric 4 — Onboarding Friction
Segments users into three `time_to_first_agent_minutes` buckets: **Fast (0–10m)**, **Medium (10–60m)**, **Slow (60–1440m)**. Computes churn rate and count per bucket. Note: the `−1` sentinel was replaced with the median `0.0` in the upstream block, so all users fall into the Fast/Medium/Slow buckets (no "Never" bucket populated).

### Metric 5 — Estimated CLV by Persona
Assigns each user to one of eight engagement personas based on `has_used_agent`, `onboarding_complete`, `has_deployed`, `is_python_user` flags. Computes CLV proxy = `mean_credits_used × (mean_obs_days / 30)` per persona.

### Metric 6 — Retention Curve
Computes Day-1, Day-7, and Day-30 retention rates (% of users with `observation_days ≥ threshold`).

### Metric 7 — Revenue Saved
Business ROI model: `High-risk users (churn_probability > 0.7) × 20% intervention success rate × mean credits per retained user`.

### Metric 8 — DAU/MAU Stickiness
`_dau_mau = days_active / observation_days.clip(lower=1)`. Compares mean DAU/MAU for retained vs. churned cohorts. Threshold of 0.5 identifies highly sticky users.

### Final Summary — 22-Metric Report
Saves all computed values to `ZervePulse_22_Metrics.txt` (combined model + behavioral + platform health + business impact metrics).

## Visualisations Rendered
`cumulative_gain.png` · `shap_beeswarm.png` · `onboarding_friction.png` · `clv_by_persona.png` · `retention_curve.png` · `dau_mau_stickiness.png`
