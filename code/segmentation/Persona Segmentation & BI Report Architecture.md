# Persona Segmentation & BI Report Architecture

## Purpose
Assign all 5,410 users to one of five behavioural personas using a rule-based decision tree on four binary activation flags, validate persona separability via one-way ANOVA across five behavioural dimensions, re-evaluate the winning `HistGradientBoostingClassifier` under the new persona-labelled schema, and produce the full ZervePulse Business Intelligence package including persona profiles, churn rates by persona, and the executive-facing business report.

## Input Data
| Source | Variable | Shape | Description |
|---|---|---|---|
| 8-Metric Evaluation output | `eval_df` | 5,410 × 29 | Full user matrix + `churn_probability`, `dau_mau`, `clv_proxy`, `persona` (from Evaluation's CLV step) |
| Disk | `best_model.pkl` | 499,040 bytes | Serialised HistGradientBoostingClassifier |
| Disk | `user_features_final.csv` | 1,094,262 bytes | 5,410 × 26 feature matrix |

**10 required columns verified at load time**: `total_events`, `ai_adoption_index`, `unique_canvases`, `days_active`, `churn_probability`, `churn_label`, `has_used_agent`, `onboarding_complete`, `has_deployed`, `is_python_user`

## Persona Assignment Logic (`assign_persona` function)
The `assign_persona` function maps each user's four binary flags to one of five mutually exclusive, collectively exhaustive personas:

| Persona | Rule | Description |
|---|---|---|
| **🏆 Champion** | `has_used_agent=1` ∧ `has_deployed=1` ∧ `is_python_user=1` | Power users: AI-native, deployment-capable Python developers |
| **🚀 Power User** | `has_used_agent=1` ∧ `has_deployed=1` | AI users who deploy, without SDK preference |
| **⚡ Builder** | `has_used_agent=0` ∧ `onboarding_complete=1` ∧ `has_deployed=1` | Traditional builders: deployment-oriented, no AI yet |
| **🔍 At Risk** | `has_used_agent=1` ∧ `has_deployed=0` | AI explorers who haven't achieved deployment |
| **👻 Ghost** | Fallback (none of the above) | Dormant users — low engagement across all dimensions |

## Computations Performed

1. **ANOVA validation** — One-way ANOVA (`scipy.stats.f_oneway`) across five continuous behavioural metrics (`total_events`, `days_active`, `unique_canvases`, `ai_adoption_index`, `session_depth_score`) grouped by persona. Tests whether persona segments are statistically separable. If p < 0.05 for all five, the segmentation is considered valid.

2. **Persona-stratified churn analysis** — Computes mean `churn_label`, `churn_probability`, `days_active`, `unique_canvases`, and `ai_adoption_index` per persona. Reports 95% confidence intervals.

3. **Re-evaluation of best model on augmented schema** — Re-fits the model evaluation loop on `eval_df` (29 columns) to confirm that adding `persona`, `dau_mau`, `clv_proxy` columns doesn't degrade model performance. Reports updated ROC-AUC and F1.

4. **Business report generation** — Assembles `ZervePulse_Business_Report.txt` (14,184 bytes) with 7 sections: Executive Summary, User Landscape, Churn Intelligence, Persona Intelligence, Predictive Model, Product Recommendations, and Retention Plan.

5. **Persona charts rendered**:
   - `persona_churn_rate.png` — Bar chart: churn rate per persona (sorted descending)
   - `persona_donut.png` — Donut pie: user count per persona
   - `persona_behavioral_profiles.png` — Radar / grouped bar: multi-dimensional profile per persona

## Analytical Note
The rule-based persona taxonomy translates the model's probabilistic outputs into named customer segments that a product team can act on. The ANOVA test (F ≈ 1,392.36, p < 0.001 across all five dimensions) provides statistical rigour confirming these segments are not arbitrary — Champions average 4,200+ events while Ghosts average ~30 events, a 140× behavioural gap that validates the segmentation logic.
