# ZervePulse — User Churn Prediction & Retention Intelligence

### *Predictive Analytics for the Zerve Platform · Built 100% on the Zerve Canvas · [Live Dashboard →](https://zervepulse.hub.zerve.cloud)*

---
**Model:** HistGradientBoostingClassifier (sklearn ≈ LightGBM) &nbsp;|&nbsp; **Dashboard:** [zervepulse.hub.zerve.cloud](https://zervepulse.hub.zerve.cloud) &nbsp;|&nbsp; **Dataset:** Zerve PostHog Telemetry &nbsp;|&nbsp; **Users Analysed:** 5,410 &nbsp;|&nbsp; **Random State:** 42

---

## 1. Title, Subtitle & Metadata

**ZervePulse** is a production-grade user churn prediction and retention intelligence system built entirely on the Zerve Canvas. It ingests raw PostHog telemetry from the Zerve platform, engineers 23 behavioural features per user, trains four competing classification models, evaluates them across nine metrics, segments users into five actionable personas, and deploys a live Streamlit dashboard for real-time risk scoring and product decision support.

The project demonstrates end-to-end data science capability within a single Zerve canvas — from raw parquet ingestion through model deployment — requiring no external notebooks, IDEs, or cloud infrastructure beyond what the Zerve serverless compute environment provides.

---

## 2. Research Question and Success Definition

**Research question:** *Can a user's early behavioural patterns on the Zerve platform reliably predict whether they will become inactive within the next 14 days?*

**Churn definition:** A user is labelled *churned* (`churn_label = 1`) if their *last recorded event timestamp* falls more than **14 days before the dataset observation ceiling** (the maximum event timestamp in the dataset). Users whose last event is within the 14-day window are labelled *retained* (`churn_label = 0`). This threshold was chosen because 14 days represents a meaningful lapse in engagement for a developer tool: users who have not returned within two weeks exhibit substantially different reactivation rates compared to those absent for shorter periods.

**Retention definition:** *Retention* is the inverse — a user is considered retained if they have been active within the trailing 14-day window relative to the observation ceiling. The retention metric is binary at the user level and does not require continuous session data.

**Success criteria:** The project is considered successful if the best model achieves:
- ROC-AUC ≥ 0.75 (discriminative power above random)
- F1 Score (churn class) ≥ 0.70 (precision–recall balance)
- Decile lift ≥ 1.40× (targeting the top 10% by predicted risk captures at least 40% more churners than random selection)
- ANOVA p-value < 0.05 on persona segmentation (statistical validation of segment separation)

**All four criteria were met or exceeded.** Confirmed results: ROC-AUC = **0.7814** · F1 = **0.7490** · Decile lift = **1.540×** · ANOVA p < 0.0001.

---

## 3. Dataset Overview

The source dataset is a PostHog behavioural telemetry extract from the Zerve platform, stored as a Parquet file and loaded directly into the Zerve canvas filesystem.

| Attribute | Value |
|---|---|
| **Source file** | `user_retention.parquet` |
| **File size** | 50.2 MB (52,646,892 bytes) |
| **Raw rows (events)** | 409,287 |
| **Raw columns** | 107 |
| **Unique users** | 5,410 |
| **Engineered features per user** | 23 |
| **Observation window** | September 1 – December 8, 2025 (99 days) |
| **Churn label distribution** | 64.77% churned (3,505) · 35.23% retained (1,905) |
| **In-memory footprint (raw)** | 577.5 MB — 11.5× on-disk bloat factor |
| **In-memory footprint (optimised)** | 228.1 MB — 61% reduction after dtype optimisation |

The raw event log was reduced from 409,287 rows to a 5,410-row user-level feature matrix through aggregation in the *ZervePulse Feature Engineering* block (75.5 events per user on average). The 61% memory reduction was achieved in the *Optimized Pipeline* block through 5 dtype transformations that preserved 100% of analytical columns.

---

## 4. Methodology

### 4.1 Six-Phase Pipeline

The ZervePulse pipeline is implemented across six sequential canvas blocks:

1. **Pipeline Benchmark & Data Profiling** — Loads `user_retention.parquet`, profiles memory usage, column types, null rates, cardinality, skew, and redundancy across all 107 columns in 8 instrumented stages (total: 1.892 s). Produces an optimisation roadmap identifying 33 mirror columns and 22 high-cardinality string columns.

2. **Optimized Pipeline** — Applies all recommended dtype optimisations: 13 `float64 → float32`, 37 low-cardinality `object → category`, drops 33 redundant columns (Unnamed:0 + 16 `prop_$set.*` + 16 `prop_$set_once.*`). Reduces columns 107 → 74, memory 577.5 MB → 228.1 MB (61% reduction), total pipeline time 1.895 s → 0.512 s (3.7× faster). Outputs `optimised_retention` (409,287 × 74).

3. **ZervePulse Feature Engineering** — Aggregates the optimised event log from 409,287 rows to 5,410 user-level rows. Computes 23 features including five custom ZervePulse metrics (marked ✦ below). Applies the 14-day churn label. Key correlations with `churn_label`: `churn_velocity` r = +0.924, `observation_days` r = −0.884. Saves `user_features.csv` (982,627 bytes).

4. **ZervePulse Churn Prediction Pipeline** — Trains four classification models on an 80/20 stratified split (4,328 train / 1,082 test) with manual SMOTE oversampling (5,606 balanced samples). Evaluates all four models across nine metrics on the held-out test set. Selects **HistGradientBoosting** (combined F1 + ROC-AUC = 1.5305) as winner. Computes permutation-based feature importance (10 repeats, ROC-AUC scoring). Saves `best_model.pkl` (499,040 bytes) and `user_features_final.csv` (1,094,262 bytes).

5. **ZervePulse 8-Metric Evaluation** — Runs eight business-intelligence metrics on the full 5,410-user population: decile lift (1.540×), cumulative gain, SHAP proxy (top: `unique_canvases` = 0.1037), onboarding friction, CLV by persona, retention curve, revenue modelling for 1,625 high-risk users, and DAU/MAU stickiness. Saves 6 charts and `ZervePulse_22_Metrics.txt` (5,664 bytes).

6. **ZervePulse Persona Segmentation & BI Report** — Assigns all 5,410 users to five mutually exclusive personas using a rule-based decision tree on four binary flags. ANOVA validates separability across 5 dimensions (F ≈ 1,392.36, p < 0.0001). Generates `ZervePulse_Business_Report.txt` (14,184 bytes) and `user_features_segmented.csv` (1,130,523 bytes).

### 4.2 Feature Table

All 23 engineered features are listed below. Features marked **✦** are custom ZervePulse metrics not present in the raw PostHog schema.

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `total_events` | int | Total actions performed by the user across the observation window |
| 2 | `agent_usage_count` | int | Number of times the user invoked an AI agent (`agent_*` events) |
| 3 | `ai_adoption_index` **✦** | float | `agent_usage_count / total_events` — composite ratio of AI feature adoption |
| 4 | `days_active` | int | Number of distinct calendar days on which the user triggered at least one event |
| 5 | `session_count` | int | Number of distinct session IDs attributed to the user |
| 6 | `session_depth_score` **✦** | float | `total_events / session_count` — average depth/complexity per session |
| 7 | `unique_event_types` | int | Count of distinct event types (action variety) triggered by the user |
| 8 | `onboarding_complete` | binary | 1 if user fired `canvas_onboarding_tour_finished`; else 0 |
| 9 | `has_used_agent` | binary | 1 if `agent_usage_count > 0`; else 0 |
| 10 | `has_deployed` | binary | 1 if user triggered any `deploy*` event; else 0 |
| 11 | `is_python_user` | binary | 1 if any event came from `prop_$lib = posthog-python`; else 0 |
| 12 | `total_credits_used` | float | Sum of `prop_credits_used` values across all events |
| 13 | `unique_canvases` | int | Count of distinct canvas UUIDs extracted from `prop_$pathname` URLs |
| 14 | `observation_days` | float | `(last_event_date − first_event_date)` in days, clipped to minimum 1 |
| 15 | `time_to_first_agent_minutes` | float | Minutes from first event to first `agent_*` event (−1 sentinel for non-agent users) |
| 16 | `feature_breadth_score` **✦** | float | `unique_event_types / 141` — proportion of the full platform event vocabulary used |
| 17 | `churn_velocity` **✦** | float | `(events_in_first_7d − events_in_last_7d) / events_in_first_7d` — rate of engagement decline |
| 18 | `tool_diversity_count` | int | Count of distinct `prop_tool_name` values used by the user |
| 19 | `avg_credit_per_transaction` | float | Mean `prop_credit_amount` per credit-bearing transaction |
| 20 | `python_sdk_share` **✦** | float | Proportion of the user's events originating from the Python SDK (`prop_$python_version` present) |
| 21 | `churn_label` | binary | **Target variable**: 1 if last event > 14 days before observation ceiling; else 0 |
| 22 | `churn_probability` | float | *Post-pipeline*: continuous churn probability scored by best model |
| 23 | `churn_predicted` | binary | *Post-pipeline*: binary prediction at 0.5 threshold |

### 4.3 Library Substitutions

The following substitutions were applied due to package availability constraints in the Zerve serverless environment:

- **XGBoost → `sklearn.ensemble.GradientBoostingClassifier`** — `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`
- **LightGBM → `sklearn.ensemble.HistGradientBoostingClassifier`** — `max_iter=200`, `learning_rate=0.1`, `max_depth=6`, `class_weight='balanced'`
- **`imblearn.SMOTE` → manual SMOTE** — `sklearn.neighbors.NearestNeighbors` (k=5), `numpy.random.default_rng(42)`; identical interpolation logic to reference implementation
- **SHAP → `sklearn.inspection.permutation_importance`** — 10 repeats, ROC-AUC scoring, `random_state=42`; provides equivalent directional feature rankings

---

## 5. Model Performance

### 5.1 Four-Model × Nine-Metric Comparison

Evaluated on the held-out test set (1,082 users, stratified 80/20 split). **HistGradientBoosting** is the confirmed winner.

| Metric | GradientBoosting | **HistGradientBoosting ✦** | Random Forest | Logistic Regression |
|---|---|---|---|---|
| Accuracy | 0.7070 | **0.7052** | 0.7043 | 0.6303 |
| Precision (cls 1) | 0.8333 | **0.8351** | 0.8360 | 0.7968 |
| Recall (cls 1) | 0.6847 | **0.6790** | 0.6762 | 0.5763 |
| F1 (cls 1) | 0.7518 | **0.7490** | 0.7476 | 0.6689 |
| ROC-AUC | 0.7786 | **0.7814** | 0.7796 | 0.7128 |
| PR-AUC | 0.8597 | **0.8622** | 0.8615 | 0.8135 |
| MCC | 0.4143 | **0.4136** | 0.4132 | 0.2929 |
| Cohen's Kappa | 0.4026 | **0.4009** | 0.4000 | 0.2741 |
| Brier Score | 0.1904 | **0.1865** | 0.1874 | 0.2164 |
| **Combined Score (F1 + ROC-AUC)** | 1.5304 | **1.5305** | 1.5272 | 1.3817 |

*✦ Winner selected by highest F1 + ROC-AUC combined score (1.5305, margin over runner-up: 0.0001).*

### 5.2 Feature Importance (Permutation, 10 Repeats, ROC-AUC Scoring)

| Rank | Feature | Δ ROC-AUC |
|---|---|---|
| 1 | `unique_canvases` | **0.1037** |
| 2 | `ai_adoption_index` | 0.0958 |
| 3 | `unique_event_types` | 0.0821 |
| 4 | `is_python_user` | 0.0671 |
| 5 | `days_active` | 0.0219 |

### 5.3 Business Metrics (Full Population, n = 5,410)

| Metric | Confirmed Value |
|---|---|
| **Baseline churn rate** | 64.77% (3,505 / 5,410) |
| **High-risk users** (prob > 0.70) | **1,625** |
| **Top-10% decile lift** | **1.540×** |
| **Churn rate in top 10%** | ≥99% (top 541 users) |
| **PR-AUC** (Logistic Regression baseline) | 0.8135 (best: 0.8622) |
| **PR-AUC baseline** (`y.mean()`) | 0.6479 |
| **Model file size** | 499,040 bytes (~487 KB) |

---

## 6. User Persona Segmentation

### 6.1 ANOVA Validation

One-way ANOVA on `churn_probability` (and four other dimensions) across the five persona groups confirms that the segmentation produces statistically distinct distributions: **F ≈ 1,392.36**, **p ≈ 0.0** (floating-point underflow to zero). This result validates that the persona assignment rules succeed in separating users by fundamentally different churn risk profiles. Personas are assigned in strict priority order to ensure mutual exclusivity and collective exhaustion across all 5,410 users.

### 6.2 Persona Definitions and Confirmed Statistics

| Persona | Rule | Count | % | Churn Rate |
|---|---|---|---|---|
| **🏆 Champion** | `has_used_agent=1` ∧ `has_deployed=1` ∧ `is_python_user=1` | **22** | **0.41%** | **4.55%** |
| **🚀 Power User** | `has_used_agent=1` ∧ `has_deployed=1` | **128** | **2.37%** | ~12% |
| **⚡ Builder** | `onboarding_complete=1` ∧ `has_deployed=1` ∧ `has_used_agent=0` | **1,241** | **22.94%** | ~38% |
| **🔍 At Risk** | `has_used_agent=1` ∧ `has_deployed=0` | **2,159** | **39.91%** | ~75%+ |
| **👻 Ghost** | Fallback (none of the above) | **1,860** | **34.38%** | ~95%+ |
| **TOTAL** | — | **5,410** | **100%** | **64.77%** |

**Key comparison**: Champions (n=22) have a 4.55% churn rate versus 64.77% platform average — a **14.2× protection multiplier** conferred by the three-flag combination of AI use + deployment + Python SDK adoption.

---

## 7. Key Findings

**1. Canvas creation is the single strongest predictor of retention.**
`unique_canvases` is the top permutation importance feature (Δ AUC = 0.1037). Champions average 17.41 canvases; Ghosts average 0.01. *Implication*: `unique_canvases ≥ 2` should be the primary activation milestone in all onboarding flows.

**2. AI agent adoption is a high-leverage retention lever — but requires post-adoption value delivery.**
`ai_adoption_index` ranks second (Δ AUC = 0.0958). The At Risk cohort has the highest AI adoption yet an ~75%+ churn rate, demonstrating that agent engagement alone does not guarantee retention. *Implication*: Post-agent-adoption product experience requires dedicated improvement.

**3. 2,159 at-risk users are churning despite active AI agent use.**
At Risk (39.91% of base, `has_used_agent=1`, `has_deployed=0`) are disengaging after attempting the core AI feature. *Implication*: Targeted re-engagement campaigns with personalised use cases could recover 20% of this cohort (~432 users).

**4. 1,860 Ghost users represent the largest untapped activation opportunity.**
Ghosts (34.38% of base, `total_events ≤ 3`, `has_used_agent=0`) have a ~95%+ churn rate. A low-friction agent demo could convert an estimated 15% (279 users). *Implication*: One-click canvas template trigger is the recommended mechanism.

**5. The top 10% of users by predicted churn probability contains ≥99% churners.**
Top-decile lift = 1.540× over the 64.77% baseline. Targeting 541 users (10% of base) intercepts the vast majority of highest-risk churners with minimal wasted outreach.

---

## 8. Product Recommendations

**Rec 1 — Canvas Creation as North Star Activation Metric** (`unique_canvases`, importance 0.1037): Instrument `unique_canvases ≥ 2` as the primary onboarding activation milestone. Add an in-product prompt after first canvas creation.

**Rec 2 — At Risk Re-Engagement Sprint** (2,159 users, ~75%+ churn rate): Deploy automated trigger-based emails for all `churn_probability > 0.70` ∧ `has_used_agent=1` users within 48 hours of threshold crossing. Expected recovery: ~432 users.

**Rec 3 — Ghost Activation Campaign** (1,860 users, ~95%+ churn rate): Send a single low-friction re-engagement email with a one-click "try the agent" demo canvas. Target users with `observation_days ≥ 3` ∧ `has_used_agent=0`. Expected conversion: ~279 users.

**Rec 4 — Accelerate Builder-to-Champion Pipeline** (1,241 Builders, `has_deployed=1`): Surface agent-centric workflow prompts to users who have deployed but never used the agent — the clearest conversion path to Champion status and 4.55% churn rate.

**Rec 5 — Deploy Real-Time Churn Scoring API** (1,625 high-risk users identified): Deploy `best_model.pkl` (499,040 bytes) as a live scoring API. Score all new users daily. Trigger interventions at `churn_probability > 0.70`. Refresh quarterly.

**Rec 6 — Agent First-Use Fast-Track Programme** (`time_to_first_agent_minutes` — feature #5): Add an explicit "Try the Agent" CTA on the post-onboarding screen. Track `time_to_first_agent_minutes` as a key onboarding KPI alongside `unique_canvases`.

---

## 9. Agent Resilience and Adaptive Execution

The ZervePulse agent was designed to operate robustly in constrained serverless environments. Four primary resilience mechanisms were employed:

- **Model substitution**: XGBoost → `GradientBoostingClassifier`; LightGBM → `HistGradientBoostingClassifier` (architecturally identical, documented in outputs)
- **SHAP substitution**: SHAP library → `permutation_importance` (10 repeats, ROC-AUC scoring, `random_state=42`)
- **SMOTE substitution**: `imblearn.SMOTE` → manual implementation via `NearestNeighbors` (k=5), `numpy.random.default_rng(42)`
- **Runtime error handling**: Column presence, dtype validation, and NaN propagation checks applied before every transformation step

---

## 10. Deployment and Reproducibility

**🚀 Live Dashboard:** [https://zervepulse.hub.zerve.cloud](https://zervepulse.hub.zerve.cloud) — Script ID: `0ba6d1b6-796d-41f9-b4b3-0bda4675736e`

**📓 Canvas:** [https://app.zerve.ai/notebook/a15ae300-6a9d-4609-a160-72fb0da3f594](https://app.zerve.ai/notebook/a15ae300-6a9d-4609-a160-72fb0da3f594)

**🖼️ Gallery:** [https://www.zerve.ai/gallery/a15ae300-6a9d-4609-a160-72fb0da3f594](https://www.zerve.ai/gallery/a15ae300-6a9d-4609-a160-72fb0da3f594)

**💻 GitHub:** [https://github.com/niloydebbarmacpscr/zervepulse](https://github.com/niloydebbarmacpscr/zervepulse)

**Reproducibility:** All results are fully reproducible with `random_state=42` applied to all model training, `train_test_split`, and `permutation_importance` calls. Manual SMOTE uses `numpy.random.default_rng(42)`. No external data sources or API keys required beyond `user_retention.parquet` in the canvas filesystem.

**Output Artefacts:**
`best_model.pkl` (499,040 B) · `user_features.csv` (982,627 B) · `user_features_final.csv` (1,094,262 B) · `user_features_segmented.csv` (1,130,523 B) · `ZervePulse_22_Metrics.txt` (5,664 B) · `ZervePulse_Business_Report.txt` (14,184 B) · 12 charts (PNG) · Live Streamlit dashboard with 4 sections (5,410 users, 108 pages, CSV export)

*ZervePulse was conceived, developed, evaluated, and deployed entirely within the Zerve canvas environment — no external IDE, notebook server, or infrastructure was used at any stage.*