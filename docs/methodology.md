# Methodology

## Overview

ZervePulse follows a structured 8-phase analytical pipeline,
executed entirely within the Zerve AI-native environment using
AI agent prompts. No external tools were used at any stage.

---

## Phase 1 — Data Processing Pipeline

**Tool:** Zerve built-in Data Processing Pipeline

- Loaded raw dataset: 409,287 rows × 107 columns
- Applied dtype optimization — converted str columns to category
- Reduced memory footprint significantly
- Identified 107 columns → kept 18 high-signal columns
- Dropped all prop_$set.*, prop_$set_once.*, prop_$sdk_debug_* columns
- Filled prop_credits_used and prop_credit_amount nulls with 0
- Extracted canvas UUIDs from prop_$pathname using regex

---

## Phase 2 — Exploratory Data Analysis

- 5,410 unique users identified via distinct_id
- 141 unique event types catalogued
- 13,519 unique browser sessions
- 11 unique AI tool names in prop_tool_name
- Null audit confirmed event and timestamp have zero nulls
- prop_$lib has exactly 2 values: web and posthog-python
- Python SDK users account for 79.5% of all rows

---

## Phase 3 — Feature Engineering

**23 custom behavioral features engineered per user**

All 409,287 event rows were aggregated by distinct_id to produce
one row per user. Features were designed specifically for
platform engagement analytics — not standard churn templates.

Key custom metrics:
- **AI Adoption Index**: agent_events / total_events
- **Churn Velocity**: activity decline rate week-over-week
- **Session Depth Score**: events per session
- **Time-to-First-Agent**: minutes from signup to first AI use
- **Feature Breadth Score**: unique_event_types / 141
- **Tool Diversity Count**: unique AI tools used
- **Python SDK Share**: API usage proportion

**Churn label definition:**
A user is labelled churned (1) if their last_event_date is
more than 14 days before the maximum date in the dataset.
Active within last 14 days = retained (0).

---

## Phase 4 — Model Training

**4-model competitive framework:**

| Model | Notes |
|---|---|
| GradientBoosting | sklearn equivalent of XGBoost |
| HistGradientBoosting | sklearn equivalent of LightGBM — suggested by Zerve agent |
| Random Forest | class_weight='balanced' |
| Logistic Regression | baseline comparison |

**Class imbalance handling:**
SMOTE applied to training set only. Never applied to test set.
Class distribution: 64.8% churned (3,506) vs 35.2% retained (1,904).

**Train/test split:**
80/20 split with stratify=y and random_state=42.

---

## Phase 5 — Evaluation (22 Metrics)

All four models evaluated on the original non-SMOTE test set
(1,082 users). Winner selected based on combined F1 + ROC-AUC.

**HistGradientBoosting selected as winning model.**

Complete evaluation framework covers:
- Standard ML metrics (10)
- Advanced research metrics including SHAP, Decile Lift (4)
- Custom ZervePulse behavioral metrics (4)
- SaaS business metrics: CLV, retention curve, DAU/MAU (4)

---

## Phase 6 — User Segmentation

**5 personas assigned via rule-based priority system:**

Rules applied in strict priority order:
1. Champion — ai_adoption_index ≥ 0.3 AND days_active ≥ 5 AND unique_canvases ≥ 2 AND churn_predicted = 0
2. Explorer — onboarding_complete = 1 AND unique_event_types ≥ 4 AND has_used_agent = 1
3. At_Risk — has_used_agent = 1 AND churn_predicted = 1
4. Ghost — has_used_agent = 0 AND total_events ≤ 3
5. Casual — all remaining users

**Statistical validation:**
One-way ANOVA confirmed statistically significant differences
across personas: F = 1,392.36, p < 0.0001.

---

## Phase 7 — Business Intelligence Report

Computed 12 specific values (A through L) from user_features_final.csv.
Generated 6 product recommendations each with:
- FINDING: exact number from analysis
- ACTION: specific product change
- EXPECTED IMPACT: quantified outcome

---

## Phase 8 — Deployment

Streamlit dashboard deployed on Zerve with:
- Command Center (6 metric cards + 3 charts)
- User Intelligence Lookup (persona + risk + action)
- Full Churn Risk Leaderboard (5,410 users, paginated)
- Insights Explorer (4 charts + 6 recommendations)

Deployment URL: https://zervepulse.hub.zerve.cloud
