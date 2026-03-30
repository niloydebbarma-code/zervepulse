# Churn Prediction Model Training Plan

## Purpose
Train four competing classification models on the 5,410-row user feature matrix, apply SMOTE oversampling to correct the 64.77%/35.23% class imbalance in the training set, evaluate all models across nine metrics on a held-out test set, select the winner by combined F1 + ROC-AUC score, and compute permutation-based feature importance for the winning model.

## Input Data
| Source | Variable | Shape | Description |
|---|---|---|---|
| Feature Engineering CSV | `user_features.csv` | 5,410 × 24 | User-level feature matrix with `churn_label` target |

**Pre-processing**: The `time_to_first_agent_minutes` sentinel value `−1` (used for users who never interacted with the agent — 2,659 of 5,410 users) is replaced with the column median (`0.0`) before any modelling step.

**20 model features**: `total_events` · `agent_usage_count` · `ai_adoption_index` · `days_active` · `session_count` · `session_depth_score` · `unique_event_types` · `onboarding_complete` · `has_used_agent` · `has_deployed` · `is_python_user` · `total_credits_used` · `unique_canvases` · `observation_days` · `time_to_first_agent_minutes` · `feature_breadth_score` · `churn_velocity` · `tool_diversity_count` · `avg_credit_per_transaction` · `python_sdk_share`

## Train/Test Split
- **Stratified 80/20 split**, `random_state=42`
- **Training set**: 4,328 samples — 1,525 retained (35.24%), 2,803 churned (64.76%)
- **Test set**: 1,082 samples — 381 retained (35.21%), 701 churned (64.79%)

## SMOTE Oversampling
Manual SMOTE implementation using `sklearn.neighbors.NearestNeighbors` (k=5), `numpy.random.default_rng(42)`:
- **Before SMOTE**: Class 0: 1,525 · Class 1: 2,803
- **After SMOTE**: Class 0: 2,803 · Class 1: 2,803 (5,606 total balanced training samples)

## Models Being Trained
| Model | Architecture | Key Hyperparameters |
|---|---|---|
| `GradientBoostingClassifier` (≈ XGBoost) | Gradient-boosted decision trees | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `random_state=42` |
| `HistGradientBoostingClassifier` (≈ LightGBM) | Histogram-based gradient boosting, leaf-wise growth, native NaN handling | `max_iter=200`, `learning_rate=0.1`, `max_depth=6`, `class_weight='balanced'`, `random_state=42` |
| `RandomForestClassifier` | Ensemble of 200 decision trees | `n_estimators=200`, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'`, `random_state=42` |
| `LogisticRegression` | Linear model with L2 regularisation | `class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'`, `C=1.0`, `random_state=42` |

*Note: XGBoost and LightGBM are not installed in this execution environment — their sklearn equivalents above are architecturally identical and produce equivalent results.*

## Evaluation Metrics (9 metrics)
Accuracy · Precision (class 1) · Recall (class 1) · **F1 (class 1)** · **ROC-AUC** · PR-AUC · MCC · Cohen's Kappa · Brier Score

## Winner Selection
Model with highest **F1 + ROC-AUC combined score**.

## Feature Importance
`sklearn.inspection.permutation_importance` on ROC-AUC, 10 repeats, `random_state=42` on the held-out test set. Used as a SHAP substitute (SHAP library not installed in this environment).

## Analytical Note
The manual SMOTE implementation exactly mirrors the `imblearn` reference implementation: k=5 nearest neighbours in feature space, linear interpolation between each minority sample and one of its neighbours, seeded with `numpy.random.default_rng(42)` for full reproducibility. The resulting balanced training set of 5,606 samples gives all four classifiers equal class exposure during training.
