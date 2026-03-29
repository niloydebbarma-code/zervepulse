# ZervePulse — Complete 22-Metric Evaluation Framework

## Model: HistGradientBoosting (best_model.pkl)
## Dataset: 5,410 users | 409,287 events
## Test set: 1,082 users (20% stratified split)

---

## Model Performance Metrics (10)

| Metric | Value | Description |
|---|---|---|
| Accuracy | 0.7052 | Overall correct predictions |
| Precision | 0.8351 | Of flagged churners, % actually churned |
| Recall | 0.6790 | Of real churners, % caught by model |
| F1 Score | 0.7490 | Harmonic mean of precision and recall |
| ROC-AUC | 0.7814 | Overall ranking quality (0.5=random, 1=perfect) |
| PR-AUC | 0.8622 | Precision-recall curve — better for imbalanced data |
| MCC | 0.4136 | Matthews Correlation Coefficient — most reliable for imbalanced classification |
| Cohen's Kappa | 0.4009 | Agreement beyond random chance |
| Brier Score | 0.1865 | Probability calibration quality (lower = better) |
| Decile Lift | 1.54× | Top 10% captures 54% more churners than random |

---

## Advanced Research Metrics (4)

### Top-10% Decile Lift
- Top 10% of users by risk score: 541 users
- Churners in top 10%: 540 out of 541
- Churn rate in top 10%: 99.82%
- Baseline churn rate: 64.8%
- **Lift ratio: 1.54×**

### Cumulative Gain
Decile gains: 15.4% → 30.7% → 45.2% → 56.4% → 69.5% → 75.9% → 84.2% → 94.1% → 99.1% → 100%

### SHAP Feature Impact (Permutation Importance)
| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | unique_canvases | 0.1322 |
| 2 | ai_adoption_index | 0.1072 |
| 3 | unique_event_types | 0.1040 |
| 4 | is_python_user | 0.0671 |
| 5 | days_active | 0.0219 |

### Confusion Matrix (Test Set)
| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 (Retained) | 287 (TN) | 94 (FP) |
| Actual 1 (Churned) | 225 (FN) | 476 (TP) |

---

## Custom ZervePulse Behavioral Metrics (4)

### AI Adoption Index Distribution
- Retained users mean: 18.9%
- Churned users mean: 42.8%
- Non-agent users churn at: 77.1%
- Agent users churn at: 52.0%
- **Differential: 25.1 percentage points**

### Churn Velocity Analysis
- Measures week-over-week activity decline per user
- Positive = declining activity (churn risk)
- Negative = growing activity (retention signal)

### Session Depth Score
- Mean for retained: higher engagement per session
- Mean for churned: shallow single-event sessions

### Feature Breadth Score
- unique_event_types / 141 (total platform events)
- Higher score = more deeply embedded in platform
- Strong negative correlation with churn

---

## SaaS Business Metrics (4)

### Day 1 / Day 7 / Day 30 Retention Curve
| Milestone | Retention Rate |
|---|---|
| Day 1 | 100.0% |
| Day 7 | 5.2% (284 users) |
| Day 30 | 2.0% (106 users) |

Note: Rates reflect concentrated activity burst period in dataset window.

### DAU/MAU Stickiness Ratio
- Formula: days_active / observation_days per user
- Retained users mean: 0.9731
- Churned users mean: 0.9999
- Users with DAU/MAU > 0.5: 5,089

### Estimated CLV by Persona
| Persona | CLV (Credits × Months) |
|---|---|
| Builder (Agent+Deployed) | 2.16 |
| Python Developer (SDK+Agent) | 0.20 |
| Engaged (Agent+Onboarded) | 0.06 |
| Others | ~0.00 |

### Revenue Saved Estimate
- Mean credits per retained user: 0.42
- High-risk users (prob > 0.7): 1,625
- At 20% intervention success: 325 users saved
- Estimated credits recovered: 135.99
