# System Design

## Architecture Overview

ZervePulse is structured as a layered intelligence system:

```
Layer 1: Data Ingestion & Optimization
    └── Raw 409,287 events × 107 columns
    └── Zerve built-in Data Processing Pipeline
    └── Output: 18 optimized columns

Layer 2: Feature Engineering
    └── 23 custom behavioral features per user
    └── 409,287 rows → 5,410 user rows
    └── Output: user_features.csv

Layer 3: Predictive Modeling
    └── 4-model competition with SMOTE balancing
    └── HistGradientBoosting wins
    └── Output: best_model.pkl + user_features_final.csv

Layer 4: Evaluation Intelligence
    └── 22 metrics across 4 evaluation frameworks
    └── SHAP explainability + decile lift analysis
    └── Output: ZervePulse_22_Metrics.txt

Layer 5: Behavioral Segmentation
    └── 5 personas via rule-based priority system
    └── ANOVA statistical validation
    └── Output: user_features_segmented.csv

Layer 6: Business Intelligence
    └── 6 product recommendations with exact numbers
    └── CLV estimation, revenue impact, retention curves
    └── Output: ZervePulse_Business_Report.txt

Layer 7: Production Deployment
    └── Streamlit dashboard on Zerve
    └── Real-time user lookup + full leaderboard
    └── Output: https://zervepulse.hub.zerve.cloud
```

---

## Data Flow

```
user_retention.parquet (raw)
        │
        ▼ [Data Processing Pipeline]
optimised_retention (dtype-optimized)
        │
        ▼ [Feature Engineering]
user_features.csv (5,410 × 23)
        │
        ▼ [ML Model Training]
best_model.pkl + user_features_final.csv
        │
        ├── ▼ [22-Metric Evaluation]
        │   ZervePulse_22_Metrics.txt
        │
        └── ▼ [Persona Segmentation + BI Report]
            user_features_segmented.csv
            ZervePulse_Business_Report.txt
                    │
                    ▼ [Dashboard Deployment]
            https://zervepulse.hub.zerve.cloud
```

---

## Key Design Decisions

### Why 14-day churn window?
The dataset observation period spans a concentrated activity
burst. 14 days captures meaningful inactivity while avoiding
false positives from natural usage gaps.

### Why HistGradientBoosting?
The Zerve AI agent recommended HistGradientBoosting based on
dataset characteristics — large size, moderate dimensionality,
class imbalance. This was validated by competitive evaluation
across 4 models with 9 metrics each.

### Why SMOTE on training only?
Applying SMOTE to test data would give artificially inflated
metrics. SMOTE was applied only to the training set to balance
classes (1,525 retained → 2,803 synthetic retained) while
preserving the natural 64.8/35.2 distribution in evaluation.

### Why rule-based personas instead of clustering?
Rule-based personas are interpretable, actionable, and directly
map to product interventions. K-means clusters require human
labeling after the fact. Named personas with explicit conditions
are immediately usable by Zerve's product team.

---

## Deployment Architecture

```
Zerve Canvas
    └── user_features_segmented.csv (1.13 MB)
    └── best_model.pkl (499 KB)
    └── app/main.py (Streamlit application)
            │
            ▼ [Zerve Deployment]
    Streamlit (Org) executor
    Port 8080
    Small instance (1 CPU, 8GB, 1.12 credits/hr)
            │
            ▼
    https://zervepulse.hub.zerve.cloud
```

---

## Reproducibility

All analysis is fully reproducible within the Zerve canvas:

1. Load user_retention.parquet into Zerve
2. Run Data Processing Pipeline block
3. Run Feature Engineering block
4. Run ML Model block
5. Run 22-Metric Evaluation block
6. Run Segmentation + BI Report block
7. Run Dashboard Deployment block

Each block is independent and can be re-run in sequence.
All outputs are deterministic with random_state=42.
