# Dashboard Deployment Summary

## Variables Exported
| Variable | Type | Value |
|---|---|---|
| `SCRIPT_ID` | `str` | `"0ba6d1b6-796d-41f9-b4b3-0bda4675736e"` |
| `SCRIPT_NAME` | `str` | `"ZervePulse Dashboard"` |
| `RUN_CMD` | `str` | `"streamlit run app/main.py --server.port 8080 --server.address 0.0.0.0"` |
| `TEMPLATE` | `str` | `"Streamlit"` |
| `ENVIRONMENT` | `str` | `"Streamlit (c47aefed)"` |
| `COMPUTE` | `str` | `"medium"` |

## Confirmed Dashboard Implementation Status

| Section | Status | Key Content |
|---|---|---|
| **Section 1 — Command Center** | ✅ Implemented | 6 KPI cards: 5,410 users · 64.8% churn · 1,625 high-risk · 0.7814 ROC-AUC · 22 Champions · 1,860 Ghosts; 3 charts (donut, risk bar, onboarding bar) |
| **Section 2 — User Intelligence Lookup** | ✅ Implemented | Live `distinct_id` lookup: persona badge + churn probability + 4 stat boxes + top 3 SHAP risk drivers with population comparison + persona-specific action box |
| **Section 3 — Churn Risk Leaderboard** | ✅ Implemented | 5,410 users · 50/page (108 pages) · filter: All/High Risk/Persona dropdown · 10-column table · row-coloured risk tiers · CSV export |
| **Section 4 — Insights Explorer** | ✅ Implemented | 4 analytical charts: avg canvases (retained vs. churned), AI adoption comparison, agent usage impact, churn probability histogram (20 bins + thresholds) + 6 product recommendations |

## Design System Compliance
Zerve dark theme applied throughout:
- Background: `#1D1D20` 
- Primary text: `#fbfbff`
- Gold accent: `#ffd400`
- Official Zerve colour palette for all data series

## Pipeline Completion Summary
The ZervePulse analytics pipeline spans **6 processing blocks** and produces a fully operational churn intelligence platform:

| Stage | Output |
|---|---|
| Pipeline Benchmark | 409,287 × 107 profiled → 18-key benchmark report |
| Optimized Pipeline | 409,287 × 74, 228.1 MB (61% memory reduction, 3.7× faster) |
| Feature Engineering | 5,410 × 24 user matrix, 14-day churn label |
| Churn Prediction | ROC-AUC=0.7814, F1=0.7490, lift=1.540×, 1,625 high-risk users |
| 8-Metric Evaluation | 22 business metrics + 6 diagnostic charts |
| Persona Segmentation | 5 personas, ANOVA F≈1,392 p<0.001, BI report (14,184 bytes) |
| **Dashboard** | 4-section Streamlit app · deployed as script `0ba6d1b6` |
