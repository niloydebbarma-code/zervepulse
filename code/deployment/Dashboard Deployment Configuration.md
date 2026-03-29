# Dashboard Deployment Configuration

## Purpose
Register the deployment configuration metadata for the ZervePulse Streamlit dashboard script (`script-id: 0ba6d1b6-796d-41f9-b4b3-0bda4675736e`) and confirm that all four sections — Command Center, User Intelligence Lookup, Churn Risk Leaderboard, and Insights Explorer — are correctly mapped to their respective data sources from upstream blocks.

## Dashboard Data Sources
| Variable | Source Block | Shape | Usage |
|---|---|---|---|
| `seg_df` | ZervePulse Persona Segmentation & BI Report | 5,410 × 27 | All four dashboard sections: user lookup, leaderboard, charts, KPI cards |
| `seg_model` | ZervePulse Persona Segmentation & BI Report | HistGradBoost | Live churn probability scoring on arbitrary user IDs |

Both variables are accessed in the Streamlit app via `from zerve import variable` — no file system reads occur at runtime.

## Section Architecture

**Section 1 — Command Center**
6 KPI metric cards (3×2 grid) sourced from `seg_df` aggregates:
- Total users: 5,410
- Overall churn rate: 64.77%
- High-risk users: 1,625
- Best ROC-AUC: 0.7814
- Champions: 22
- Ghosts: 1,860

3 charts: persona donut (5 segments), churn risk 3-bar (Low/Medium/High), onboarding impact 2-bar

**Section 2 — User Intelligence Lookup**
Text input for any `distinct_id` → live persona badge + probability score + 4 stat boxes + top 3 SHAP risk drivers (user value vs. population mean) + persona-specific retention recommendation

**Section 3 — Churn Risk Leaderboard**
All 5,410 users sorted by `churn_probability` (descending) · paginated 50/page (108 pages) · filter by risk tier and persona · 10-column table with row-coloured risk tiers (red >0.7, amber 0.4–0.7, green <0.4) · CSV export

**Section 4 — Insights Explorer**
4 analytical charts + 6 product recommendations extracted from `ZervePulse_Business_Report.txt`

## Deployment Configuration
| Setting | Value |
|---|---|
| Script ID | `0ba6d1b6-796d-41f9-b4b3-0bda4675736e` |
| Template | Streamlit |
| Environment | `Streamlit (c47aefed)` |
| Compute size | `medium` |
| Run command | `streamlit run app/main.py --server.port 8080 --server.address 0.0.0.0` |

## Analytical Note
The dashboard exposes the full 5,410-user population through a paginated leaderboard (108 pages at 50/page), enabling product and growth teams to action churn interventions at individual user resolution. The 1,625 high-risk users (`churn_probability > 0.70`) are immediately filterable on the leaderboard with CSV export for CRM ingestion.
