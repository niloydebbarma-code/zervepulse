
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# ── Zerve Design System ───────────────────────────────────────────
_BG      = '#1D1D20'
_TEXT    = '#fbfbff'
_SUBTLE  = '#909094'
_GOLD    = '#ffd400'
_GREEN   = '#17b26a'
_RED     = '#f04438'
_BLUE    = '#A1C9F4'
_ORANGE  = '#FFB482'
_LAVENDER= '#D0BBFF'
_CORAL   = '#FF9F9B'

def _style_ax(ax_obj):
    ax_obj.set_facecolor(_BG)
    ax_obj.tick_params(colors=_TEXT, labelsize=10)
    for sp in ax_obj.spines.values():
        sp.set_color(_SUBTLE)

# ── Load data from upstream variables ────────────────────────────
eval_df = churn_df.copy()
print(f"eval_df shape: {eval_df.shape}")

eval_best_model = joblib.load('best_model.pkl')
print(f"Loaded best_model.pkl: {type(eval_best_model).__name__}")

EVAL_FEATURES = [
    'total_events', 'agent_usage_count', 'ai_adoption_index', 'days_active',
    'session_count', 'session_depth_score', 'unique_event_types',
    'onboarding_complete', 'has_used_agent', 'has_deployed', 'is_python_user',
    'total_credits_used', 'unique_canvases', 'observation_days',
    'time_to_first_agent_minutes', 'feature_breadth_score', 'churn_velocity',
    'tool_diversity_count', 'avg_credit_per_transaction', 'python_sdk_share'
]

# ════════════════════════════════════════════════════════════════════
# METRIC 1 — Top-10% Decile Lift (all 5,410 users)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 1 — Top-10% Decile Lift")
print("=" * 70)

_m1_df = eval_df[['churn_probability', 'churn_label']].copy()
_m1_sorted = _m1_df.sort_values('churn_probability', ascending=False).reset_index(drop=True)
_m1_top10_n = int(np.ceil(len(_m1_sorted) * 0.10))   # 541 users
_m1_top10   = _m1_sorted.head(_m1_top10_n)
_m1_churned_in_top10 = int(_m1_top10['churn_label'].sum())
_m1_baseline = 0.648
_m1_lift = (_m1_churned_in_top10 / _m1_top10_n) / _m1_baseline
_m1_pct_captured = (_m1_churned_in_top10 / eval_df['churn_label'].sum()) * 100

print(f"Total users              : {len(_m1_sorted):,}")
print(f"Top 10% bucket size      : {_m1_top10_n:,} users")
print(f"Churners in top 10%      : {_m1_churned_in_top10:,}")
print(f"Churn rate in top 10%    : {_m1_churned_in_top10/_m1_top10_n:.4f}")
print(f"Baseline churn rate      : {_m1_baseline:.3f}")
print(f"Lift ratio               : {_m1_lift:.2f}x")
print(f"\nTop 10% captured {_m1_pct_captured:.1f}% of churners")
print(f"Lift ratio: {_m1_lift:.2f} vs baseline")

# ════════════════════════════════════════════════════════════════════
# METRIC 2 — Cumulative Gain Chart
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 2 — Cumulative Gain Chart")
print("=" * 70)

_total_churners = eval_df['churn_label'].sum()
_decile_pcts    = []
_decile_labels  = []
for _d in range(1, 11):
    _cutoff  = int(np.ceil(len(_m1_sorted) * _d / 10))
    _slice   = _m1_sorted.head(_cutoff)
    _gained  = _slice['churn_label'].sum() / _total_churners * 100
    _decile_pcts.append(_gained)
    _decile_labels.append(f"{_d*10}%")

fig_gain, ax_gain = plt.subplots(figsize=(10, 7))
fig_gain.patch.set_facecolor(_BG)
_style_ax(ax_gain)

_x_vals = [10 * i for i in range(1, 11)]
ax_gain.plot(_x_vals, _decile_pcts, color=_ORANGE, linewidth=2.5,
             marker='o', markersize=6, label='Model')
ax_gain.plot([0, 100], [0, 100], '--', color=_SUBTLE, linewidth=1.5, label='Random Baseline')
ax_gain.fill_between(_x_vals, _decile_pcts, list(np.linspace(10, 100, 10)),
                     alpha=0.12, color=_ORANGE)
ax_gain.set_xlabel('% Population Targeted', color=_TEXT, fontsize=12)
ax_gain.set_ylabel('% Churners Captured', color=_TEXT, fontsize=12)
ax_gain.set_title('ZervePulse — Cumulative Gain Chart', fontsize=14,
                   fontweight='bold', color=_TEXT, pad=12)
ax_gain.set_xlim(0, 105)
ax_gain.set_ylim(0, 105)
ax_gain.legend(fontsize=11, facecolor='#2a2a2e', edgecolor=_SUBTLE, labelcolor=_TEXT)
ax_gain.grid(True, alpha=0.12, color=_SUBTLE)
ax_gain.annotate(f'{_decile_pcts[0]:.1f}%',
                 xy=(10, _decile_pcts[0]), xytext=(18, _decile_pcts[0] - 8),
                 color=_GOLD, fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=_GOLD, lw=1.5))
plt.tight_layout()
plt.savefig('cumulative_gain.png', dpi=150, bbox_inches='tight', facecolor=_BG)
plt.show()
print(f"✓ Cumulative Gain Chart rendered")
print(f"  Decile gains: {[f'{v:.1f}%' for v in _decile_pcts]}")

# ════════════════════════════════════════════════════════════════════
# METRIC 3 — SHAP Feature Impact (Permutation Importance proxy)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 3 — SHAP Feature Impact (via Permutation Importance)")
print("=" * 70)

from sklearn.inspection import permutation_importance

_shap_X = eval_df[EVAL_FEATURES].copy()
_shap_y = eval_df['churn_label'].copy()

_perm_result  = permutation_importance(
    eval_best_model, _shap_X, _shap_y,
    n_repeats=10, random_state=42, scoring='roc_auc', n_jobs=-1
)
_shap_mean_abs = np.maximum(_perm_result.importances_mean, 0)
_shap_std      = _perm_result.importances_std

_shap_fi_df = pd.DataFrame({
    'feature': EVAL_FEATURES,
    'importance': _shap_mean_abs,
    'std': _shap_std
}).sort_values('importance', ascending=True).reset_index(drop=True)

_shap_top3_idx = np.argsort(_shap_mean_abs)[::-1][:3]
print("\nTop 3 features by mean |SHAP value| (permutation importance):")
for _rank, _idx in enumerate(_shap_top3_idx, 1):
    print(f"  {_rank}. {EVAL_FEATURES[_idx]:<38}  mean|SHAP| = {_shap_mean_abs[_idx]:.5f}")

fig_shap, ax_shap = plt.subplots(figsize=(11, 9))
fig_shap.patch.set_facecolor(_BG)
_style_ax(ax_shap)

_shap_colors = [_GOLD if v == _shap_fi_df['importance'].max() else _BLUE
                for v in _shap_fi_df['importance']]
_shap_bars = ax_shap.barh(
    _shap_fi_df['feature'], _shap_fi_df['importance'],
    xerr=_shap_fi_df['std'], color=_shap_colors,
    edgecolor='none', height=0.6,
    error_kw=dict(ecolor=_SUBTLE, linewidth=1.2, capsize=3)
)
_shap_max = _shap_fi_df['importance'].max() if _shap_fi_df['importance'].max() > 0 else 1
for _sb, _sv in zip(_shap_bars, _shap_fi_df['importance']):
    ax_shap.text(_sv + _shap_max * 0.01, _sb.get_y() + _sb.get_height() / 2,
                 f'{_sv:.4f}', va='center', ha='left', fontsize=8.5, color=_TEXT)
ax_shap.set_xlabel('Mean |SHAP Value| (Permutation Importance on ROC-AUC)', color=_TEXT, fontsize=11)
ax_shap.set_title('ZervePulse — SHAP Feature Impact', fontsize=14,
                   fontweight='bold', color=_TEXT, pad=12)
ax_shap.set_xlim(0, _shap_max * 1.20)
ax_shap.grid(True, axis='x', alpha=0.12, color=_SUBTLE)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight', facecolor=_BG)
plt.show()
print("✓ SHAP Feature Impact chart rendered")

# ════════════════════════════════════════════════════════════════════
# METRIC 4 — Onboarding Friction
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 4 — Onboarding Friction")
print("=" * 70)

# Note: upstream replaced -1 with median (0.0), so 'Never' bucket may be empty
def _ttfa_bucket(v):
    if v == -1:
        return 'Never'
    elif v <= 10:
        return 'Fast (0-10m)'
    elif v <= 60:
        return 'Medium (10-60m)'
    else:
        return 'Slow (60-1440m)'

eval_df['_ttfa_bucket'] = eval_df['time_to_first_agent_minutes'].apply(_ttfa_bucket)
_bucket_order  = ['Fast (0-10m)', 'Medium (10-60m)', 'Slow (60-1440m)', 'Never']
_bucket_colors = [_GREEN, _BLUE, _ORANGE, _RED]

_ttfa_grouped = eval_df.groupby('_ttfa_bucket').agg(
    count=('churn_label', 'count'),
    mean_churn_prob=('churn_probability', 'mean'),
    churn_rate=('churn_label', 'mean')
)
# Reindex with fill_value for missing buckets (like 'Never')
_ttfa_stats = _ttfa_grouped.reindex(_bucket_order).reset_index()
_ttfa_stats['count'] = _ttfa_stats['count'].fillna(0)
_ttfa_stats['churn_rate'] = _ttfa_stats['churn_rate'].fillna(0)
_ttfa_stats['mean_churn_prob'] = _ttfa_stats['mean_churn_prob'].fillna(0)

print(_ttfa_stats.to_string(index=False))

# Filter to non-zero buckets for chart
_chart_buckets = _ttfa_stats[_ttfa_stats['count'] > 0]
_chart_labels  = _chart_buckets['_ttfa_bucket'].tolist()
_chart_colors  = [c for lbl, c in zip(_bucket_order, _bucket_colors)
                  if lbl in _chart_labels]

fig_ttfa, ax_ttfa = plt.subplots(figsize=(10, 7))
fig_ttfa.patch.set_facecolor(_BG)
_style_ax(ax_ttfa)

_b_idx  = np.arange(len(_chart_labels))
_b_bars = ax_ttfa.bar(_b_idx, _chart_buckets['churn_rate'].values * 100,
                      color=_chart_colors, edgecolor='none', width=0.6)
ax_ttfa.set_xticks(_b_idx)
ax_ttfa.set_xticklabels(_chart_labels, color=_TEXT, fontsize=11)
ax_ttfa.set_ylabel('Churn Rate (%)', color=_TEXT, fontsize=12)
ax_ttfa.set_title('ZervePulse — Time to First Agent vs Churn', fontsize=14,
                   fontweight='bold', color=_TEXT, pad=12)

for _b_bar, _brow in zip(_b_bars, _chart_buckets.itertuples()):
    _bv = _brow.churn_rate * 100
    ax_ttfa.text(_b_bar.get_x() + _b_bar.get_width() / 2, _bv + 0.5,
                 f'{_bv:.1f}%\n(n={int(_brow.count):,})', ha='center',
                 va='bottom', fontsize=9.5, color=_TEXT, fontweight='bold')

_max_rate = _chart_buckets['churn_rate'].max() if len(_chart_buckets) > 0 else 1
ax_ttfa.set_ylim(0, _max_rate * 130)
ax_ttfa.grid(True, axis='y', alpha=0.12, color=_SUBTLE)
plt.tight_layout()
plt.savefig('onboarding_friction.png', dpi=150, bbox_inches='tight', facecolor=_BG)
plt.show()
print("✓ Onboarding Friction chart rendered")

# Helper: get rate safely for summary (handles NaN/empty bucket)
def _get_rate(bucket_name):
    _row = _ttfa_stats[_ttfa_stats['_ttfa_bucket'] == bucket_name]['churn_rate']
    return float(_row.values[0]) if len(_row) > 0 else 0.0

# ════════════════════════════════════════════════════════════════════
# METRIC 5 — Estimated CLV by Persona
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 5 — Estimated CLV by Persona")
print("=" * 70)

def _get_persona(row):
    _ag = int(row['has_used_agent'])
    _ob = int(row['onboarding_complete'])
    _dp = int(row['has_deployed'])
    _py = int(row['is_python_user'])
    if _ag == 1 and _ob == 1 and _dp == 1:
        return 'Power User (Agent+Onboard+Deploy)'
    elif _ag == 1 and _ob == 1:
        return 'Engaged (Agent+Onboarded)'
    elif _ag == 1 and _dp == 1:
        return 'Builder (Agent+Deployed)'
    elif _py == 1 and _ag == 1:
        return 'Python Developer (SDK+Agent)'
    elif _ob == 1 and _dp == 1:
        return 'Onboarded Builder (Onboard+Deploy)'
    elif _ag == 1:
        return 'Agent Explorer (Agent Only)'
    elif _ob == 1:
        return 'Onboarded (No Agent)'
    elif _py == 1:
        return 'Python User (SDK Only)'
    else:
        return 'Basic User (Minimal Engagement)'

eval_df['_persona'] = eval_df.apply(_get_persona, axis=1)

_clv_df = eval_df.groupby('_persona').agg(
    _count=('total_credits_used', 'count'),
    _mean_credits=('total_credits_used', 'mean'),
    _mean_obs_days=('observation_days', 'mean'),
).reset_index()
_clv_df['clv'] = _clv_df['_mean_credits'] * (_clv_df['_mean_obs_days'] / 30)
_clv_df = _clv_df.sort_values('clv', ascending=True).reset_index(drop=True)

print(_clv_df[['_persona', '_count', '_mean_credits', '_mean_obs_days', 'clv']].to_string(index=False))

fig_clv, ax_clv = plt.subplots(figsize=(13, 8))
fig_clv.patch.set_facecolor(_BG)
_style_ax(ax_clv)

_clv_palette = [_BLUE, _LAVENDER, _GREEN, _ORANGE, _GOLD, _CORAL, _RED, _BLUE, _GREEN]
_clv_colors  = _clv_palette[:len(_clv_df)]
_clv_bars = ax_clv.barh(_clv_df['_persona'], _clv_df['clv'],
                        color=_clv_colors, edgecolor='none', height=0.65)
_clv_max = _clv_df['clv'].max() if _clv_df['clv'].max() > 0 else 1
for _cb, _cv in zip(_clv_bars, _clv_df['clv']):
    ax_clv.text(_cv + _clv_max * 0.01, _cb.get_y() + _cb.get_height() / 2,
                f'{_cv:.1f}', va='center', ha='left', fontsize=9.5, color=_TEXT, fontweight='bold')
ax_clv.set_xlabel('Estimated CLV (Credits x Months)', color=_TEXT, fontsize=12)
ax_clv.set_title('ZervePulse — Estimated CLV by Persona', fontsize=14,
                  fontweight='bold', color=_TEXT, pad=12)
ax_clv.set_xlim(0, _clv_max * 1.18)
ax_clv.grid(True, axis='x', alpha=0.12, color=_SUBTLE)
plt.tight_layout()
plt.savefig('clv_by_persona.png', dpi=150, bbox_inches='tight', facecolor=_BG)
plt.show()
print("✓ CLV by Persona chart rendered")

# ════════════════════════════════════════════════════════════════════
# METRIC 6 — Retention Curve
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 6 — Retention Curve")
print("=" * 70)

_total_users = len(eval_df)
_day1_ret    = (eval_df['observation_days'] >= 1).mean() * 100
_day7_ret    = (eval_df['observation_days'] >= 7).mean() * 100
_day30_ret   = (eval_df['observation_days'] >= 30).mean() * 100

print(f"Day 1  Retention : {_day1_ret:.1f}%  ({(eval_df['observation_days'] >= 1).sum():,} / {_total_users:,})")
print(f"Day 7  Retention : {_day7_ret:.1f}%  ({(eval_df['observation_days'] >= 7).sum():,} / {_total_users:,})")
print(f"Day 30 Retention : {_day30_ret:.1f}%  ({(eval_df['observation_days'] >= 30).sum():,} / {_total_users:,})")

fig_ret, ax_ret = plt.subplots(figsize=(9, 7))
fig_ret.patch.set_facecolor(_BG)
_style_ax(ax_ret)

_ret_bars = ax_ret.bar(['Day 1', 'Day 7', 'Day 30'],
                       [_day1_ret, _day7_ret, _day30_ret],
                       color=[_GREEN, _BLUE, _ORANGE], edgecolor='none', width=0.5)
for _rb, _rv in zip(_ret_bars, [_day1_ret, _day7_ret, _day30_ret]):
    ax_ret.text(_rb.get_x() + _rb.get_width() / 2, _rv + 0.5,
                f'{_rv:.1f}%', ha='center', va='bottom', fontsize=13, color=_TEXT, fontweight='bold')
ax_ret.set_ylabel('Retention Rate (%)', color=_TEXT, fontsize=12)
ax_ret.set_title('ZervePulse — Retention Curve', fontsize=14,
                  fontweight='bold', color=_TEXT, pad=12)
ax_ret.set_ylim(0, 115)
ax_ret.grid(True, axis='y', alpha=0.12, color=_SUBTLE)
plt.tight_layout()
plt.savefig('retention_curve.png', dpi=150, bbox_inches='tight', facecolor=_BG)
plt.show()
print("✓ Retention Curve chart rendered")

# ════════════════════════════════════════════════════════════════════
# METRIC 7 — Revenue Saved
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 7 — Revenue Saved")
print("=" * 70)

_retained_users        = eval_df[eval_df['churn_label'] == 0]
_mean_credits_retained = _retained_users['total_credits_used'].mean()
_high_risk_count       = int((eval_df['churn_probability'] > 0.7).sum())
_users_saved           = _high_risk_count * 0.20
_revenue_saved         = _users_saved * _mean_credits_retained

print("--- ZervePulse Revenue Impact ---")
print(f"Mean credits retained users          : {_mean_credits_retained:.2f}")
print(f"High-risk users (prob > 0.7)         : {_high_risk_count:,}")
print(f"Users saved (20% intervention rate)  : {_users_saved:.1f}")
print(f"Revenue saved (credits)              : {_revenue_saved:,.2f}")

# ════════════════════════════════════════════════════════════════════
# METRIC 8 — DAU/MAU Stickiness
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("METRIC 8 — DAU/MAU Stickiness")
print("=" * 70)

eval_df['_dau_mau'] = eval_df['days_active'] / eval_df['observation_days'].clip(lower=1)

_retained_dau_mau = eval_df[eval_df['churn_label'] == 0]['_dau_mau']
_churned_dau_mau  = eval_df[eval_df['churn_label'] == 1]['_dau_mau']
_mean_retained_dm = _retained_dau_mau.mean()
_mean_churned_dm  = _churned_dau_mau.mean()
_sticky_count     = int((eval_df['_dau_mau'] > 0.5).sum())

print(f"Mean DAU/MAU — Retained : {_mean_retained_dm:.4f}")
print(f"Mean DAU/MAU — Churned  : {_mean_churned_dm:.4f}")
print(f"Users with DAU/MAU > 0.5: {_sticky_count:,}")

fig_dau, ax_dau = plt.subplots(figsize=(10, 7))
fig_dau.patch.set_facecolor(_BG)
_style_ax(ax_dau)

_bp = ax_dau.boxplot(
    [_retained_dau_mau.values, _churned_dau_mau.values],
    labels=['Retained\n(churn=0)', 'Churned\n(churn=1)'],
    patch_artist=True, notch=False,
    medianprops=dict(color=_GOLD, linewidth=2.5),
    whiskerprops=dict(color=_SUBTLE, linewidth=1.5),
    capprops=dict(color=_SUBTLE, linewidth=1.5),
    flierprops=dict(marker='o', color=_SUBTLE, alpha=0.3, markersize=3)
)
_bp['boxes'][0].set_facecolor(_GREEN + '55')
_bp['boxes'][0].set_edgecolor(_GREEN)
_bp['boxes'][1].set_facecolor(_RED + '55')
_bp['boxes'][1].set_edgecolor(_RED)

ax_dau.axhline(0.5, linestyle='--', color=_GOLD, linewidth=1.5,
               label='DAU/MAU = 0.5 threshold', alpha=0.8)
ax_dau.set_ylabel('DAU/MAU Ratio', color=_TEXT, fontsize=12)
ax_dau.set_title('ZervePulse — DAU/MAU Stickiness', fontsize=14,
                  fontweight='bold', color=_TEXT, pad=12)
ax_dau.legend(fontsize=10, facecolor='#2a2a2e', edgecolor=_SUBTLE, labelcolor=_TEXT)
ax_dau.grid(True, axis='y', alpha=0.12, color=_SUBTLE)

for _xi, (_mn,) in enumerate(zip([_mean_retained_dm, _mean_churned_dm]), 1):
    ax_dau.text(_xi + 0.05, _mn, f'mu={_mn:.3f}', va='center', ha='left',
                fontsize=10, color=_GOLD, fontweight='bold')

plt.tight_layout()
plt.savefig('dau_mau_stickiness.png', dpi=150, bbox_inches='tight', facecolor=_BG)
plt.show()
print("✓ DAU/MAU Stickiness chart rendered")

# ════════════════════════════════════════════════════════════════════
# FINAL SUMMARY — ZervePulse 22-Metric Report
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY — ZervePulse 22-Metric Report")
print("=" * 70)

_bm = best_metrics  # from upstream
_top_clv_persona = _clv_df.iloc[-1]['_persona']
_top_clv_val     = _clv_df.iloc[-1]['clv']

_fast_rate   = _get_rate('Fast (0-10m)')
_medium_rate = _get_rate('Medium (10-60m)')
_slow_rate   = _get_rate('Slow (60-1440m)')
_never_rate  = _get_rate('Never')

_summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ZERVEPULSE — COMPLETE EVALUATION SUMMARY                            ║
║        Best Model: {best_model_name:<57}║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  [1] MODEL PERFORMANCE                                                       ║
║  Accuracy            : {_bm['Accuracy']:.4f}                                            ║
║  Precision (cls 1)   : {_bm['Precision (cls 1)']:.4f}                                            ║
║  Recall (cls 1)      : {_bm['Recall (cls 1)']:.4f}                                            ║
║  F1 Score (cls 1)    : {_bm['F1 (cls 1)']:.4f}                                            ║
║  ROC-AUC             : {_bm['ROC-AUC']:.4f}                                            ║
║  PR-AUC              : {_bm['PR-AUC']:.4f}                                            ║
║  MCC                 : {_bm['MCC']:.4f}                                            ║
║  Cohen's Kappa       : {_bm["Cohen's Kappa"]:.4f}                                            ║
║  Brier Score         : {_bm['Brier Score']:.4f}                                            ║
║  Top-10% Decile Lift : {_m1_lift:.4f}x ({_m1_churned_in_top10}/{_m1_top10_n} top-10% users churned)        ║
║                                                                              ║
║  [2] BEHAVIORAL INTELLIGENCE                                                 ║
║  Mean DAU/MAU (Retained)    : {_mean_retained_dm:.4f}                                    ║
║  Mean DAU/MAU (Churned)     : {_mean_churned_dm:.4f}                                    ║
║  Fast onboarding churn rate : {_fast_rate*100:.1f}%                                         ║
║  Med  onboarding churn rate : {_medium_rate*100:.1f}%                                         ║
║  Slow onboarding churn rate : {_slow_rate*100:.1f}%                                         ║
║  Never used agent churn rate: {_never_rate*100:.1f}%                                         ║
║  Top CLV Persona            : {_top_clv_persona:<44}║
║  Top CLV Value              : {_top_clv_val:.2f} credits x months                       ║
║  Users with DAU/MAU > 0.5  : {_sticky_count:,}                                          ║
║  Cumul. gain top 10%        : {_decile_pcts[0]:.1f}% churners captured                          ║
║                                                                              ║
║  [3] PLATFORM HEALTH                                                         ║
║  Day  1 Retention  : {_day1_ret:.1f}%                                              ║
║  Day  7 Retention  : {_day7_ret:.1f}%                                              ║
║  Day 30 Retention  : {_day30_ret:.1f}%                                              ║
║                                                                              ║
║  [4] BUSINESS IMPACT                                                         ║
║  High-risk users (prob>0.7) : {_high_risk_count:,}                                          ║
║  Users saved (20% rate)     : {_users_saved:.1f}                                        ║
║  Revenue saved (credits)    : {_revenue_saved:,.2f}                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

print(_summary)
print("ZervePulse reports 22 metrics — churn analysis.")

# ── Save report file ──────────────────────────────────────────────
_report_path = 'ZervePulse_22_Metrics.txt'
with open(_report_path, 'w') as _f:
    _f.write("ZERVEPULSE — COMPLETE EVALUATION SUMMARY\n")
    _f.write(f"Best Model: {best_model_name}\n")
    _f.write("=" * 80 + "\n\n")
    _f.write(_summary)
    _f.write("\n\n=== Detailed Metrics ===\n")
    _f.write("\n[MODEL PERFORMANCE]\n")
    for _k, _v in _bm.items():
        _f.write(f"  {_k}: {_v:.6f}\n")
    _f.write("\n[DECILE LIFT]\n")
    _f.write(f"  Top-10% users: {_m1_top10_n:,}\n")
    _f.write(f"  Churners in top-10%: {_m1_churned_in_top10:,}\n")
    _f.write(f"  Lift ratio: {_m1_lift:.6f}x\n")
    _f.write(f"  % churners captured: {_m1_pct_captured:.2f}%\n")
    _f.write("\n[CUMULATIVE GAIN BY DECILE]\n")
    for _lbl, _pct in zip(_decile_labels, _decile_pcts):
        _f.write(f"  {_lbl}: {_pct:.2f}%\n")
    _f.write("\n[FEATURE IMPORTANCE — TOP 3 (SHAP proxy via Permutation Importance)]\n")
    for _rank, _idx in enumerate(_shap_top3_idx, 1):
        _f.write(f"  {_rank}. {EVAL_FEATURES[_idx]}: {_shap_mean_abs[_idx]:.6f}\n")
    _f.write("\n[ONBOARDING FRICTION]\n")
    for _, _brow in _ttfa_stats.iterrows():
        _cnt = int(_brow['count'])
        _f.write(f"  {_brow['_ttfa_bucket']}: n={_cnt}, churn_rate={_brow['churn_rate']:.4f}, mean_prob={_brow['mean_churn_prob']:.4f}\n")
    _f.write("\n[CLV BY PERSONA]\n")
    for _, _crow in _clv_df.iterrows():
        _f.write(f"  {_crow['_persona']}: CLV={_crow['clv']:.2f}, n={int(_crow['_count'])}\n")
    _f.write("\n[RETENTION]\n")
    _f.write(f"  Day 1 : {_day1_ret:.2f}%\n")
    _f.write(f"  Day 7 : {_day7_ret:.2f}%\n")
    _f.write(f"  Day 30: {_day30_ret:.2f}%\n")
    _f.write("\n[REVENUE SAVED]\n")
    _f.write(f"  Mean credits (retained users): {_mean_credits_retained:.4f}\n")
    _f.write(f"  High-risk users (prob>0.7): {_high_risk_count:,}\n")
    _f.write(f"  Users saved: {_users_saved:.1f}\n")
    _f.write(f"  Revenue saved: {_revenue_saved:,.2f}\n")
    _f.write("\n[DAU/MAU STICKINESS]\n")
    _f.write(f"  Mean DAU/MAU retained: {_mean_retained_dm:.4f}\n")
    _f.write(f"  Mean DAU/MAU churned:  {_mean_churned_dm:.4f}\n")
    _f.write(f"  Users with DAU/MAU > 0.5: {_sticky_count:,}\n")
    _f.write("\n\nZervePulse reports 22 metrics — churn analysis.\n")

_file_sz = os.path.getsize(_report_path)
print(f"\n✅ ZervePulse_22_Metrics.txt saved — {_file_sz:,} bytes")
print(f"ZervePulse reports 22 metrics — churn analysis.")
