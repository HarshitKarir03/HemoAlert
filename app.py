import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HemoAlert",
    layout="wide",
    page_icon="🫀"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] .stSlider label {
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Header */
.hemo-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 4px;
}
.hemo-title {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    color: #e6edf3;
}
.hemo-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #58a6ff;
    background: #0d2a4a;
    border: 1px solid #1f4a7a;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}
.hemo-subtitle {
    font-size: 12px;
    color: #6e7681;
    font-family: 'DM Mono', monospace;
    margin-bottom: 28px;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
}
.metric-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #6e7681;
    text-transform: uppercase;
}
.metric-card .value {
    font-family: 'DM Mono', monospace;
    font-size: 1.7rem;
    color: #e6edf3;
}
.metric-card .sub {
    font-size: 11px;
    color: #6e7681;
}

/* Risk banner */
.risk-banner {
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.risk-banner.high {
    background: #1a0a0a;
    border: 1px solid #6e1a1a;
}
.risk-banner.moderate {
    background: #1a1300;
    border: 1px solid #6e5000;
}
.risk-banner.low {
    background: #0a1a0f;
    border: 1px solid #1a5c2a;
}
.risk-title {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    text-transform: uppercase;
}
.risk-banner.high .risk-title { color: #f85149; }
.risk-banner.moderate .risk-title { color: #e3b341; }
.risk-banner.low .risk-title { color: #3fb950; }

.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6e7681;
    text-transform: uppercase;
    margin: 24px 0 14px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
}

/* Hide Streamlit chrome */
#MainMenu, footer {
    visibility: hidden;
}
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/xgb_calibrated.pkl')
    feat_cols = pd.read_csv('data/processed/feature_cols.csv').iloc[:, 0].tolist()
    with open('models/metadata.json') as f:
        meta = json.load(f)
    return model, feat_cols, meta

model, feature_cols, meta = load_model()
THRESHOLD = meta['threshold']

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hemo-header">
    <span class="hemo-title">HemoAlert</span>
    <span class="hemo-badge">XGBoost · MIMIC-IV</span>
</div>
<div class="hemo-subtitle">
    ICU Hemodynamic Instability Early Warning · 
    AUC-ROC 0.834 · 
    Median Lead Time {meta['median_lead_h']:.1f}h ·
    Threshold {THRESHOLD:.1%}
</div>
""", unsafe_allow_html=True)

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    map_now = st.slider("MAP now", 40, 120, 82)
    map_1h = st.slider("MAP 1h ago", 40, 120, 86)
    map_3h = st.slider("MAP 3h ago", 40, 120, 90)
    hr_now = st.slider("HR now", 40, 160, 88)
    hr_1h = st.slider("HR 1h ago", 40, 160, 82)
    sbp = st.slider("SBP", 60, 200, 118)
    dbp = st.slider("DBP", 30, 120, 72)
    spo2 = st.slider("SpO2", 70, 100, 96)
    resp_rate = st.slider("Resp Rate", 8, 40, 18)
    lactate = st.slider("Lactate", 0.5, 10.0, 1.4)
    creatinine = st.slider("Creatinine", 0.3, 10.0, 1.1)
    vaso_on = st.toggle("Vasopressor Running")

# ── Feature Engineering ────────────────────────────────────────────────────────
map_slope = (map_now - map_3h) / 3
hr_slope = hr_now - hr_1h

input_dict = {col: 0 for col in feature_cols}
input_dict.update({
    'map': map_now,
    'map_lag1': map_1h,
    'map_lag3': map_3h,
    'heart_rate': hr_now,
    'heart_rate_lag1': hr_1h,
    'spo2': spo2,
    'resp_rate': resp_rate,
    'lactate': lactate,
    'creatinine': creatinine,
    'sbp': sbp,
    'dbp': dbp,
    'map_slope': map_slope,
    'hr_slope': hr_slope,
    'shock_index': hr_now / sbp,
    'pulse_pressure': sbp - dbp,
    'hr_map_product': hr_now * map_now,
    'compensation_signal': hr_slope - map_slope,
    'vasopressor_on': int(vaso_on),
    'map_mean6h': np.mean([map_now, map_1h, map_3h]),
    'heart_rate_mean6h': np.mean([hr_now, hr_1h]),
})

X_input = pd.DataFrame([input_dict])[feature_cols]
risk_score = model.predict_proba(X_input)[0][1]

# ── Risk Banner ────────────────────────────────────────────────────────────────
if risk_score >= THRESHOLD:
    banner_class = "high"
    title = "HIGH RISK"
elif risk_score >= THRESHOLD * 0.6:
    banner_class = "moderate"
    title = "MODERATE RISK"
else:
    banner_class = "low"
    title = "LOW RISK"

st.markdown(f"""
<div class="risk-banner {banner_class}">
    <div>
        <div class="risk-title">{title}</div>
        <div>{risk_score:.1%} predicted probability</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_gauge, col_shap = st.columns([1, 2])

# ── Gauge ──────────────────────────────────────────────────────────────────────
with col_gauge:
    st.markdown('<div class="section-header">Risk Gauge</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw={'projection':'polar'})
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    green_end = THRESHOLD * 0.6
    yellow_end = THRESHOLD

    def prob_to_theta(p):
        return np.pi * (1 - p)

    segments = [
        (0, green_end, '#1a5c2a'),
        (green_end, yellow_end, '#6e5000'),
        (yellow_end, 1.0, '#6e1a1a')
    ]

    for start, end, color in segments:
        theta = np.linspace(prob_to_theta(end), prob_to_theta(start), 100)
        ax.plot(theta, [1]*100, color=color, linewidth=18)

    needle_angle = prob_to_theta(risk_score)

    # Needle
    ax.plot(
        [needle_angle, needle_angle],
        [0, 0.82],
        color='white',
        linewidth=3.5,
        zorder=5
    )

    # Fixed center pivot
    ax.scatter(
        [np.pi/2],
        [0],
        s=180,
        color='white',
        zorder=6
    )

    ax.set_ylim(0, 1.25)
    ax.axis('off')

    st.pyplot(fig)
    plt.close()


# ── SHAP Plot ──────────────────────────────────────────────────────────────────
with col_shap:
    st.markdown('<div class="section-header">Feature Contributions</div>', unsafe_allow_html=True)

    try:
        raw_model = model.calibrated_classifiers_[0].estimator
        raw_explainer = shap.TreeExplainer(raw_model)
        shap_vals = raw_explainer.shap_values(X_input)[0]

        contrib_df = pd.DataFrame({
            'feature': feature_cols,
            'shap': shap_vals,
            'value': X_input.iloc[0].values
        })

        contrib_df['abs_shap'] = contrib_df['shap'].abs()
        contrib_df = contrib_df.sort_values('abs_shap').tail(12)

        name_map = {
            'map_mean6h': 'MAP mean (6h)',
            'map': 'MAP (current)',
            'vasopressor_on': 'Vasopressor running',
            'shock_index': 'Shock index',
            'lactate': 'Lactate',
            'hr_slope': 'HR trend',
            'map_slope': 'MAP trend',
            'heart_rate': 'Heart rate',
            'creatinine': 'Creatinine',
            'pulse_pressure': 'Pulse pressure',
            'compensation_signal': 'Compensation signal',
            'map_lag3': 'MAP (3h ago)',
            'heart_rate_lag1': 'HR (1h ago)',
        }

        contrib_df['display'] = contrib_df['feature'].map(
            lambda x: name_map.get(x, x.replace('_', ' ').title())
        )

        fig2, ax2 = plt.subplots(figsize=(6,5.2))
        fig2.patch.set_facecolor('#161b22')
        ax2.set_facecolor('#161b22')

        colors = ['#f85149' if v > 0 else '#58a6ff' for v in contrib_df['shap']]

        ax2.barh(
            range(len(contrib_df)),
            contrib_df['shap'],
            color=colors,
            height=0.62
        )

        ax2.set_yticks(range(len(contrib_df)))
        ax2.set_yticklabels(
            contrib_df['display'],
            fontsize=9,
            color='#c9d1d9',
            fontfamily='monospace'
        )

        ax2.axvline(0, color='#30363d', linewidth=1)

        ax2.tick_params(axis='x', colors='#6e7681')
        ax2.tick_params(axis='y', length=0)

        for spine in ax2.spines.values():
            spine.set_visible(False)

        ax2.grid(axis='x', color='#21262d', linewidth=0.5)

        red_patch = mpatches.Patch(color='#f85149', label='Increases risk')
        blue_patch = mpatches.Patch(color='#58a6ff', label='Decreases risk')

        legend = ax2.legend(
            handles=[red_patch, blue_patch],
            loc='lower right',
            fontsize=8,
            facecolor='#161b22',
            edgecolor='#30363d',
            framealpha=0.95
        )

        for text in legend.get_texts():
            text.set_color('#c9d1d9')
            text.set_fontfamily('monospace')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    except Exception as e:
        st.info(f"Feature chart unavailable: {e}")