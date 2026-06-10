# HemoAlert
### ICU Hemodynamic Instability Early Warning System

> Predicting circulatory collapse up to **8 hours** before it happens — using EHR vitals, lab trajectories, and temporal machine learning on 8 million ICU timesteps.

---

## What it does

HemoAlert monitors ICU patients and predicts hemodynamic instability (shock) 4–8 hours before it occurs — using only routine EHR data (vitals + labs). No special hardware required.

Standard ICU alarms fire **when** MAP drops below 65 mmHg. By then, organ damage may already be occurring. HemoAlert fires **hours before**, giving clinicians time to intervene.

---

## Results

| Metric | HemoAlert (XGBoost) | AHI-PI benchmark |
|---|---|---|
| AUC-ROC | **0.834** | 0.87 (continuous ECG) |
| AUC-PR | **0.540** | 0.37 |
| Sensitivity | **85%** | 86–90% |
| **Median lead time** | **8.0 hours** | 1.1 hours |
| ≥4h warning coverage | **73.7%** | — |

Trained on **8 million timesteps** from 50,000+ MIMIC-IV ICU stays. Time-based 80/20 train/test split.

---

## Key innovations over prior work

- **EHR-native** — uses only nurse-charted vitals and lab draws. Deployable in any ward with an EHR, not just high-tech surgical suites.
- **Temporal modeling** — 6-hour rolling slopes, lag features, and interaction terms capture physiological *trajectory*, not just instantaneous values.
- **Vasopressor co-labeling** — instability label includes new vasopressor initiation, preventing mislabeling of patients who were treated before MAP crashed.
- **Calibrated probabilities** — Platt scaling converts raw scores to true clinical probabilities (a 70% score = ~70% real risk).

---

## Tech stack

| Layer | Tool |
|---|---|
| Data | MIMIC-IV via BigQuery |
| Preprocessing | Python, Pandas, NumPy |
| Primary model | XGBoost + SHAP |
| Secondary model | PyTorch LSTM (Kaggle GPU) |
| Calibration | scikit-learn Platt scaling |
| Dashboard | Streamlit |

---

## Project structure

```
hemoalert/
├── app.py                        ← Streamlit dashboard
├── requirements.txt
├── models/
│   ├── xgb_calibrated.pkl        ← Calibrated XGBoost
│   ├── xgb_raw.pkl
│   ├── shap_explainer.pkl
│   └── metadata.json             ← AUC, threshold, lead time
└── data/
    └── processed/
        └── feature_cols.csv
```


## Data access

Requires credentialed MIMIC-IV access via [PhysioNet](https://physionet.org). Complete CITI training and sign the data use agreement to apply.

---

*HemoAlert v2 · May 2026 · For research use only — not a clinical device*
