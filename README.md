# Unified Device Fingerprinting — Risk-Based Authentication (RBA)

A data science endeavor that is dedicated to the evaluation of login risk by utilizing indications such as device type and browser, as well as behavioral traits.
  The entire workflow is facilitated by the notebook, which includes the following steps: importing the data, doing exploratory data analysis, developing features, training and evaluating the model, and conducting both batch and real-time scoring.

---

## 1. Project Contents
- **Notebook (HTML):** end-to-end pipeline with code cells and printed summaries.
- **Security Report Section:** operational recommendations — MFA, monitoring, retraining cadence, alerting.

> Note: Keep raw data out of git. Commit only code, configs, and sample CSVs.

---

## 2. Pipeline Overview

1. **Data Loading & Cleaning** — normalize timestamps, IPs, and device info; handle missing values.
2. **EDA** — visualize distributions, correlations, and outliers to guide feature creation.
3. **Feature Engineering** — generate fingerprints, geo/IP patterns, and temporal velocity metrics.
4. **Modeling** — train and compare multiple ML models using Accuracy, ROC-AUC, PR metrics.
5. **Evaluation** — set thresholds to balance false positives vs false negatives.
6. **Batch Scoring** — score historical datasets for retrospective analysis.
7. **Real-Time Scoring** — implement via REST (e.g., FastAPI) for login-time risk evaluation.
8. **Ops Guidance** — monitoring, MFA triggers, retraining schedule, and alert integration.

---

## 3. Quick Start (Local)

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

jupyter lab
```

---

## 4. Suggested Repository Structure

```
.
├── notebooks/
│   ├── Unified_Device_Fingerprinting_RBA_clean.ipynb
│   └── exported/Unified_Device_Fingerprinting_RBA_clean.html
├── src/
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── serve_api.py
├── models/
│   ├── preprocessor.pkl
│   └── model.pkl
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

---

## 5. Reproducible Workflow

```bash
python src/train.py --config configs/train.yaml
python src/evaluate.py --run-id <id>
python src/score_batch.py --input data/processed/events.csv --output outputs/scores.csv
uvicorn src.serve_api:app --host 0.0.0.0 --port 8000
```

---

## 6. Model Metrics & Selection
- Evaluate models using ROC-AUC and PR curves.
- Log all metrics in `outputs/metrics.csv` and plots in `outputs/plots/`.
- Iterate on feature sets when metrics plateau.

---

## 7. Operations & Security Guidelines
- Enable MFA when the risk score exceeds a defined threshold.
- Monitor login patterns and trigger alerts on spikes.
- Schedule retraining with new data and maintain audit logs for transparency.

---

## 8. Notes & Limitations
- The HTML export includes MathJax/Mermaid loaders for rendering only — logic unaffected.
- Placeholder metrics (e.g., 0.500 AUC) indicate synthetic/minimal data; verify label integrity.

---

## 9. License & Citation
If you use or extend this project, please cite the notebook export:

**Unified_Device_Fingerprinting_RBA_clean.ipynb** (Notebook Export)

---

**Maintainer:** Kevin Kiding  
