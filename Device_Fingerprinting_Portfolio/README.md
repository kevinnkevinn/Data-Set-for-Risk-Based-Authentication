# Device Fingerprinting Portfolio (with RBA Signals)

## Quickstart

# 1) Create venv (Windows)
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

# 2) Train model (simpan model.pkl)
python src/models/train.py --input data/sample_events.csv --out model.pkl

# 3) Run API
uvicorn src.api.app:app --reload --port 8000
# POST /predict with event JSON
\\\

## Docker
\\\bash
docker build -t device-fp .
docker run -p 8000:8000 device-fp
\\\

## Structure
- data/: sample data for EDA & training
- notebooks/: EDA + modeling + RBA enrichment
- src/: features, models, utils, FastAPI
- models/random_forest_risk_model/: pretrained artifacts (optional)
- scripts/: setup scripts (Windows)

