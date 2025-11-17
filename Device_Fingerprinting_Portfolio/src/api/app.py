from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.features.device_fingerprint import build_features, select_feature_cols

app = FastAPI(title="Device Fingerprinting Inference API")

# Load model at startup (expect model.pkl in working dir)
try:
    MODEL = joblib.load("model.pkl")
except Exception:
    MODEL = None

class Event(BaseModel):
    event_time: int
    user_id: str
    ip: str
    country: str
    region: str
    city: str
    user_agent: str
    browser_name: str
    browser_version: str
    os_name: str
    os_version: str
    screen_res: str
    device_type: str
    request_per_min_from_ip: int
    is_vpn: int
    is_proxy: int

@app.post("/predict")
def predict(e: Event):
    global MODEL
    if MODEL is None:
        return {"error":"Model not loaded. Train and place model.pkl in cwd."}
    df = pd.DataFrame([e.dict()])
    df = build_features(df)
    feats = select_feature_cols(df)
    proba = MODEL.predict_proba(df[feats])[:,1][0]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}
