"""
Train baseline & advanced models for Device Fingerprinting.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

from src.features.device_fingerprint import build_features, select_feature_cols

def build_preprocessor(df, feature_cols):
    num_cols = [c for c in feature_cols if df[c].dtype != "object"]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols),
        ]
    )
    return pre

def main(args):
    df = pd.read_csv(args.input)
    df = build_features(df)
    feature_cols = select_feature_cols(df)

    # supervised label
    y = df["label_is_same_device"].astype(int)
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = build_preprocessor(df, feature_cols)

    # Baseline
    baseline = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200))
    ])
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    y_proba = baseline.predict_proba(X_test)[:,1]
    print("=== Baseline (LogReg) ===")
    print(classification_report(y_test, y_pred, digits=4))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    # Advanced
    advanced = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=4, tree_method="hist"
        ))
    ])
    advanced.fit(X_train, y_train)
    y_pred2 = advanced.predict(X_test)
    y_proba2 = advanced.predict_proba(X_test)[:,1]
    print("\n=== Advanced (XGBoost) ===")
    print(classification_report(y_test, y_pred2, digits=4))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_proba2))
    except Exception:
        pass

    # Save the better model (for demo we save advanced)
    joblib.dump(advanced, args.out)
    print(f"Saved model → {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="model.pkl")
    main(p.parse_args())
