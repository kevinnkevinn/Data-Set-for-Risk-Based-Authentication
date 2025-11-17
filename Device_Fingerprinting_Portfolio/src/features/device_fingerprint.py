"""
Device fingerprint feature engineering utilities.
Code: English. Penjelasan: komentar Indonesia singkat.
"""
from typing import Dict, Any, List, Tuple
import re
import hashlib
import math
import numpy as np
import pandas as pd

UA_BROWSER_PATTERNS = {
    "Chrome": r"Chrome/([0-9.]+)",
    "Safari": r"Version/([0-9.]+).*Safari",
    "Firefox": r"Firefox/([0-9.]+)",
    "Edge": r"Edg/([0-9.]+)",
}

def parse_user_agent(ua: str) -> Dict[str, str]:
    """Parse sederhana UA → ekstrak browser_version jika ada. (Indonesia: parsing UA dasar)"""
    if not isinstance(ua, str):
        return {"ua_browser_version": ""}
    for name, pat in UA_BROWSER_PATTERNS.items():
        m = re.search(pat, ua)
        if m:
            return {"ua_browser_version": m.group(1)}
    return {"ua_browser_version": ""}

def screen_res_to_nums(res: str) -> Tuple[int,int,int]:
    """Konversi '1920x1080' → (w,h,pixels)."""
    if not isinstance(res, str) or "x" not in res:
        return 0,0,0
    try:
        w,h = res.split("x")
        w,h = int(w), int(h)
        return w,h,w*h
    except Exception:
        return 0,0,0

def categorical_safemap(series: pd.Series) -> pd.Series:
    """Map kategori jarang menjadi 'OTHER' untuk stabilitas. (Indonesia: reduksi kategori langka)"""
    if series.isna().all():
        return series.fillna("UNKNOWN")
    freq = series.value_counts(normalize=True)
    rare = set(freq[freq < 0.01].index)
    return series.apply(lambda v: "OTHER" if v in rare else ("UNKNOWN" if pd.isna(v) else v))

def device_hash(row: pd.Series) -> str:
    """Hash stabil dari beberapa atribut perangkat (Indonesia: fingerprint kasar)."""
    parts = [
        str(row.get("browser_name","")),
        str(row.get("browser_version","")),
        str(row.get("os_name","")),
        str(row.get("os_version","")),
        str(row.get("screen_res","")),
        str(row.get("device_type","")),
    ]
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bangun fitur untuk device fingerprinting. (Indonesia: fitur utama)"""
    df = df.copy()

    # Categorical normalization
    for col in ["country","region","city","browser_name","os_name","device_type"]:
        if col in df.columns:
            df[col] = categorical_safemap(df[col])

    # UA parse
    ua_parsed = df.get("user_agent","").apply(parse_user_agent)
    df["ua_browser_version"] = [d.get("ua_browser_version","") for d in ua_parsed]

    # Screen resolution numeric
    w,h,p = zip(*df.get("screen_res","").apply(screen_res_to_nums))
    df["screen_w"] = w
    df["screen_h"] = h
    df["screen_pixels"] = p

    # Stability / entropy-ish signals (Indonesia: indikasi konsistensi perangkat)
    # e.g., perubahan UA versi dalam user_id/24h tidak dihitung di sini (untuk notebook/SQL)
    df["ip_octet_1"] = df["ip"].fillna("0.0.0.0").astype(str).str.split(".").str[0].astype(int, errors="ignore")

    # Binary indicators
    for c in ["is_vpn","is_proxy"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # Simple hash-based fingerprint
    df["device_fingerprint_hash"] = df.apply(device_hash, axis=1)

    # Numeric safe fill
    for c in ["request_per_min_from_ip","screen_w","screen_h","screen_pixels"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Ready to model
    return df

def select_feature_cols(df: pd.DataFrame) -> List[str]:
    """Pilih fitur final untuk model supervised. (Indonesia: subset kolom)"""
    base = [
        "request_per_min_from_ip","is_vpn","is_proxy",
        "screen_w","screen_h","screen_pixels","ip_octet_1"
    ]
    # one-hot candidates (bisa di-encode di pipeline)
    cats = ["country","region","city","browser_name","os_name","device_type","ua_browser_version"]
    return [c for c in base + cats if c in df.columns]
