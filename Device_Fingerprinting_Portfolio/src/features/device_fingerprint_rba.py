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


# ===========================
# RBA-style helpers (Indonesia: komponen dari notebook RBA lama)
# ===========================
def _coerce_bool_from_text(series: pd.Series) -> pd.Series:
    """Map text tokens to binary (1 risky, 0 safe, NaN unknown)."""
    pos_tokens = {
        "high", "risky", "risk", "attack", "fraud", "malicious",
        "anomal", "anomaly", "fail", "failed", "denied", "blocked",
        "reject", "rejected", "unauthorized", "alert"
    }
    neg_tokens = {
        "low", "safe", "normal", "benign", "success", "passed",
        "approved", "allow", "allowed", "authorized", "ok"
    }
    def map_token(x):
        if not isinstance(x, str):
            return np.nan
        xl = x.strip().lower()
        if any(tok in xl for tok in pos_tokens):
            return 1
        if any(tok in xl for tok in neg_tokens):
            return 0
        return np.nan
    return series.apply(map_token)

def _coerce_bool_from_numeric(series: pd.Series, true_values={1, "1", True}, false_values={0, "0", False}) -> pd.Series:
    """Map numeric-like values to 0/1 robustly."""
    def map_num(x):
        if x in true_values:
            return 1
        if x in false_values:
            return 0
        try:
            xv = float(x)
            return 1 if xv > 0.5 else 0
        except Exception:
            return np.nan
    return series.apply(map_num)

# Optional GPU hooks
try:
    import cupy as cp  # type: ignore
    _GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _GPU_AVAILABLE = False

def _safe_array(a):
    """Return CuPy array when GPU available, else NumPy."""
    if _GPU_AVAILABLE and cp is not None:
        return cp.asarray(a)
    return np.asarray(a)

def enrich_with_rba_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan sinyal RBA generik:
    - high_rpm_flag: req/min di atas ambang
    - vpn_or_proxy: gabungan boolean
    - ua_os_mismatch (proxy untuk inkonsistensi UA vs kolom eksplisit)
    - simple risk_score: kombinasi berbobot
    """
    df = df.copy()
    rpm = pd.to_numeric(df.get("request_per_min_from_ip", 0), errors="coerce").fillna(0.0)
    is_vpn = _coerce_bool_from_numeric(df.get("is_vpn", 0)).fillna(0).astype(int)
    is_proxy = _coerce_bool_from_numeric(df.get("is_proxy", 0)).fillna(0).astype(int)

    # Flags
    df["high_rpm_flag"] = (rpm >= 30).astype(int)
    df["vpn_or_proxy"] = ((is_vpn == 1) | (is_proxy == 1)).astype(int)

    # UA vs explicit browser/os mismatch heuristic
    ua_ver = df.get("ua_browser_version", "").astype(str).str.split(".").str[0]
    bver = df.get("browser_version", "").astype(str).str.split(".").str[0]
    df["ua_os_mismatch"] = ((ua_ver != bver) & (bver!="") & (ua_ver!="")).astype(int)

    # Simple numeric risk score (0..1-ish)
    # weights: rpm (0.4), vpn/proxy (0.4), mismatch (0.2)
    score = 0.4*(rpm/60.0).clip(0,1) + 0.4*df["vpn_or_proxy"] + 0.2*df["ua_os_mismatch"]
    df["rba_risk_score"] = score.clip(0,1)

    return df

def build_features_with_rba(df: pd.DataFrame) -> pd.DataFrame:
    """Build base device features + RBA enrichment."""
    df2 = build_features(df)
    df2 = enrich_with_rba_signals(df2)
    return df2

def select_feature_cols_with_rba(df: pd.DataFrame) -> List[str]:
    """Select base + RBA features for modeling."""
    base = select_feature_cols(df)
    extra = ["high_rpm_flag","vpn_or_proxy","ua_os_mismatch","rba_risk_score"]
    return [c for c in base + extra if c in df.columns]
