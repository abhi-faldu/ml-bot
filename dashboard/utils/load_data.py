"""
dashboard/utils/load_data.py

Cached data-loading helpers shared across all dashboard pages.

All heavy I/O and model inference is wrapped in @st.cache_data / @st.cache_resource
so that Streamlit re-runs do not re-trigger the backtest or walk-forward validation.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List

import joblib
import pandas as pd
import streamlit as st

ROOT     = Path(__file__).resolve().parents[2]
DASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH_DIR))

_FEATURES_FILE = "features_binance_BTCUSDT_1h.csv"
_RAW_FILE      = "binance_BTCUSDT_1h.csv"


# ── Raw data ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading feature matrix…")
def load_features() -> pd.DataFrame:
    """Return the pre-built feature matrix from data/processed/."""
    path = ROOT / "data" / "processed" / _FEATURES_FILE
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data(show_spinner="Loading raw OHLCV data…")
def load_raw_ohlcv() -> pd.DataFrame:
    """Return the raw OHLCV price data from data/raw/."""
    path = ROOT / "data" / "raw" / _RAW_FILE
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ── Backtest ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running OOS backtest…")
def load_backtest() -> pd.DataFrame:
    """
    Run the vectorised OOS backtest and return the annotated DataFrame.
    Cached after the first run so page re-renders are instant.
    """
    from src.backtest.backtest import backtest
    return backtest(_FEATURES_FILE, _RAW_FILE)


# ── OOS classification metrics ────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing OOS classification metrics…")
def load_oos_metrics() -> Dict[str, Any]:
    """
    Run the trained model on the held-out test set (last 20 %) and return
    accuracy, confusion matrix, and feature importances.

    The model is loaded directly via joblib here (rather than calling the
    cached load_model()) to avoid nesting st.cache_resource inside
    st.cache_data, which can cause serialisation issues in some Streamlit
    versions.

    Returns
    -------
    dict with keys:
        accuracy         : float  — OOS accuracy (0–1)
        confusion_matrix : list[list[int]]  — [[TN, FP], [FN, TP]]
        feature_names    : list[str]
        importances      : list[float]  — aligned with feature_names
    """
    from sklearn.metrics import accuracy_score, confusion_matrix
    from src.data.pipeline import split_temporal
    from src.data.make_features import get_feature_cols

    # Load model — prefer dashboard alias, fall back to original artefact
    primary  = ROOT / "models" / "lgbm_model.pkl"
    fallback = ROOT / "models" / "rf_crypto_next_candle.pkl"
    path = primary if primary.exists() else fallback
    model = joblib.load(path)

    # Same 80/20 temporal split used by train_model.py and backtest.py
    df_feat = load_features()
    _, df_test = split_temporal(df_feat)

    feature_cols = get_feature_cols(df_test)
    preds = model.predict(df_test[feature_cols])
    acc   = accuracy_score(df_test["target"], preds)
    cm    = confusion_matrix(df_test["target"], preds)

    return {
        "accuracy":         float(acc),
        "confusion_matrix": cm.tolist(),
        "feature_names":    feature_cols,
        "importances":      model.feature_importances_.tolist(),
    }


# ── Walk-forward results ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running walk-forward validation (this may take ~30 s)…")
def load_walk_forward() -> Dict[str, Any]:
    """
    Run 5-fold walk-forward validation and return the aggregated result dict.

    This trains 5 separate models; the first call takes ~30 seconds.
    Subsequent re-runs are served from the Streamlit cache.
    """
    from src.models.walk_forward import walk_forward_validate
    return walk_forward_validate(_RAW_FILE)
