"""
dashboard/utils/load_model.py

Cached model-loading helper.
Tries the dashboard alias (lgbm_model.pkl) first, then falls back to the
original trained artefact so the dashboard works before a re-train.
"""
from pathlib import Path
import sys

import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_PRIMARY  = ROOT / "models" / "lgbm_model.pkl"
_FALLBACK = ROOT / "models" / "rf_crypto_next_candle.pkl"


@st.cache_resource(show_spinner="Loading model…")
def load_model() -> RandomForestClassifier:
    """
    Load the trained classifier from disk.

    Tries lgbm_model.pkl first (written by the updated train_model.py),
    then falls back to rf_crypto_next_candle.pkl for backwards compatibility.
    Raises FileNotFoundError if neither is present — run train_model.py first.
    """
    path = _PRIMARY if _PRIMARY.exists() else _FALLBACK
    if not path.exists():
        raise FileNotFoundError(
            "No trained model found in models/. "
            "Run: python src/models/train_model.py"
        )
    return joblib.load(path)
