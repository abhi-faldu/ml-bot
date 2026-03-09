"""
src/data/make_features.py

Feature engineering for the ML trading model.

Features built
--------------
Lagged returns      : ret_lag_1 … ret_lag_N   (price momentum)
RSI-14              : rsi_14                  (overbought/oversold)
MACD line           : macd                    (trend direction)
MACD signal line    : macd_signal             (trend confirmation)
MACD histogram      : macd_hist               (momentum change)
ATR-14              : atr_14                  (volatility)
Volume ratio        : vol_ratio               (volume vs 20-period avg)

All indicators are implemented in pure pandas/numpy — no extra libraries
needed. Every indicator uses only past data so there is zero look-ahead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import add_lag_features

RAW_DIR  = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ── Individual indicators ─────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    """
    Relative Strength Index.
    RSI > 70 → overbought, RSI < 30 → oversold.
    Uses Wilder's smoothing (exponential with alpha = 1/period).
    """
    delta = df[col].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    df[f"rsi_{period}"] = rsi
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.
    macd       : fast EMA - slow EMA  (trend direction)
    macd_signal: 9-period EMA of macd (trigger line)
    macd_hist  : macd - macd_signal   (momentum strength)
    """
    ema_fast    = df[col].ewm(span=fast,   adjust=False).mean()
    ema_slow    = df[col].ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df["macd"]        = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"]   = macd_line - signal_line
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range — measures market volatility.
    High ATR = volatile, low ATR = quiet.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df[f"atr_{period}"] = tr.rolling(period).mean()
    return df


def add_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Volume ratio = current volume / rolling mean volume.
    > 1.0 = above-average activity, < 1.0 = quiet.
    """
    vol_ma = df["volume"].rolling(period).mean()
    df["vol_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)
    return df


# ── Full feature pipeline ─────────────────────────────────────────────────────

def make_features(csv_name: str, lookback: int = 10) -> pd.DataFrame:
    """
    Load raw OHLCV and build the full feature matrix.
    Returns dataframe with all features + 'target' column.
    """
    df = pd.read_csv(RAW_DIR / csv_name, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    df["return"] = df["close"].pct_change()
    df = add_lag_features(df, "return", lookback)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_volume_ratio(df)

    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    feature_cols = get_feature_cols(df)
    out = df[feature_cols + ["target"]]

    out_path = PROC_DIR / f"features_{csv_name}"
    out.to_csv(out_path)
    print(f"features saved → {out_path}  ({len(out)} rows, {len(feature_cols)} features)")
    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature column names in consistent order."""
    lag_cols  = sorted([c for c in df.columns if c.startswith("ret_lag_")])
    tech_cols = [c for c in ["rsi_14", "macd", "macd_signal", "macd_hist", "atr_14", "vol_ratio"]
                 if c in df.columns]
    return lag_cols + tech_cols


if __name__ == "__main__":
    df = make_features("binance_BTCUSDT_1h.csv")
    print(f"\nFeatures: {get_feature_cols(df)}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
# macd added Feb 13
# atr and vol_ratio added Feb 14
