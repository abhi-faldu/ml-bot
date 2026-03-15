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
ATR-14              : atr_14                  (volatility absolute)
ATR %               : atr_pct                 (volatility relative to price)
Volume ratio        : vol_ratio               (volume vs 20-period avg)
Bollinger %B        : bb_pct                  (position within bands)
Bollinger width     : bb_width                (band width / MA — regime)
EMA-20 deviation    : ema_20_dev              (close / EMA20 - 1)
EMA-50 deviation    : ema_50_dev              (close / EMA50 - 1)
Candle body %       : body_pct                ((close-open)/(high-low))

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


def add_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    n_std: float = 2.0,
    col: str = "close",
) -> pd.DataFrame:
    """
    Bollinger Bands — two derived features.

    bb_pct   : %B = (close - lower) / (upper - lower).
               0 = at lower band, 1 = at upper band; can exceed [0, 1].
               Captures mean-reversion potential.
    bb_width : (upper - lower) / MA — normalised band width.
               High values = volatile regime, low values = compression.
    """
    ma  = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    band_range = (upper - lower).replace(0, np.nan)
    df["bb_pct"]   = (df[col] - lower) / band_range
    df["bb_width"] = band_range / ma.replace(0, np.nan)
    return df


def add_ema_features(
    df: pd.DataFrame,
    spans: tuple = (20, 50),
    col: str = "close",
) -> pd.DataFrame:
    """
    Price deviation from exponential moving averages.

    ema_20_dev : close / EMA(20) - 1  (short-term trend position)
    ema_50_dev : close / EMA(50) - 1  (medium-term trend position)
    Positive = price above EMA (bullish), negative = below (bearish).
    """
    for span in spans:
        ema = df[col].ewm(span=span, adjust=False).mean()
        df[f"ema_{span}_dev"] = df[col] / ema.replace(0, np.nan) - 1
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle structure and normalised volatility features.

    body_pct : (close - open) / (high - low).
               +1 = full bullish candle, -1 = full bearish candle.
               Near 0 = indecision / doji.
    atr_pct  : atr_14 / close — ATR as a fraction of price.
               Normalises volatility across different price levels.
               Requires add_atr() to have been called first.
    """
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / hl_range
    df["atr_pct"]  = df["atr_14"] / df["close"].replace(0, np.nan)
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
    df = add_bollinger(df)
    df = add_ema_features(df)
    df = add_candle_features(df)   # must follow add_atr

    # 4h forward label — longer horizon reduces single-candle noise
    df["target"] = (df["close"].shift(-4) > df["close"]).astype(int)
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
    tech_cols = [c for c in [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "atr_14", "atr_pct", "vol_ratio",
        "bb_pct", "bb_width",
        "ema_20_dev", "ema_50_dev",
        "body_pct",
    ] if c in df.columns]
    return lag_cols + tech_cols


if __name__ == "__main__":
    df = make_features("binance_BTCUSDT_1h.csv")
    print(f"\nFeatures: {get_feature_cols(df)}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
# macd added Feb 13
# atr and vol_ratio added Feb 14
