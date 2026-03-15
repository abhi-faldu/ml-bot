"""
src/data/pipeline.py

Temporal-safe data pipeline.

Key guarantee: ALL transforms are fit ONLY on training data.
No future information ever leaks into the training window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.make_features import (
    add_rsi, add_macd, add_atr, add_volume_ratio,
    add_bollinger, add_ema_features, add_candle_features,
    get_feature_cols,
)
from src.utils import add_lag_features

RAW_DIR  = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_raw(csv_name: str) -> pd.DataFrame:
    """Load raw OHLCV, parse timestamps, sort ascending."""
    df = pd.read_csv(RAW_DIR / csv_name, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    assert df.index.is_monotonic_increasing, "Index is not sorted — data integrity issue"
    return df


def build_features(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Build full feature matrix from raw OHLCV dataframe.

    Features: lagged returns + RSI-14 + MACD + ATR-14 + ATR% + volume ratio
              + Bollinger %B/width + EMA deviations + candle body% + target.
    All indicators use only past data — zero look-ahead bias.
    Splitting is handled separately by split_temporal().

    Label: 1 if close is higher 4 candles from now (4h horizon).
    A 4h label is substantially less noisy than a single-candle label
    because small random fluctuations average out over a longer window.
    """
    df = df.copy()
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
    return df


def split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe preserving time order — NEVER shuffle.

    Shuffling on time series causes look-ahead bias because test
    samples end up sandwiched between training samples.
    """
    n = len(df)
    split_idx  = int(n * (1 - test_ratio))
    split_date = df.index[split_idx]

    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    print(f"Train: {len(train)} rows  [{train.index[0].date()} -> {train.index[-1].date()}]")
    print(f"Test : {len(test)} rows   [{test.index[0].date()} -> {test.index[-1].date()}]")
    print(f"Split date: {split_date.date()}")
    return train, test


def build_and_split(
    csv_name: str,
    lookback: int = 10,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load -> build features -> split temporally.
    Single entry point used by train_model.py and walk_forward.py.
    """
    df_raw  = load_raw(csv_name)
    df_feat = build_features(df_raw, lookback)
    train, test = split_temporal(df_feat, test_ratio)
    df_feat.to_csv(PROC_DIR / f"features_{csv_name}")
    return train, test


if __name__ == "__main__":
    train, test = build_and_split("binance_BTCUSDT_1h.csv")
    print(f"\nFeature columns: {get_feature_cols(train)}")
    print(f"Target distribution (train):\n{train['target'].value_counts()}")
