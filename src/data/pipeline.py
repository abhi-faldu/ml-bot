"""
src/data/pipeline.py

Temporal-safe data pipeline.

Key guarantee: ALL transforms (scaling, encoding) are fit ONLY on
training data and then applied to test data. No future information
ever leaks into the training window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import add_lag_features

RAW_DIR  = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_raw(csv_name: str) -> pd.DataFrame:
    """Load raw OHLCV, parse timestamps, sort ascending."""
    df = pd.read_csv(RAW_DIR / csv_name, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)          # enforce chronological order
    assert df.index.is_monotonic_increasing, "Index is not sorted — data integrity issue"
    return df


def build_features(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Add lagged return features and binary target.

    IMPORTANT: target uses shift(-1) which looks one step ahead.
    This is intentional and correct — we are predicting the NEXT candle.
    The leakage risk is in how the dataset is SPLIT, not in the target itself.
    This function only builds the matrix; splitting is handled by split_temporal().
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df = add_lag_features(df, "return", lookback)

    # target: 1 if next candle closes higher, else 0
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    # drop rows with any NaN — caused by lagging and the final row (no future target)
    df.dropna(inplace=True)
    return df


def split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train / test preserving time order.

    NEVER shuffle — shuffling on time series causes severe look-ahead bias
    because test samples end up sandwiched between training samples.

    Parameters
    ----------
    df         : feature matrix with 'target' column
    test_ratio : fraction of MOST RECENT rows reserved for test

    Returns
    -------
    (train_df, test_df) — both contain features + target column
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    split_date = df.index[split_idx]

    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    print(f"Train: {len(train)} rows  [{train.index[0].date()} → {train.index[-1].date()}]")
    print(f"Test : {len(test)} rows   [{test.index[0].date()} → {test.index[-1].date()}]")
    print(f"Split date: {split_date.date()}")

    return train, test


def build_and_split(
    csv_name: str,
    lookback: int = 10,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load → build features → split temporally.
    Single entry point used by train_model.py and walk_forward.py.
    """
    df_raw  = load_raw(csv_name)
    df_feat = build_features(df_raw, lookback)
    train, test = split_temporal(df_feat, test_ratio)

    # save full feature matrix for backtest use
    df_feat.to_csv(PROC_DIR / f"features_{csv_name}")
    return train, test


if __name__ == "__main__":
    train, test = build_and_split("binance_BTCUSDT_1h.csv")
    print(f"\nFeature columns: {[c for c in train.columns if c != 'target']}")
    print(f"Target distribution (train):\n{train['target'].value_counts()}")
