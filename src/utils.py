import pandas as pd


def add_lag_features(df: pd.DataFrame, col: str, lookback: int) -> pd.DataFrame:
    """Add lagged versions of `col` as new columns ret_lag_1 … ret_lag_N."""
    for i in range(1, lookback + 1):
        df[f"ret_lag_{i}"] = df[col].shift(i)
    return df
