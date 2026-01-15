from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FEE, INITIAL_CAPITAL, MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"


def backtest(
    features_file: str,
    raw_file: str,
    fee: float = FEE,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Vectorised backtest: signal=1 → long, signal=0 → flat."""
    df_feat = pd.read_csv(PROC_DIR / features_file, index_col=0, parse_dates=True)
    df_price = pd.read_csv(RAW_DIR / raw_file, index_col=0, parse_dates=True)
    model = joblib.load(MODEL_DIR / MODEL_NAME)

    feature_cols = [c for c in df_feat.columns if c.startswith("ret_lag_")]
    df_feat["signal"] = model.predict(df_feat[feature_cols])
    df_feat["price"] = df_price.loc[df_feat.index, "close"]
    df_feat["ret"] = df_feat["price"].pct_change().fillna(0)
    df_feat["trade"] = df_feat["signal"].diff().abs().fillna(0)
    df_feat["strat_ret"] = (
        df_feat["signal"].shift(1).fillna(0) * df_feat["ret"]
        - fee * df_feat["trade"]
    )
    df_feat["equity"] = (1 + df_feat["strat_ret"]).cumprod() * initial_capital
    return df_feat


def summary(df: pd.DataFrame) -> None:
    total_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100
    n_trades = int(df["trade"].sum())
    ann_sharpe = (df["strat_ret"].mean() / df["strat_ret"].std()) * np.sqrt(8760)
    print(f"Total return : {total_ret:+.1f}%")
    print(f"Trades       : {n_trades}")
    print(f"Sharpe (ann) : {ann_sharpe:.2f}")


if __name__ == "__main__":
    df = backtest("features_binance_BTCUSDT_1h.csv", "binance_BTCUSDT_1h.csv")
    summary(df)
    df[["equity"]].plot(title="Equity curve", ylabel="USD")
