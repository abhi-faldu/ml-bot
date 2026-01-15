from pathlib import Path

import pandas as pd

from src.utils import add_lag_features

# go up from src/data/ to project root
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def make_features(csv_name: str, lookback: int = 10) -> pd.DataFrame:
    """Build lagged-return feature matrix and binary direction target."""
    df = pd.read_csv(RAW_DIR / csv_name, index_col=0, parse_dates=True)
    df["return"] = df["close"].pct_change()
    df = add_lag_features(df, "return", lookback)
    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns if c.startswith("ret_lag_")]
    out = df[feature_cols + ["target"]]
    out_path = PROC_DIR / f"features_{csv_name}"
    out.to_csv(out_path)
    print(f"features saved → {out_path}  ({len(out)} rows)")
    return out


if __name__ == "__main__":
    make_features("binance_BTCUSDT_1h.csv")
