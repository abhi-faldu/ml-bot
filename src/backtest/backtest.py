"""
src/backtest/backtest.py

Vectorised OOS backtest for the LightGBM crypto direction classifier.

Design choices
--------------
* Only the last 20 % of the feature matrix is evaluated — training rows
  are never touched to avoid look-ahead bias.
* Predictions are gated by a confidence threshold so low-conviction bars
  are left as FLAT rather than traded.
* All 16 engineered features are passed to the model via get_feature_cols()
  instead of only the 10 lagged returns.
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display window
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd

from config import FEE, INITIAL_CAPITAL, MODEL_NAME
from src.data.make_features import get_feature_cols
from src.data.pipeline import split_temporal

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"


def backtest(
    features_file: str,
    raw_file: str,
    fee: float = FEE,
    initial_capital: float = INITIAL_CAPITAL,
    confidence_threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Run a vectorised backtest on the true out-of-sample portion of data.

    Loads the pre-built feature matrix and raw OHLCV prices, restricts
    evaluation to the last 20 % of rows via a strict temporal split, then
    generates signals using the trained model's predicted probabilities.
    Only bars where P(up) > ``confidence_threshold`` are treated as BUY;
    all other bars are FLAT (no position).

    Parameters
    ----------
    features_file : str
        Filename inside data/processed/ (e.g. ``features_binance_BTCUSDT_1h.csv``).
    raw_file : str
        Filename inside data/raw/ for raw OHLCV close prices.
    fee : float
        One-way taker fee applied on every position change (entry or exit).
    initial_capital : float
        Starting capital in USD used to scale the ``equity`` column.
    confidence_threshold : float
        Minimum predicted probability of class 1 to generate a BUY signal.
        Lowering this increases trade frequency; raising it increases selectivity.

    Returns
    -------
    pd.DataFrame
        Test-period dataframe with the following columns appended:
        signal, proba, price, ret, trade, strat_ret, equity, equity_norm,
        bh_equity.

    Raises
    ------
    FileNotFoundError
        If any of the required files (features, raw prices, model) are absent.
    """
    feat_path = PROC_DIR / features_file
    raw_path = RAW_DIR / raw_file
    model_path = MODEL_DIR / MODEL_NAME

    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw price file not found: {raw_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    df_feat = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df_price = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    model = joblib.load(model_path)

    # Strict temporal split — never evaluate on training rows
    _, df_test = split_temporal(df_feat)

    # Use all engineered features, not just lagged returns
    feature_cols = get_feature_cols(df_test)
    proba = model.predict_proba(df_test[feature_cols])[:, 1]
    df_test["proba"] = proba
    df_test["signal"] = (proba > confidence_threshold).astype(int)

    n_transitions = int(df_test["signal"].diff().abs().fillna(0).sum())
    if n_transitions == 0:
        print(
            f"WARNING: No trades generated with confidence_threshold="
            f"{confidence_threshold:.2f}. "
            "Try lowering the threshold."
        )
        return df_test

    df_test["price"] = df_price.loc[df_test.index, "close"]
    df_test["ret"] = df_test["price"].pct_change().fillna(0)
    df_test["trade"] = df_test["signal"].diff().abs().fillna(0)
    df_test["strat_ret"] = (
        df_test["signal"].shift(1).fillna(0) * df_test["ret"]
        - fee * df_test["trade"]
    )

    # Equity in USD and normalised (starts at 1.0) for the plot
    df_test["equity"] = (1 + df_test["strat_ret"]).cumprod() * initial_capital
    df_test["equity_norm"] = (1 + df_test["strat_ret"]).cumprod()

    # Passive buy-and-hold benchmark, normalised to 1.0 at test start
    df_test["bh_equity"] = df_test["price"] / df_test["price"].iloc[0]

    return df_test


def summary(df: pd.DataFrame) -> None:
    """
    Print a detailed performance summary for the backtested period.

    Metrics
    -------
    Total return     : net strategy P&L over the test window (%)
    Buy-and-hold     : passive long return over the same window (%)
    Sharpe (ann)     : annualised Sharpe ratio (hourly bars * sqrt(8760))
    Max drawdown     : largest peak-to-trough equity decline (%)
    Win rate         : % of active-position bars with a positive net return
    Trades           : total number of position-change events
    Fees paid        : total cost of all transitions as % of starting capital
    Avg holding (h)  : mean number of hours per long position

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`backtest`. Must contain strat_ret,
        equity, bh_equity, signal, and trade columns.
    """
    if "strat_ret" not in df.columns:
        print("No trades were generated — nothing to summarise.")
        return

    # ── Returns ──────────────────────────────────────────────────────────────
    total_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100
    bh_ret = (df["bh_equity"].iloc[-1] - 1) * 100

    # ── Risk metrics ─────────────────────────────────────────────────────────
    std = df["strat_ret"].std()
    ann_sharpe = (df["strat_ret"].mean() / std) * np.sqrt(8760) if std > 0 else 0.0

    roll_max = df["equity"].cummax()
    max_dd = ((df["equity"] - roll_max) / roll_max).min() * 100

    # ── Trade statistics ──────────────────────────────────────────────────────
    n_trades = int(df["trade"].sum())
    fees_pct = n_trades * FEE * 100

    # Win rate: among bars where we held a long position, % with positive return
    held_long = df["signal"].shift(1).fillna(0) == 1
    if held_long.sum() > 0:
        win_rate = (df.loc[held_long, "strat_ret"] > 0).mean() * 100
    else:
        win_rate = 0.0

    # Average holding period: total long bars / number of entries
    entries = int((df["signal"].diff() == 1).sum())
    bars_long = int((df["signal"] == 1).sum())
    avg_hold = bars_long / entries if entries > 0 else 0.0

    print(f"Total return     : {total_ret:+.1f}%")
    print(f"Buy-and-hold     : {bh_ret:+.1f}%")
    print(f"Sharpe (ann)     : {ann_sharpe:.2f}")
    print(f"Max drawdown     : {max_dd:.1f}%")
    print(f"Win rate         : {win_rate:.1f}%")
    print(f"Trades           : {n_trades}")
    print(f"Fees paid        : {fees_pct:.2f}%")
    print(f"Avg holding (h)  : {avg_hold:.1f}")


def plot_equity(df: pd.DataFrame) -> None:
    """
    Save a normalised equity-curve chart (strategy vs buy-and-hold) to disk.

    Both curves are normalised to 1.0 at the start of the test period.
    A dashed horizontal line marks the break-even level (y = 1.0).
    The chart is saved to ``<project_root>/backtest_equity.png`` and
    the figure is closed afterwards — ``plt.show()`` is never called.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by :func:`backtest`. Must contain equity_norm
        and bh_equity columns.
    """
    if "equity_norm" not in df.columns:
        print("No equity data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df.index, df["equity_norm"], label="Strategy", color="steelblue", linewidth=1.2)
    ax.plot(df.index, df["bh_equity"], label="Buy & Hold", color="darkorange",
            linewidth=1.0, alpha=0.8)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Break-even")

    ax.set_title("OOS Equity Curve (normalised to 1.0)")
    ax.set_ylabel("Equity (normalised)")
    ax.legend()
    fig.autofmt_xdate()

    output_path = ROOT / "backtest_equity.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved → {output_path}")


if __name__ == "__main__":
    df = backtest("features_binance_BTCUSDT_1h.csv", "binance_BTCUSDT_1h.csv")

    if "strat_ret" not in df.columns:
        # backtest() already printed the no-trades warning
        sys.exit(0)

    summary(df)
    plot_equity(df)
