"""
dashboard/pages/1_Overview.py

Overview page — OOS equity curve and key performance metrics.

Metrics displayed
-----------------
- OOS accuracy      : classification accuracy on the held-out test set
- Sharpe ratio      : annualised Sharpe (hourly bars × √8760)
- Max drawdown      : largest peak-to-trough equity decline in the OOS window
- Total trades      : number of position-change events (entries + exits)
- Strategy return   : net P&L over the OOS period
- Buy-and-hold      : passive long benchmark over the same window
- Win rate          : % of held bars with a positive net return
"""
from pathlib import Path
import sys

import numpy as np
import streamlit as st

# Ensure project root and dashboard/ are importable from any working directory
ROOT     = Path(__file__).resolve().parents[2]
DASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DASH_DIR))

from utils.load_data import load_backtest, load_oos_metrics
from utils.charts import equity_curve_chart, drawdown_chart

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Overview — ML Crypto",
    page_icon="📊",
    layout="wide",
)

st.title("Overview")
st.caption("Out-of-sample backtest · last 20 % of data · BTC/USDT 1h")

# ── Load data ─────────────────────────────────────────────────────────────────

df  = load_backtest()
oos = load_oos_metrics()

if "strat_ret" not in df.columns:
    st.warning(
        "The backtest generated no trades with the current confidence threshold. "
        "Try lowering `confidence_threshold` in `src/backtest/backtest.py`.",
        icon="⚠️",
    )
    st.stop()

# ── Derived metrics ───────────────────────────────────────────────────────────

total_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100
bh_ret    = (df["bh_equity"].iloc[-1] - 1) * 100

std = df["strat_ret"].std()
ann_sharpe = (df["strat_ret"].mean() / std) * np.sqrt(8760) if std > 0 else 0.0

roll_max = df["equity"].cummax()
max_dd   = ((df["equity"] - roll_max) / roll_max).min() * 100

n_trades = int(df["trade"].sum())

held_long = df["signal"].shift(1).fillna(0) == 1
win_rate  = (
    (df.loc[held_long, "strat_ret"] > 0).mean() * 100
    if held_long.sum() > 0
    else 0.0
)

oos_acc_pct = oos["accuracy"] * 100

# ── Primary metric cards ──────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "OOS Accuracy",
    f"{oos_acc_pct:.1f}%",
    help="Classification accuracy on the held-out test set (last 20 % of data).",
)
c2.metric(
    "Sharpe Ratio (ann.)",
    f"{ann_sharpe:.2f}",
    help="Annualised Sharpe ratio — hourly bar returns × √8760.",
)
c3.metric(
    "Max Drawdown",
    f"{max_dd:.1f}%",
    help="Largest peak-to-trough equity decline in the OOS window.",
)
c4.metric(
    "Total Trades",
    str(n_trades),
    help="Number of position-change events (entries + exits combined).",
)

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────────────

st.subheader("Equity Curve")
st.plotly_chart(equity_curve_chart(df), width="stretch")

# Secondary metrics below the chart
ca, cb, cc = st.columns(3)
ca.metric("Strategy Return (OOS)", f"{total_ret:+.1f}%")
cb.metric("Buy & Hold Return (OOS)", f"{bh_ret:+.1f}%")
cc.metric("Win Rate (held bars)", f"{win_rate:.1f}%")

st.divider()

# ── Drawdown chart ────────────────────────────────────────────────────────────

st.subheader("Drawdown")
st.plotly_chart(drawdown_chart(df), width="stretch")

# ── Raw backtest data ─────────────────────────────────────────────────────────

with st.expander("Raw backtest data (last 200 rows)"):
    display_cols = ["proba", "signal", "price", "ret", "strat_ret", "equity_norm", "bh_equity"]
    available    = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].tail(200), width="stretch")
