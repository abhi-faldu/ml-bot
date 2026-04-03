"""
dashboard/app.py

Entry point for the ML Crypto Trading Streamlit dashboard.

Usage
-----
    streamlit run dashboard/app.py

    (run from the project root so that src/ and config.py are importable)
"""
import streamlit as st

st.set_page_config(
    page_title="ML Crypto Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ML Crypto Trading Dashboard")
st.caption("BTC/USDT · 1h · Random Forest · Binance Testnet")

# ── Project summary ───────────────────────────────────────────────────────────

st.markdown("""
## Research Summary

End-to-end ML directional trading system for BTC/USDT built to test whether
standard OHLCV-derived features carry a predictive edge at the 1-hour timeframe.
""")

col_model, col_results = st.columns(2)

with col_model:
    st.markdown("""
**Model details**

| Attribute | Value |
|:---|:---|
| Classifier | Random Forest |
| Trees | 300 |
| Max depth | 6 |
| Min samples / leaf | 50 |
| Label | 4-hour forward direction |
| Features | 22 (technical indicators + lags) |
| Validation | 5-fold walk-forward |
""")

with col_results:
    st.markdown("""
**Results at a glance**

| Metric | Value |
|:---|:---|
| Walk-forward OOS accuracy | ~50% |
| Annualised Sharpe ratio | near 0 |
| Max drawdown | see Overview page |
| Live trading | Binance Testnet only |
| Confidence threshold | P(up) > 0.55 |
""")

st.divider()

# ── Navigation ────────────────────────────────────────────────────────────────

st.markdown("""
## Navigation

Use the sidebar to explore the three analysis pages:

- **1 Overview** — OOS equity curve, Sharpe ratio, max drawdown, total trades
- **2 Model Analysis** — feature importances, walk-forward fold table, confusion matrix
- **3 Risk Management** — ATR stop-loss visualisation, drawdown chart, risk parameters
""")

st.divider()

# ── Key finding ───────────────────────────────────────────────────────────────

st.markdown("""
## Key Finding

Public OHLCV features carry **no detectable predictive edge** at the 1-hour timeframe
after fees. The 50 % accuracy result is an empirical finding about market efficiency —
not a modelling failure. BTC/USDT is one of the most heavily arbitraged markets in
the world; any signal in these features is priced away almost instantly.

Recommended next steps (see `FINDINGS.md`):

- Alternative data — funding rates, liquidation heatmaps, order-book imbalance
- Longer timeframes — daily / weekly bars carry more signal
- Regime-conditional models — separate bull / bear / sideways regimes
- Cross-asset signals — DXY, equity futures, BTC dominance
""")

st.info(
    "**Testnet only** — all live trades execute on Binance Testnet. "
    "No real funds are at risk.",
    icon="ℹ️",
)
