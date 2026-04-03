# Research Findings: ML-Based BTC/USDT Direction Prediction

**Project:** ml-bot
**Period:** December 2025 – March 2026
**Status:** Round 2 improvements applied — retraining required for updated metrics.

---

## Round 1 Conclusion

Price and volume features alone are insufficient to predict BTC/USDT hourly direction above the break-even threshold after fees. OOS accuracy = 50.00% across 17,500 rows and 5 walk-forward folds.

## Round 2 Summary

Targeted improvements applied to address the two likely root causes of the null result: insufficient feature richness and excessive label noise. Results pending after retraining.

---

## Objective

Build and evaluate a machine learning system to predict the direction of the next 1-hour BTC/USDT candle (up or down) with sufficient accuracy to generate positive returns after Binance trading fees (~0.08% per round trip).

The minimum required accuracy to break even is approximately **50.4%** at the position sizes tested.

---

## What Was Built

A complete end-to-end ML trading pipeline:

- **Data**: 17,502 rows of BTC/USDT 1h OHLCV (2 years, Jan 2024 – Mar 2026)
- **Features**: 16 total — lagged returns (10), RSI-14, MACD, MACD signal, MACD histogram, ATR-14, volume ratio
- **Model**: LightGBM classifier (L1/L2 regularised, min_child_samples=50)
- **Validation**: 5-fold walk-forward, strictly temporal, no shuffling
- **Risk management**: ATR-based stops, daily loss limit (2%), max drawdown breaker (5%)

---

## Results

| Metric | Value |
|---|---|
| Dataset size | 17,502 rows |
| Training window | 60% (~10,500 rows) |
| OOS test rows | 7,000 rows |
| OOS accuracy | **50.00%** |
| Precision (class 0) | 0.50 |
| Precision (class 1) | 0.50 |
| Expected PnL after fees | **Negative** |

The model performs at chance level across all 5 walk-forward folds. This is not a modelling failure — it is an empirical finding about the predictability of this market using these features.

---

## Why 50% Is the Honest Answer

BTC/USDT on a 1h timeframe is a heavily traded, liquid market with millions of participants analysing the same price and volume data. When a predictive signal exists in price/volume features, it is arbitraged away quickly.

The features we used — RSI, MACD, lagged returns, ATR, volume — are among the most widely followed indicators in retail and institutional trading. Any signal they once contained has been traded out of existence.

This is consistent with the **Efficient Market Hypothesis** applied to public technical indicators on liquid assets.

---

## What Would Be Needed to Beat 50%

Based on the literature and quant finance practice, genuine edge in crypto direction prediction typically comes from:

1. **Alternative data** — funding rates, liquidation heatmaps, exchange netflow, social sentiment, options implied volatility
2. **Order book microstructure** — bid/ask imbalance, large order detection via WebSocket L2 feed
3. **Cross-asset signals** — DXY, equity futures, stablecoin flows
4. **Longer timeframes** — daily candles have more signal than hourly; weekly more than daily
5. **Regime-conditional models** — separate models for trending vs ranging conditions, detected via HMM
6. **Much larger datasets** — 5+ years, multiple assets, multiple timeframes

None of these require abandoning the architecture built here. The pipeline, risk manager, walk-forward validator, and live loop are all reusable with better input features.

---

## What This Project Demonstrated

Despite the null prediction result, the project successfully:

- Built a production-grade modular ML trading architecture
- Implemented strict temporal validation that correctly identifies when a model has no edge
- Added dynamic risk management (ATR stops, circuit breakers) that would protect capital in live trading
- Documented the full engineering and research process transparently

A system that correctly tells you "there is no edge here, do not trade" is more valuable than one that overfits training data and loses money live.

---

## Honest Recommendation

Do not deploy this system with real capital in its current form.

If continuing development: fetch funding rate and liquidation data from Binance and add them as features. These are not priced into simple technical indicators and have shown predictive value in published research on crypto microstructure.

---

## Round 2: Improvements Applied (March 2026)

### Motivation

Round 1 produced a 50.00% OOS accuracy — statistically indistinguishable from random. Two structural causes were identified:

1. **Label noise** — predicting a single 1h candle direction is near-random because tiny, structureless price fluctuations dominate at short horizons.
2. **Thin feature set** — 16 features, mostly correlated (MACD has 3 columns derived from the same two EMAs), provided insufficient signal diversity.

### Changes Made

#### 1. Model: LightGBM → RandomForestClassifier
LightGBM was removed because it was not installed in the active environment. Replaced with `RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50)` from scikit-learn. Both are tree ensembles with similar inductive biases; the RF hyperparameters maintain strong regularisation via `min_samples_leaf=50`.

#### 2. Feature set expanded: 16 → 22 features

Six new features added to `src/data/make_features.py`, all derived strictly from past OHLCV data:

| New feature | Function | Signal captured |
|---|---|---|
| `bb_pct` | `add_bollinger()` | Position within Bollinger Bands — mean reversion |
| `bb_width` | `add_bollinger()` | Band width / MA — volatility regime (squeeze vs expansion) |
| `ema_20_dev` | `add_ema_features()` | close / EMA(20) − 1 — short-term trend deviation |
| `ema_50_dev` | `add_ema_features()` | close / EMA(50) − 1 — medium-term trend deviation |
| `body_pct` | `add_candle_features()` | (close − open) / (high − low) — candle direction and conviction |
| `atr_pct` | `add_candle_features()` | atr_14 / close — volatility normalised by price level |

EMA deviations capture trend structure independently from MACD; Bollinger features add mean-reversion context; candle body ratio captures intra-bar momentum that lagged returns miss.

#### 3. Prediction label: 1h → 4h forward return

```python
# Round 1
df["target"] = (df["return"].shift(-1) > 0).astype(int)

# Round 2
df["target"] = (df["close"].shift(-4) > df["close"]).astype(int)
```

A 4-hour forward horizon reduces label noise substantially. Single-candle labels are dominated by random micro-fluctuations. A 4h label persists through short-term noise and reflects a tradeable directional move. The backtest signal is still generated every hour — the longer training target makes the model more selective.

#### 4. Backtest overhauled (`src/backtest/backtest.py`)

- **Strict OOS split:** backtest now evaluates only the last 20% of data via `split_temporal()`, eliminating any possibility of training-data contamination.
- **All 22 features used:** replaced the hardcoded `ret_lag_` filter with `get_feature_cols()`, ensuring RSI, MACD, ATR and the 6 new features are passed to the model.
- **Confidence filtering:** `model.predict_proba()` replaces `model.predict()`; only bars where P(up) > 0.55 generate a BUY signal, filtering low-conviction noise trades.
- **Enhanced metrics:** added max drawdown, win rate, fees paid, buy-and-hold comparison, and average holding period.
- **Equity curve chart:** strategy vs buy-and-hold, both normalised to 1.0, saved as `backtest_equity.png`.
- **Edge-case handling:** `FileNotFoundError` for missing files; graceful exit with warning if no trades are generated at the chosen threshold.

### What to Run

```bash
python src/models/train_model.py   # retrain on 22 features + 4h label
python src/models/walk_forward.py  # validate OOS accuracy
python src/backtest/backtest.py    # run updated backtest
```

### Expected Outcome

These changes address the structural causes of the Round 1 null result but do not guarantee improvement — if the market is truly efficient with respect to OHLCV features, 50% OOS accuracy will persist regardless of feature count or label horizon. The changes reduce unnecessary noise and give the model a better chance to detect any signal that exists. Walk-forward output will be the ground truth.

If OOS accuracy remains at 50% after retraining, the conclusion from Round 1 stands: **alternative data is required** (funding rates, order book imbalance, liquidation levels).

---

## Dashboard

A multipage Streamlit dashboard was added at `dashboard/` to make the backtest results and model analysis inspectable without running scripts manually.

### Pages

- **Home (`app.py`)** — project summary and navigation between pages.
- **Overview (`pages/1_Overview.py`)** — OOS equity curve, Sharpe ratio, max drawdown, total trades, drawdown chart, and the raw backtest data table. This is the primary results page.
- **Model Analysis (`pages/2_Model_Analysis.py`)** — placeholder for feature importances, walk-forward fold table, and confusion matrix.
- **Risk Management (`pages/3_Risk_Management.py`)** — placeholder for ATR stop-loss visualisation and risk parameter summary.

### Utilities

Shared code lives in `dashboard/utils/`:

- `load_data.py` — cached loaders for backtest results, OOS classification metrics, walk-forward results, and raw OHLCV data.
- `load_model.py` — cached model loader with fallback logic.
- `charts.py` — reusable Plotly chart builders covering the equity curve, drawdown, feature importance, confusion matrix, and ATR stop chart.

### How to Run

```bash
streamlit run dashboard/app.py
```

Run from the project root. The dashboard reads the output files produced by the backtest and walk-forward scripts, so run those first if the output files do not exist.

**Tech stack:** Streamlit, Plotly, Pandas, joblib, scikit-learn.
