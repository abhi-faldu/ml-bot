# Research Findings: ML-Based BTC/USDT Direction Prediction

**Project:** ml-bot  
**Period:** December 2025 – March 2026  
**Conclusion:** Price and volume features alone are insufficient to predict BTC/USDT hourly direction above the break-even threshold after fees.

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
