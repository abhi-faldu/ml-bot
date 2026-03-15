# ml-bot

A Python machine learning trading bot for BTC/USDT using the Binance API.

Built as a research and engineering project across January–March 2026.

> **Round 1 finding:** Price and volume features (RSI, MACD, ATR, lagged returns) could not predict BTC/USDT hourly direction above break-even after fees. OOS accuracy = 50.00% across 17,500 rows and 5 walk-forward folds.
>
> **Round 2 (current):** Applied targeted improvements — expanded feature set (16 → 22), switched model to RandomForestClassifier, changed prediction label to a 4-hour forward return to reduce single-candle noise, added confidence-filtered signals, and improved the backtest with strict OOS split, enhanced metrics, and an equity curve chart. Results pending retraining.
>
> See [FINDINGS.md](FINDINGS.md) for full analysis.

---

## Project Structure

```
ml-bot/
├── src/
│   ├── data/
│   │   ├── download_crypto.py   # fetches 2 years of OHLCV via ccxt
│   │   ├── make_features.py     # RSI, MACD, ATR, Bollinger, EMA deviations, candle features
│   │   └── pipeline.py          # temporal-safe train/test split, 4h label
│   ├── models/
│   │   ├── train_model.py       # RandomForestClassifier
│   │   └── walk_forward.py      # 5-fold walk-forward validation
│   ├── risk/
│   │   └── risk_manager.py      # ATR stops, daily loss limit, drawdown breaker
│   ├── backtest/
│   │   └── backtest.py          # vectorised OOS backtest, confidence filtering, equity chart
│   ├── live/
│   │   └── live_trade.py        # hourly trading loop
│   └── utils.py
├── config.py
├── requirements.txt
├── FINDINGS.md                  # full research findings and null result analysis
└── README.md
```

---

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. set env vars
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
export PYTHONPATH=/path/to/ml-bot

# 3. download 2 years of data (~17,500 rows)
python src/data/download_crypto.py

# 4. train model (rebuilds features CSV automatically)
python src/models/train_model.py

# 5. validate OOS accuracy (5-fold walk-forward)
python src/models/walk_forward.py

# 6. run backtest on true OOS data (saves backtest_equity.png)
python src/backtest/backtest.py

# 7. run live loop (testnet)
python src/live/live_trade.py
```

---

## Features (22 total)

| Feature | Description |
|---|---|
| `ret_lag_1..10` | Lagged 1h returns (momentum) |
| `rsi_14` | Relative Strength Index (overbought/oversold) |
| `macd` | MACD line (trend direction) |
| `macd_signal` | MACD signal line (trend confirmation) |
| `macd_hist` | MACD histogram (momentum change) |
| `atr_14` | Average True Range — absolute volatility |
| `atr_pct` | ATR as % of close — normalised volatility |
| `vol_ratio` | Volume vs 20-period average |
| `bb_pct` | Bollinger %B — position within bands |
| `bb_width` | Bollinger band width / MA — volatility regime |
| `ema_20_dev` | close / EMA(20) − 1 — short-term trend deviation |
| `ema_50_dev` | close / EMA(50) − 1 — medium-term trend deviation |
| `body_pct` | (close − open) / (high − low) — candle direction and strength |

---

## Model

**RandomForestClassifier** (scikit-learn)
- `n_estimators=300`, `max_depth=6`, `min_samples_leaf=50`
- Trained on 80% of data, validated on the last 20% (strict temporal split)
- Signals filtered by confidence: only trade when P(up) > 0.55

**Label:** 1 if `close[t+4] > close[t]`, else 0 (4-hour forward horizon reduces single-candle noise)

---

## Backtest

- Strictly evaluates only the OOS portion (last 20%) — no training data contamination
- Confidence filtering: only BUY when predicted probability exceeds threshold
- Metrics: total return, Sharpe ratio, max drawdown, win rate, fees paid, buy-and-hold comparison, avg holding period
- Saves normalised equity curve chart as `backtest_equity.png`

---

## Risk Management

- ATR-based stop loss: `entry − 2 × ATR`
- ATR-based take profit: `entry + 3 × ATR`
- Daily loss limit: halts bot if down >2% on the day
- Max drawdown circuit breaker: halts if down >5% from equity peak

---

## What Was Learned

Building the system was valuable. The honest OOS result (50.00%) from Round 1 is the
correct output of a well-constructed validation framework. A system that correctly
identifies the absence of edge is more useful than one that overfits and loses money live.

Round 2 improvements target the two most likely causes of the null result: an
insufficiently rich feature set and excessive label noise from single-candle targets.

To move beyond 50% sustainably, the next step is alternative data: funding rates,
liquidation levels, and order book imbalance. These are not encoded in public OHLCV
and have shown predictive value in published research.

---

## Tech Stack

Python · scikit-learn · ccxt · python-binance · pandas · numpy · matplotlib
