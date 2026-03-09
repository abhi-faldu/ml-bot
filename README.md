# ml-bot

A Python machine learning trading bot for BTC/USDT using the Binance API.

Built as a research and engineering project across January–March 2026.

> **Finding:** Price and volume features (RSI, MACD, ATR, lagged returns) do not predict BTC/USDT hourly direction above the break-even threshold after fees. OOS accuracy = 50.00% across 17,500 rows and 5 walk-forward folds. See [FINDINGS.md](FINDINGS.md) for full analysis.

---

## Project Structure

```
ml-bot/
├── src/
│   ├── data/
│   │   ├── download_crypto.py   # fetches 2 years of OHLCV via ccxt
│   │   ├── make_features.py     # RSI, MACD, ATR, volume ratio, lagged returns
│   │   └── pipeline.py          # temporal-safe train/test split
│   ├── models/
│   │   ├── train_model.py       # LightGBM classifier
│   │   └── walk_forward.py      # 5-fold walk-forward validation
│   ├── risk/
│   │   └── risk_manager.py      # ATR stops, daily loss limit, drawdown breaker
│   ├── backtest/
│   │   └── backtest.py          # vectorised backtest + Sharpe ratio
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

# 4. build features
python src/data/make_features.py

# 5. validate model (walk-forward)
python src/models/walk_forward.py

# 6. train final model
python src/models/train_model.py

# 7. run live loop (testnet)
python src/live/live_trade.py
```

---

## Features (16 total)

| Feature | Description |
|---|---|
| `ret_lag_1..10` | Lagged 1h returns |
| `rsi_14` | Relative Strength Index |
| `macd` | MACD line |
| `macd_signal` | MACD signal line |
| `macd_hist` | MACD histogram |
| `atr_14` | Average True Range |
| `vol_ratio` | Volume vs 20-period average |

---

## Risk Management

- ATR-based stop loss: `entry - 2 × ATR`
- ATR-based take profit: `entry + 3 × ATR`
- Daily loss limit: halts bot if down >2% on the day
- Max drawdown circuit breaker: halts if down >5% from equity peak

---

## What Was Learned

Building the system was valuable. The honest OOS result (50.00%) is the
correct output of a well-constructed validation framework. A system that
correctly identifies the absence of edge is more useful than one that
overfits and loses money live.

To move beyond 50%, the next step is alternative data: funding rates,
liquidation levels, and order book imbalance. These are not encoded in
public OHLCV and have shown predictive value in published research.

---

## Tech Stack

Python · LightGBM · scikit-learn · ccxt · python-binance · pandas · numpy
