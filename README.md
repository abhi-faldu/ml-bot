# ML Bot — Crypto Trading Bot

Automated BTC/USDT direction prediction and live trading using a Random Forest classifier on hourly candles.

## Overview

This bot predicts whether the next hourly BTC/USDT candle will close **up or down** using lagged return features:

- **Signal 1**: Long — buy market order
- **Signal 0**: Flat — no trade

The pipeline covers data download, feature engineering, model training, backtesting, and live execution on Binance.

## Features

- **Automated OHLCV download** via ccxt with pagination
- **Lagged return features** (10 lags by default) for direction prediction
- **Random Forest classifier** with tuned hyperparameters (300 trees, max_depth=8)
- **Vectorized backtest** with fee deduction and equity curve
- **Annualised Sharpe ratio** reporting
- **Live trading loop** with Binance market orders and structured logging

## Project Structure

```
ml-bot/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_crypto.py    # pull OHLCV candles via ccxt
│   │   └── make_features.py      # build lagged-return feature matrix
│   ├── models/
│   │   ├── __init__.py
│   │   └── train_model.py        # train and save RF model
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtest.py           # vectorized backtest with equity curve
│   ├── live/
│   │   ├── __init__.py
│   │   └── live_trade.py         # live hourly trading loop
│   ├── __init__.py
│   └── utils.py                  # shared helpers
├── data/
│   ├── raw/                      # CSV output from download_crypto.py
│   └── processed/                # feature CSVs from make_features.py
├── models/                       # saved .pkl model files
├── tests/                        # unit tests
├── docs/                         # documentation
├── config.py                     # all tuneable settings in one place
├── requirements.txt
└── README.md
```

## Quickstart

### Prerequisites

- Python 3.8 or higher
- Binance account (for live trading only)

### 1. Clone Repository

```bash
git clone https://github.com/abhi-faldu/ml-bot
cd ml-bot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1 — Download Data

```bash
python src/data/download_crypto.py
```

Downloads hourly BTC/USDT and ETH/USDT candles and saves to `data/raw/`.

### Step 2 — Build Features

```bash
python src/data/make_features.py
```

Generates 10 lagged return features and direction target, saved to `data/processed/`.

### Step 3 — Train Model

```bash
python src/models/train_model.py
```

Trains a Random Forest (300 trees, max_depth=8) and saves to `models/`.

### Step 4 — Backtest

```bash
python src/backtest/backtest.py
```

Runs a vectorized backtest and prints total return, trade count, and annualised Sharpe ratio.

### Step 5 — Live Trading

```bash
# Set your Binance API keys first
export BINANCE_API_KEY=your_key       # Windows: $env:BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET=your_secret # Windows: $env:BINANCE_API_SECRET="your_secret"

python src/live/live_trade.py
```

Runs every hour — fetches latest candles, predicts direction, places market order if signal is long.

## Technical Details

### Feature Pipeline

1. Download raw OHLCV candles (ccxt)
2. Compute 1-period percentage returns
3. Generate 10 lagged return features (`ret_lag_1` … `ret_lag_10`)
4. Binary target: 1 if next candle closes up, else 0

### Model

- Algorithm: Random Forest Classifier
- Estimators: 300 trees
- Max depth: 8
- Train/test split: 80/20 (no shuffling — preserves time order)
- All cores used: `n_jobs=-1`

### Backtest

- Signal: model prediction (1 = long, 0 = flat)
- Fee: 0.04% charged on every position change
- Metrics: total return %, trade count, annualised Sharpe ratio

## Configuration

All settings are in `config.py` at the project root:

```python
SYMBOL = "BTC/USDT"       # trading pair
TIMEFRAME = "1h"           # candle interval
LOOKBACK = 10              # number of lag features
EXCHANGE_ID = "binance"    # ccxt exchange
FEE = 0.0004               # taker fee
INITIAL_CAPITAL = 10_000   # starting equity for backtest
MODEL_NAME = "rf_crypto_next_candle.pkl"
```

## Requirements

```
ccxt>=4.2
pandas>=2.0
scikit-learn>=1.4
joblib>=1.3
python-binance>=1.0.19
matplotlib>=3.8
numpy>=1.26
```

## Troubleshooting

**Issue: `BINANCE_API_KEY` not found**

```bash
# Windows PowerShell
$env:BINANCE_API_KEY="your_key"
$env:BINANCE_API_SECRET="your_secret"
```

**Issue: Model file not found**

```
# Run steps in order — train_model.py must run before backtest or live_trade
python src/models/train_model.py
```

**Issue: ModuleNotFoundError**

```bash
# Always run scripts from the project root, not from inside src/
cd ml-bot
python src/data/download_crypto.py   # correct
```

**Issue: Low prediction accuracy**

- Increase `LOOKBACK` in `config.py` for more lag features
- Add technical indicators (RSI, MACD) inside `src/data/make_features.py`

## About

Personal learning project exploring ML-based crypto trading strategies.

> ⚠️ Not financial advice. Use at your own risk.
