# project-wide settings — edit this before running anything
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LOOKBACK = 10           # number of lagged return features
EXCHANGE_ID = "binance"
# backtest
FEE = 0.0004            # ~0.04% taker fee on Binance
INITIAL_CAPITAL = 10_000
# model
MODEL_NAME = "rf_crypto_next_candle.pkl"
# live trading
POSITION_SIZE = 0.001   # BTC per trade (minimum size)
TRADE_SYMBOL = "BTCUSDT"
# testnet — set to False when moving to real trading
TESTNET = True
TESTNET_BASE_URL = "https://testnet.binance.vision"
# risk management
ATR_PERIOD       = 14    # ATR lookback in candles
ATR_MULTIPLIER   = 2.0   # stop = entry - ATR_MULTIPLIER * ATR
TP_MULTIPLIER    = 3.0   # take profit = entry + TP_MULTIPLIER * ATR
MAX_DAILY_LOSS   = 0.02  # halt if down 2% on the day
MAX_DRAWDOWN     = 0.05  # halt if down 5% from equity peak

