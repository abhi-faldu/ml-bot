"""
src/data/download_crypto.py

Downloads historical OHLCV data going back a specified number of days.

Default: 730 days (2 years) of 1h candles = ~17,520 rows per symbol.
This is the minimum required for the ML model to find reliable patterns.

With only 500 rows the model sees too few examples to generalise —
it memorises training noise and fails on unseen data.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 730,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Download OHLCV candles starting from `days` ago until now.

    Paginates automatically in batches of 500 respecting rate limits.
    With days=730 and timeframe=1h this fetches ~17,500 rows.

    Parameters
    ----------
    symbol      : e.g. "BTC/USDT"
    timeframe   : e.g. "1h", "4h", "1d"
    days        : how many days of history to fetch (default 730 = 2 years)
    exchange_id : ccxt exchange name
    """
    exchange = getattr(ccxt, exchange_id)()

    # start timestamp in milliseconds
    start_dt = datetime.utcnow() - timedelta(days=days)
    since_ms  = int(start_dt.timestamp() * 1000)

    print(f"Fetching {symbol} {timeframe} from {start_dt.date()} ...")

    all_candles: list = []
    since = since_ms

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=500)
        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]

        # stop once we have reached current time
        now_ms = int(datetime.utcnow().timestamp() * 1000)
        if last_ts >= now_ms - 3_600_000:  # within 1 hour of now
            break

        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    fname = f"{exchange_id}_{symbol.replace('/', '')}_{timeframe}.csv"
    df.to_csv(DATA_DIR / fname)
    print(f"saved {len(df)} rows -> {DATA_DIR / fname}")
    return df


if __name__ == "__main__":
    for sym in ["BTC/USDT", "ETH/USDT"]:
        fetch_ohlcv(symbol=sym, days=730)
