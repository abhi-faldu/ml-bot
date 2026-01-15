import time
from pathlib import Path

import ccxt
import pandas as pd

# go up from src/data/ to project root, then into data/raw
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1000,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Download OHLCV candles and save to data/raw/.

    Paginates in batches of 500 until `limit` candles are collected.
    """
    exchange = getattr(ccxt, exchange_id)()

    all_candles: list = []
    since = None

    while len(all_candles) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=500)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)  # avoid hitting rate limits

    all_candles = all_candles[:limit]
    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    fname = f"{exchange_id}_{symbol.replace('/', '')}_{timeframe}.csv"
    df.to_csv(DATA_DIR / fname)
    print(f"saved {len(df)} rows → {DATA_DIR / fname}")
    return df


if __name__ == "__main__":
    for sym in ["BTC/USDT", "ETH/USDT"]:
        fetch_ohlcv(symbol=sym)
