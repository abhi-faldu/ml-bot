"""
Live trading loop — runs every hour, predicts direction, fires market orders.

"""
import logging
import os
import sys
from pathlib import Path
from time import sleep

import joblib
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import LOOKBACK, MODEL_NAME, POSITION_SIZE, TESTNET, TRADE_SYMBOL
from src.data.pipeline import build_features
from src.data.make_features import get_feature_cols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

MODEL_PATH = ROOT / "models" / MODEL_NAME
INTERVAL = Client.KLINE_INTERVAL_1HOUR



# Binance Client


def get_client():

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("Binance API keys not found in environment variables")

    if TESTNET:
        log.info("Running in TESTNET mode")

        client = Client(api_key, api_secret, testnet=True)
        client.API_URL = "https://testnet.binance.vision/api"

    else:
        log.warning("LIVE TRADING ENABLED")
        client = Client(api_key, api_secret)

    client.ping()
    return client



# Market Data


def get_recent_klines(client, limit=100):

    raw = client.get_klines(
        symbol=TRADE_SYMBOL,
        interval=INTERVAL,
        limit=limit
    )

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ]

    df = pd.DataFrame(raw, columns=cols)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)

    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    return df[["open","high","low","close","volume"]]



# Feature Engineering


def get_live_features(df):

    df_feat = build_features(df.copy(), lookback=LOOKBACK)

    if df_feat.empty:
        raise ValueError("Feature engineering produced empty DataFrame — need more data")

    latest = df_feat.iloc[[-1]]
    feature_cols = get_feature_cols(latest)

    return latest[feature_cols]



# Quantity Precision


def get_step_size(client):

    info = client.get_symbol_info(TRADE_SYMBOL)

    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            return float(f["stepSize"])

    return 0.000001


def adjust_qty(qty, step):
    # str(step) can be scientific notation (e.g. "1e-05") which has no "."
    # Decimal parses it correctly regardless of representation
    from decimal import Decimal
    precision = abs(Decimal(str(step)).as_tuple().exponent)
    return round(qty, precision)



# Order Execution


def place_order(client, qty):

    step = get_step_size(client)
    qty = adjust_qty(qty, step)

    log.info(f"Placing BUY order: {qty} {TRADE_SYMBOL}")

    try:

        resp = client.create_order(
            symbol=TRADE_SYMBOL,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=qty
        )

        log.info(f"Order filled | id={resp['orderId']}")

    except Exception as e:

        log.error(f"Order failed: {e}")



# Trading Loop


def main():

    client = get_client()

    if not MODEL_PATH.exists():
        log.error("Model not found")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    log.info("Model loaded")
    log.info(f"Trading {TRADE_SYMBOL}")
    log.info("Starting live trading loop")

    while True:

        try:

            df = get_recent_klines(client)

            X = get_live_features(df)

            pred = model.predict(X)[0]

            price = df["close"].iloc[-1]

            log.info(f"Price ${price:.2f} | Signal {pred}")

            if pred == 1:
                place_order(client, POSITION_SIZE)

            else:
                log.info("No trade signal")

        except Exception as e:

            log.error(f"Loop error: {e}")

        log.info("Sleeping 1 hour")

        sleep(3600)


if __name__ == "__main__":
    main()