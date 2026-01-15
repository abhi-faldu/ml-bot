"""Live trading loop with ATR stop loss, take profit, and circuit breakers.

Required env vars:
    BINANCE_API_KEY
    BINANCE_API_SECRET
"""
import logging
import os
import sys
from pathlib import Path
from time import sleep

import joblib
import pandas as pd
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import LOOKBACK, MODEL_NAME, POSITION_SIZE, TESTNET, TRADE_SYMBOL
from src.utils import add_lag_features
from src.risk.risk_manager import (
    RiskConfig, AccountRiskState,
    get_stop_and_tp, should_exit
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = ROOT / "models" / MODEL_NAME
INTERVAL   = Client.KLINE_INTERVAL_1HOUR
TESTNET_API_URL = "https://testnet.binance.vision/api"

# risk config — adjust in config.py if needed
RISK = RiskConfig()


def get_client() -> Client:
    api_key    = os.environ["BINANCE_API_KEY"]
    api_secret = os.environ["BINANCE_API_SECRET"]
    if TESTNET:
        log.info("TESTNET mode — no real money involved")
        client = Client(api_key, api_secret)
        client.API_URL = TESTNET_API_URL
        client.ping()
        log.info("Testnet ping OK")
    else:
        log.info("LIVE mode — real money at risk!")
        client = Client(api_key, api_secret)
    return client


def get_recent_klines(client: Client, limit: int = 50) -> pd.DataFrame:
    raw = client.get_klines(symbol=TRADE_SYMBOL, interval=INTERVAL, limit=limit)
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df = add_lag_features(df, "return", LOOKBACK)
    df.dropna(inplace=True)
    latest = df.iloc[[-1]]
    return latest[[c for c in latest.columns if c.startswith("ret_lag_")]]


def get_account_equity(client: Client) -> float:
    """Fetch USDT balance as proxy for account equity."""
    try:
        balances = client.get_account()["balances"]
        usdt = next((b for b in balances if b["asset"] == "USDT"), None)
        return float(usdt["free"]) if usdt else 0.0
    except Exception as e:
        log.error(f"Could not fetch equity: {e}")
        return 0.0


def place_order(client: Client, side: str, qty: float) -> dict:
    log.info(f"placing {side} {qty} {TRADE_SYMBOL}")
    try:
        resp = client.create_order(
            symbol=TRADE_SYMBOL, side=side,
            type=ORDER_TYPE_MARKET, quantity=qty,
        )
        log.info(f"order filled: id={resp.get('orderId')} status={resp.get('status')}")
        return resp
    except Exception as e:
        log.error(f"order failed: {e}")
        return {}


def main() -> None:
    client = get_client()

    if not MODEL_PATH.exists():
        log.error(f"Model not found — run train_model.py first")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    log.info(f"Model loaded | {TRADE_SYMBOL} | size={POSITION_SIZE} BTC")

    # state
    risk_state  = AccountRiskState()
    in_position = False
    entry_price = 0.0
    stop_price  = 0.0
    tp_price    = 0.0

    # seed peak equity
    equity = get_account_equity(client)
    risk_state.peak_equity       = equity
    risk_state.daily_start_equity = equity
    log.info(f"Starting equity: {equity:.2f} USDT")
    log.info("Live loop started — runs every hour")

    while True:
        try:
            df    = get_recent_klines(client)
            price = float(df["close"].iloc[-1])

            # ── account-level circuit breakers ──────────────────────────────
            equity = get_account_equity(client)
            if risk_state.check_circuit_breakers(equity, RISK):
                log.warning(f"HALTED: {risk_state.halt_reason}")
                log.warning("Bot paused. Restart manually after reviewing.")
                break

            # ── position exit check (stop / TP) ─────────────────────────────
            if in_position:
                exit_now, reason = should_exit(price, entry_price, stop_price, tp_price)
                if exit_now:
                    log.info(f"Exiting position: {reason}")
                    resp = place_order(client, SIDE_SELL, POSITION_SIZE)
                    if resp:
                        in_position = False
                        entry_price = stop_price = tp_price = 0.0

            # ── signal & entry ───────────────────────────────────────────────
            if not in_position:
                X    = build_features(df)
                pred = model.predict(X)[0]
                log.info(f"Price: ${price:,.2f} | Signal: {'LONG' if pred == 1 else 'FLAT'}")

                if pred == 1:
                    resp = place_order(client, SIDE_BUY, POSITION_SIZE)
                    if resp:
                        entry_price = price
                        stop_price, tp_price = get_stop_and_tp(entry_price, df, RISK)
                        in_position = True
                        log.info(
                            f"Position opened | entry={entry_price:.2f} "
                            f"stop={stop_price:.2f} TP={tp_price:.2f}"
                        )
                else:
                    log.info("Signal=FLAT — no entry")

        except Exception as e:
            log.error(f"Loop error: {e}")

        log.info("Sleeping 1 hour...")
        sleep(3600)


if __name__ == "__main__":
    main()
