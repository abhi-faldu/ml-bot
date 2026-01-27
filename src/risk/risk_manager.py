"""
src/risk/risk_manager.py

Dynamic risk management module.

Implements three critical protections:
1. ATR-based stop loss    — stop proportional to current volatility
2. Daily loss limit       — halt trading if day's PnL drops too far
3. Max drawdown breaker   — halt if equity falls too far from peak
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """All risk parameters in one place — override in config.py if needed."""

    # ATR stop loss
    atr_period: int   = 14       # ATR lookback in candles
    atr_multiplier: float = 2.0  # stop = entry - atr_multiplier * ATR

    # Take profit
    tp_multiplier: float = 3.0   # TP = entry + tp_multiplier * ATR (risk:reward = 1:1.5)

    # Daily loss limit
    max_daily_loss_pct: float = 0.02    # halt if day down more than 2% of starting equity

    # Max drawdown circuit breaker
    max_drawdown_pct: float = 0.05      # halt if equity drops 5% from peak


# ── ATR Calculation ───────────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute Average True Range from OHLCV dataframe.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = rolling mean of TR over `period` candles.

    Returns the most recent ATR value.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


def get_stop_and_tp(
    entry_price: float,
    df: pd.DataFrame,
    cfg: RiskConfig = RiskConfig(),
) -> tuple[float, float]:
    """
    Compute ATR-based stop loss and take profit prices for a long position.

    Parameters
    ----------
    entry_price : fill price of the buy order
    df          : recent OHLCV dataframe (needs at least atr_period + 1 rows)
    cfg         : RiskConfig instance

    Returns
    -------
    (stop_price, take_profit_price)
    """
    atr = calculate_atr(df, cfg.atr_period)
    stop_price       = entry_price - cfg.atr_multiplier * atr
    take_profit_price = entry_price + cfg.tp_multiplier * atr

    log.info(
        f"ATR={atr:.2f}  "
        f"entry={entry_price:.2f}  "
        f"stop={stop_price:.2f}  "
        f"TP={take_profit_price:.2f}"
    )
    return stop_price, take_profit_price


# ── Position Exit Check ───────────────────────────────────────────────────────

def should_exit(
    current_price: float,
    entry_price: float,
    stop_price: float,
    take_profit_price: float,
) -> tuple[bool, str]:
    """
    Check whether current price triggers a stop or take profit.

    Returns
    -------
    (should_exit: bool, reason: str)
    """
    if current_price <= stop_price:
        pnl_pct = (current_price - entry_price) / entry_price * 100
        return True, f"STOP LOSS hit — price={current_price:.2f} stop={stop_price:.2f} pnl={pnl_pct:+.2f}%"

    if current_price >= take_profit_price:
        pnl_pct = (current_price - entry_price) / entry_price * 100
        return True, f"TAKE PROFIT hit — price={current_price:.2f} tp={take_profit_price:.2f} pnl={pnl_pct:+.2f}%"

    return False, ""


# ── Account-level Circuit Breakers ───────────────────────────────────────────

@dataclass
class AccountRiskState:
    """
    Tracks account-level risk metrics across the trading session.
    Reset daily_start_equity at the beginning of each trading day.
    """
    peak_equity: float              = 0.0
    daily_start_equity: float       = 0.0
    daily_start_date: Optional[date] = None
    halted: bool                    = False
    halt_reason: str                = ""

    def update_peak(self, equity: float) -> None:
        if equity > self.peak_equity:
            self.peak_equity = equity

    def reset_day_if_needed(self, equity: float) -> None:
        today = date.today()
        if self.daily_start_date != today:
            self.daily_start_date  = today
            self.daily_start_equity = equity
            log.info(f"New trading day — daily equity reset to {equity:.2f}")

    def check_circuit_breakers(
        self,
        equity: float,
        cfg: RiskConfig = RiskConfig(),
    ) -> bool:
        """
        Run all account-level checks.
        Returns True if trading should be HALTED.
        """
        if self.halted:
            return True

        self.update_peak(equity)
        self.reset_day_if_needed(equity)

        # daily loss limit
        if self.daily_start_equity > 0:
            daily_loss = (equity - self.daily_start_equity) / self.daily_start_equity
            if daily_loss < -cfg.max_daily_loss_pct:
                self.halted = True
                self.halt_reason = (
                    f"DAILY LOSS LIMIT — down {daily_loss*100:.2f}% today "
                    f"(limit {cfg.max_daily_loss_pct*100:.1f}%)"
                )
                log.warning(f"CIRCUIT BREAKER: {self.halt_reason}")
                return True

        # max drawdown from peak
        if self.peak_equity > 0:
            drawdown = (equity - self.peak_equity) / self.peak_equity
            if drawdown < -cfg.max_drawdown_pct:
                self.halted = True
                self.halt_reason = (
                    f"MAX DRAWDOWN — down {drawdown*100:.2f}% from peak "
                    f"(limit {cfg.max_drawdown_pct*100:.1f}%)"
                )
                log.warning(f"CIRCUIT BREAKER: {self.halt_reason}")
                return True

        return False


# ── Quick Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # simulate ATR stop calculation
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=50, freq="1h")
    price = 85000 + np.cumsum(np.random.randn(50) * 200)
    df_demo = pd.DataFrame({
        "open":  price - 100,
        "high":  price + 300,
        "low":   price - 300,
        "close": price,
        "volume": np.random.randint(100, 500, 50),
    }, index=dates)

    entry = float(df_demo["close"].iloc[-1])
    stop, tp = get_stop_and_tp(entry, df_demo)

    # simulate circuit breaker
    state = AccountRiskState(peak_equity=10000, daily_start_equity=10000)
    print(f"\nCircuit breaker check at equity=9800: halted={state.check_circuit_breakers(9800)}")
    print(f"Circuit breaker check at equity=9400: halted={state.check_circuit_breakers(9400)}")
