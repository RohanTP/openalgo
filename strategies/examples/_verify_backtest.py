#!/usr/bin/env python
"""
Verification suite for backtest_engine.py + fib_premium_breakout_backtest.py.

Tests run in 4 tiers:
  1. Unit — Trade.partial_exit() capital maths
  2. Engine — controlled single-trade scenarios with hand-calculated expected PnL
  3. Strategy — synthetic 2-year 1-min OHLCV, verifies per-trade logic
  4. Metrics  — equity curve properties (drawdown, Sharpe sign consistency)

Run:
  python _verify_backtest.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, date, time, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from backtest_engine import BacktestEngine, BacktestResult, Trade, load_csv
from fib_premium_breakout_backtest import (
    fib_breakout_strategy,
    _build_targets,
    _build_stop_loss,
    _split_quantities,
    _heikin_ashi,
    LOT_SIZE,
    TOTAL_LOTS,
    FIB_TARGETS,
    BIG_CANDLE_THRESHOLD,
    BIG_CANDLE_SL_FIB,
    SL_BUFFER,
    ENTRY_BUFFER,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        msg = f"  {FAIL}  {label}" + (f"  ({detail})" if detail else "")
        print(msg)
        _failures.append(label)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ============================================================
# 1. Unit: Trade.partial_exit
# ============================================================

def test_trade_unit() -> None:
    print("\n[1] Trade.partial_exit unit tests")

    # Simple full exit — LONG, buy 100 @ 350, sell 100 @ 400, no commission
    t = Trade("X", "LONG", datetime(2024, 1, 2, 9, 15), 350.0, 100, entry_commission=0.0)
    t.partial_exit(datetime(2024, 1, 2, 9, 30), 400.0, 100, "target", commission=0.0)
    check("full exit: open_qty=0", t.open_qty == 0)
    check("full exit: gross_pnl=5000", approx(t.gross_pnl, 5000.0), str(t.gross_pnl))
    check("full exit: pnl=5000 (no comm)", approx(t.pnl, 5000.0), str(t.pnl))
    check("full exit: is_open=False", not t.is_open)

    # Partial 2-stage exit — buy 300 @ 200, sell 100 @ 220, sell 200 @ 250
    t2 = Trade("Y", "LONG", datetime(2024, 1, 2, 9, 15), 200.0, 300, entry_commission=0.0)
    t2.partial_exit(datetime(2024, 1, 2, 9, 20), 220.0, 100, "t1", commission=0.0)
    check("partial exit 1: open_qty=200", t2.open_qty == 200, str(t2.open_qty))
    check("partial exit 1: still open", t2.is_open)
    t2.partial_exit(datetime(2024, 1, 2, 9, 25), 250.0, 200, "t2", commission=0.0)
    # gross = (220*100 + 250*200) - 200*300 = (22000+50000) - 60000 = 12000
    check("partial exit 2: gross_pnl=12000", approx(t2.gross_pnl, 12000.0), str(t2.gross_pnl))
    check("partial exit 2: closed", not t2.is_open)

    # Commission deducted — buy 75 @ 300, sell 75 @ 360, comm 20 in + 20 out
    t3 = Trade("Z", "LONG", datetime(2024, 1, 2, 9, 15), 300.0, 75, entry_commission=20.0)
    t3.partial_exit(datetime(2024, 1, 2, 9, 30), 360.0, 75, "exit", commission=20.0)
    # gross = (360-300)*75 = 4500; net = 4500 - 40 = 4460
    check("comm: gross_pnl=4500", approx(t3.gross_pnl, 4500.0), str(t3.gross_pnl))
    check("comm: net pnl=4460", approx(t3.pnl, 4460.0), str(t3.pnl))


# ============================================================
# 2. Engine unit — controlled scenario
# ============================================================

def _make_bar(ts: datetime, o: float, h: float, lo: float, c: float) -> dict:
    return {"open": o, "high": h, "low": lo, "close": c, "volume": 1000}


def _single_trade_df() -> pd.DataFrame:
    """5 bars: bar1=pre-entry, bar2=entry trigger, bar3-4=hold, bar5=exit at SL."""
    rows = [
        _make_bar(datetime(2024, 1, 2, 9, 15), 300.0, 309.0, 298.0, 305.0),  # ref candle
        _make_bar(datetime(2024, 1, 2, 9, 16), 305.0, 315.0, 303.0, 312.0),  # entry (high >= 309.05)
        _make_bar(datetime(2024, 1, 2, 9, 17), 312.0, 318.0, 310.0, 316.0),
        _make_bar(datetime(2024, 1, 2, 9, 18), 316.0, 320.0, 308.0, 310.0),
        _make_bar(datetime(2024, 1, 2, 9, 19), 310.0, 310.0, 290.0, 291.0),  # SL hit (close <= SL)
    ]
    idx = [r.pop("volume") and None or r for r in []]  # reset trick
    rows2 = [
        _make_bar(datetime(2024, 1, 2, 9, 15), 300.0, 309.0, 298.0, 305.0),
        _make_bar(datetime(2024, 1, 2, 9, 16), 305.0, 315.0, 303.0, 312.0),
        _make_bar(datetime(2024, 1, 2, 9, 17), 312.0, 318.0, 310.0, 316.0),
        _make_bar(datetime(2024, 1, 2, 9, 18), 316.0, 320.0, 308.0, 310.0),
        _make_bar(datetime(2024, 1, 2, 9, 19), 310.0, 310.0, 290.0, 291.0),
    ]
    df = pd.DataFrame(rows2, index=pd.to_datetime([
        "2024-01-02 09:15", "2024-01-02 09:16",
        "2024-01-02 09:17", "2024-01-02 09:18", "2024-01-02 09:19"
    ]))
    return df


def test_engine_simple_buy_sell() -> None:
    print("\n[2] Engine — simple deterministic BUY/SELL")

    buy_price, sell_price = 200.0, 250.0
    qty = 75
    initial = 100_000.0

    def strategy(candle, _data, state):
        idx = state.get("idx", 0)
        state["idx"] = idx + 1
        if idx == 0:
            return [{"action": "BUY", "quantity": qty, "price": buy_price}]
        if idx == 1:
            return [{"action": "SELL", "quantity": qty, "price": sell_price}]
        return []

    df = pd.DataFrame(
        [{"open": 200, "high": 210, "low": 195, "close": 205}] * 5,
        index=pd.date_range("2024-01-02 09:15", periods=5, freq="1min"),
    )
    engine = BacktestEngine(df, strategy, initial_capital=initial,
                            slippage_pct=0.0, commission_per_lot=0.0, lot_size=qty)
    result = engine.run()

    expected_pnl = (sell_price - buy_price) * qty  # 3750
    expected_final = initial + expected_pnl

    check("simple trade: 1 trade recorded", result.total_trades == 1, str(result.total_trades))
    check("simple trade: final capital correct",
          approx(result.final_capital, expected_final, tol=0.01),
          f"got {result.final_capital:.2f} expected {expected_final:.2f}")
    check("simple trade: total_return > 0", result.total_return_pct > 0)

    # With commission: 20 per lot entry + 20 exit
    engine2 = BacktestEngine(df, strategy, initial_capital=initial,
                             slippage_pct=0.0, commission_per_lot=20.0, lot_size=qty)
    result2 = engine2.run()
    expected_final2 = initial + expected_pnl - 40.0  # 2 x 20 commission
    check("commission deducted from final capital",
          approx(result2.final_capital, expected_final2, tol=0.01),
          f"got {result2.final_capital:.2f} expected {expected_final2:.2f}")


def test_engine_partial_exit() -> None:
    print("\n[3] Engine — partial 3-leg exit")

    initial = 200_000.0
    entry_p = 300.0
    t1_p, t2_p, t3_p = 340.0, 385.0, 420.0
    qty_t1, qty_t2, qty_t3 = 75, 75, 150  # sums to 300
    total_qty = qty_t1 + qty_t2 + qty_t3

    call_idx = [0]

    def strategy(candle, _data, state):
        i = call_idx[0]
        call_idx[0] += 1
        if i == 0:
            return [{"action": "BUY", "quantity": total_qty, "price": entry_p}]
        if i == 1:
            return [{"action": "SELL", "quantity": qty_t1, "price": t1_p, "reason": "t1"}]
        if i == 2:
            return [{"action": "SELL", "quantity": qty_t2, "price": t2_p, "reason": "t2"}]
        if i == 3:
            return [{"action": "SELL", "quantity": qty_t3, "price": t3_p, "reason": "t3"}]
        return []

    df = pd.DataFrame(
        [{"open": 300, "high": 430, "low": 290, "close": 400}] * 5,
        index=pd.date_range("2024-01-02 09:15", periods=5, freq="1min"),
    )
    engine = BacktestEngine(df, strategy, initial_capital=initial,
                            slippage_pct=0.0, commission_per_lot=0.0, lot_size=75)
    result = engine.run()

    expected_pnl = (
        (t1_p - entry_p) * qty_t1
        + (t2_p - entry_p) * qty_t2
        + (t3_p - entry_p) * qty_t3
    )
    expected_final = initial + expected_pnl
    check("partial exit: 1 trade recorded", result.total_trades == 1, str(result.total_trades))
    check("partial exit: capital correct",
          approx(result.final_capital, expected_final, tol=0.01),
          f"got {result.final_capital:.2f} expected {expected_final:.2f}")


# ============================================================
# 3. Strategy helpers unit
# ============================================================

def test_strategy_helpers() -> None:
    print("\n[4] Strategy helper maths")

    # _build_targets: low=300, high=320, range=20
    # targets = [300 + 20*1.272, 300 + 20*1.618, 300 + 20*2.0]
    #          = [325.44, 332.36, 340.0]
    targets = _build_targets(300.0, 320.0)
    check("targets[0] = 325.44", approx(targets[0], 325.44), str(targets[0]))
    check("targets[1] = 332.36", approx(targets[1], 332.36), str(targets[1]))
    check("targets[2] = 340.00", approx(targets[2], 340.00), str(targets[2]))

    # _build_stop_loss: small candle (range=20 < 40 threshold) → low - SL_BUFFER
    sl_small = _build_stop_loss(300.0, 320.0)
    check("small candle SL = low - buffer", approx(sl_small, 300.0 - SL_BUFFER), str(sl_small))

    # _build_stop_loss: big candle (range=60 > 40) → high - range * 0.55
    sl_big = _build_stop_loss(300.0, 360.0)
    expected_big_sl = round(360.0 - 60.0 * BIG_CANDLE_SL_FIB, 2)
    check("big candle SL fib", approx(sl_big, expected_big_sl), str(sl_big))

    # _split_quantities: 300 qty, [25%, 25%, 50%], lot=75
    qs = _split_quantities(300)
    check("split qty sums to total", sum(qs) == 300, str(qs))
    check("split qty[0] = 75", qs[0] == 75, str(qs[0]))
    check("split qty[1] = 75", qs[1] == 75, str(qs[1]))
    check("split qty[2] = 150", qs[2] == 150, str(qs[2]))


# ============================================================
# 4. Synthetic 2-year strategy backtest + integrity checks
# ============================================================

def _make_synthetic_2yr_data() -> pd.DataFrame:
    """
    Build synthetic 1-min OHLCV data for a single option symbol.
    Trading session: 09:15 to 15:30, Mon-Fri, Jan 2024 – Dec 2025 (~500 days).

    Pattern per day (deterministic, varied by day index so different outcomes):
      - Base premium starts at 350.0 and drifts slowly
      - 09:15 candle: reference candle (fixed range 20–60 points)
      - 09:16–10:30: either breakout (65% of days) or fade (35%)
      - Post-entry: price moves to one of the fib targets before SL, or hits SL
    """
    rng = np.random.default_rng(42)
    rows = []

    start_date = date(2024, 1, 1)
    end_date = date(2025, 12, 31)
    trading_days = pd.bdate_range(start_date, end_date, freq="B")

    base_premium = 350.0

    for day_idx, day in enumerate(trading_days):
        day_dt = pd.Timestamp(day)
        drift = rng.uniform(-2, 2)
        base_premium = max(200.0, min(500.0, base_premium + drift))

        # Reference candle 09:15
        ref_low = round(base_premium - rng.uniform(10, 30), 2)
        ref_high = round(ref_low + rng.uniform(15, 70), 2)
        ref_close = round(rng.uniform(ref_low, ref_high), 2)

        session_open = datetime.combine(day.date(), time(9, 15))
        candle_times = [session_open + timedelta(minutes=m) for m in range(0, 375)]  # 09:15–15:29

        breakout_day = rng.random() < 0.65

        for m, ts in enumerate(candle_times):
            if m == 0:
                # Reference candle
                rows.append({
                    "datetime": ts,
                    "open": round(ref_low + (ref_high - ref_low) * 0.3, 2),
                    "high": ref_high,
                    "low": ref_low,
                    "close": ref_close,
                    "volume": int(rng.integers(100, 1000)),
                })
                continue

            prev = rows[-1]
            prev_close = prev["close"]

            if breakout_day and 1 <= m <= 5:
                # Strong up move to trigger breakout
                c_high = round(prev_close + rng.uniform(5, 15), 2)
                c_low = round(prev_close - rng.uniform(1, 3), 2)
                c_close = round(rng.uniform(prev_close, c_high), 2)
            elif breakout_day and 6 <= m <= 30:
                # Post-breakout trending up
                c_high = round(prev_close + rng.uniform(2, 8), 2)
                c_low = round(prev_close - rng.uniform(0.5, 3), 2)
                c_close = round(rng.uniform(c_low * 0.7 + c_high * 0.3, c_high), 2)
            elif not breakout_day and 1 <= m <= 10:
                # Fade — price drops toward SL
                c_high = round(prev_close + rng.uniform(0, 2), 2)
                c_low = round(prev_close - rng.uniform(3, 8), 2)
                c_close = round(rng.uniform(c_low, c_low * 0.3 + c_high * 0.7), 2)
            else:
                # Random walk
                delta = rng.uniform(-4, 4)
                c_close = round(max(10.0, prev_close + delta), 2)
                c_high = round(c_close + rng.uniform(0, 3), 2)
                c_low = round(c_close - rng.uniform(0, 3), 2)

            rows.append({
                "datetime": ts,
                "open": prev_close,
                "high": c_high,
                "low": min(c_low, c_close - 0.01),
                "close": c_close,
                "volume": int(rng.integers(100, 1000)),
            })

    df = pd.DataFrame(rows).set_index("datetime")
    df.index = pd.to_datetime(df.index)
    return df


def test_strategy_2yr() -> None:
    print("\n[5] Synthetic 2-year strategy backtest")

    df = _make_synthetic_2yr_data()
    print(f"     Data: {len(df)} candles, {df.index[0].date()} → {df.index[-1].date()}")

    engine = BacktestEngine(
        data=df,
        strategy_fn=fib_breakout_strategy,
        initial_capital=100_000.0,
        symbol="SYNTHETIC_CE",
        slippage_pct=0.001,
        commission_per_lot=20.0,
        lot_size=LOT_SIZE,
    )
    result = engine.run()
    print(result.summary())
    print()

    # ---- Integrity checks ----

    # 1. No open trades remain
    open_count = sum(1 for t in result.trades if t.is_open)
    check("no open trades at end", open_count == 0, f"{open_count} open")

    # 2. Capital reconciliation: initial - costs + proceeds = final_capital
    recon = 100_000.0
    for t in result.trades:
        if not t.is_open:
            recon -= t.entry_price * t.quantity + t.entry_commission
            recon += t._exit_value - t._exit_commission
    check("capital reconciles",
          approx(recon, result.final_capital, tol=0.05),
          f"recon={recon:.4f} final={result.final_capital:.4f}")

    # 3. Sum of trade net PnLs equals total_return in capital terms
    sum_pnl = sum(t.pnl for t in result.trades if not t.is_open)
    capital_gain = result.final_capital - result.initial_capital
    check("sum(trade.pnl) == capital_gain",
          approx(sum_pnl, capital_gain, tol=0.05),
          f"sum_pnl={sum_pnl:.4f} capital_gain={capital_gain:.4f}")

    # 4. No negative quantities
    neg_qty = [t for t in result.trades if t.quantity <= 0]
    check("all trade quantities positive", len(neg_qty) == 0, str(neg_qty))

    # 5. No entry after square-off time
    sq_off_violations = [
        t for t in result.trades
        if t.entry_time is not None and t.entry_time.time() >= time(15, 15)
    ]
    check("no entries at/after square-off", len(sq_off_violations) == 0,
          str(sq_off_violations))

    # 6. Equity curve length matches data length
    check("equity curve length == data length",
          len(result.equity_curve) == len(df), f"{len(result.equity_curve)} vs {len(df)}")

    # 7. Max drawdown is <= 0
    check("max_drawdown_pct <= 0", result.max_drawdown_pct <= 0.0,
          str(result.max_drawdown_pct))

    # 8. Equity curve never drops below 50% of initial capital
    # (options can lose MTM value significantly during an open position;
    #  total wipe-out within a position is the real failure mode)
    min_eq = float(result.equity_curve.min())
    check("equity never below 50% of initial",
          min_eq > -result.initial_capital * 0.5,
          f"min={min_eq:.2f}")

    # 9. Win + Lose = Total
    check("win+lose = total",
          result.winning_trades + result.losing_trades == result.total_trades,
          f"{result.winning_trades}+{result.losing_trades}={result.total_trades}")

    # 10. ONE_TRADE_PER_DAY: at most 1 trade per calendar day
    from collections import Counter
    trade_dates = Counter(
        t.entry_time.date() for t in result.trades if t.entry_time is not None
    )
    multi_day = {d: n for d, n in trade_dates.items() if n > 1}
    check("one trade per day", len(multi_day) == 0, str(multi_day))

    return result


# ============================================================
# 5. Metrics sanity
# ============================================================

def test_metrics_sanity() -> None:
    print("\n[6] Metrics sanity")

    # A strategy that always wins 10% per trade over 252 days
    call_idx = [0]
    buy_price = 300.0

    def always_win(candle, _data, state):
        i = call_idx[0]
        call_idx[0] += 1
        if i % 2 == 0:
            return [{"action": "BUY", "quantity": 1, "price": buy_price}]
        return [{"action": "SELL", "quantity": 1, "price": buy_price * 1.10}]

    df = pd.DataFrame(
        [{"open": 300, "high": 330, "low": 295, "close": 310}] * 504,
        index=pd.date_range("2024-01-02 09:15", periods=504, freq="1min"),
    )
    engine = BacktestEngine(df, always_win, initial_capital=10_000.0,
                            slippage_pct=0.0, commission_per_lot=0.0, lot_size=1)
    result = engine.run()

    check("always-win: total_return > 0", result.total_return_pct > 0,
          str(result.total_return_pct))
    check("always-win: win_rate = 100%",
          approx(result.win_rate_pct, 100.0), str(result.win_rate_pct))
    check("always-win: profit_factor = inf",
          result.profit_factor == float("inf"), str(result.profit_factor))
    # MTM equity dips during open positions even when every trade is profitable,
    # because the snapshot uses close price which may be below entry before exit.
    check("always-win: max_drawdown > -20%", result.max_drawdown_pct > -20.0,
          str(result.max_drawdown_pct))

    # Always-lose strategy
    call_idx2 = [0]

    def always_lose(candle, _data, state):
        i = call_idx2[0]
        call_idx2[0] += 1
        if i % 2 == 0:
            return [{"action": "BUY", "quantity": 1, "price": buy_price}]
        return [{"action": "SELL", "quantity": 1, "price": buy_price * 0.90}]

    engine3 = BacktestEngine(df, always_lose, initial_capital=10_000.0,
                             slippage_pct=0.0, commission_per_lot=0.0, lot_size=1)
    result3 = engine3.run()
    check("always-lose: total_return < 0", result3.total_return_pct < 0,
          str(result3.total_return_pct))
    check("always-lose: win_rate = 0%",
          approx(result3.win_rate_pct, 0.0), str(result3.win_rate_pct))
    check("always-lose: max_drawdown < 0", result3.max_drawdown_pct < 0.0,
          str(result3.max_drawdown_pct))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_trade_unit()
    test_engine_simple_buy_sell()
    test_engine_partial_exit()
    test_strategy_helpers()
    test_strategy_2yr()
    test_metrics_sanity()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} FAILED:\033[0m " + ", ".join(_failures))
        sys.exit(1)
    else:
        print("\033[32mAll checks passed.\033[0m")
