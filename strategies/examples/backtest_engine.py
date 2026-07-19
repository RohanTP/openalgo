#!/usr/bin/env python
"""
Backtest engine for OpenAlgo strategies.

Loads historical OHLCV data (from OpenAlgo API or CSV files), runs an event-driven
strategy tick-by-tick, and produces a performance report.

Capital accounting is simple and correct:
  BUY:  capital -= fill_price * qty + commission
  SELL: capital += fill_price * qty - commission

Partial exits are fully supported — a single BUY can be followed by multiple
SELL signals (e.g., fib target booking) without creating phantom short trades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV. Expects a datetime/date/timestamp column plus open/high/low/close."""
    df = pd.read_csv(path)
    for col in ("datetime", "timestamp", "date", "time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def load_from_openalgo(
    client,
    symbol: str,
    exchange: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV data from the OpenAlgo history API."""
    df = client.history(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        raise ValueError(f"No data returned for {symbol} {start_date}→{end_date}")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol: str
    direction: str          # "LONG" or "SHORT"
    entry_time: datetime
    entry_price: float      # fill price (after slippage)
    quantity: int           # original entry quantity
    entry_commission: float = 0.0

    # mutable — updated by partial_exit()
    open_qty: int = field(init=False)
    exit_time: datetime | None = field(default=None, init=False)
    exit_price: float | None = field(default=None, init=False)   # weighted-avg
    exit_reason: str = field(default="", init=False)
    gross_pnl: float = field(default=0.0, init=False)  # before commissions
    pnl: float = field(default=0.0, init=False)         # net (after commissions)
    pnl_pct: float = field(default=0.0, init=False)

    # private accumulators
    _exit_value: float = field(default=0.0, init=False, repr=False)
    _exit_commission: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.open_qty = self.quantity

    @property
    def is_open(self) -> bool:
        return self.open_qty > 0

    def partial_exit(
        self,
        exit_time: datetime,
        price: float,
        qty: int,
        reason: str,
        commission: float = 0.0,
    ) -> None:
        qty = min(qty, self.open_qty)
        if qty <= 0:
            return
        self._exit_value += price * qty
        self._exit_commission += commission
        self.open_qty -= qty

        if self.open_qty <= 0:
            self.open_qty = 0
            self.exit_time = exit_time
            self.exit_price = round(self._exit_value / self.quantity, 4)
            self.exit_reason = reason
            entry_value = self.entry_price * self.quantity
            total_commission = self.entry_commission + self._exit_commission
            if self.direction == "LONG":
                self.gross_pnl = self._exit_value - entry_value
            else:
                self.gross_pnl = entry_value - self._exit_value
            self.pnl = self.gross_pnl - total_commission
            if entry_value > 0:
                self.pnl_pct = self.pnl / entry_value * 100


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    initial_capital: float
    final_capital: float

    # filled by _compute_metrics
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    def __post_init__(self) -> None:
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        closed = [t for t in self.trades if not t.is_open]
        self.total_trades = len(closed)
        if self.total_trades == 0:
            return

        pnls = [t.pnl for t in closed]  # net (after commissions)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate_pct = self.winning_trades / self.total_trades * 100
        self.avg_win = sum(wins) / len(wins) if wins else 0.0
        self.avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        self.total_return_pct = (
            (self.final_capital - self.initial_capital) / self.initial_capital * 100
        )

        if len(self.equity_curve) >= 2:
            start_dt = self.equity_curve.index[0]
            end_dt = self.equity_curve.index[-1]
            years = max((end_dt - start_dt).days / 365.25, 1 / 365.25)
            if self.initial_capital > 0:
                self.cagr_pct = (
                    (self.final_capital / self.initial_capital) ** (1 / years) - 1
                ) * 100

        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max * 100
        self.max_drawdown_pct = float(drawdown.min())

        daily_eq = self.equity_curve.resample("D").last().dropna()
        if len(daily_eq) >= 2:
            daily_ret = daily_eq.pct_change().dropna()
            mean_ret = float(daily_ret.mean())
            std_ret = float(daily_ret.std())
            downside = daily_ret[daily_ret < 0]
            downside_std = float(downside.std()) if len(downside) > 1 else 0.0
            ann = math.sqrt(252)
            self.sharpe_ratio = mean_ret / std_ret * ann if std_ret > 0 else 0.0
            self.sortino_ratio = (
                mean_ret / downside_std * ann if downside_std > 0 else 0.0
            )

        if self.max_drawdown_pct < 0:
            self.calmar_ratio = self.cagr_pct / abs(self.max_drawdown_pct)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  BACKTEST RESULTS",
            "=" * 55,
            f"  Initial capital  : ₹{self.initial_capital:,.2f}",
            f"  Final capital    : ₹{self.final_capital:,.2f}",
            f"  Total return     : {self.total_return_pct:+.2f}%",
            f"  CAGR             : {self.cagr_pct:+.2f}%",
            f"  Max drawdown     : {self.max_drawdown_pct:.2f}%",
            f"  Sharpe ratio     : {self.sharpe_ratio:.3f}",
            f"  Sortino ratio    : {self.sortino_ratio:.3f}",
            f"  Calmar ratio     : {self.calmar_ratio:.3f}",
            "-" * 55,
            f"  Total trades     : {self.total_trades}",
            f"  Win rate         : {self.win_rate_pct:.1f}%",
            f"  Profit factor    : {self.profit_factor:.2f}",
            f"  Avg win (net)    : ₹{self.avg_win:+,.2f}",
            f"  Avg loss (net)   : ₹{self.avg_loss:+,.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def trades_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            rows.append({
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "gross_pnl": round(t.gross_pnl, 2),
                "net_pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Strategy callback: (current_candle, history_df, state_dict) -> list of signal dicts
# Signal dict keys:
#   "action"   : "BUY" or "SELL" (required)
#   "quantity" : int — number of units (default: lot_size)
#   "price"    : float — desired fill price (default: candle close)
#   "reason"   : str — label for trade log (optional)
StrategyFn = Callable[[pd.Series, pd.DataFrame, dict], list[dict]]


class BacktestEngine:
    """
    Event-driven candle-by-candle backtester with partial-exit support.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV dataframe with a DatetimeIndex (1-min or any interval).
    strategy_fn : StrategyFn
        Called on each candle; returns a list of order signals.
    initial_capital : float
        Starting capital in rupees.
    symbol : str
        Instrument label used in trade records.
    slippage_pct : float
        Slippage fraction applied to every fill (0.001 = 0.1%).
    commission_per_lot : float
        Flat commission charged per lot (entry AND exit separately).
    lot_size : int
        Units per lot — used to calculate commission.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_fn: StrategyFn,
        initial_capital: float = 100_000.0,
        symbol: str = "INSTRUMENT",
        slippage_pct: float = 0.001,
        commission_per_lot: float = 20.0,
        lot_size: int = 1,
    ) -> None:
        self.data = data.copy()
        self.strategy_fn = strategy_fn
        self.initial_capital = initial_capital
        self.symbol = symbol
        self.slippage_pct = slippage_pct
        self.commission_per_lot = commission_per_lot
        self.lot_size = lot_size

    def _fill_price(self, price: float, action: str) -> float:
        slip = price * self.slippage_pct
        return price + slip if action == "BUY" else price - slip

    def _commission(self, qty: int) -> float:
        lots = max(qty // self.lot_size, 1)
        return lots * self.commission_per_lot

    def run(self) -> BacktestResult:
        capital = self.initial_capital
        open_trade: Trade | None = None
        trades: list[Trade] = []
        equity_series: dict = {}
        state: dict = {}

        for i, (ts, candle) in enumerate(self.data.iterrows()):
            # Pass only the index position — strategies that need history
            # should accumulate rows in `state` to avoid O(n²) slicing.
            state["_i"] = i
            state["_data"] = self.data
            signals = self.strategy_fn(candle, self.data, state)

            for sig in signals:
                action = sig.get("action", "").upper()
                qty = int(sig.get("quantity", self.lot_size))
                raw_price = float(sig.get("price", candle["close"]))
                fill = self._fill_price(raw_price, action)
                comm = self._commission(qty)

                if action == "BUY" and open_trade is None:
                    cost = fill * qty + comm
                    if capital >= cost:
                        capital -= cost
                        open_trade = Trade(
                            symbol=self.symbol,
                            direction="LONG",
                            entry_time=ts,
                            entry_price=fill,
                            quantity=qty,
                            entry_commission=comm,
                        )
                        trades.append(open_trade)

                elif action == "SELL" and open_trade is not None and open_trade.is_open:
                    qty = min(qty, open_trade.open_qty)
                    proceeds = fill * qty - comm
                    capital += proceeds
                    open_trade.partial_exit(ts, fill, qty, sig.get("reason", "signal"), comm)
                    if not open_trade.is_open:
                        open_trade = None

            # Mark-to-market equity snapshot
            if open_trade is not None and open_trade.is_open:
                current_price = float(candle["close"])
                unrealised = (current_price - open_trade.entry_price) * open_trade.open_qty
                equity_series[ts] = capital + unrealised
            else:
                equity_series[ts] = capital

        # Force-close any remaining open position at last close
        if open_trade is not None and open_trade.is_open:
            last_ts = self.data.index[-1]
            last_close = float(self.data["close"].iloc[-1])
            comm = self._commission(open_trade.open_qty)
            capital += last_close * open_trade.open_qty - comm
            open_trade.partial_exit(
                last_ts, last_close, open_trade.open_qty, "end of data", comm
            )

        equity_curve = pd.Series(equity_series).sort_index()

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            final_capital=capital,
        )
