#!/usr/bin/env python
"""
Fibonacci option premium breakout strategy.

Default idea:
- At SELECT_TIME, pick ITM CE/PE option premiums between PREMIUM_MIN and PREMIUM_MAX.
- Use the first regular 1-minute candle as the reference candle.
- Buy the option whose premium breaks above its reference candle high.
- For small candles, SL is the candle low. For big candles, SL is a fib retracement.
- Book partial quantities at fib extension targets and trail the balance with Heikin Ashi.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any

import sys
import pandas as pd  # type: ignore[reportMissingImports]
from openalgo import api  # type: ignore[reportMissingImports]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_time(name: str, default: str) -> dtime:
    value = os.getenv(name, default)
    return datetime.strptime(value, "%H:%M").time()


def env_float_list(name: str, default: str) -> list[float]:
    value = os.getenv(name, default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def env_int_list(name: str, default: str) -> list[int]:
    value = os.getenv(name, default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


API_KEY = os.getenv("OPENALGO_API_KEY")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL") or (
    f"ws://{os.getenv('WEBSOCKET_HOST', '127.0.0.1')}:{os.getenv('WEBSOCKET_PORT', '8765')}"
)

if not API_KEY and (__name__ == "__main__" or (len(sys.argv) > 1 and "live" in sys.argv)):
    print("Error: OPENALGO_API_KEY environment variable not set")
    raise SystemExit(1)


STRATEGY_ID = os.getenv("STRATEGY_ID", "fib_premium_breakout")
STRATEGY_NAME = os.getenv("STRATEGY_NAME", "Fib Premium Breakout")
UNDERLYING = os.getenv("UNDERLYING", "NIFTY").upper()
INDEX_EXCHANGE = os.getenv("INDEX_EXCHANGE", "NSE_INDEX").upper()
DERIVATIVE_EXCHANGE = os.getenv("DERIVATIVE_EXCHANGE", "NFO").upper()
PRODUCT = os.getenv("PRODUCT", "MIS").upper()

SELECT_TIME = env_time("SELECT_TIME", "09:07")
REFERENCE_CANDLE_TIME = env_time("REFERENCE_CANDLE_TIME", "09:15")
TRADE_END_TIME = env_time("TRADE_END_TIME", "10:30")
SQUARE_OFF_TIME = env_time("SQUARE_OFF_TIME", "15:15")

PREMIUM_MIN = env_float("PREMIUM_MIN", 300.0)
PREMIUM_MAX = env_float("PREMIUM_MAX", 400.0)
STRIKE_COUNT = env_int("STRIKE_COUNT", 30)

TOTAL_LOTS = env_int("TOTAL_LOTS", 4)
LOT_SIZE = env_int("LOT_SIZE", 75)
BIG_CANDLE_THRESHOLD = env_float("BIG_CANDLE_THRESHOLD", 40.0)
BIG_CANDLE_SL_FIB = env_float("BIG_CANDLE_SL_FIB", 0.55)
ENTRY_BUFFER = env_float("ENTRY_BUFFER", 0.05)
SL_BUFFER = env_float("SL_BUFFER", 0.05)
ENTRY_TRIGGER_BUFFER = env_float("ENTRY_TRIGGER_BUFFER", 5.0)

FIB_TARGETS = env_float_list("FIB_TARGETS", "1.272,1.618,2.0")
BOOKING_PCTS = env_int_list("BOOKING_PCTS", "25,25,50")
ALLOW_BOTH_CE_PE = env_bool("ALLOW_BOTH_CE_PE", True)
ONE_TRADE_PER_DAY = env_bool("ONE_TRADE_PER_DAY", True)
POLL_SECONDS = env_int("POLL_SECONDS", 3)

# HA trail activation: wait for target N hit and/or N minutes after entry (0 = disabled)
HA_TRAIL_START_TARGET = env_int("HA_TRAIL_START_TARGET", 0)
HA_TRAIL_DELAY_MINUTES = env_int("HA_TRAIL_DELAY_MINUTES", 0)

# Capital / order robustness
MAX_CAPITAL_PER_DAY = env_float("MAX_CAPITAL_PER_DAY", 150000.0)
ORDER_MAX_RETRIES = env_int("ORDER_MAX_RETRIES", 3)
ORDER_RETRY_DELAY_SEC = env_float("ORDER_RETRY_DELAY_SEC", 0.2)
# Off by default: openposition is account-wide and can confuse this strategy
# with manual / other-strategy positions on the same symbol.
BROKER_RECONCILE = env_bool("BROKER_RECONCILE", False)

_LOG_LEVEL = os.getenv("STRATEGY_LOG_LEVEL", "INFO").upper()  # DISABLED | INFO | DEBUG

# Running capital ledger for the trading day (premium notional).
remaining_capital = MAX_CAPITAL_PER_DAY


def log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVEL == "DISABLED":
        return
    if _LOG_LEVEL == "INFO" and level == "DEBUG":
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


client = None
if API_KEY:
    client = api(api_key=API_KEY, host=HOST, ws_url=WS_URL)


@dataclass
class Candidate:
    symbol: str
    option_type: str
    premium: float
    label: str
    strike: float
    lotsize: int
    expiry: str | None = None


@dataclass
class TradePlan:
    candidate: Candidate
    entry: float
    stop_loss: float
    targets: list[float]
    remaining_lot: int
    target_lots: list[int]
    booked_targets: set[int] = field(default_factory=set)
    active_stop_loss: float = 0.0
    entered: bool = False
    entry_price: float | None = None
    entry_time: datetime | None = None
    plan_id: str | None = None
    ha_trail_active: bool = False
    skip_reason: str | None = None


@dataclass
class RuntimeSession:
    trading_day: str
    candidates: list[Candidate] = field(default_factory=list)
    plans: list[TradePlan] = field(default_factory=list)


_session: RuntimeSession | None = None


def reset_day_capital(max_capital: float | None = None) -> None:
    """Reset remaining capital for a new trading day."""
    global remaining_capital, MAX_CAPITAL_PER_DAY
    if max_capital is not None:
        MAX_CAPITAL_PER_DAY = float(max_capital)
    remaining_capital = float(MAX_CAPITAL_PER_DAY)


def capital_gate_enabled() -> bool:
    return MAX_CAPITAL_PER_DAY > 0


def can_afford(cost: float) -> bool:
    if not capital_gate_enabled():
        return True
    return cost <= remaining_capital + 1e-9


def debit_capital(amount: float) -> None:
    global remaining_capital
    if not capital_gate_enabled():
        return
    remaining_capital = round(remaining_capital - amount, 2)


def credit_capital(amount: float) -> None:
    global remaining_capital
    if not capital_gate_enabled():
        return
    remaining_capital = round(remaining_capital + amount, 2)


def _order_ok(response: Any) -> bool:
    return isinstance(response, dict) and response.get("status") == "success"


def _order_id(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    for key in ("orderid", "order_id", "id"):
        value = response.get(key)
        if value not in (None, ""):
            return str(value)
    data = response.get("data")
    if isinstance(data, dict):
        for key in ("orderid", "order_id", "id"):
            value = data.get(key)
            if value not in (None, ""):
                return str(value)
    return None


class OrderExecutor:
    def place_order(
        self,
        plan: TradePlan,
        action: str,
        quantity: int,
        position_size: int,
        price: float | None = None,
        ts: datetime | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class LiveOrderExecutor(OrderExecutor):
    def place_order(
        self,
        plan: TradePlan,
        action: str,
        quantity: int,
        position_size: int,
        price: float | None = None,
        ts: datetime | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if client is None:
            return {
                "ok": False,
                "orderid": None,
                "fill_price": None,
                "response": None,
                "message": "API client is not initialized.",
            }

        last_response: Any = None
        last_message = "unknown error"
        attempts = max(1, ORDER_MAX_RETRIES)

        for attempt in range(1, attempts + 1):
            try:
                response = client.placesmartorder(
                    strategy=STRATEGY_NAME,
                    symbol=plan.candidate.symbol,
                    action=action,
                    exchange=DERIVATIVE_EXCHANGE,
                    price_type="MARKET",
                    product=PRODUCT,
                    quantity=quantity,
                    position_size=position_size,
                )
                last_response = response
                log(
                    f"{action} {quantity} {plan.candidate.symbol} "
                    f"attempt={attempt}/{attempts} response: {response}",
                    "DEBUG",
                )
                if _order_ok(response):
                    orderid = _order_id(response)
                    # Simple: ACK = done. Use our strategy price for capital/PnL.
                    # Exact broker fill can be looked up later via orderid if needed.
                    if price is None:
                        return {
                            "ok": False,
                            "orderid": orderid,
                            "fill_price": None,
                            "response": response,
                            "message": "order accepted but strategy price missing",
                        }
                    return {
                        "ok": True,
                        "orderid": orderid,
                        "fill_price": float(price),
                        "response": response,
                        "message": None,
                    }
                last_message = str(
                    response.get("message") if isinstance(response, dict) else response
                )
            except Exception as exc:
                last_message = str(exc)
                log(
                    f"{action} {quantity} {plan.candidate.symbol} "
                    f"attempt={attempt}/{attempts} error: {exc}",
                    "INFO",
                )

            if attempt < attempts:
                time.sleep(ORDER_RETRY_DELAY_SEC)

        log(
            f"ORDER FAILED after {attempts} attempts: {action} {quantity} "
            f"{plan.candidate.symbol} reason={reason} message={last_message}",
            "INFO",
        )
        return {
            "ok": False,
            "orderid": None,
            "fill_price": None,
            "response": last_response,
            "message": last_message,
        }


class BacktestOrderExecutor(OrderExecutor):
    def __init__(self) -> None:
        self.orders: list[dict] = []

    def place_order(
        self,
        plan: TradePlan,
        action: str,
        quantity: int,
        position_size: int,
        price: float | None = None,
        ts: datetime | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if price is None:
            raise ValueError(
                f"price is required for backtest {action} on {plan.candidate.symbol}"
            )
        if ts is None:
            raise ValueError(
                f"ts is required for backtest {action} on {plan.candidate.symbol}"
            )

        if action == "BUY":
            if plan.entry_price is None:
                plan.entry_price = price
            if plan.entry_time is None:
                plan.entry_time = ts

        pnl = 0.0
        if action == "SELL":
            if plan.entry_price is None:
                raise ValueError(
                    f"entry_price missing for SELL on {plan.candidate.symbol}"
                )
            pnl = round((price - plan.entry_price) * quantity, 2)

        lotsize = plan.candidate.lotsize
        # position_size is the post-order open qty (placesmartorder contract).
        order_row = {
            "plan_id": getattr(plan, "plan_id", None),
            "symbol": plan.candidate.symbol,
            "option_type": plan.candidate.option_type,
            "strike": plan.candidate.strike,
            "expiry": getattr(plan.candidate, "expiry", None),
            "action": action,
            "quantity": quantity,
            "lots": quantity // lotsize,
            "price": price,
            "position_size": position_size,
            "reason": reason,
            "remaining_lots": position_size // lotsize,
            "remaining_quantity": position_size,
            "entry_price": plan.entry_price,
            "realized_pnl": pnl,
            "timestamp": ts,
        }
        self.orders.append(order_row)
        return {
            "ok": True,
            "orderid": f"bt_{len(self.orders)}",
            "fill_price": price,
            "response": order_row,
            "message": None,
        }


# ---------------------------------------------------------------------------
# Runtime persistence (candidates + plans)
# ---------------------------------------------------------------------------

def runtime_state_path() -> Path:
    return Path("strategies") / "runtime" / f"{STRATEGY_ID}_state.json"


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return asdict(candidate)


def candidate_from_dict(data: dict[str, Any]) -> Candidate:
    return Candidate(
        symbol=str(data["symbol"]),
        option_type=str(data["option_type"]),
        premium=float(data["premium"]),
        label=str(data.get("label", "")),
        strike=float(data["strike"]),
        lotsize=int(data.get("lotsize") or LOT_SIZE),
        expiry=data.get("expiry"),
    )


def plan_to_dict(plan: TradePlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "candidate": candidate_to_dict(plan.candidate),
        "entry": plan.entry,
        "stop_loss": plan.stop_loss,
        "targets": list(plan.targets),
        "remaining_lot": plan.remaining_lot,
        "target_lots": list(plan.target_lots),
        "booked_targets": sorted(plan.booked_targets),
        "active_stop_loss": plan.active_stop_loss,
        "entered": plan.entered,
        "entry_price": plan.entry_price,
        "entry_time": plan.entry_time.isoformat() if plan.entry_time else None,
        "ha_trail_active": plan.ha_trail_active,
        "skip_reason": plan.skip_reason,
    }


def plan_from_dict(data: dict[str, Any]) -> TradePlan:
    entry_time = None
    if data.get("entry_time"):
        entry_time = datetime.fromisoformat(str(data["entry_time"]))
    return TradePlan(
        candidate=candidate_from_dict(data["candidate"]),
        entry=float(data["entry"]),
        stop_loss=float(data["stop_loss"]),
        targets=[float(x) for x in data.get("targets", [])],
        remaining_lot=int(data["remaining_lot"]),
        target_lots=[int(x) for x in data.get("target_lots", [])],
        booked_targets=set(int(x) for x in data.get("booked_targets", [])),
        active_stop_loss=float(data.get("active_stop_loss") or data["stop_loss"]),
        entered=bool(data.get("entered", False)),
        entry_price=float(data["entry_price"]) if data.get("entry_price") is not None else None,
        entry_time=entry_time,
        plan_id=data.get("plan_id"),
        ha_trail_active=bool(data.get("ha_trail_active", False)),
        skip_reason=data.get("skip_reason"),
    )


def save_runtime_state(session: RuntimeSession | None = None) -> None:
    session = session or _session
    if session is None:
        return
    path = runtime_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trading_day": session.trading_day,
        "remaining_capital": remaining_capital,
        "max_capital_per_day": MAX_CAPITAL_PER_DAY,
        "candidates": [candidate_to_dict(c) for c in session.candidates],
        "plans": [plan_to_dict(p) for p in session.plans],
        "updated_at": datetime.now().isoformat(),
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_runtime_state() -> dict[str, Any] | None:
    path = runtime_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"Failed to load runtime state {path}: {exc}", "INFO")
        return None


def persist_runtime() -> None:
    save_runtime_state(_session)


def parse_expiry_date(expiry: str) -> datetime:
    for fmt in ("%d-%b-%y", "%d%b%y", "%d-%B-%y", "%d%B%y"):
        try:
            return datetime.strptime(expiry.upper().strip(), fmt)
        except ValueError:
            continue
    return datetime.max


def get_current_weekly_expiry() -> tuple[str, bool]:
    response = client.expiry(
        symbol=UNDERLYING,
        exchange=DERIVATIVE_EXCHANGE,
        instrumenttype="options",
    )
    if response.get("status") != "success":
        raise RuntimeError(f"Failed to fetch expiry dates: {response.get('message')}")

    expiries = response.get("data", [])
    if not expiries:
        raise RuntimeError(f"No option expiries found for {UNDERLYING}")

    sorted_expiries = sorted(expiries, key=parse_expiry_date)
    expiry_raw = sorted_expiries[0]
    # optionchain API expects DDMMMYY (e.g. 28JUL26), expiry API returns DD-MMM-YY
    expiry = expiry_raw.replace("-", "")
    expiry_date = parse_expiry_date(expiry_raw).date()
    return expiry, expiry_date == datetime.now().date()


def is_itm_option(option_data: dict) -> bool:
    return str(option_data.get("label", "")).upper().startswith("ITM")


def option_ltp(option_data: dict) -> float | None:
    for key in ("ltp", "LTP", "last_price", "close"):
        value = option_data.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def select_option_candidates() -> list[Candidate]:
    expiry, is_expiry_day = get_current_weekly_expiry()
    chain = client.optionchain(
        underlying=UNDERLYING,
        exchange=INDEX_EXCHANGE,
        expiry_date=expiry,
        strike_count=STRIKE_COUNT,
    )
    if chain.get("status") != "success":
        raise RuntimeError(f"Failed to fetch option chain: {chain.get('message')}")

    candidates: list[Candidate] = []
    for row in chain.get("chain", []):
        strike = float(row.get("strike", 0))
        for option_type in ("CE", "PE"):
            option_data = row.get(option_type.lower()) or {}
            premium = option_ltp(option_data)
            symbol = option_data.get("symbol")
            if not symbol or premium is None:
                continue
            if not is_itm_option(option_data):
                continue
            if PREMIUM_MIN <= premium <= PREMIUM_MAX:
                candidates.append(
                    Candidate(
                        symbol=symbol,
                        option_type=option_type,
                        premium=premium,
                        label=str(option_data.get("label", "")),
                        strike=strike,
                        lotsize=int(option_data.get("lotsize") or LOT_SIZE),
                        expiry=expiry,
                    )
                )

    if not candidates:
        raise RuntimeError(
            f"No ITM {UNDERLYING} options found between premiums {PREMIUM_MIN}-{PREMIUM_MAX}"
        )

    selected: list[Candidate] = []
    for option_type in ("CE", "PE"):
        typed = [candidate for candidate in candidates if candidate.option_type == option_type]
        if not typed:
            continue
        selected.append(min(typed, key=lambda candidate: candidate.premium))

    if not ALLOW_BOTH_CE_PE and selected:
        selected = [min(selected, key=lambda candidate: abs(candidate.premium))]

    log(f"Selected expiry: {expiry} | Expiry day: {is_expiry_day}", "INFO")
    for candidate in selected:
        log(
            f"Selected {candidate.option_type}: {candidate.symbol} "
            f"premium={candidate.premium} label={candidate.label} strike={candidate.strike}",
            "INFO",
        )
    return selected


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        for column in ("datetime", "timestamp", "time", "date"):
            if column in normalized.columns:
                normalized[column] = pd.to_datetime(normalized[column])
                normalized = normalized.set_index(column)
                break

    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    return normalized


def fetch_today_history(symbol: str) -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%d")
    df = client.history(
        symbol=symbol,
        exchange=DERIVATIVE_EXCHANGE,
        interval="1m",
        start_date=today,
        end_date=today,
    )
    if isinstance(df, dict):
        raise RuntimeError(
            f"History API error for {symbol}: {df.get('message', df)} "
            f"| exchange={DERIVATIVE_EXCHANGE} interval=1m "
            f"start={today} end={today} status={df.get('status')} "
            f"keys={list(df.keys())}"
        )
    df = normalize_history(df)
    required_columns = {"open", "high", "low", "close"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{symbol} history missing columns: {missing_columns}")
    return df


def fetch_ltp(symbol: str) -> float:
    response = client.quotes(
        symbol=symbol,
        exchange=DERIVATIVE_EXCHANGE,
    )

    if response.get("status") != "success":
        raise RuntimeError(
            f"Failed to fetch LTP for {symbol}: {response.get('message')}"
        )

    data = response.get("data", {})

    for key in ("ltp", "LTP", "last_price", "lastPrice", "close"):
        value = data.get(key)
        if value not in (None, ""):
            return float(value)

    raise KeyError(f"LTP not found in quote response for {symbol}")


def fetch_open_quantity(symbol: str) -> int | None:
    """Return open qty for symbol, or None if lookup failed."""
    if client is None:
        return None
    try:
        response = client.openposition(
            strategy=STRATEGY_NAME,
            symbol=symbol,
            exchange=DERIVATIVE_EXCHANGE,
            product=PRODUCT,
        )
    except Exception as exc:
        log(f"openposition error for {symbol}: {exc}", "INFO")
        return None

    if not isinstance(response, dict):
        return None
    if response.get("status") not in (None, "success") and "quantity" not in response:
        return None
    qty = response.get("quantity", 0)
    try:
        return int(float(qty))
    except (TypeError, ValueError):
        return None


def completed_candles(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame()
    return df.iloc[:-1].copy()


def reference_candle(df: pd.DataFrame) -> pd.Series | None:
    candles = completed_candles(df)
    if candles.empty:
        return None

    first_date = candles.index[0].date()
    day_candles = candles[candles.index.date == first_date]
    day_candles = day_candles[day_candles.index.time >= REFERENCE_CANDLE_TIME]
    if day_candles.empty:
        return None
    return day_candles.iloc[0]


def build_targets(low: float, high: float) -> list[float]:
    candle_range = high - low
    return [round(low + candle_range * level, 2) for level in FIB_TARGETS]


def build_stop_loss(low: float, high: float) -> float:
    candle_range = high - low
    if candle_range > BIG_CANDLE_THRESHOLD:
        return round(high - candle_range * BIG_CANDLE_SL_FIB, 2)
    return round(low - SL_BUFFER, 2)


def split_target_lots(total_lots: int) -> list[int]:
    lots: list[int] = []
    allocated = 0

    for index, pct in enumerate(BOOKING_PCTS):
        if index == len(BOOKING_PCTS) - 1:
            lot_count = total_lots - allocated
        else:
            lot_count = round(total_lots * pct / 100)
            lot_count = max(0, min(lot_count, total_lots - allocated))

        lots.append(lot_count)
        allocated += lot_count

    return lots


def create_trade_plans(
    candidates: list[Candidate],
    existing_plans: list[TradePlan] | None = None,
) -> list[TradePlan]:
    """Build plans for candidates; keep existing plans and only fill missing symbols."""
    plans: list[TradePlan] = list(existing_plans or [])
    planned_symbols = {plan.candidate.symbol for plan in plans}

    for candidate in candidates:
        if candidate.symbol in planned_symbols:
            continue
        try:
            df = fetch_today_history(candidate.symbol)
        except RuntimeError as e:
            log(
                f"Waiting for history data for {candidate.symbol} | "
                f"option_type={candidate.option_type} strike={candidate.strike} "
                f"exchange={DERIVATIVE_EXCHANGE} | {e}",
                "INFO",
            )
            continue
        candle = reference_candle(df)
        if candle is None:
            log(f"Waiting for reference candle for {candidate.symbol}", "INFO")
            continue

        high = round(float(candle["high"]), 2)
        low = round(float(candle["low"]), 2)
        candle_range = round(high - low, 2)
        target_lots = split_target_lots(TOTAL_LOTS)

        plan = TradePlan(
            candidate=candidate,
            entry=round(high + ENTRY_BUFFER, 2),
            stop_loss=build_stop_loss(low, high),
            targets=build_targets(low, high),
            remaining_lot=TOTAL_LOTS,
            target_lots=target_lots,
        )
        plan.active_stop_loss = plan.stop_loss
        plan.plan_id = (
            f"plan_{datetime.now().strftime('%Y-%m-%d')}_{candidate.symbol}_"
            f"{candidate.option_type}_{candidate.strike}"
        )
        plans.append(plan)
        planned_symbols.add(candidate.symbol)

        log(
            f"{candidate.symbol} reference high={high} low={low} range={candle_range} "
            f"entry={plan.entry} sl={plan.stop_loss} targets={plan.targets}",
            "INFO",
        )

    return plans


def plans_ready_for_all_candidates(
    candidates: list[Candidate],
    plans: list[TradePlan],
) -> bool:
    if not candidates:
        return False
    planned = {plan.candidate.symbol for plan in plans}
    return all(candidate.symbol in planned for candidate in candidates)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    candles = df[["open", "high", "low", "close"]].astype(float).copy()
    ha = pd.DataFrame(index=candles.index)
    ha["close"] = candles[["open", "high", "low", "close"]].mean(axis=1)
    ha["open"] = 0.0
    if candles.empty:
        return ha

    ha.iloc[0, ha.columns.get_loc("open")] = (candles["open"].iloc[0] + candles["close"].iloc[0]) / 2
    for i in range(1, len(candles)):
        ha.iloc[i, ha.columns.get_loc("open")] = (ha["open"].iloc[i - 1] + ha["close"].iloc[i - 1]) / 2

    ha["high"] = pd.concat([candles["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([candles["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


def enter_trade(
    plan: TradePlan,
    order_executor: OrderExecutor,
    price: float,
    ts: datetime,
) -> bool:
    if plan.entered:
        return True

    quantity = plan.remaining_lot * plan.candidate.lotsize
    cost = float(price) * quantity

    if not can_afford(cost):
        plan.skip_reason = "NO_CAPITAL"
        log(
            f"Skip BUY {plan.candidate.symbol} qty={quantity} cost={cost:.2f} "
            f"> remaining_capital={remaining_capital:.2f}",
            "INFO",
        )
        return False

    result = order_executor.place_order(
        plan,
        action="BUY",
        quantity=quantity,
        position_size=quantity,
        price=price,
        ts=ts,
        reason="ENTRY",
    )
    if not result.get("ok"):
        log(
            f"BUY not confirmed for {plan.candidate.symbol}: {result.get('message')}",
            "INFO",
        )
        return False

    fill_price = float(result["fill_price"])
    plan.entered = True
    plan.entry_price = fill_price
    plan.entry_time = ts
    plan.skip_reason = None
    debit_capital(fill_price * quantity)
    persist_runtime()
    log(
        f"Entered BUY {plan.candidate.symbol} qty={quantity} lots={plan.remaining_lot} "
        f"price={fill_price} orderid={result.get('orderid')} sl={plan.active_stop_loss} "
        f"targets={plan.targets} remaining_capital={remaining_capital:.2f}",
        "INFO",
    )
    return True


def exit_quantity(
    plan: TradePlan,
    lot: int,
    reason: str,
    order_executor: OrderExecutor,
    price: float,
    ts: datetime,
) -> bool:
    if lot <= 0 or plan.remaining_lot <= 0:
        return False

    lot = min(lot, plan.remaining_lot)
    quantity = lot * plan.candidate.lotsize
    new_remaining_lot = plan.remaining_lot - lot
    remaining_qty = new_remaining_lot * plan.candidate.lotsize

    result = order_executor.place_order(
        plan,
        action="SELL",
        quantity=quantity,
        position_size=remaining_qty,
        price=price,
        ts=ts,
        reason=reason,
    )
    if not result.get("ok"):
        log(
            f"SELL not confirmed for {plan.candidate.symbol} qty={quantity}: {result.get('message')}",
            "INFO",
        )
        return False

    fill_price = float(result["fill_price"])
    plan.remaining_lot = new_remaining_lot
    credit_capital(fill_price * quantity)
    persist_runtime()
    log(
        f"Exited {quantity}(lot {lot}) {plan.candidate.symbol} due to {reason}. "
        f"price={fill_price} orderid={result.get('orderid')} "
        f"Remaining={remaining_qty}(lot {plan.remaining_lot}) "
        f"remaining_capital={remaining_capital:.2f}",
        "INFO",
    )
    return True


def update_heikin_ashi_stop(plan: TradePlan, df: pd.DataFrame, ts: datetime) -> None:
    if plan.entry_time is None:
        log(
            f"ERROR: {plan.candidate.symbol} missing entry_time while managing position; "
            f"refusing HA trail update",
            "INFO",
        )
        return

    if not plan.ha_trail_active:
        target_hit = len(plan.booked_targets) >= HA_TRAIL_START_TARGET
        delay_elapsed = (ts - plan.entry_time).total_seconds() / 60 >= HA_TRAIL_DELAY_MINUTES
        if target_hit or delay_elapsed:
            plan.ha_trail_active = True
            log(
                f"{plan.candidate.symbol} HA trail activated at {ts} | "
                f"target_hit={target_hit} delay_elapsed={delay_elapsed}",
                "INFO",
            )
        else:
            return

    candles = completed_candles(df)
    if len(candles) < 2:
        return

    ha = heikin_ashi(candles)
    previous_ha = ha.iloc[-2]
    trailed_sl = round(float(previous_ha["low"]) - SL_BUFFER, 2)
    old_sl = plan.active_stop_loss
    plan.active_stop_loss = max(plan.active_stop_loss, trailed_sl)
    if plan.active_stop_loss != old_sl:
        log(f"{plan.candidate.symbol} HA trail SL updated {old_sl} -> {plan.active_stop_loss}", "INFO")
        persist_runtime()


def handle_plan(
    plan: TradePlan,
    candle: pd.Series,
    ltp: float,
    ts: datetime,
    order_executor: OrderExecutor,
    history_df: pd.DataFrame | None = None,
    is_backtest: bool = False,
    trade_end_time: dtime | None = None,
) -> bool:
    if not plan.entered:
        past_entry_window = trade_end_time is not None and ts.time() > trade_end_time
        if not past_entry_window:
            use_buffer = is_backtest or env_bool("LIVE_ENTRY_BUFFER_ENABLED", True)

            should_enter = False
            if use_buffer:
                if plan.entry <= ltp <= plan.entry + ENTRY_TRIGGER_BUFFER:
                    should_enter = True
            else:
                if ltp >= plan.entry:
                    should_enter = True

            if should_enter:
                enter_trade(plan, order_executor, price=plan.entry, ts=ts)

    if plan.entered:
        if history_df is not None:
            update_heikin_ashi_stop(plan, history_df, ts=ts)

        latest_close = float(candle["close"])

        # SL checked first — if triggered, skip targets on this candle
        if latest_close <= plan.active_stop_loss:
            exit_quantity(
                plan,
                plan.remaining_lot,
                reason=f"SL close beyond {plan.active_stop_loss}",
                order_executor=order_executor,
                price=plan.active_stop_loss,
                ts=ts,
            )
        else:
            for index, target in enumerate(plan.targets):
                if index in plan.booked_targets:
                    continue
                if plan.remaining_lot <= 0:
                    break
                if ltp >= target:
                    lot = plan.target_lots[index] if index < len(plan.target_lots) else 0
                    if lot <= 0:
                        plan.booked_targets.add(index)
                        continue
                    if exit_quantity(
                        plan,
                        lot,
                        reason=f"fib target {target}",
                        order_executor=order_executor,
                        price=target,
                        ts=ts,
                    ):
                        plan.booked_targets.add(index)
                        persist_runtime()

        if plan.remaining_lot > 0:
            log(
                f"{plan.candidate.symbol} close={latest_close:.2f} "
                f"active_sl={plan.active_stop_loss:.2f} remaining={plan.remaining_lot}",
                "DEBUG",
            )

    return plan.entered


def monitor_plan(plan: TradePlan, order_executor: OrderExecutor) -> bool:
    now = datetime.now()
    try:
        df = fetch_today_history(plan.candidate.symbol)
    except RuntimeError as e:
        log(
            f"Transient history error for {plan.candidate.symbol}, skipping poll | "
            f"option_type={plan.candidate.option_type} strike={plan.candidate.strike} "
            f"entered={plan.entered} remaining_lot={plan.remaining_lot} "
            f"entry={plan.entry} active_sl={plan.active_stop_loss} "
            f"plan_id={plan.plan_id} poll_ts={now.isoformat(timespec='seconds')} | {e}",
            "INFO",
        )
        return plan.entered

    candles = completed_candles(df)
    if candles.empty:
        log(
            f"No completed candles yet for {plan.candidate.symbol} | "
            f"rows={len(df)} poll_ts={now.isoformat(timespec='seconds')}",
            "DEBUG",
        )
        return plan.entered

    latest = candles.iloc[-1]

    try:
        ltp = fetch_ltp(plan.candidate.symbol)
    except Exception as e:
        log(
            f"Transient LTP error for {plan.candidate.symbol}, skipping poll | "
            f"option_type={plan.candidate.option_type} strike={plan.candidate.strike} "
            f"entered={plan.entered} remaining_lot={plan.remaining_lot} "
            f"exchange={DERIVATIVE_EXCHANGE} "
            f"poll_ts={now.isoformat(timespec='seconds')} | {e}",
            "INFO",
        )
        return plan.entered

    return handle_plan(
        plan=plan,
        candle=latest,
        ltp=ltp,
        ts=latest.name if hasattr(latest, "name") else datetime.now(),
        order_executor=order_executor,
        history_df=df,
        is_backtest=False,
        trade_end_time=TRADE_END_TIME,
    )


def wait_until(target_time: dtime) -> None:
    while datetime.now().time() < target_time:
        log(f"Waiting for {target_time.strftime('%H:%M')}...", "DEBUG")
        time.sleep(min(POLL_SECONDS, 60))


def reconcile_plans_with_broker(plans: list[TradePlan]) -> None:
    """Light position sync so resume does not double-BUY or manage ghosts."""
    for plan in plans:
        qty = fetch_open_quantity(plan.candidate.symbol)
        if qty is None:
            continue
        lotsize = plan.candidate.lotsize
        broker_lots = abs(qty) // lotsize

        if qty == 0 and plan.entered and plan.remaining_lot > 0:
            log(
                f"Reconcile: {plan.candidate.symbol} broker flat; clearing remaining_lot "
                f"{plan.remaining_lot} -> 0",
                "INFO",
            )
            plan.remaining_lot = 0
        elif qty > 0 and not plan.entered:
            log(
                f"Reconcile: {plan.candidate.symbol} broker qty={qty}; adopting as entered",
                "INFO",
            )
            plan.entered = True
            plan.remaining_lot = max(broker_lots, 1)
            if plan.entry_price is None:
                plan.entry_price = plan.entry
                log(
                    f"Reconcile: {plan.candidate.symbol} entry_price unknown; "
                    f"using planned entry={plan.entry}",
                    "INFO",
                )
            if plan.entry_time is None:
                plan.entry_time = datetime.now()
                log(
                    f"Reconcile: {plan.candidate.symbol} entry_time unknown; "
                    f"using resume time={plan.entry_time}",
                    "INFO",
                )
        elif qty > 0 and plan.entered and broker_lots != plan.remaining_lot:
            log(
                f"Reconcile: {plan.candidate.symbol} remaining_lot "
                f"{plan.remaining_lot} -> {broker_lots} from broker qty={qty}",
                "INFO",
            )
            plan.remaining_lot = broker_lots


def _handle_shutdown(signum: int, _frame: Any) -> None:
    log(f"Received signal {signum}; flushing runtime state", "INFO")
    persist_runtime()
    raise SystemExit(0)


def run_strategy() -> None:
    global _session, remaining_capital

    log(f"Starting {STRATEGY_NAME} strategy_id={STRATEGY_ID}", "INFO")
    log(
        f"Underlying={UNDERLYING} index_exchange={INDEX_EXCHANGE} "
        f"derivative_exchange={DERIVATIVE_EXCHANGE} product={PRODUCT} "
        f"max_capital_per_day={MAX_CAPITAL_PER_DAY}",
        "INFO",
    )

    try:
        signal.signal(signal.SIGTERM, _handle_shutdown)
        signal.signal(signal.SIGINT, _handle_shutdown)
    except Exception:
        pass

    trading_day = datetime.now().strftime("%Y-%m-%d")
    _session = RuntimeSession(trading_day=trading_day)
    reset_day_capital()

    saved = load_runtime_state()
    if saved and saved.get("trading_day") == trading_day:
        if "remaining_capital" in saved:
            remaining_capital = float(saved["remaining_capital"])
        if saved.get("candidates"):
            _session.candidates = [candidate_from_dict(c) for c in saved["candidates"]]
            log(f"Resumed {len(_session.candidates)} candidates from state", "INFO")
        if saved.get("plans"):
            _session.plans = [plan_from_dict(p) for p in saved["plans"]]
            log(f"Resumed {len(_session.plans)} plans from state", "INFO")
            if BROKER_RECONCILE:
                reconcile_plans_with_broker(_session.plans)
            else:
                log("Broker reconcile disabled (BROKER_RECONCILE=false); trusting persisted plans", "INFO")
            persist_runtime()

    if not _session.candidates:
        wait_until(SELECT_TIME)
        _session.candidates = select_option_candidates()
        persist_runtime()
    else:
        log("Skipping candidate selection; using persisted candidates", "INFO")
        if datetime.now().time() < SELECT_TIME:
            wait_until(SELECT_TIME)

    if plans_ready_for_all_candidates(_session.candidates, _session.plans):
        log(
            f"Skipping plan creation; using {len(_session.plans)} persisted plans "
            f"for {len(_session.candidates)} candidates",
            "INFO",
        )
    else:
        while (
            not plans_ready_for_all_candidates(_session.candidates, _session.plans)
            and datetime.now().time() <= TRADE_END_TIME
        ):
            before = len(_session.plans)
            _session.plans = create_trade_plans(_session.candidates, _session.plans)
            if len(_session.plans) > before:
                persist_runtime()
                log(
                    f"Plans ready {len(_session.plans)}/{len(_session.candidates)}",
                    "INFO",
                )
            if not plans_ready_for_all_candidates(_session.candidates, _session.plans):
                time.sleep(POLL_SECONDS)

        if _session.plans and not plans_ready_for_all_candidates(
            _session.candidates, _session.plans
        ):
            missing = [
                c.symbol
                for c in _session.candidates
                if c.symbol not in {p.candidate.symbol for p in _session.plans}
            ]
            log(
                f"Trade end reached with {len(_session.plans)}/{len(_session.candidates)} "
                f"plans; missing={missing}. Proceeding with ready plans.",
                "INFO",
            )
            persist_runtime()

    plans = _session.plans
    if not plans:
        log("No reference candle plans created before trade end time", "INFO")
        return

    any_trade_entered = any(plan.entered for plan in plans)
    live_executor = LiveOrderExecutor()
    while datetime.now().time() <= SQUARE_OFF_TIME:
        if datetime.now().time() > TRADE_END_TIME and not any(plan.entered for plan in plans):
            log("Trade window ended with no entry", "INFO")
            persist_runtime()
            return

        active_plans = [
            plan for plan in plans
            if plan.remaining_lot > 0 and (plan.entered or datetime.now().time() <= TRADE_END_TIME)
        ]
        if not active_plans:
            if any(plan.entered for plan in plans):
                log("All plans completed", "INFO")
            else:
                log("No active plans left", "INFO")
            persist_runtime()
            return

        for plan in active_plans:
            if ONE_TRADE_PER_DAY and any_trade_entered and not plan.entered:
                continue
            try:
                entered_now_or_before = monitor_plan(plan, live_executor)
                any_trade_entered = any_trade_entered or entered_now_or_before
            except Exception as e:
                log(f"Error monitoring {plan.candidate.symbol}, skipping poll: {e}", "INFO")

        time.sleep(POLL_SECONDS)

    for plan in plans:
        if plan.entered and plan.remaining_lot > 0:
            try:
                square_off_price = fetch_ltp(plan.candidate.symbol)
            except Exception as exc:
                log(
                    f"Square-off aborted for {plan.candidate.symbol}: "
                    f"LTP unavailable ({exc}); position left open",
                    "INFO",
                )
                continue
            exit_quantity(
                plan,
                plan.remaining_lot,
                reason="square off time",
                order_executor=live_executor,
                price=square_off_price,
                ts=datetime.now(),
            )
    persist_runtime()


if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        log("Strategy stopped", "INFO")
        persist_runtime()
    except SystemExit:
        raise
    except Exception as exc:
        log(f"Strategy error: {exc}", "INFO")
        persist_runtime()
