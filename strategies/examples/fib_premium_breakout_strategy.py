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

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime

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

if not API_KEY:
    print("Error: OPENALGO_API_KEY environment variable not set")
    raise SystemExit(1)


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
PREMIUM_TARGET = env_float("PREMIUM_TARGET", (PREMIUM_MIN + PREMIUM_MAX) / 2)
STRIKE_COUNT = env_int("STRIKE_COUNT", 30)

TOTAL_LOTS = env_int("TOTAL_LOTS", 4)
LOT_SIZE = env_int("LOT_SIZE", 75)
BIG_CANDLE_THRESHOLD = env_float("BIG_CANDLE_THRESHOLD", 40.0)
BIG_CANDLE_SL_FIB = env_float("BIG_CANDLE_SL_FIB", 0.55)
ENTRY_BUFFER = env_float("ENTRY_BUFFER", 0.05)
SL_BUFFER = env_float("SL_BUFFER", 0.05)

FIB_TARGETS = env_float_list("FIB_TARGETS", "1.272,1.618,2.0")
BOOKING_PCTS = env_int_list("BOOKING_PCTS", "25,25,50")
ALLOW_BOTH_CE_PE = env_bool("ALLOW_BOTH_CE_PE", True)
ONE_TRADE_PER_DAY = env_bool("ONE_TRADE_PER_DAY", True)
POLL_SECONDS = env_int("POLL_SECONDS", 15)

client = api(api_key=API_KEY, host=HOST, ws_url=WS_URL)


@dataclass
class Candidate:
    symbol: str
    option_type: str
    premium: float
    label: str
    strike: float
    lotsize: int


@dataclass
class TradePlan:
    candidate: Candidate
    entry: float
    stop_loss: float
    targets: list[float]
    remaining_qty: int
    target_quantities: list[int]
    booked_targets: set[int] = field(default_factory=set)
    active_stop_loss: float = 0.0
    entered: bool = False


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
    expiry = sorted_expiries[0]
    expiry_date = parse_expiry_date(expiry).date()
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
        if is_expiry_day:
            selected.append(min(typed, key=lambda candidate: candidate.premium))
        else:
            selected.append(min(typed, key=lambda candidate: abs(candidate.premium - PREMIUM_TARGET)))

    if not ALLOW_BOTH_CE_PE and selected:
        selected = [min(selected, key=lambda candidate: abs(candidate.premium - PREMIUM_TARGET))]

    print(f"Selected expiry: {expiry} | Expiry day: {is_expiry_day}")
    for candidate in selected:
        print(
            "Selected "
            f"{candidate.option_type}: {candidate.symbol} "
            f"premium={candidate.premium} label={candidate.label} strike={candidate.strike}"
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
    df = normalize_history(df)
    required_columns = {"open", "high", "low", "close"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{symbol} history missing columns: {missing_columns}")
    return df


def completed_candles(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame()
    return df.iloc[:-1].copy()


def reference_candle(df: pd.DataFrame) -> pd.Series | None:
    candles = completed_candles(df)
    if candles.empty:
        return None

    today = datetime.now().date()
    day_candles = candles[candles.index.date == today]
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


def split_target_quantities(total_qty: int, lot_size: int) -> list[int]:
    quantities: list[int] = []
    allocated = 0

    for index, pct in enumerate(BOOKING_PCTS):
        if index == len(BOOKING_PCTS) - 1:
            qty = total_qty - allocated
        else:
            raw_qty = int(total_qty * pct / 100)
            qty = (raw_qty // lot_size) * lot_size
            if qty == 0 and total_qty - allocated >= lot_size:
                qty = lot_size
        qty = max(0, min(qty, total_qty - allocated))
        quantities.append(qty)
        allocated += qty

    if allocated < total_qty and quantities:
        quantities[-1] += total_qty - allocated

    return quantities


def create_trade_plans(candidates: list[Candidate]) -> list[TradePlan]:
    plans: list[TradePlan] = []
    for candidate in candidates:
        df = fetch_today_history(candidate.symbol)
        candle = reference_candle(df)
        if candle is None:
            print(f"Waiting for reference candle for {candidate.symbol}")
            continue

        high = round(float(candle["high"]), 2)
        low = round(float(candle["low"]), 2)
        candle_range = round(high - low, 2)
        total_qty = TOTAL_LOTS * candidate.lotsize
        target_quantities = split_target_quantities(total_qty, candidate.lotsize)

        plan = TradePlan(
            candidate=candidate,
            entry=round(high + ENTRY_BUFFER, 2),
            stop_loss=build_stop_loss(low, high),
            targets=build_targets(low, high),
            remaining_qty=total_qty,
            target_quantities=target_quantities,
        )
        plan.active_stop_loss = plan.stop_loss
        plans.append(plan)

        print(
            f"{candidate.symbol} reference high={high} low={low} range={candle_range} "
            f"entry={plan.entry} sl={plan.stop_loss} targets={plan.targets}"
        )

    return plans


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


def place_order(plan: TradePlan, action: str, quantity: int, position_size: int) -> None:
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
    print(f"{action} {quantity} {plan.candidate.symbol} response:", response)


def enter_trade(plan: TradePlan) -> None:
    place_order(plan, action="BUY", quantity=plan.remaining_qty, position_size=plan.remaining_qty)
    plan.entered = True
    print(
        f"Entered BUY {plan.candidate.symbol} qty={plan.remaining_qty} "
        f"entry={plan.entry} sl={plan.active_stop_loss} targets={plan.targets}"
    )


def exit_quantity(plan: TradePlan, quantity: int, reason: str) -> None:
    if quantity <= 0 or plan.remaining_qty <= 0:
        return

    quantity = min(quantity, plan.remaining_qty)
    plan.remaining_qty -= quantity
    place_order(plan, action="SELL", quantity=quantity, position_size=plan.remaining_qty)
    print(f"Exited {quantity} {plan.candidate.symbol} due to {reason}. Remaining={plan.remaining_qty}")


def update_heikin_ashi_stop(plan: TradePlan, df: pd.DataFrame) -> None:
    candles = completed_candles(df)
    if len(candles) < 2:
        return

    ha = heikin_ashi(candles)
    previous_ha = ha.iloc[-2]
    trailed_sl = round(float(previous_ha["low"]) - SL_BUFFER, 2)
    plan.active_stop_loss = max(plan.active_stop_loss, trailed_sl)


def monitor_plan(plan: TradePlan) -> bool:
    df = fetch_today_history(plan.candidate.symbol)
    candles = completed_candles(df)
    if candles.empty:
        return False

    latest = candles.iloc[-1]
    latest_high = float(latest["high"])
    latest_close = float(latest["close"])

    if not plan.entered:
        if latest_high >= plan.entry:
            enter_trade(plan)
            return True
        return False

    update_heikin_ashi_stop(plan, df)

    for index, target in enumerate(plan.targets):
        if index in plan.booked_targets:
            continue

        if latest_high >= target:
            quantity = plan.target_quantities[index] if index < len(plan.target_quantities) else 0
            exit_quantity(plan, quantity, reason=f"fib target {target}")
            plan.booked_targets.add(index)

    if latest_close <= plan.active_stop_loss:
        exit_quantity(plan, plan.remaining_qty, reason=f"SL close beyond {plan.active_stop_loss}")

    print(
        f"{plan.candidate.symbol} close={latest_close:.2f} "
        f"active_sl={plan.active_stop_loss:.2f} remaining={plan.remaining_qty}"
    )
    return plan.entered


def wait_until(target_time: dtime) -> None:
    while datetime.now().time() < target_time:
        print(f"Waiting for {target_time.strftime('%H:%M')}...")
        time.sleep(min(POLL_SECONDS, 60))


def run_strategy() -> None:
    print(f"Starting {STRATEGY_NAME}")
    print(
        f"Underlying={UNDERLYING} index_exchange={INDEX_EXCHANGE} "
        f"derivative_exchange={DERIVATIVE_EXCHANGE} product={PRODUCT}"
    )

    wait_until(SELECT_TIME)
    candidates = select_option_candidates()

    plans: list[TradePlan] = []
    while not plans and datetime.now().time() <= TRADE_END_TIME:
        plans = create_trade_plans(candidates)
        if not plans:
            time.sleep(POLL_SECONDS)

    if not plans:
        print("No reference candle plans created before trade end time")
        return

    any_trade_entered = False
    while datetime.now().time() <= SQUARE_OFF_TIME:
        if datetime.now().time() > TRADE_END_TIME and not any(plan.entered for plan in plans):
            print("Trade window ended with no entry")
            return

        active_plans = [plan for plan in plans if plan.remaining_qty > 0]
        if not active_plans:
            print("All plans completed")
            return

        for plan in active_plans:
            if ONE_TRADE_PER_DAY and any_trade_entered and not plan.entered:
                continue
            entered_now_or_before = monitor_plan(plan)
            any_trade_entered = any_trade_entered or entered_now_or_before

        time.sleep(POLL_SECONDS)

    for plan in plans:
        if plan.entered and plan.remaining_qty > 0:
            exit_quantity(plan, plan.remaining_qty, reason="square off time")


if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        print("Strategy stopped")
    except Exception as exc:
        print(f"Strategy error: {exc}")
