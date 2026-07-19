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

FIB_TARGETS = env_float_list("FIB_TARGETS", "1.272,1.618,2.0")
BOOKING_PCTS = env_int_list("BOOKING_PCTS", "25,25,50")
ALLOW_BOTH_CE_PE = env_bool("ALLOW_BOTH_CE_PE", True)
ONE_TRADE_PER_DAY = env_bool("ONE_TRADE_PER_DAY", True)
POLL_SECONDS = env_int("POLL_SECONDS", 3)

# HA trail activation: wait for target N hit and/or N minutes after entry (0 = disabled)
HA_TRAIL_START_TARGET = env_int("HA_TRAIL_START_TARGET", 0)
HA_TRAIL_DELAY_MINUTES = env_int("HA_TRAIL_DELAY_MINUTES", 0)

_LOG_LEVEL = os.getenv("STRATEGY_LOG_LEVEL", "INFO").upper()  # DISABLED | INFO | DEBUG


def log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVEL == "DISABLED":
        return
    if _LOG_LEVEL == "INFO" and level == "DEBUG":
        return
    print(msg)

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
    ) -> None:
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
    ) -> None:
        if client is None:
            raise RuntimeError("API client is not initialized.")
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
        log(f"{action} {quantity} {plan.candidate.symbol} response: {response}", "DEBUG")


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
    ) -> None:
        if action == "BUY":
            if plan.entry_price is None:
                plan.entry_price = price if price is not None else plan.entry
            if plan.entry_time is None:
                plan.entry_time = ts

        pnl = 0.0
        if action == "SELL":
            entry_p = plan.entry_price if plan.entry_price is not None else plan.entry
            sell_p = price if price is not None else entry_p
            pnl = round((sell_p - entry_p) * quantity, 2)

        order_row = {
            "plan_id": getattr(plan, "plan_id", None),
            "symbol": plan.candidate.symbol,
            "option_type": plan.candidate.option_type,
            "strike": plan.candidate.strike,
            "expiry": getattr(plan.candidate, "expiry", None),
            "action": action,
            "quantity": quantity,
            "lots": quantity // plan.candidate.lotsize,
            "price": price if price is not None else (plan.entry if action == "BUY" else plan.active_stop_loss),
            "position_size": position_size,
            "reason": reason,
            "remaining_lots": plan.remaining_lot,
            "remaining_quantity": plan.remaining_lot * plan.candidate.lotsize,
            "entry_price": plan.entry_price,
            "realized_pnl": pnl,
            "timestamp": ts,
        }
        self.orders.append(order_row)


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


def create_trade_plans(candidates: list[Candidate]) -> list[TradePlan]:
    plans: list[TradePlan] = []
    for candidate in candidates:
        df = fetch_today_history(candidate.symbol)
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
            target_lots = target_lots
        )
        plan.active_stop_loss = plan.stop_loss
        plans.append(plan)

        log(
            f"{candidate.symbol} reference high={high} low={low} range={candle_range} "
            f"entry={plan.entry} sl={plan.stop_loss} targets={plan.targets}",
            "INFO",
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


def enter_trade(
    plan: TradePlan,
    order_executor: OrderExecutor,
    price: float | None = None,
    ts: datetime | None = None,
) -> None:
    quantity = plan.remaining_lot * plan.candidate.lotsize
    order_executor.place_order(
        plan,
        action="BUY",
        quantity=quantity,
        position_size=quantity,
        price=price,
        ts=ts,
        reason="ENTRY",
    )
    plan.entered = True
    log(
        f"Entered BUY {plan.candidate.symbol} qty={quantity} lots={plan.remaining_lot} "
        f"entry={price if price is not None else plan.entry} sl={plan.active_stop_loss} targets={plan.targets}",
        "INFO",
    )


def exit_quantity(
    plan: TradePlan,
    lot: int,
    reason: str,
    order_executor: OrderExecutor,
    price: float | None = None,
    ts: datetime | None = None,
) -> None:
    if lot <= 0 or plan.remaining_lot <= 0:
        return
    lot = min(lot, plan.remaining_lot)
    plan.remaining_lot -= lot
    quantity = lot * plan.candidate.lotsize
    remaining_qty = plan.remaining_lot * plan.candidate.lotsize
    order_executor.place_order(
        plan,
        action="SELL",
        quantity=quantity,
        position_size=remaining_qty,
        price=price,
        ts=ts,
        reason=reason,
    )
    log(
        f"Exited {quantity}(lot {lot}) {plan.candidate.symbol} due to {reason}. "
        f"Remaining={remaining_qty}(lot {plan.remaining_lot})",
        "INFO",
    )


def update_heikin_ashi_stop(plan: TradePlan, df: pd.DataFrame, ts: datetime | None = None) -> None:
    if not plan.ha_trail_active:
        target_hit = len(plan.booked_targets) >= HA_TRAIL_START_TARGET
        delay_elapsed = (ts - plan.entry_time).total_seconds() / 60 >= HA_TRAIL_DELAY_MINUTES
        if target_hit or delay_elapsed:
            plan.ha_trail_active = True
            log(f"{plan.candidate.symbol} HA trail activated at {ts} | target_hit={target_hit} delay_elapsed={delay_elapsed}", "INFO")
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
        log(f"{plan.candidate.symbol} HA trail SL updated {old_sl} -> {plan.active_stop_loss}", "DEBUG")


def handle_plan(
    plan: TradePlan,
    candle: pd.Series,
    ltp: float,
    ts: datetime,
    order_executor: OrderExecutor,
    history_df: pd.DataFrame | None = None,
    is_backtest: bool = False,
) -> bool:
    if not plan.entered:
        use_buffer = is_backtest or env_bool("LIVE_ENTRY_BUFFER_ENABLED", True)
        entry_trigger_buffer = env_float("ENTRY_TRIGGER_BUFFER", 5.0)

        should_enter = False
        if use_buffer:
            if plan.entry <= ltp <= plan.entry + entry_trigger_buffer:
                should_enter = True
        else:
            if ltp >= plan.entry:
                should_enter = True

        if should_enter:
            enter_price = plan.entry
            enter_trade(plan, order_executor, price=enter_price, ts=ts)

    if plan.entered:
        if history_df is not None:
            update_heikin_ashi_stop(plan, history_df, ts=ts)

        for index, target in enumerate(plan.targets):
            if index in plan.booked_targets:
                continue

            if ltp >= target:
                lot = plan.target_lots[index] if index < len(plan.target_lots) else 0
                exit_price = target
                exit_quantity(
                    plan,
                    lot,
                    reason=f"fib target {target}",
                    order_executor=order_executor,
                    price=exit_price,
                    ts=ts,
                )
                plan.booked_targets.add(index)

        latest_close = float(candle["close"])
        if latest_close <= plan.active_stop_loss:
            exit_price = plan.active_stop_loss
            exit_quantity(
                plan,
                plan.remaining_lot,
                reason=f"SL close beyond {plan.active_stop_loss}",
                order_executor=order_executor,
                price=exit_price,
                ts=ts,
            )

        if plan.remaining_lot > 0:
            log(
                f"{plan.candidate.symbol} close={latest_close:.2f} "
                f"active_sl={plan.active_stop_loss:.2f} remaining={plan.remaining_lot}",
                "DEBUG",
            )

    return plan.entered


def monitor_plan(plan: TradePlan, order_executor: OrderExecutor) -> bool:
    df = fetch_today_history(plan.candidate.symbol)
    candles = completed_candles(df)
    if candles.empty:
        return False

    latest = candles.iloc[-1]
    ltp = fetch_ltp(plan.candidate.symbol)

    return handle_plan(
        plan=plan,
        candle=latest,
        ltp=ltp,
        ts=latest.name if hasattr(latest, "name") else datetime.now(),
        order_executor=order_executor,
        history_df=df,
        is_backtest=False,
    )


def wait_until(target_time: dtime) -> None:
    while datetime.now().time() < target_time:
        log(f"Waiting for {target_time.strftime('%H:%M')}...", "DEBUG")
        time.sleep(min(POLL_SECONDS, 60))


def run_strategy() -> None:
    log(f"Starting {STRATEGY_NAME}", "INFO")
    log(
        f"Underlying={UNDERLYING} index_exchange={INDEX_EXCHANGE} "
        f"derivative_exchange={DERIVATIVE_EXCHANGE} product={PRODUCT}",
        "INFO",
    )

    wait_until(SELECT_TIME)
    candidates = select_option_candidates()

    plans: list[TradePlan] = []
    while not plans and datetime.now().time() <= TRADE_END_TIME:
        plans = create_trade_plans(candidates)
        if not plans:
            time.sleep(POLL_SECONDS)

    if not plans:
        log("No reference candle plans created before trade end time", "INFO")
        return

    any_trade_entered = False
    live_executor = LiveOrderExecutor()
    while datetime.now().time() <= SQUARE_OFF_TIME:
        if datetime.now().time() > TRADE_END_TIME and not any(plan.entered for plan in plans):
            log("Trade window ended with no entry", "INFO")
            return

        active_plans = [plan for plan in plans if plan.remaining_lot > 0]
        if not active_plans:
            log("All plans completed", "INFO")
            return

        for plan in active_plans:
            if ONE_TRADE_PER_DAY and any_trade_entered and not plan.entered:
                continue
            entered_now_or_before = monitor_plan(plan, live_executor)
            any_trade_entered = any_trade_entered or entered_now_or_before

        time.sleep(POLL_SECONDS)

    for plan in plans:
        if plan.entered and plan.remaining_lot > 0:
            exit_quantity(
                plan,
                plan.remaining_lot,
                reason="square off time",
                order_executor=live_executor,
                ts=datetime.now(),
            )


if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        log("Strategy stopped", "INFO")
    except Exception as exc:
        log(f"Strategy error: {exc}", "INFO")
