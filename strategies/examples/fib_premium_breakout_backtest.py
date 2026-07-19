#!/usr/bin/env python
"""
Parquet-based backtest runner for the Fibonacci breakout strategy.
"""

import os
import sys
import json
import argparse
from datetime import datetime, date, time as dtime
import pandas as pd
import duckdb

# Add workspace directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import strategies.examples.fib_premium_breakout_strategy as strat


def validate_parquet_columns(conn: duckdb.DuckDBPyConnection, table_path: str, expected_cols: list[str]) -> None:
    try:
        desc = conn.execute(f"DESCRIBE SELECT * FROM '{table_path}' LIMIT 1").df()
        existing_cols = set(desc["column_name"].tolist())
    except Exception as e:
        raise ValueError(f"Failed to read parquet file at '{table_path}': {e}")
        
    missing = [col for col in expected_cols if col not in existing_cols]
    if missing:
        raise ValueError(f"Parquet file at '{table_path}' is missing expected columns: {missing}")


def apply_config_to_strategy(config: dict) -> None:
    # Map time parameters
    for time_key in ["SELECT_TIME", "REFERENCE_CANDLE_TIME", "TRADE_END_TIME", "SQUARE_OFF_TIME"]:
        if time_key in config:
            setattr(strat, time_key, datetime.strptime(config[time_key], "%H:%M").time())
            
    # Float parameters
    for float_key in [
        "PREMIUM_MIN", "PREMIUM_MAX",
        "BIG_CANDLE_THRESHOLD", "BIG_CANDLE_SL_FIB", "ENTRY_BUFFER", "SL_BUFFER",
        "ENTRY_TRIGGER_BUFFER",
    ]:
        if float_key in config:
            setattr(strat, float_key, float(config[float_key]))

    # Int parameters
    for int_key in ["TOTAL_LOTS", "LOT_SIZE", "HA_TRAIL_START_TARGET", "HA_TRAIL_DELAY_MINUTES"]:
        if int_key in config:
            setattr(strat, int_key, int(config[int_key]))

    # Lists
    if "FIB_TARGETS" in config:
        strat.FIB_TARGETS = [float(x) for x in config["FIB_TARGETS"]]
    if "BOOKING_PCTS" in config:
        strat.BOOKING_PCTS = [int(x) for x in config["BOOKING_PCTS"]]

    # Bools
    for bool_key in ["ALLOW_BOTH_CE_PE", "ONE_TRADE_PER_DAY"]:
        if bool_key in config:
            setattr(strat, bool_key, bool(config[bool_key]))

    # String parameters
    if "STRATEGY_LOG_LEVEL" in config:
        strat._LOG_LEVEL = str(config["STRATEGY_LOG_LEVEL"]).upper()

class back_test:
    def __init__(self, config: dict, conn: duckdb.DuckDBPyConnection):
        self.test_name = config["name"]
        self.instrument = config["instrument"]
        self.options_parquet = config["options_parquet"]
        self.index_parquet = config["index_parquet"]
        self.conn = conn

        # Date range from config
        ranges = config.get("range", [])
        if ranges:
            self.start_date = ranges[0]["start"]
            self.end_date = ranges[0]["end"]
        else:
            self.start_date = "1900-01-01"
            self.end_date = "9999-12-31"

        # Strategy settings
        self.premium_min = config.get("PREMIUM_MIN", 300)
        self.premium_max = config.get("PREMIUM_MAX", 400)
        self.total_lots = config.get("TOTAL_LOTS", 4)
        self.entry_buffer = config.get("ENTRY_BUFFER", 0.05)
        self.select_time_str = config.get("SELECT_TIME", "09:07")
        self.ref_time_str = config.get("REFERENCE_CANDLE_TIME", "09:15")
        self.trade_end_time_str = config.get("TRADE_END_TIME", "10:30")
        self.square_off_time_str = config.get("SQUARE_OFF_TIME", "15:15")
        self.allow_both = config.get("ALLOW_BOTH_CE_PE", True)
        self.one_trade_per_day = config.get("ONE_TRADE_PER_DAY", False)

        self.config = config


    def get_weekly_expiries(self) -> list[dict]:
        """
        If options_parquet is a folder, list expiry parquet files.
        Filename should be YYYY-MM-DD.parquet.
        """
        expiries = []

        for file in os.listdir(self.options_parquet):
            if not file.endswith(".parquet"):
                continue

            expiry = file.replace(".parquet", "")

            try:
                expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            except ValueError:
                continue

            expiries.append({
                "expiry": expiry,
                "expiry_date": expiry_date,
                "options_path": os.path.join(self.options_parquet, file),
            })

        expiries.sort(key=lambda x: x["expiry_date"])
        return expiries

    def get_trading_days_for_expiry(
            self,
            options_path: str,
            expiry_date,
        ) -> list[str]:
            expiry_str = expiry_date.strftime("%Y-%m-%d") if hasattr(expiry_date, "strftime") else str(expiry_date)[:10]
            query = f"""
                SELECT DISTINCT trading_day
                FROM '{options_path}'
                WHERE expiry = '{expiry_str}'
                ORDER BY trading_day
            """

            df = self.conn.execute(query).df()

            return [
                d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                for d in df["trading_day"].tolist()
            ]

    def get_spot_for_day(self, trading_day: str) -> float | None:
        query = f"""
            SELECT close
            FROM '{self.index_parquet}'
            WHERE symbol = '{self.instrument}'
            AND trading_day = '{trading_day}'
            AND strftime('%H:%M', timezone('Asia/Kolkata', timestamp)) = '{self.select_time_str}'
            LIMIT 1
        """

        df = self.conn.execute(query).df()

        if df.empty:
            return None

        return float(df.iloc[0]["close"])
    
    def get_itm_candidates(
            self,
            options_path: str,
            expiry_date_str: str,
            trading_day: str,
            spot: float,
        ) -> list:
            res_options = pd.DataFrame()

            # Try SELECT_TIME first, fallback to 09:15 if 09:07 candle missing
            for t_str in [self.select_time_str, self.ref_time_str]:

                query = f"""
                    SELECT symbol, strike, option_type, close AS premium
                    FROM '{options_path}'
                    WHERE expiry = '{expiry_date_str}'
                    AND trading_day = '{trading_day}'
                    AND strftime('%H:%M', timezone('Asia/Kolkata', timestamp)) = '{t_str}'
                    AND close >= {self.premium_min}
                    AND close <= {self.premium_max}
                """

                res_options = self.conn.execute(query).df()

                if not res_options.empty:
                    break

            candidates = []

            for _, row in res_options.iterrows():
                strike = float(row["strike"])
                opt_type = str(row["option_type"]).upper()
                premium = float(row["premium"])
                symbol = row["symbol"]
                lotsize = int(self.config.get("LOT_SIZE", 75))

                is_itm = (
                    (opt_type == "CE" and strike < spot)
                    or (opt_type == "PE" and strike > spot)
                )

                if not is_itm:
                    continue

                candidates.append(
                    strat.Candidate(
                        symbol=symbol,
                        option_type=opt_type,
                        premium=premium,
                        label="ITM",
                        strike=strike,
                        lotsize=lotsize,
                        expiry=expiry_date_str,
                    )
                )

            return candidates
    
    def select_candidates(
            self,
            candidates: list[strat.Candidate],
            trading_day: str,
            expiry_date: str,
        ) -> list[strat.Candidate]:
            """
            Select final CE/PE candidates using the same logic as live strategy.
            """

            is_expiry_day = trading_day == expiry_date
            selected: list[strat.Candidate] = []

            for option_type in ("CE", "PE"):
                typed = [c for c in candidates if c.option_type == option_type]

                if not typed:
                    continue

                if is_expiry_day:
                    selected.append(
                        min(typed, key=lambda c: c.premium)
                    )
                else:
                    selected.append(
                        min(
                            typed,
                            key=lambda c: c.premium,
                        )
                    )

            if not self.allow_both and selected:
                selected = [
                    min(
                        selected,
                        key=lambda c: c.premium,
                    )
                ]

            return selected

    def create_plans_for_day(
            self,
            options_path: str,
            trading_day: str,
            selected_candidates: list[strat.Candidate],
        ) -> list[strat.TradePlan]:
            if not selected_candidates:
                return []

            symbols_str = ", ".join(f"'{c.symbol}'" for c in selected_candidates)

            query = f"""
                SELECT symbol, strike, option_type, timestamp, open, high, low, close
                FROM '{options_path}'
                WHERE symbol IN ({symbols_str})
                AND trading_day = '{trading_day}'
                AND strftime('%H:%M', timezone('Asia/Kolkata', timestamp)) = '{self.ref_time_str}'
            """

            ref_df = self.conn.execute(query).df()

            plans_for_day = []

            for candidate in selected_candidates:
                cand_ref = ref_df[
                    (ref_df["symbol"] == candidate.symbol)
                    & (ref_df["strike"].astype(float) == float(candidate.strike))
                    & (ref_df["option_type"].str.upper() == candidate.option_type)
                ]

                if cand_ref.empty:
                    continue

                row = cand_ref.iloc[0]

                high = round(float(row["high"]), 2)
                low = round(float(row["low"]), 2)

                target_lots = strat.split_target_lots(self.total_lots)

                plan = strat.TradePlan(
                    candidate=candidate,
                    entry=round(high + self.entry_buffer, 2),
                    stop_loss=strat.build_stop_loss(low, high),
                    targets=strat.build_targets(low, high),
                    remaining_lot=self.total_lots,
                    target_lots=target_lots,
                )

                plan.active_stop_loss = plan.stop_loss
                plan.plan_id = (
                    f"plan_{trading_day}_{candidate.symbol}_"
                    f"{candidate.option_type}_{candidate.strike}"
                )

                plan.ref_ts = row["timestamp"]
                plan.ref_open = float(row["open"])
                plan.ref_high = high
                plan.ref_low = low
                plan.ref_close = float(row["close"])

                plans_for_day.append(plan)

            return plans_for_day

    def get_daily_candles(
            self,
            options_path: str,
            trading_day: str,
            plans_for_day: list[strat.TradePlan],
        ) -> pd.DataFrame:
            if not plans_for_day:
                return pd.DataFrame()

            symbols_str = ", ".join(
                f"'{plan.candidate.symbol}'"
                for plan in plans_for_day
            )

            query = f"""
                SELECT timezone('Asia/Kolkata', timestamp) AS timestamp, symbol, strike, option_type, open, high, low, close
                FROM '{options_path}'
                WHERE symbol IN ({symbols_str})
                AND trading_day = '{trading_day}'
                AND strftime('%H:%M', timezone('Asia/Kolkata', timestamp)) >= '09:00'
                AND strftime('%H:%M', timezone('Asia/Kolkata', timestamp)) <= '{self.square_off_time_str}'
                ORDER BY timestamp
            """

            df = self.conn.execute(query).df()

            if df.empty:
                return df

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df["option_type"] = df["option_type"].str.upper()
            df["strike"] = df["strike"].astype(float)

            return df
    
    def run_back_test_week(
            self,
            options_path: str,
            expiry_date_str: str,
            days: list[str],
        ) -> tuple[list[dict], list[dict], list[dict]]:
            all_plans: list[dict] = []
            all_orders: list[dict] = []
            all_skipped_days: list[dict] = []


            for trading_day in days:
                spot = self.get_spot_for_day(trading_day)

                if spot is None:
                    all_skipped_days.append({
                        "test_name": self.test_name,
                        "trading_day": trading_day,
                        "instrument": self.instrument,
                        "expiry": expiry_date_str,
                        "reason": "no spot candle",
                        "details": f"No spot candle found at {self.select_time_str}",
                    })
                    continue

                candidates = self.get_itm_candidates(
                    options_path=options_path,
                    expiry_date_str=expiry_date_str,
                    trading_day=trading_day,
                    spot=spot,
                )

                if not candidates:
                    all_skipped_days.append({
                        "test_name": self.test_name,
                        "trading_day": trading_day,
                        "instrument": self.instrument,
                        "expiry": expiry_date_str,
                        "reason": "no ITM candidates",
                        "details": (
                            f"No ITM options between {self.premium_min}-"
                            f"{self.premium_max} at {self.select_time_str}"
                        ),
                    })
                    continue

                selected_candidates = self.select_candidates(
                    candidates=candidates,
                    trading_day=trading_day,
                    expiry_date=expiry_date_str,
                )

                if not selected_candidates:
                    all_skipped_days.append({
                        "test_name": self.test_name,
                        "trading_day": trading_day,
                        "instrument": self.instrument,
                        "expiry": expiry_date_str,
                        "reason": "no selected candidates",
                        "details": "No CE/PE candidates selected after filtering",
                    })
                    continue

                plans_for_day = self.create_plans_for_day(
                    options_path=options_path,
                    trading_day=trading_day,
                    selected_candidates=selected_candidates,
                )

                if not plans_for_day:
                    all_skipped_days.append({
                        "test_name": self.test_name,
                        "trading_day": trading_day,
                        "instrument": self.instrument,
                        "expiry": expiry_date_str,
                        "reason": "no reference candle",
                        "details": f"No reference candle at {self.ref_time_str}",
                    })
                    continue

                daily_candles_df = self.get_daily_candles(
                    options_path=options_path,
                    trading_day=trading_day,
                    plans_for_day=plans_for_day,
                )

                if daily_candles_df.empty:
                    all_skipped_days.append({
                        "test_name": self.test_name,
                        "trading_day": trading_day,
                        "instrument": self.instrument,
                        "expiry": expiry_date_str,
                        "reason": "missing option data",
                        "details": "No intraday candles found for selected plans",
                    })
                    continue

                day_plans, day_orders = self.run_back_test_day(
                    trading_day=trading_day,
                    expiry_date_str=expiry_date_str,
                    plans_for_day=plans_for_day,
                    daily_candles_df=daily_candles_df,
                )

                all_plans.extend(day_plans)
                all_orders.extend(day_orders)

            return all_plans, all_orders, all_skipped_days
    
    def run_back_test_day(
        self,
        trading_day: str,
        expiry_date_str: str,
        plans_for_day: list[strat.TradePlan],
        daily_candles_df: pd.DataFrame,
    ) -> tuple[list[dict], list[dict]]:
        day_plans: list[dict] = []
        day_orders: list[dict] = []

        trade_end_time = datetime.strptime(
            self.trade_end_time_str, "%H:%M"
        ).time()

        square_off_time = datetime.strptime(
            self.square_off_time_str, "%H:%M"
        ).time()

        trade_start_time = datetime.strptime("09:16", "%H:%M").time()

        backtest_executor = strat.BacktestOrderExecutor()
        any_trade_entered = False

        timestamps = sorted(set(daily_candles_df.index))

        trade_timestamps = [
            ts for ts in timestamps
            if trade_start_time <= ts.time() <= square_off_time
        ]

        for ts in trade_timestamps:
            active_plans = [p for p in plans_for_day if p.remaining_lot > 0]

            if not active_plans:
                break

            if ts.time() > trade_end_time and not any(p.entered for p in plans_for_day):
                break

            for plan in active_plans:
                if self.one_trade_per_day and any_trade_entered and not plan.entered:
                    continue

                candle_rows = daily_candles_df[
                    (daily_candles_df.index == ts)
                    & (daily_candles_df["symbol"] == plan.candidate.symbol)
                    & (daily_candles_df["strike"].astype(float) == float(plan.candidate.strike))
                    & (daily_candles_df["option_type"].str.upper() == plan.candidate.option_type)
                ]

                if candle_rows.empty:
                    continue

                candle = candle_rows.iloc[0]

                # Backtest assumption: high is used as simulated LTP
                ltp = float(candle["high"])

                history_df = daily_candles_df[
                    (daily_candles_df.index <= ts)
                    & (daily_candles_df["symbol"] == plan.candidate.symbol)
                    & (daily_candles_df["strike"].astype(float) == float(plan.candidate.strike))
                    & (daily_candles_df["option_type"].str.upper() == plan.candidate.option_type)
                ]

                entered_now_or_before = strat.handle_plan(
                    plan=plan,
                    candle=candle,
                    ltp=ltp,
                    ts=ts,
                    order_executor=backtest_executor,
                    history_df=history_df,
                    is_backtest=True,
                )

                any_trade_entered = any_trade_entered or entered_now_or_before

                if (
                    ts.time() == square_off_time
                    and plan.entered
                    and plan.remaining_lot > 0
                ):
                    strat.exit_quantity(
                        plan=plan,
                        lot=plan.remaining_lot,
                        reason="square off time",
                        order_executor=backtest_executor,
                        price=float(candle["close"]),
                        ts=ts,
                    )

        for plan in plans_for_day:
            plan_orders = [
                o for o in backtest_executor.orders
                if o["plan_id"] == plan.plan_id
            ]

            final_status = self.get_final_plan_status(plan, plan_orders)

            day_plans.append({
                "plan_id": plan.plan_id,
                "test_name": self.test_name,
                "instrument": self.instrument,
                "trading_day": trading_day,
                "expiry": expiry_date_str,
                "symbol": plan.candidate.symbol,
                "option_type": plan.candidate.option_type,
                "strike": plan.candidate.strike,
                "selected_premium": plan.candidate.premium,
                "planned_entry": plan.entry,
                "planned_stop_loss": plan.stop_loss,
                "planned_targets": ";".join(map(str, plan.targets)),
                "target_lots": ";".join(map(str, plan.target_lots)),
                "final_status": final_status,
            })

            cumulative_pnl = 0.0

            for order in plan_orders:
                realized_pnl = float(order.get("realized_pnl", 0) or 0)

                if order["action"] == "SELL":
                    cumulative_pnl += realized_pnl

                day_orders.append({
                    "plan_id": plan.plan_id,
                    "test_name": self.test_name,
                    "trading_day": trading_day,
                    "timestamp": order["timestamp"],
                    "symbol": order["symbol"],
                    "option_type": order["option_type"],
                    "strike": order["strike"],
                    "expiry": expiry_date_str,
                    "action": order["action"],
                    "quantity": order["quantity"],
                    "lots": order["lots"],
                    "price": order["price"],
                    "reason": self.map_order_reason(order, plan),
                    "remaining_lots": order["remaining_lots"],
                    "remaining_quantity": order["remaining_quantity"],
                    "entry_price": order["entry_price"],
                    "realized_pnl": realized_pnl,
                    "cumulative_plan_pnl": cumulative_pnl,
                })

        return day_plans, day_orders
    
    def get_final_plan_status(
        self,
        plan: strat.TradePlan,
        plan_orders: list[dict],
    ) -> str:
        if not plan.entered:
            return "NOT_ENTERED"

        if plan.remaining_lot > 0:
            return "OPEN"

        if not plan_orders:
            return "COMPLETED"

        last_order = plan_orders[-1]
        last_reason = str(last_order.get("reason", "")).lower()

        if "sl" in last_reason or "stop" in last_reason:
            last_exit_price = float(last_order.get("price") or 0)
            entry_price = float(last_order.get("entry_price") or 0)
            if entry_price > 0 and last_exit_price > entry_price:
                return "TRAILING_SL"
            return "STOPLOSS"

        if "square" in last_reason:
            return "SQUARE_OFF"

        return "COMPLETED"
    
    def map_order_reason(
        self,
        order: dict,
        plan: strat.TradePlan,
    ) -> str:
        raw_reason = str(order.get("reason", ""))

        if order["action"] == "BUY":
            return "ENTRY"

        if "fib target" in raw_reason:
            try:
                target = float(raw_reason.split("fib target")[-1].strip())
                target_index = plan.targets.index(target)
                return f"TARGET_{target_index + 1}"
            except Exception:
                return "TARGET"

        if "sl" in raw_reason.lower() or "stop" in raw_reason.lower():
            return "STOPLOSS"

        if "square" in raw_reason.lower():
            return "SQUARE_OFF"

        return raw_reason

    def run_back_test(self) -> tuple[list[dict], list[dict], list[dict]]:
        all_plans: list[dict] = []
        all_orders: list[dict] = []
        all_skipped_days: list[dict] = []

        weekly_expiries = self.get_weekly_expiries()

        for expiry_info in weekly_expiries:
            expiry_date_str = expiry_info["expiry"]
            options_path = expiry_info["options_path"]

            expiry_date = expiry_info.get("expiry_date")
            if expiry_date is not None:
                if str(expiry_date) < self.start_date or str(expiry_date) > self.end_date:
                    continue

            days = self.get_trading_days_for_expiry(
                options_path=options_path,
                expiry_date=expiry_date_str,
            )

            if not days:
                continue

            week_plans, week_orders, week_skipped = self.run_back_test_week(
                options_path=options_path,
                expiry_date_str=expiry_date_str,
                days=days,
            )

            all_plans.extend(week_plans)
            all_orders.extend(week_orders)
            all_skipped_days.extend(week_skipped)

        return all_plans, all_orders, all_skipped_days
def main() -> None:
    parser = argparse.ArgumentParser(description="Fib Premium Breakout Backtest Runner")
    parser.add_argument("--config", type=str, default="backtest.json", help="Path to backtest JSON config file")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        configs = json.load(f)
        
    conn = duckdb.connect()
    
    for config in configs:
        name = config["name"]
        print(f"Running backtest config: {name}...")
        
        # Apply config parameters to strat module
        apply_config_to_strategy(config)
        bt = back_test(config, conn)

        plans, orders, skipped_days = bt.run_back_test()
        
        # Save output files
        output_dir = config.get("output_dir", "./backtest_results")
        test_dir = os.path.join(output_dir, name)
        os.makedirs(test_dir, exist_ok=True)
        
        # Write plans.csv
        plans_df = pd.DataFrame(plans)
        plans_path = os.path.join(test_dir, "plans.csv")
        plans_df.to_csv(plans_path, index=False)
        print(f"Saved plans to {plans_path} ({len(plans_df)} rows)")
        
        # Write orders.csv
        orders_df = pd.DataFrame(orders)
        orders_path = os.path.join(test_dir, "orders.csv")
        orders_df.to_csv(orders_path, index=False)
        print(f"Saved orders to {orders_path} ({len(orders_df)} rows)")
        
        # Write skipped_days.csv
        skipped_df = pd.DataFrame(skipped_days)
        if skipped_df.empty:
            skipped_df = pd.DataFrame(columns=["test_name", "trading_day", "instrument", "expiry", "reason", "details"])
        skipped_path = os.path.join(test_dir, "skipped_days.csv")
        skipped_df.to_csv(skipped_path, index=False)
        print(f"Saved skipped days to {skipped_path} ({len(skipped_df)} rows)")


if __name__ == "__main__":
    main()
