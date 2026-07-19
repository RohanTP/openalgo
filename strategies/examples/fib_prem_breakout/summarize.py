#!/usr/bin/env python
"""
Summarize backtest results from a results folder.
Usage: uv run strategies/examples/fib_prem_breakout/summarize.py <results_dir>
       uv run strategies/examples/fib_prem_breakout/summarize.py  (uses default)
"""

import os
import sys
import argparse
import pandas as pd


DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")


def max_consecutive(series: pd.Series, condition: bool) -> int:
    max_run = current = 0
    for val in series:
        if (val > 0) == condition:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def summarize(results_dir: str) -> None:
    test_dirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])

    if not test_dirs:
        print(f"No test result folders found in {results_dir}")
        return

    rows = []

    for test_name in test_dirs:
        test_dir = os.path.join(results_dir, test_name)
        orders_path = os.path.join(test_dir, "orders.csv")
        plans_path = os.path.join(test_dir, "plans.csv")
        skipped_path = os.path.join(test_dir, "skipped_days.csv")

        if not os.path.exists(orders_path) or not os.path.exists(plans_path):
            print(f"Skipping {test_name}: missing orders.csv or plans.csv")
            continue

        orders = pd.read_csv(orders_path)
        plans = pd.read_csv(plans_path)
        skipped = pd.read_csv(skipped_path) if os.path.exists(skipped_path) else pd.DataFrame()

        sells = orders[orders["action"] == "SELL"]
        buys = orders[orders["action"] == "BUY"]

        # Per-plan PnL
        plan_pnl = sells.groupby("plan_id")["realized_pnl"].sum()

        # Entry info per plan
        entry_info = buys.groupby("plan_id").first().reset_index()[["plan_id", "trading_day", "price", "quantity"]]
        entry_info["capital"] = entry_info["price"] * entry_info["quantity"]
        entry_info = entry_info.merge(plan_pnl.rename("pnl"), on="plan_id", how="left").fillna(0)

        # Daily aggregation
        daily = entry_info.groupby("trading_day").agg(
            trades=("plan_id", "count"),
            capital=("capital", "sum"),
            pnl=("pnl", "sum"),
        ).reset_index()

        total_trading_days = len(daily)
        win_days = (daily["pnl"] > 0).sum()
        loss_days = (daily["pnl"] < 0).sum()
        breakeven_days = (daily["pnl"] == 0).sum()

        max_consec_wins = max_consecutive(daily["pnl"], condition=True)
        max_consec_losses = max_consecutive(daily["pnl"], condition=False)

        total_pnl = daily["pnl"].sum()
        avg_pnl_per_day = daily["pnl"].mean()
        avg_win = daily.loc[daily["pnl"] > 0, "pnl"].mean() if win_days > 0 else 0
        avg_loss = daily.loc[daily["pnl"] < 0, "pnl"].mean() if loss_days > 0 else 0
        best_day_pnl = daily["pnl"].max()
        worst_day_pnl = daily["pnl"].min()
        best_day = daily.loc[daily["pnl"].idxmax(), "trading_day"]
        worst_day = daily.loc[daily["pnl"].idxmin(), "trading_day"]

        # Sharpe-like: avg / std of daily pnl
        std_pnl = daily["pnl"].std()
        sharpe = round(avg_pnl_per_day / std_pnl, 3) if std_pnl > 0 else 0

        # Plan status breakdown
        status_counts = plans["final_status"].value_counts().to_dict()
        total_plans = len(plans)
        not_entered = status_counts.get("NOT_ENTERED", 0)
        entered_plans = total_plans - not_entered

        # Skip stats
        total_skipped = len(skipped)
        skip_reasons = skipped["reason"].value_counts().to_dict() if not skipped.empty else {}

        # Date range
        date_range = f"{daily['trading_day'].min()} to {daily['trading_day'].max()}"

        rows.append({
            "test_name": test_name,
            "date_range": date_range,
            "trading_days": total_trading_days,
            "total_plans": total_plans,
            "entered": entered_plans,
            "not_entered": not_entered,
            "skipped_days": total_skipped,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_day": round(avg_pnl_per_day, 2),
            "win_days": win_days,
            "loss_days": loss_days,
            "breakeven_days": breakeven_days,
            "win_rate_%": round(win_days / total_trading_days * 100, 1) if total_trading_days else 0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "best_day_pnl": round(best_day_pnl, 2),
            "best_day": best_day,
            "worst_day_pnl": round(worst_day_pnl, 2),
            "worst_day": worst_day,
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,
            "sharpe_like": sharpe,
            **{f"status_{k}": v for k, v in status_counts.items()},
            **{f"skip_{k.replace(' ', '_')}": v for k, v in skip_reasons.items()},
        })

    if not rows:
        print("No valid results found.")
        return

    df = pd.DataFrame(rows)

    # Print each test as a vertical key-value block for readability
    for _, row in df.iterrows():
        print("=" * 60)
        for col, val in row.items():
            print(f"  {col:<25} {val}")
    print("=" * 60)

    # If multiple tests, also print a compact comparison table
    if len(rows) > 1:
        compare_cols = [
            "test_name", "trading_days", "total_pnl", "win_rate_%",
            "avg_pnl_day", "best_day_pnl", "worst_day_pnl",
            "max_consec_wins", "max_consec_losses", "sharpe_like",
        ]
        print("\nCOMPARISON TABLE")
        print(df[compare_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fib backtest results")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=DEFAULT_RESULTS_DIR,
        help="Path to backtest_results folder (default: ./backtest_results next to this script)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: results directory not found: {args.results_dir}")
        sys.exit(1)

    summarize(args.results_dir)


if __name__ == "__main__":
    main()
