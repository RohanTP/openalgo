Modify `strategies/examples/fib_premium_breakout_strategy.py` with minimal refactor and add a separate parquet backtest runner.

## Main requirements

### 1. Add only one interface: `OrderExecutor`

Do **not** create interfaces for `enter_trade()` or `exit_quantity()`.

Only abstract `place_order()`.

Create:

```python
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
```

Create two implementations:

```python
class LiveOrderExecutor(OrderExecutor):
    # uses existing client.placesmartorder()
```

```python
class BacktestOrderExecutor(OrderExecutor):
    # does not place real orders
    # records orders, fills, pnl, remaining qty, reason, timestamp
```

Keep existing `enter_trade()` and `exit_quantity()` as shared strategy functions. They should call:

```python
order_executor.place_order(...)
```

instead of directly calling `client.placesmartorder()`.

Live behavior must remain unchanged.

---

### 2. Change monitor function to accept candle and ltp params

Change current `monitor_plan()` so it does not always fetch history/LTP internally.

Add a new param-based function like:

```python
def handle_plan(
    plan: TradePlan,
    candle: pd.Series,
    ltp: float,
    ts: datetime,
    order_executor: OrderExecutor,
    history_df: pd.DataFrame | None = None,
    is_backtest: bool = False,
) -> bool:
    ...
```

Backtest will pass:

* current 1-minute option candle
* `ltp = candle["high"]`
* timestamp
* historical candles up to current timestamp for Heikin Ashi trailing if needed

Live mode can still fetch current data and then call `handle_plan()` internally.

Important behavior change:
After entry happens, do **not** immediately return.

Currently after `enter_trade()` it returns. Change this so that in the same candle/minute, after entry, targets and stop loss are also checked.

Reason: in backtest, a candle can cross entry and target/SL in the same minute.

---

### 3. Entry condition with buffer

For backtest entry, do not enter only because `ltp >= plan.entry`.

Use a configurable entry trigger buffer.

Example:

```python
ENTRY_TRIGGER_BUFFER = 5.0
```

Backtest entry should happen when:

```python
plan.entry <= ltp <= plan.entry + ENTRY_TRIGGER_BUFFER
```

or equivalent candle-based logic.

Because in backtest we use `ltp = candle.high`, this means the candle high should be close to the planned entry and not far beyond it.

Make this buffer configurable from env/config/json.

For live mode, keep existing behavior unless config says otherwise.

---

### 4. Add separate backtest file

Add a new file:

```text
strategies/examples/fib_premium_breakout_backtest.py
```

This file should run parquet-based backtests.

It should read config from:

```text
backtest.json
```

`backtest.json` should contain an array of backtest configs.

Example structure:

```json
[
  {
    "name": "nifty_may_2023_test",
    "instrument": "NIFTY",
    "options_parquet": "/path/to/options.parquet",
    "index_parquet": "/path/to/nifty_index.parquet",
    "weeks": [
      {
        "start": "2023-05-08",
        "end": "2023-05-12",
        "expiry": "2023-05-11"
      }
    ],
    "output_dir": "./backtest_results",

    "SELECT_TIME": "09:07",
    "REFERENCE_CANDLE_TIME": "09:15",
    "TRADE_END_TIME": "10:30",
    "SQUARE_OFF_TIME": "15:15",

    "PREMIUM_MIN": 300,
    "PREMIUM_MAX": 400,
    "PREMIUM_TARGET": 350,
    "TOTAL_LOTS": 4,
    "LOT_SIZE": 75,
    "BIG_CANDLE_THRESHOLD": 40,
    "BIG_CANDLE_SL_FIB": 0.55,
    "ENTRY_BUFFER": 0.05,
    "ENTRY_TRIGGER_BUFFER": 5,
    "SL_BUFFER": 0.05,
    "FIB_TARGETS": [1.272, 1.618, 2.0],
    "BOOKING_PCTS": [25, 25, 50],
    "ALLOW_BOTH_CE_PE": true,
    "ONE_TRADE_PER_DAY": true
  }
]
```

Support multiple tests in one JSON file.

Also support instruments like:

* `NIFTY`
* `SENSEX`

Each test should define its own parquet paths, instrument, weeks, and strategy params.

---

### 5. Backtest loop

For each config in `backtest.json`:

```text
for each week:
  for each trading day:
    at 09:07:
      get index spot from index parquet
      get option candles from options parquet
      find ITM options with premium between 300 and 400

    construct candidates

    for selected candidates:
      get 09:15 candle
      construct TradePlan

    for every minute after 09:15 until square off:
      for every active plan:
        get that option's current candle
        ltp = candle.high
        call handle_plan(plan, candle, ltp, ts, backtest_executor, history_df, is_backtest=True)

    at square off:
      exit remaining open quantity at square-off candle close
```

ITM rules:

* CE is ITM when `strike < spot`
* PE is ITM when `strike > spot`

Candidate selection:

* filter premium between `PREMIUM_MIN` and `PREMIUM_MAX` at `SELECT_TIME`
* choose CE and PE candidates using same current logic:

  * on expiry day: choose lowest premium
  * non-expiry day: choose closest to `PREMIUM_TARGET`
* respect `ALLOW_BOTH_CE_PE`

Reference candle:

* use option candle at `REFERENCE_CANDLE_TIME`, default 09:15
* entry = reference high + `ENTRY_BUFFER`
* SL and targets should use existing logic

Minute loop:

* start from 09:16
* pass candle and `ltp = candle.high`
* do not fetch data inside `handle_plan()`

---

### 6. Backtest `place_order()` behavior

In `BacktestOrderExecutor.place_order()`:

* capture every BUY/SELL as an order row
* include:

  * test name
  * instrument
  * week start/end
  * trading day
  * timestamp
  * symbol
  * option type
  * strike
  * expiry
  * action
  * quantity
  * lots
  * price
  * position size
  * reason
  * realized pnl for SELL
  * cumulative pnl
  * remaining lots
  * remaining quantity

For BUY:

* store entry price and entry timestamp on the plan if not already stored.

For SELL:

* calculate realized P&L:

```python
pnl = (sell_price - entry_price) * quantity
```

Accumulate:

* per plan
* per day
* per week
* per month
* per full test

---

### 7. Output files

For every backtest config, create:

{output_dir}/{test_name}/
1. plans.csv

One row per TradePlan created.

Columns:

plan_id (unique id)
test_name
instrument
week_start
week_end
trading_day
expiry
symbol
option_type
strike
selected_premium
reference_timestamp
reference_open
reference_high
reference_low
reference_close
planned_entry
planned_stop_loss
planned_targets
target_lots
final_status
NOT_ENTERED
OPEN
COMPLETED
STOPLOSS
SQUARE_OFF

This represents the strategy decision before any orders are executed.

2. orders.csv

One row per simulated BUY/SELL order.

Columns:

plan_id
test_name
trading_day
timestamp
symbol
option_type
strike
expiry
action (BUY / SELL)
quantity
lots
price
reason
ENTRY
TARGET_1
TARGET_2
TARGET_3
STOPLOSS
SQUARE_OFF
remaining_lots
remaining_quantity
entry_price
realized_pnl
cumulative_plan_pnl

Every partial exit should generate a separate SELL row.

3. skipped_days.csv

One row for every skipped day.

Columns:

test_name
trading_day
instrument
expiry
reason
details

Examples:

no spot candle
no ITM candidates
no reference candle
missing option data
invalid configuration

### 8. DuckDB/parquet requirements

Use DuckDB for parquet reads.

Do not load the full 5-year parquet into pandas unnecessarily.

Push filters into SQL:

* instrument/symbol
* trading_day
* expiry
* timestamp/date range
* option_type
* strike when available

Expected option parquet columns:

* timestamp
* open
* high
* low
* close
* volume
* open_interest
* trading_day
* symbol
* strike
* option_type
* expiry

Expected index parquet columns:

* timestamp
* open
* high
* low
* close
* volume
* trading_day
* symbol

Add validation and clear errors for missing columns.

Handle timezone-aware timestamps safely.

---

### 9. Preserve live mode

`strategies/examples/fib_premium_breakout_strategy.py` should still work live.

Existing behavior should remain the default when running:

```bash
python strategies/examples/fib_premium_breakout_strategy.py
```

or:

```bash
python strategies/examples/fib_premium_breakout_strategy.py live
```

Backtest should be run separately:

```bash
python strategies/examples/fib_premium_breakout_backtest.py --config backtest.json
```

---

### 10. Code quality

* Keep the refactor minimal.
* Do not over-engineer with too many interfaces.
* Only abstract `place_order()`.
* Keep strategy state updates in `enter_trade()` and `exit_quantity()`.
* Keep the existing fib target, target lot split, stop loss, and Heikin Ashi logic.
* Add comments where backtest fill assumptions differ from live trading.
* Make sure `BOOKING_PCTS` sum to 100.
* Make sure target lots sum to `TOTAL_LOTS`.
* Do not crash full backtest for one bad day; record skipped day and continue.

After implementation, show:

1. Files changed.
2. New class/function structure.
3. How live execution works.
4. How backtest execution works.
5. Sample `backtest.json`.
6. Output CSV files generated.
