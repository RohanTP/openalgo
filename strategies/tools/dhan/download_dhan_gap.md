# Download Dhan history → HF-style parquet

Script: [`download_dhan_gap.py`](./download_dhan_gap.py)

Fills **1-minute** NIFTY index + options bars from **Dhan** (via OpenAlgo) into the same folder layout as the Hugging Face dataset [`thetrademarkk/india-index-options-1m`](https://huggingface.co/datasets/thetrademarkk/india-index-options-1m).

Use this when HF parquet stops at a date (e.g. `2026-07-02`) and you need later days for backtests.

## Prerequisites

1. OpenAlgo running: `uv run app.py`
2. Logged in to **Dhan** in the OpenAlgo UI (valid broker session)
3. `OPENALGO_API_KEY` set in `.env`
4. Master contracts loaded (NFO symbols in `db/openalgo.db`)

## Output layout

```text
<out>/
  index/NIFTY.parquet
  options/NIFTY/{YYYY-MM-DD}.parquet   # one file per expiry (weekly/monthly)
```

Options schema (HF-compatible):

`timestamp, open, high, low, close, volume, open_interest, trading_day, symbol, strike, option_type, expiry`

## How it works

1. Downloads NIFTY index (`NSE_INDEX`) for `--start` → `--end`
2. Lists near-term NIFTY CE/PE from `symtoken` (skips far LEAPs and `NIFTYNXT*`)
3. Filters strikes to `--strike-min` / `--strike-max`
4. Calls OpenAlgo `client.history(..., interval="1m")` in ~25-day chunks (Dhan limit / rate limit)
5. Writes one parquet per **expiry date** (same convention as HF weeklies)

## Run

From the OpenAlgo repo root:

```bash
PYTHONUNBUFFERED=1 uv run python -u \
  strategies/tools/dhan/download_dhan_gap.py \
  --start 2026-07-01 \
  --end 2026-08-09 \
  --out ../HistoricalData/india-options-data-dhan \
  --strike-min 23000 \
  --strike-max 25500
```

Reuse an existing index file:

```bash
... --skip-index
```

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--start` / `--end` | `2026-07-01` / today | Trading-day range (IST) |
| `--out` | `../HistoricalData/india-options-data-dhan` if that folder exists | Output root |
| `--strike-min` / `--strike-max` | `23000` / `25500` | ATM band (narrower = faster) |
| `--skip-index` | off | Keep existing `index/NIFTY.parquet` |
| `--db` | `db/openalgo.db` | Symtoken DB for contract list |

## Limits / gotchas

- **Expired weeklies** (already removed from master contracts) cannot be fetched by OpenAlgo symbol lookup — HF will still have those older expiry files; Dhan fill only covers still-listed expiries.
- Dhan intraday: up to ~5 years retention, max ~90 days per raw request; script chunks ranges.
- Expect ~0.5 symbols/sec (broker rate limit). A 100-contract expiry takes ~3–4 minutes.
- Point `backtest.json` `options_parquet` / `index_parquet` at the Dhan folder (or a merged dataset) when backtesting the gap period.

## Related

- Strategy backtest config: [`backtest.json`](../../examples/fib_prem_breakout/backtest.json)
- HF dataset path used by backtests: `../HistoricalData/india-options-data/`
- Example Dhan fill used for Jul–Aug 2026 gap: `../HistoricalData/india-options-data-dhan/`
