#!/usr/bin/env python3
"""Download NIFTY index + options 1m bars from Dhan via OpenAlgo into HF-like parquet layout.

Output:
  <out>/index/NIFTY.parquet
  <out>/options/NIFTY/{YYYY-MM-DD}.parquet
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

# Live logs even when piped
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# strategies/tools/dhan -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
OPENALGO_ROOT = REPO_ROOT
sys.path.insert(0, str(OPENALGO_ROOT))
load_dotenv(OPENALGO_ROOT / ".env", override=True)

from openalgo import api  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_expiry_token(expiry: str) -> date:
    """Convert symtoken expiry like '11-AUG-26' -> date(2026, 8, 11)."""
    return datetime.strptime(expiry, "%d-%b-%y").date()


def chunk_ranges(start: date, end: date, max_days: int = 25) -> list[tuple[date, date]]:
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=max_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"])
            out = out.set_index("timestamp")
        else:
            out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("Asia/Kolkata")
    else:
        out.index = out.index.tz_convert("Asia/Kolkata")
    out = out.reset_index()
    ts_col = "timestamp" if "timestamp" in out.columns else out.columns[0]
    out = out.rename(columns={ts_col: "timestamp"})
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = 0
    if "oi" in out.columns and "open_interest" not in out.columns:
        out["open_interest"] = out["oi"]
    if "open_interest" not in out.columns:
        out["open_interest"] = 0
    out["trading_day"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    return out


def fetch_history(client, symbol: str, exchange: str, start: date, end: date) -> pd.DataFrame:
    frames = []
    for c_start, c_end in chunk_ranges(start, end, max_days=25):
        resp = client.history(
            symbol=symbol,
            exchange=exchange,
            interval="1m",
            start_date=c_start.isoformat(),
            end_date=c_end.isoformat(),
        )
        if isinstance(resp, dict):
            msg = str(resp.get("message", resp))
            if "No data" in msg or "not found" in msg.lower():
                continue
            raise RuntimeError(f"{symbol}: {msg}")
        part = normalize_history(resp)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def download_index(client, out_dir: Path, start: date, end: date) -> int:
    log(f"[index] NIFTY {start} -> {end}")
    df = fetch_history(client, "NIFTY", "NSE_INDEX", start, end)
    if df.empty:
        log("[index] no rows")
        return 0
    df["symbol"] = "NIFTY"
    path = out_dir / "index" / "NIFTY.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log(
        f"[index] wrote {len(df)} rows -> {path} "
        f"({df['trading_day'].min()} .. {df['trading_day'].max()})"
    )
    return len(df)


def list_near_expiries(db_path: Path, start: date, end: date) -> list[tuple[str, date]]:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT DISTINCT expiry FROM symtoken
        WHERE exchange='NFO'
          AND symbol LIKE 'NIFTY%'
          AND symbol NOT LIKE 'NIFTYNXT%'
          AND (instrumenttype='CE' OR instrumenttype='PE')
        """
    ).fetchall()
    con.close()
    out = []
    for (expiry_token,) in rows:
        try:
            exp_date = parse_expiry_token(expiry_token)
        except ValueError:
            continue
        if exp_date < start:
            continue
        if exp_date > end + timedelta(days=120):
            continue
        out.append((expiry_token, exp_date))
    out.sort(key=lambda x: x[1])
    return out


def list_contracts(
    db_path: Path,
    expiry_token: str,
    strike_min: float | None,
    strike_max: float | None,
) -> list[tuple[str, float, str]]:
    con = sqlite3.connect(db_path)
    sql = """
        SELECT symbol, strike, instrumenttype FROM symtoken
        WHERE exchange='NFO'
          AND expiry=?
          AND symbol LIKE 'NIFTY%'
          AND symbol NOT LIKE 'NIFTYNXT%'
          AND (instrumenttype='CE' OR instrumenttype='PE')
    """
    params: list = [expiry_token]
    if strike_min is not None:
        sql += " AND strike >= ?"
        params.append(strike_min)
    if strike_max is not None:
        sql += " AND strike <= ?"
        params.append(strike_max)
    sql += " ORDER BY strike, instrumenttype"
    rows = con.execute(sql, params).fetchall()
    con.close()
    return [(s, float(k), t) for s, k, t in rows]


def download_expiry(
    client,
    out_dir: Path,
    expiry_token: str,
    expiry_date: date,
    start: date,
    end: date,
    db_path: Path,
    strike_min: float | None,
    strike_max: float | None,
) -> int:
    contracts = list_contracts(db_path, expiry_token, strike_min, strike_max)
    req_end = min(end, expiry_date)
    req_start = start
    if req_start > req_end:
        log(f"[options] skip {expiry_token}: start after expiry")
        return 0

    log(
        f"[options] {expiry_token} ({expiry_date}) contracts={len(contracts)} "
        f"strikes=[{strike_min},{strike_max}] {req_start}->{req_end}"
    )
    frames = []
    ok = 0
    empty = 0
    errors = 0
    t0 = time.time()
    for i, (symbol, strike, opt_type) in enumerate(contracts, 1):
        try:
            df = fetch_history(client, symbol, "NFO", req_start, req_end)
        except Exception as exc:
            errors += 1
            if errors <= 8:
                log(f"  ERR {symbol}: {exc}")
            continue
        if df.empty:
            empty += 1
        else:
            ok += 1
            df["symbol"] = "NIFTY"
            df["strike"] = int(strike) if float(strike).is_integer() else strike
            df["option_type"] = opt_type
            df["expiry"] = expiry_date.isoformat()
            frames.append(df)
        if i % 10 == 0 or i == len(contracts):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            log(
                f"  progress {i}/{len(contracts)} ok={ok} empty={empty} err={errors} "
                f"({rate:.2f} sym/s, {elapsed:.0f}s)"
            )

    if not frames:
        log(f"[options] no data for {expiry_token}")
        return 0

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp", "strike", "option_type"]).reset_index(drop=True)
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "trading_day",
        "symbol",
        "strike",
        "option_type",
        "expiry",
    ]
    out = out[cols]
    path = out_dir / "options" / "NIFTY" / f"{expiry_date.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log(
        f"[options] wrote {len(out)} rows / {ok} contracts -> {path} "
        f"({out['trading_day'].min()} .. {out['trading_day'].max()})"
    )
    return len(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default=date.today().isoformat())
    default_out = REPO_ROOT.parent / "HistoricalData" / "india-options-data-dhan"
    if not default_out.parent.exists():
        default_out = Path(__file__).resolve().parent / "india-options-data-dhan"
    parser.add_argument(
        "--out",
        default=str(default_out),
        help="Output root (HF-like index/ + options/ layout)",
    )
    parser.add_argument(
        "--db",
        default=str(OPENALGO_ROOT / "db" / "openalgo.db"),
    )
    parser.add_argument("--strike-min", type=float, default=23000.0)
    parser.add_argument("--strike-max", type=float, default=25500.0)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Reuse existing index parquet if present",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = Path(args.out)
    db_path = Path(args.db)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENALGO_API_KEY")
    host = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST") or "http://127.0.0.1:5000"
    if not api_key:
        raise SystemExit("OPENALGO_API_KEY missing")

    client = api(api_key=api_key, host=host)
    log(f"Downloading {start} -> {end} into {out_dir}")
    log(f"Strike band: {args.strike_min} .. {args.strike_max}")

    index_path = out_dir / "index" / "NIFTY.parquet"
    if args.skip_index and index_path.exists():
        log(f"[index] skip existing {index_path}")
    else:
        download_index(client, out_dir, start, end)

    expiries = list_near_expiries(db_path, start, end)
    log(f"Near-term expiries: {[f'{t}({d})' for t, d in expiries]}")
    total = 0
    for expiry_token, expiry_date in expiries:
        total += download_expiry(
            client,
            out_dir,
            expiry_token,
            expiry_date,
            start,
            end,
            db_path,
            args.strike_min,
            args.strike_max,
        )

    log(f"Done. option rows={total}")
    log(
        "Note: expired weeklies (e.g. 07/14/21/28 Jul, 04 Aug) are usually absent "
        "from master contracts and cannot be fetched via OpenAlgo/Dhan symbol lookup."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
