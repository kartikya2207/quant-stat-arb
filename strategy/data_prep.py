"""
data_prep.py — Step 1 of the PCA-residual stat-arb build.

WHAT THIS DOES
--------------
Downloads ADJUSTED daily OHLCV for a universe of NSE stocks via yfinance,
validates the data (catches bad ticks / gaps / corporate-action artifacts),
and saves a clean per-symbol parquet plus a combined wide-format parquet that
the cross-sectional harness will read.

WHY ADJUSTED (read GUIDELINES.md §4.3 gap 2)
--------------------------------------------
PCA runs on RETURNS. Unadjusted prices contain fake jumps on split/dividend
days (a 1-for-2 split looks like a -50% crash that never happened). Those fake
returns dominate the covariance matrix and corrupt the PCA factors -> garbage
signal. yfinance auto_adjust=True returns split/dividend-adjusted prices, which
is exactly what we need. (NOTE: for MULTI-symbol downloads yfinance historically
defaulted auto_adjust=False, so we set it EXPLICITLY below.)

USAGE
-----
    python data_prep.py --start 2019-01-01 --end 2024-12-31 \
        --symbols-file universe.txt --out-dir ./data

`universe.txt` = one NSE symbol per line WITHOUT the .NS suffix, e.g.
    RELIANCE
    TCS
    HDFCBANK
    ...

If --symbols-file is omitted, a small built-in demo list is used so you can
smoke-test the pipeline before assembling the full ~250-300 name universe.

OUTPUT
------
    <out-dir>/per_symbol/<SYMBOL>.parquet     # one tidy file per stock
    <out-dir>/prices_wide.parquet             # wide: index=date, cols=symbols (adj close)
    <out-dir>/volume_wide.parquet             # wide: index=date, cols=symbols (volume)
    <out-dir>/data_quality_report.csv         # per-symbol validation summary

This is a DATA-PREP tool, run once (or whenever you refresh data). The harness
reads the parquet files; it never hits the network. That keeps backtests fast
and reproducible.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed. Run:  pip install yfinance --upgrade")


# --------------------------------------------------------------------------- #
# Validation thresholds — deliberately conservative. These FLAG suspicious data
# for review; they don't silently delete it. Honesty about data quality is part
# of the project (GUIDELINES.md §1.10).
# --------------------------------------------------------------------------- #
MAX_ABS_DAILY_RETURN = 0.50   # a >50% single-day move is suspicious for a large/mid cap
MIN_TRADING_DAYS_PER_YEAR = 200  # NSE has ~250 trading days/yr; far fewer => gaps
MAX_CONSECUTIVE_FLAT_DAYS = 5    # many identical closes in a row => stale/illiquid


DEMO_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT",
]


def load_symbols(symbols_file: str | None) -> list[str]:
    if symbols_file is None:
        print(f"[info] no --symbols-file given; using built-in demo list "
              f"({len(DEMO_SYMBOLS)} names) for a smoke test.")
        return DEMO_SYMBOLS
    path = Path(symbols_file)
    if not path.exists():
        sys.exit(f"symbols file not found: {path}")
    syms = [ln.strip().upper() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    # de-dupe, preserve order
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    print(f"[info] loaded {len(out)} symbols from {path}")
    return out


def download_one(symbol: str, start: str, end: str,
                 max_retries: int = 3) -> pd.DataFrame | None:
    """Download one symbol's ADJUSTED daily OHLCV. Returns tidy DataFrame or None."""
    ticker = f"{symbol}.NS"
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,      # <-- CRITICAL: split/dividend adjusted
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                print(f"  [warn] {symbol}: empty response (attempt {attempt})")
                time.sleep(1.0 * attempt)
                continue
            # yfinance may return a MultiIndex column frame for single tickers in
            # some versions; flatten to simple OHLCV.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str).rename(columns={
                "Open": "Open", "High": "High", "Low": "Low",
                "Close": "Close", "Volume": "Volume",
            })
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)  # tz-naive (framework wants this)
            df.index.name = "Date"
            return df
        except Exception as e:  # noqa: BLE001 - we want to keep going across symbols
            print(f"  [warn] {symbol}: error on attempt {attempt}: {e}")
            time.sleep(1.0 * attempt)
    print(f"  [fail] {symbol}: gave up after {max_retries} attempts")
    return None


def validate(symbol: str, df: pd.DataFrame) -> dict:
    """Return a dict of data-quality flags for this symbol. Does not mutate df."""
    out = {"symbol": symbol, "rows": len(df), "ok": True, "issues": []}

    if df.empty:
        out["ok"] = False
        out["issues"].append("empty")
        return out

    close = df["Close"].astype(float)
    rets = close.pct_change()

    # 1) extreme daily returns (possible bad tick or unhandled corporate action)
    extreme = rets.abs() > MAX_ABS_DAILY_RETURN
    n_extreme = int(extreme.sum())
    if n_extreme > 0:
        out["issues"].append(f"{n_extreme} day(s) with |return|>{MAX_ABS_DAILY_RETURN:.0%}")
        worst_date = rets.abs().idxmax()
        out["worst_return"] = float(rets.loc[worst_date])
        out["worst_date"] = str(worst_date.date())

    # 2) coverage — enough trading days per calendar year?
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    days_per_year = len(df) / years
    out["days_per_year"] = round(days_per_year, 1)
    if days_per_year < MIN_TRADING_DAYS_PER_YEAR:
        out["issues"].append(f"low coverage ({days_per_year:.0f} days/yr)")

    # 3) zero / negative prices or volumes
    if (close <= 0).any():
        out["issues"].append("non-positive close present")
    if (df["Volume"].astype(float) < 0).any():
        out["issues"].append("negative volume present")
    n_zero_vol = int((df["Volume"].astype(float) == 0).sum())
    if n_zero_vol > 0:
        out["issues"].append(f"{n_zero_vol} zero-volume day(s)")

    # 4) long flat runs (stale/illiquid)
    flat = (close.diff() == 0)
    max_run = _max_consecutive_true(flat)
    out["max_flat_run"] = int(max_run)
    if max_run > MAX_CONSECUTIVE_FLAT_DAYS:
        out["issues"].append(f"{max_run} consecutive flat closes")

    # 5) missing values
    n_nan = int(df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
    if n_nan > 0:
        out["issues"].append(f"{n_nan} row(s) with NaN OHLC")

    out["ok"] = len(out["issues"]) == 0
    return out


def _max_consecutive_true(mask: pd.Series) -> int:
    best = run = 0
    for v in mask.to_numpy():
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Download & validate adjusted daily NSE data.")
    ap.add_argument("--start", default="2019-01-01", help="YYYY-MM-DD (>=5yr recommended)")
    ap.add_argument("--end", default="2024-12-31", help="YYYY-MM-DD")
    ap.add_argument("--symbols-file", default=None, help="one NSE symbol per line, no .NS")
    ap.add_argument("--out-dir", default="./data", help="output directory")
    ap.add_argument("--sleep", type=float, default=0.4, help="pause between symbols (sec)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    per_symbol_dir = out_dir / "per_symbol"
    per_symbol_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_symbols(args.symbols_file)

    close_cols: dict[str, pd.Series] = {}
    volume_cols: dict[str, pd.Series] = {}
    open_cols: dict[str, pd.Series] = {}
    reports: list[dict] = []

    print(f"\n[info] downloading {len(symbols)} symbols, {args.start} -> {args.end}\n")
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")
        df = download_one(sym, args.start, args.end)
        if df is None:
            reports.append({"symbol": sym, "rows": 0, "ok": False,
                            "issues": ["download failed"]})
            continue

        rep = validate(sym, df)
        rep["issues"] = "; ".join(rep["issues"]) if rep["issues"] else ""
        reports.append(rep)

        # save per-symbol tidy file regardless of flags (flags are for review)
        df.to_parquet(per_symbol_dir / f"{sym}.parquet")
        close_cols[sym] = df["Close"].astype(float)
        volume_cols[sym] = df["Volume"].astype(float)
        if "Open" in df.columns:
            open_cols[sym] = df["Open"].astype(float)

        time.sleep(args.sleep)  # be polite to the data source

    if not close_cols:
        sys.exit("\n[fail] no data downloaded for any symbol. Check connectivity/symbols.")

    # combined wide frames (index=date, columns=symbols). Outer-join on dates so
    # every trading day present for ANY symbol is a row; missing cells are NaN
    # and must be handled in the harness (forward-fill / drop, by design choice).
    prices_wide = pd.DataFrame(close_cols).sort_index()
    volume_wide = pd.DataFrame(volume_cols).sort_index()
    prices_wide.to_parquet(out_dir / "prices_wide.parquet")
    volume_wide.to_parquet(out_dir / "volume_wide.parquet")
    if open_cols:
        open_wide = pd.DataFrame(open_cols).sort_index()
        open_wide.to_parquet(out_dir / "open_wide.parquet")

    report_df = pd.DataFrame(reports)
    report_df.to_parquet(out_dir / "data_quality_report.parquet") if False else None
    report_df.to_csv(out_dir / "data_quality_report.csv", index=False)

    # ---- summary to console ----
    n_ok = int(report_df["ok"].sum()) if "ok" in report_df else 0
    n_flagged = len(report_df) - n_ok
    print("\n" + "=" * 60)
    print(f"DONE. {len(close_cols)} symbols saved.")
    print(f"  clean: {n_ok}   flagged for review: {n_flagged}")
    print(f"  prices_wide: {prices_wide.shape[0]} dates x {prices_wide.shape[1]} symbols")
    print(f"  files in: {out_dir.resolve()}")
    if n_flagged:
        print("\n  Flagged symbols (see data_quality_report.csv):")
        flagged = report_df[~report_df["ok"]][["symbol", "issues"]]
        for _, r in flagged.iterrows():
            print(f"    {r['symbol']:<15} {r['issues']}")
        print("\n  NOTE: flags are for REVIEW, not auto-deletion. Inspect before"
              " including/excluding a name from the PCA universe.")
    print("=" * 60)


if __name__ == "__main__":
    main()
