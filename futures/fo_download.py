"""
fo_download.py — Stage 1a: bulk download + raw cache of the NSE archives we need.

Three archives, all free, all from archives.nseindia.com:

  fo   F&O bhavcopy      -> single-stock futures (FUTSTK / STF) per contract per day
  cm   cash bhavcopy     -> EQ close, used as the INDEPENDENT spot reference for
                            basis + corporate-action cross-checks
  pr   "price report" zip -> contains Bc<ddmmyy>.csv, NSE's own CORPORATE ACTION
                            calendar (SYMBOL, EX_DT, PURPOSE). This is the real
                            NSE corp-action source the futures README asked for;
                            yfinance .splits is only a fallback cross-check.

Both bhavcopy formats are handled by URL: the old per-month CSV archive before
2024-07-08 and the UDiFF layout after. If one 404s we try the other, because the
two formats overlapped for a while around the cutover.

WHAT IS CACHED: the raw .zip bytes, exactly as NSE served them, one file per
(archive, date). Parsing happens later in fo_build.py, so a parser change never
costs another download. A 404 (market holiday) writes a zero-byte `.missing`
marker so holidays are never re-requested.

CACHE LOCATION: the raw cache is ~2 GB. The repo lives inside a OneDrive folder,
so by default we write OUTSIDE it (%LOCALAPPDATA%/quant-stat-arb/fo_cache) rather
than pushing 2 GB into the user's cloud storage. Override with QSA_CACHE_DIR.

Usage:
    python fo_download.py --what fo --start 2015-01-01 --end 2024-12-31
    python fo_download.py --what cm
    python fo_download.py --what pr        # weekly cadence, see PR_EVERY_N_DAYS
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

CUTOVER = pd.Timestamp("2024-07-08")     # UDiFF replaces the old CSV layout

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# be polite: a small pool + a per-request pause, not a flood
MAX_WORKERS = 4
PAUSE_PER_REQUEST = 0.15      # seconds, per worker
MAX_RETRIES = 4
TIMEOUT = 45

# the Bc corp-action file inside each PR zip lists only a rolling window of
# upcoming ex-dates and drops entries once the ex-date arrives (verified: the
# RELIANCE-Jio demerger appears in the 17-Jul-2023 file but not the 20-Jul one),
# so we take EVERY trading day and union them. Sampling weekly would silently
# drop corporate actions.
PR_EVERY_N_DAYS = 1


def cache_root() -> Path:
    env = os.environ.get("QSA_CACHE_DIR")
    if env:
        root = Path(env)
    else:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
        root = Path(base) / "quant-stat-arb" / "fo_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def urls_for(what: str, d: pd.Timestamp) -> list[str]:
    """Candidate URLs for one archive on one date, best guess first."""
    ymd = d.strftime("%Y%m%d")
    mon = d.strftime("%b").upper()
    dd = d.strftime("%d")
    old_fo = (f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
              f"{d.year}/{mon}/fo{dd}{mon}{d.year}bhav.csv.zip")
    new_fo = (f"https://archives.nseindia.com/content/fo/"
              f"BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip")
    old_cm = (f"https://archives.nseindia.com/content/historical/EQUITIES/"
              f"{d.year}/{mon}/cm{dd}{mon}{d.year}bhav.csv.zip")
    new_cm = (f"https://archives.nseindia.com/content/cm/"
              f"BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip")
    if what == "fo":
        return [new_fo, old_fo] if d >= CUTOVER else [old_fo, new_fo]
    if what == "cm":
        return [new_cm, old_cm] if d >= CUTOVER else [old_cm, new_cm]
    if what == "pr":
        return [f"https://archives.nseindia.com/archives/equities/bhavcopy/pr/"
                f"PR{d.strftime('%d%m%y')}.zip"]
    raise ValueError(what)


_print_lock = threading.Lock()
_session_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _session_local.s = s
    return s


def fetch_one(what: str, d: pd.Timestamp, root: Path) -> str:
    """Download one date into the cache. Returns 'cached'|'ok'|'missing'|'error'."""
    out = root / what / f"{d.strftime('%Y%m%d')}.zip"
    marker = out.with_suffix(".missing")
    if out.exists() and out.stat().st_size > 0:
        return "cached"
    if marker.exists():
        return "cached"
    out.parent.mkdir(parents=True, exist_ok=True)

    sess = _session()
    saw_404 = False
    for attempt in range(1, MAX_RETRIES + 1):
        for url in urls_for(what, d):
            try:
                r = sess.get(url, timeout=TIMEOUT)
            except Exception:
                time.sleep(1.5 * attempt)
                continue
            if r.status_code == 200 and len(r.content) > 500:
                tmp = out.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(out)
                time.sleep(PAUSE_PER_REQUEST)
                return "ok"
            if r.status_code == 404:
                saw_404 = True
                continue
            time.sleep(1.5 * attempt)
        if saw_404 and attempt >= 2:
            break                      # a real holiday: every candidate URL 404s
        time.sleep(1.0 * attempt)

    if saw_404:
        marker.write_bytes(b"")        # holiday / genuinely absent
        return "missing"
    return "error"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download & cache NSE archives.")
    ap.add_argument("--what", required=True, choices=["fo", "cm", "pr"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    root = cache_root()
    days = pd.bdate_range(args.start, args.end)     # weekdays; holidays 404 -> marker
    if args.what == "pr":
        days = days[::PR_EVERY_N_DAYS]

    print(f"[info] archive={args.what}  dates={len(days)}  "
          f"{days[0].date()} -> {days[-1].date()}")
    print(f"[info] cache: {root / args.what}")
    sys.stdout.flush()

    counts = {"cached": 0, "ok": 0, "missing": 0, "error": 0}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status in ex.map(lambda d: fetch_one(args.what, d, root), days):
            counts[status] += 1
            done += 1
            if done % 100 == 0 or done == len(days):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (len(days) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(days)}] ok={counts['ok']} cached={counts['cached']} "
                      f"missing={counts['missing']} error={counts['error']} "
                      f"| {el/60:.1f}m elapsed, ETA {eta/60:.1f}m")
                sys.stdout.flush()

    print(f"[done] {args.what}: {counts}  in {(time.time()-t0)/60:.1f} min")
    if counts["error"]:
        print(f"[warn] {counts['error']} date(s) errored (not 404) — re-run to retry them.")


if __name__ == "__main__":
    main()
