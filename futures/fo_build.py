"""
fo_build.py — Stage 1b: raw cached zips -> tidy per-contract panel + membership.

Reads the raw cache written by fo_download.py and produces, under futures/data/:

  contracts.parquet   one row per (date, symbol, expiry): OHLC, settle, volume,
                      turnover, open interest, and LOT SIZE.
  membership.parquet  the point-in-time universe: (date, symbol) for every name
                      that actually had a single-stock futures contract that day.
                      This is the futures analogue of index membership and is the
                      whole answer to survivorship bias here — it is read off the
                      file, not off a current list.
  cash.parquet        NSE cash-segment EQ close/open per (date, symbol), the
                      independent spot reference for validation.

FORMAT HANDLING
---------------
old CSV (pre 2024-07-08)   INSTRUMENT==FUTSTK, EXPIRY_DT, OPEN..CLOSE, SETTLE_PR,
                           CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI
UDiFF   (2024-07-08 on)    FinInstrmTp==STF, XpryDt, OpnPric..ClsPric, SttlmPric,
                           TtlTradgVol, TtlTrfVal, OpnIntrst, ChngInOpnIntrst,
                           NewBrdLotQty, UndrlygPric

UNITS (verified across the cutover in fo_validate.py):
  open interest  SHARES (units) in both formats. OI in shares depends on the lot
                 size, which changes over time, so for LIQUIDITY RANKING we use
                 OI NOTIONAL = oi_shares x price (rupees) — the economically
                 meaningful measure of how much open interest a name carries.
  volume         CONTRACTS (lots) in both formats (old CONTRACTS, new
                 TtlTradgVol), confirmed by turnover/price/lot arithmetic. We
                 rank on OI as specified and keep volume only as a diagnostic.
  turnover       rupees (old VAL_INLAKH is in lakh, so x1e5).

LOT SIZE
--------
UDiFF publishes it (NewBrdLotQty). The old format does not, so we derive it, PER
CONTRACT (symbol, expiry) rather than per symbol-day — the lot belongs to the
contract, and two expiries of one name genuinely differ while a revision phases
in. Two independent estimates, reconciled over the contract's life:
  (a) value estimate:  lot ~ turnover / (contracts x typical price)
  (b) gcd of that contract's open interest and change-in-OI. OI is always an
      exact multiple of the lot, so the true lot DIVIDES this gcd.
We take the divisor of (b) closest to the median of (a). A contract that never
traded has no (a) and falls back to (b).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fo_download import cache_root

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# --------------------------------------------------------------------------- #
# parsing one day
# --------------------------------------------------------------------------- #
# column sets that identify a layout; a zip may hold several CSVs (the market
# activity report has ~12), so the CSV is chosen by layout, not by position.
# Taking "the first CSV" silently skipped 2021-03-30 for the whole project.
_LAYOUTS = ({"INSTRUMENT", "SYMBOL", "EXPIRY_DT", "SETTLE_PR"},        # old bhavcopy
            {"FinInstrmTp", "TckrSymb", "XpryDt"},                      # UDiFF
            {"INSTRUMENT", "SYMBOL", "EXP_DATE", "OPEN_INT*"},          # market activity report
            {"SERIES", "SYMBOL", "CLOSE"}, {"SctySrs", "TckrSymb"})     # cash, both layouts


def _read_zip(path: Path) -> pd.DataFrame | None:
    try:
        z = zipfile.ZipFile(path)
    except Exception:
        return None
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    for n in names:
        try:
            df = pd.read_csv(z.open(n), low_memory=False)
        except Exception:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if any(cols <= set(df.columns) for cols in _LAYOUTS):
            return df
    return None


def parse_fo_day(path: Path, date: pd.Timestamp) -> pd.DataFrame | None:
    raw = _read_zip(path)
    if raw is None or raw.empty:
        return None

    if "INSTRUMENT" in raw.columns and "EXPIRY_DT" in raw.columns:   # old layout
        f = raw[raw["INSTRUMENT"].astype(str).str.strip() == "FUTSTK"]
        if f.empty:
            return None
        out = pd.DataFrame({
            "symbol": f["SYMBOL"].astype(str).str.strip(),
            "expiry": pd.to_datetime(f["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce"),
            "open": pd.to_numeric(f["OPEN"], errors="coerce"),
            "high": pd.to_numeric(f["HIGH"], errors="coerce"),
            "low": pd.to_numeric(f["LOW"], errors="coerce"),
            "close": pd.to_numeric(f["CLOSE"], errors="coerce"),
            "settle": pd.to_numeric(f["SETTLE_PR"], errors="coerce"),
            "volume_lots": pd.to_numeric(f["CONTRACTS"], errors="coerce"),
            "turnover": pd.to_numeric(f["VAL_INLAKH"], errors="coerce") * 1e5,
            "oi_shares": pd.to_numeric(f["OPEN_INT"], errors="coerce"),
            "chg_oi_shares": pd.to_numeric(f["CHG_IN_OI"], errors="coerce"),
        })
        out["lot_size"] = np.nan            # derived below
        out["underlying"] = np.nan
        out["fmt"] = "old"
    elif "EXP_DATE" in raw.columns and "OPEN_INT*" in raw.columns:   # market activity report
        f = raw[raw["INSTRUMENT"].astype(str).str.strip() == "FUTSTK"]
        if f.empty:
            return None
        # no settlement price in this file: the close stands in for it (for a
        # front contract the two agree to the tick on a normal day); no change
        # in OI either, left NaN (the OI-reset detector then cannot fire).
        close = pd.to_numeric(f["CLOSE_PRICE"], errors="coerce")
        out = pd.DataFrame({
            "symbol": f["SYMBOL"].astype(str).str.strip(),
            "expiry": pd.to_datetime(f["EXP_DATE"].astype(str).str.strip(), format="%d/%m/%Y", errors="coerce"),
            "open": pd.to_numeric(f["OPEN_PRICE"], errors="coerce"),
            "high": pd.to_numeric(f["HI_PRICE"], errors="coerce"),
            "low": pd.to_numeric(f["LO_PRICE"], errors="coerce"),
            "close": close,
            "settle": close,
            "volume_lots": pd.to_numeric(f["NO_OF_CONT"], errors="coerce"),
            "turnover": pd.to_numeric(f["TRD_VAL"], errors="coerce"),
            "oi_shares": pd.to_numeric(f["OPEN_INT*"], errors="coerce"),
            "chg_oi_shares": np.nan,
        })
        out["lot_size"] = np.nan            # derived below, like the old format
        out["underlying"] = np.nan
        out["fmt"] = "old"                  # derived lot, old-era units
    elif "FinInstrmTp" in raw.columns:                    # UDiFF layout
        f = raw[raw["FinInstrmTp"].astype(str).str.strip() == "STF"]
        if f.empty:
            return None
        out = pd.DataFrame({
            "symbol": f["TckrSymb"].astype(str).str.strip(),
            "expiry": pd.to_datetime(f["XpryDt"], errors="coerce"),
            "open": pd.to_numeric(f["OpnPric"], errors="coerce"),
            "high": pd.to_numeric(f["HghPric"], errors="coerce"),
            "low": pd.to_numeric(f["LwPric"], errors="coerce"),
            "close": pd.to_numeric(f["ClsPric"], errors="coerce"),
            "settle": pd.to_numeric(f["SttlmPric"], errors="coerce"),
            "volume_lots": pd.to_numeric(f["TtlTradgVol"], errors="coerce"),
            "turnover": pd.to_numeric(f["TtlTrfVal"], errors="coerce"),
            "oi_shares": pd.to_numeric(f["OpnIntrst"], errors="coerce"),
            "chg_oi_shares": pd.to_numeric(f["ChngInOpnIntrst"], errors="coerce"),
            "lot_size": pd.to_numeric(f["NewBrdLotQty"], errors="coerce"),
            "underlying": pd.to_numeric(f["UndrlygPric"], errors="coerce"),
        })
        out["fmt"] = "udiff"
    else:
        return None

    out["date"] = date
    out = out[out["expiry"].notna() & out["symbol"].ne("")]
    # a zero OHLC print means "no trade", not a real price
    for c in ("open", "high", "low", "close"):
        out.loc[out[c] <= 0, c] = np.nan
    out.loc[out["settle"] <= 0, "settle"] = np.nan
    return out.reset_index(drop=True)


def parse_cm_day(path: Path, date: pd.Timestamp) -> pd.DataFrame | None:
    raw = _read_zip(path)
    if raw is None or raw.empty:
        return None
    if "SERIES" in raw.columns:
        e = raw[raw["SERIES"].astype(str).str.strip() == "EQ"]
        if e.empty:
            return None
        out = pd.DataFrame({
            "symbol": e["SYMBOL"].astype(str).str.strip(),
            "cash_open": pd.to_numeric(e["OPEN"], errors="coerce"),
            "cash_close": pd.to_numeric(e["CLOSE"], errors="coerce"),
            "cash_prevclose": pd.to_numeric(e.get("PREVCLOSE"), errors="coerce"),
            "cash_volume": pd.to_numeric(e.get("TOTTRDQTY"), errors="coerce"),
        })
    elif "SctySrs" in raw.columns:
        e = raw[raw["SctySrs"].astype(str).str.strip() == "EQ"]
        if e.empty:
            return None
        out = pd.DataFrame({
            "symbol": e["TckrSymb"].astype(str).str.strip(),
            "cash_open": pd.to_numeric(e["OpnPric"], errors="coerce"),
            "cash_close": pd.to_numeric(e["ClsPric"], errors="coerce"),
            "cash_prevclose": pd.to_numeric(e.get("PrvsClsgPric"), errors="coerce"),
            "cash_volume": pd.to_numeric(e.get("TtlTradgVol"), errors="coerce"),
        })
    else:
        return None
    out["date"] = date
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# lot size for the old format
# --------------------------------------------------------------------------- #
def _divisors(n: int) -> list[int]:
    n = int(n)
    if n <= 0:
        return []
    ds: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return sorted(ds)


def derive_lots_old(contracts: pd.DataFrame) -> pd.Series:
    """Lot size for every old-format row, derived PER CONTRACT.

    The lot is a property of a CONTRACT — (symbol, expiry) — not of a name on a
    day. Different expiries of one name really do carry different lots while a
    revision is phasing in: on 2023-05-03 AARTIIND's May and June contracts were
    on 850 while the newly listed July contract had open interest of 7000, which
    850 does not divide. Deriving one lot per symbol-day mixes those together and
    produces a series that jitters by a few percent every day (851, 850, 852,
    849, 857, 832...), which then trips the corporate-action detector constantly.

    Per contract we combine the same two independent estimates as before, but now
    over the contract's whole life, which is what makes them stable:
      (a) value estimate  turnover / (volume x price). Accurate to a few percent
          on any one day — the price proxy is not the true VWAP — but unbiased,
          so its MEDIAN over the contract's life is tight.
      (b) gcd of the contract's own open interest and change-in-OI over its life.
          OI is an exact multiple of the lot, so the true lot DIVIDES this.
    The answer is the divisor of (b) closest to (a).

    Returns a Series aligned to `contracts.index`.
    """
    d = contracts
    px = d[["open", "high", "low", "close"]].mean(axis=1)
    px = px.fillna(d["close"]).fillna(d["settle"])
    with np.errstate(divide="ignore", invalid="ignore"):
        est_row = np.where((d["volume_lots"] > 0) & (px > 0),
                           d["turnover"] / (d["volume_lots"] * px), np.nan)
    work = pd.DataFrame({"symbol": d["symbol"], "expiry": d["expiry"],
                         "est": est_row,
                         "oi": d["oi_shares"].abs().fillna(0),
                         "chg": d["chg_oi_shares"].abs().fillna(0)},
                        index=d.index)

    def _gcd_all(vals) -> int:
        g = 0
        for v in vals:
            if v and np.isfinite(v):
                g = math.gcd(g, int(v))
        return g

    lots: dict[tuple, float] = {}
    for key, g in work.groupby(["symbol", "expiry"], sort=False):
        est = np.nanmedian(g["est"].values) if g["est"].notna().any() else np.nan
        gcd = _gcd_all(np.concatenate([g["oi"].values, g["chg"].values]))
        if not np.isfinite(est) or est <= 0:
            lots[key] = float(gcd) if gcd > 0 else np.nan
            continue
        if gcd > 0:
            best = min(_divisors(gcd), key=lambda x: abs(x - est))
            lots[key] = float(best) if abs(best - est) / est < 0.05 else float(round(est))
        else:
            lots[key] = float(round(est))

    idx = pd.MultiIndex.from_arrays([d["symbol"], d["expiry"]])
    return pd.Series(pd.Series(lots).reindex(idx).to_numpy(), index=d.index,
                     name="lot_size")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(what: str) -> None:
    root = cache_root() / what
    files = sorted(root.glob("*.zip"))
    print(f"[info] {what}: {len(files)} cached day files")
    DATA.mkdir(parents=True, exist_ok=True)

    parts, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        date = pd.Timestamp(p.stem)
        df = parse_fo_day(p, date) if what == "fo" else parse_cm_day(p, date)
        if df is not None and not df.empty:
            parts.append(df)
        if i % 250 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {(time.time() - t0) / 60:.1f}m")
            sys.stdout.flush()

    if not parts:
        sys.exit(f"[fail] nothing parsed for {what}")
    sort_keys = ["date", "symbol"] + (["expiry"] if what == "fo" else [])
    all_df = (pd.concat(parts, ignore_index=True)
              .sort_values(sort_keys).reset_index(drop=True))

    if what == "fo":
        old = all_df["fmt"] == "old"
        if old.any():
            print(f"[info] deriving lot sizes for {int(old.sum()):,} old-format rows "
                  f"({all_df.loc[old, 'expiry'].nunique():,} distinct expiries)...")
            all_df.loc[old, "lot_size"] = derive_lots_old(all_df[old])
        # a contract listed but untraded still needs a lot; fill within the
        # contract only, never across contracts
        all_df["lot_size"] = (all_df.groupby(["symbol", "expiry"])["lot_size"]
                              .transform(lambda s: s.ffill().bfill()))
        all_df.to_parquet(DATA / "contracts.parquet", index=False)
        memb = (all_df[["date", "symbol"]].drop_duplicates()
                .sort_values(["date", "symbol"]).reset_index(drop=True))
        memb.to_parquet(DATA / "membership.parquet", index=False)
        print(f"\n[done] contracts: {len(all_df):,} rows, "
              f"{all_df['symbol'].nunique()} symbols, {all_df['date'].nunique()} days "
              f"({all_df['date'].min().date()} -> {all_df['date'].max().date()})")
        per_year = (memb.assign(y=memb["date"].dt.year)
                    .groupby("y")["symbol"].nunique())
        print("[done] distinct single-stock-futures names per year:")
        print(per_year.to_string())
    else:
        all_df.to_parquet(DATA / "cash.parquet", index=False)
        print(f"\n[done] cash: {len(all_df):,} rows, {all_df['symbol'].nunique()} "
              f"symbols, {all_df['date'].nunique()} days")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["fo", "cm"])
    build(ap.parse_args().what)
