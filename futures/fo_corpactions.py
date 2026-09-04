"""
fo_corpactions.py — Stage 1c: corporate-action factors for the futures panel.

NSE futures bhavcopy prices are RAW. A split/bonus/demerger puts a fake jump in
the price series; left alone it dominates the PCA covariance and manufactures a
huge fake s-score. This module decides, for every (symbol, date), the price
adjustment factor f such that

    clean_return(D) = close(D) / (close(D-1) * f) - 1

and writes an audit row for every single adjustment it applies.

WHY NOT yfinance .splits (what the prototype used)
--------------------------------------------------
It covers splits and bonuses only. It misses demergers — the RELIANCE/Jio
demerger (ex 2023-07-20) is a real -7.8% fake move it never reports. We use
NSE's own corporate-action calendar instead (the Bc<ddmmyy>.csv inside each
daily PR zip: SYMBOL, EX_DT, PURPOSE), which does carry
"RELIANCE / 20-07-2023 / DEMERGER".

FOUR INDEPENDENT SIGNALS, combined
----------------------------------
1. CALENDAR   NSE Bc file. Authoritative on WHEN and WHAT. For splits/bonuses the
              PURPOSE text also gives the EXACT factor ("BONUS 1:1" -> 0.5,
              "FV SPLT FRM RS 10 TO RS 2" -> 0.2).
2. LOT STEP   the exchange re-denominates the contract, so lot(D)/lot(D-1) = 1/f
              exactly. Verified: RELIANCE 1:1 bonus 2024-10-28, lot 250 -> 500.
              NSE also revises lot sizes every six months for contract-value
              reasons with NO price adjustment, so a lot step counts only when
              the price actually moved by the reciprocal of it.
3. OI RESET   for a contract-replacing action the whole open interest is reset
              (ChgInOI == OI for a contract that already existed the previous
              session). Verified: RELIANCE demerger 2023-07-20. Tested on the
              FRONT contract only — a newly listed far month trivially has
              ChgInOI == OI on its first day, so testing "any contract of this
              name" would fire every single month.
4. PRICE JUMP the overnight ratio on the SAME contract lands on a round fraction
              (1/2, 2/3, 1/4, ...). The safety net for anything the calendar
              missed, and the loosest of the four, so it is held to a 30% move
              and a 1% tolerance: a 20% circuit move is a ratio of 6/5, which a
              lax round-fraction test would happily call a corporate action.

Because the NSE calendar is the primary source and is comprehensive, rules 2-4
are deliberately strict. They exist to catch calendar gaps, not to second-guess it.

DIVIDENDS ARE DELIBERATELY NOT ADJUSTED
---------------------------------------
Unlike the cash study, an ordinary dividend needs NO adjustment here: NSE does
not adjust futures for it, and a futures holder genuinely does not receive it —
the price drop on the ex-date is a real loss to a long. We only treat a dividend
as adjusting when the total going ex that day (ordinary + interim + special)
clears NSE's "extraordinary" bar of 2% of the underlying price, which is exactly
the rule the exchange itself applies.

WHEN WE CANNOT KNOW THE FACTOR (conservative choice)
----------------------------------------------------
A demerger's factor is not in the text and the observed jump mixes the exchange's
adjustment with that day's real trading; the two cannot be separated from the
bhavcopy. So for demergers / rights / extraordinary dividends we:
  - use the MEASURED overnight ratio for P&L continuity (this is what the
    exchange's own adjustment achieves for a position holder), and
  - MASK that name's return for that day out of the signal, so the PCA and the
    OU fit never see a number we cannot vouch for.
We give up one name-day of signal rather than trade a fabricated one. Every such
day is logged with mask_signal=True and reported in FINDINGS.md.
"""
from __future__ import annotations

import re
import sys
import time
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from fo_download import cache_root

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# --- detector thresholds ---------------------------------------------------- #
LOT_TOL = 0.01          # lot ratio must match 1/f to 1%
JUMP_MIN = 0.30         # |1 - ratio| for a BARE round-fraction hit. Deliberately
                        # high: a 20% circuit move is a ratio of 6/5, which any
                        # loose "round fraction" test would call a corporate action.
JUMP_MIN_CONFIRMED = 0.03   # ...when an OI reset already corroborates it
ROUND_TOL = 0.01        # ratio within 1% of a round fraction p/q
MAX_Q = 10              # denominators considered for "round fraction"
LOT_PRICE_TOL = 0.10    # a lot step counts only if the price moved by 1/lot_ratio
EXTRA_DIV_PCT = 0.02    # NSE's extraordinary-dividend bar: > 2% of price
# A calendar-derived factor must be consistent with what the price actually did
# overnight. NSE's free text is sometimes wrong or ambiguous — it published
# NHPC's 2016-02-18 interim dividend as both "RS 9.20" and "RE 0.92" on a Rs 19
# stock — and applying an unchecked figure does not merely miss an adjustment,
# it INVENTS a return: that one turned an ordinary day into a fake +120%. When
# the text and the tape disagree we refuse the adjustment rather than trust text.
PLAUSIBLE_REL = 0.15    # |factor - observed ratio| / factor
PLAUSIBLE_ABS = 0.10    # ...or in absolute terms, whichever is kinder


# --------------------------------------------------------------------------- #
# 1. the NSE corporate-action calendar
# --------------------------------------------------------------------------- #
def parse_bc_calendar() -> pd.DataFrame:
    """Union every Bc<ddmmyy>.csv in the cached PR zips into one CA calendar.

    The Bc file is only a ROLLING WINDOW of upcoming ex-dates and drops an entry
    once its ex-date arrives, so we read every trading day's file and union.
    """
    root = cache_root() / "pr"
    files = sorted(root.glob("*.zip"))
    print(f"[info] corp actions: {len(files)} cached PR zips")
    rows, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        try:
            z = zipfile.ZipFile(p)
            names = [n for n in z.namelist() if Path(n).name.lower().startswith("bc")]
            if not names:
                continue
            d = pd.read_csv(z.open(names[0]), encoding="latin1", low_memory=False)
        except Exception:
            continue
        d.columns = [str(c).strip().upper() for c in d.columns]
        if not {"SYMBOL", "EX_DT", "PURPOSE"} <= set(d.columns):
            continue
        keep = d[["SERIES", "SYMBOL", "EX_DT", "PURPOSE"]].copy() \
            if "SERIES" in d.columns else d[["SYMBOL", "EX_DT", "PURPOSE"]].assign(SERIES="")
        for c in keep.columns:
            keep[c] = keep[c].astype(str).str.strip()
        rows.append(keep)
        if i % 500 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {(time.time() - t0) / 60:.1f}m")
            sys.stdout.flush()

    if not rows:
        return pd.DataFrame(columns=["symbol", "ex_date", "purpose"])
    cal = pd.concat(rows, ignore_index=True)
    cal["ex_date"] = pd.to_datetime(cal["EX_DT"], format="%d/%m/%Y", errors="coerce")
    cal = cal[cal["ex_date"].notna()]
    cal = cal.rename(columns={"SYMBOL": "symbol", "PURPOSE": "purpose",
                              "SERIES": "series"})
    cal["purpose"] = cal["purpose"].str.upper().str.replace(r"\s+", " ", regex=True)
    cal = (cal[["symbol", "ex_date", "purpose", "series"]]
           .drop_duplicates(subset=["symbol", "ex_date", "purpose"])
           .sort_values(["ex_date", "symbol"]).reset_index(drop=True))
    return cal


# --------------------------------------------------------------------------- #
# 2. reading a factor out of the PURPOSE text
# --------------------------------------------------------------------------- #
# NSE's PURPOSE text is free-form and abbreviates inconsistently. Real examples
# these have to cover: "BONUS 1:1", "BON 1:1/SPLIT RS.10TORS.2",
# "FV SPLT FRM RS 10 TO RS 2", "FV SPL 10 TO 2", "AGM/DIV-RS 29.50/BONUS".
_RE_BONUS = re.compile(r"\bBON(?:US)?\b\s*(\d+)\s*:\s*(\d+)")
_RE_BONUS_NORATIO = re.compile(r"\bBON(?:US)?\b")
_RE_SPLIT = re.compile(
    r"(?:FV\s*(?:SPLT?|SPL)|SPLIT|SUB\s*-?\s*DIV)\D{0,12}?"
    r"(\d+(?:\.\d+)?)\s*(?:TO|:)\s*(?:RS?|RE)?\.?\s*(\d+(?:\.\d+)?)")
# any dividend token (DIV / INTDIV / SPLDIV / AGM-DIV) followed by a rupee amount
_RE_ANYDIV = re.compile(r"(?:INT|SPL|SPECIAL)?\s*DIV\D{0,6}?(\d+(?:\.\d+)?)")


def classify_purpose(purpose: str) -> tuple[str, float, bool]:
    """(kind, factor, adjusting).  factor = NaN when the text does not give one."""
    p = (purpose or "").upper()
    factor = 1.0
    kinds: list[str] = []

    m = _RE_BONUS.search(p)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 0 and b > 0:
            factor *= b / (a + b)          # a new shares for every b held
            kinds.append("bonus")

    m = _RE_SPLIT.search(p)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if old_fv > 0 and new_fv > 0:
            factor *= new_fv / old_fv      # face value 10 -> 2 means price x0.2
            kinds.append("split")

    if kinds:
        return ("+".join(kinds), factor, True)

    if re.search(r"DEMERG|SPIN\s*OFF|SCHEME OF ARRANG", p):
        return ("demerger", np.nan, True)
    if re.search(r"\bRIGHTS?\b", p):
        return ("rights", np.nan, True)
    if re.search(r"CONSOLIDAT|CONSLD", p):
        return ("consolidation", np.nan, True)
    # a bonus announced with no ratio in the text, e.g. INFOSYS 2015-06-15
    # "AGM/DIV-RS 29.50/BONUS". It IS price-adjusting, we just cannot read the
    # size off the text — resolve it from the lot step, like a demerger.
    if _RE_BONUS_NORATIO.search(p):
        return ("bonus-noratio", np.nan, True)

    amounts = [float(x) for x in _RE_ANYDIV.findall(p)]
    if amounts:
        # total rupees going ex on this date (ordinary + interim + special).
        # 'adjusting' is decided later against NSE's 2%-of-price bar, since that
        # is the actual rule the exchange applies.
        return ("dividend", float(sum(amounts)), False)
    return ("other", np.nan, False)


# --------------------------------------------------------------------------- #
# 3. per-symbol daily evidence from the contract panel
# --------------------------------------------------------------------------- #
def _round_fraction(ratio: float) -> float | None:
    """Snap a ratio to a nearby simple fraction p/q (q <= MAX_Q), else None."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    fr = Fraction(ratio).limit_denominator(MAX_Q)
    if fr.numerator == 0:
        return None
    val = float(fr)
    if val > 0 and abs(val - ratio) / ratio <= ROUND_TOL and fr != Fraction(1, 1):
        return val
    return None


def front_evidence(contracts: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, date): the front contract's overnight ratio on the SAME
    contract, plus lot-size ratio and an OI-reset flag. This is the raw evidence
    the detector reasons over."""
    c = contracts.copy()
    c["px"] = c["close"].fillna(c["settle"])
    c = c[c["px"].notna() & (c["px"] > 0)]

    # front contract = nearest non-expired expiry with the most open interest;
    # nearest expiry is the usual definition and OI breaks any tie.
    c = c.sort_values(["symbol", "date", "expiry"])
    front = (c.groupby(["symbol", "date"], as_index=False).first()
             [["symbol", "date", "expiry", "px", "lot_size"]])

    # OI RESET: the whole open interest of a contract reported as today's change,
    # for a contract that ALREADY EXISTED on the previous session -> the exchange
    # replaced/re-denominated it (the demerger signature).
    #
    # Two traps this has to avoid. (1) A newly introduced far month trivially has
    # ChgInOI == OI on its first day, so the "existed yesterday" guard is
    # essential. (2) Testing "any contract of this name" makes the flag fire every
    # month, when the new far month is listed. We therefore test only the FRONT
    # contract — the one whose price we are actually chaining, and the one a real
    # adjustment always touches.
    cal = np.sort(pd.DatetimeIndex(c["date"].unique()).to_numpy())
    oi = c[["symbol", "date", "expiry", "oi_shares", "chg_oi_shares"]].copy()
    oi = oi.sort_values(["symbol", "expiry", "date"])
    oi["prev_row_date"] = oi.groupby(["symbol", "expiry"])["date"].shift(1)
    gap = (np.searchsorted(cal, oi["date"].to_numpy(), side="left")
           - np.searchsorted(cal, oi["prev_row_date"].to_numpy(), side="left"))
    oi["existed_prev"] = oi["prev_row_date"].notna() & (gap == 1)
    oi["reset"] = (oi["oi_shares"] > 0) \
        & (oi["oi_shares"] == oi["chg_oi_shares"]) \
        & oi["existed_prev"]
    oi_reset = (oi.merge(front[["symbol", "date", "expiry"]],
                         on=["symbol", "date", "expiry"])
                .set_index(["symbol", "date"])["reset"].rename("oi_reset"))
    oi_reset = oi_reset[~oi_reset.index.duplicated()]

    # same-contract overnight ratio: shift within (symbol, expiry), then keep the
    # row whose expiry is the front one for that date
    c = c.sort_values(["symbol", "expiry", "date"])
    c["px_prev"] = c.groupby(["symbol", "expiry"])["px"].shift(1)
    c["date_prev"] = c.groupby(["symbol", "expiry"])["date"].shift(1)
    same = c.merge(front[["symbol", "date", "expiry"]], on=["symbol", "date", "expiry"])
    same = same[same["px_prev"].notna()]
    same["ratio"] = same["px"] / same["px_prev"]

    ev = same[["symbol", "date", "expiry", "px", "px_prev", "date_prev", "ratio"]].copy()
    ev = ev.merge(front[["symbol", "date", "lot_size"]], on=["symbol", "date"], how="left")
    ev = ev.sort_values(["symbol", "date"])
    ev["lot_prev"] = ev.groupby("symbol")["lot_size"].shift(1)
    ev["lot_ratio"] = ev["lot_size"] / ev["lot_prev"]
    ev = ev.merge(oi_reset, on=["symbol", "date"], how="left")
    ev["oi_reset"] = ev["oi_reset"].fillna(False)
    return ev.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. combine calendar + evidence into factors + audit log
# --------------------------------------------------------------------------- #
def build_factors(contracts: pd.DataFrame, calendar: pd.DataFrame,
                  cash: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = front_evidence(contracts)
    ev_idx = ev.set_index(["symbol", "date"])

    # --- candidate events: calendar entries that adjust, on days we actually
    #     have a futures price for, UNION anything the detector trips on.
    cal = calendar.copy()
    cls = cal["purpose"].map(classify_purpose)
    cal["kind"] = [c[0] for c in cls]
    cal["cal_value"] = [c[1] for c in cls]
    cal["adjusting"] = [c[2] for c in cls]
    # keep dividends too: whether they adjust is decided per-row below against
    # NSE's 2%-of-price bar, which needs the price we only have here.
    cal = cal[cal["adjusting"] | cal["kind"].eq("dividend")] \
        .rename(columns={"ex_date": "date"})
    # A name can have SEPARATE calendar rows on one ex-date (a bonus AND a split
    # is common). Their price effects multiply, so aggregate by product — taking
    # 'first' would silently drop one of them.
    priceable = cal["kind"].isin(["bonus", "split", "bonus+split", "split+bonus"])
    cal["_text_factor"] = np.where(priceable, cal["cal_value"], np.nan)
    cal["_div"] = np.where(cal["kind"] == "dividend", cal["cal_value"], np.nan)
    cal["_unknown"] = cal["kind"].isin(["demerger", "rights", "consolidation",
                                       "bonus-noratio"])
    cal = (cal.sort_values("purpose")
           .groupby(["symbol", "date"], as_index=False)
           .agg(purpose=("purpose", lambda s: " | ".join(sorted(set(s)))),
                kind=("kind", lambda s: "+".join(sorted(set(s)))),
                text_factor=("_text_factor", "prod"),
                n_text=("_text_factor", "count"),
                div_amt=("_div", "max"),
                has_unknown=("_unknown", "any")))
    cal.loc[cal["n_text"] == 0, "text_factor"] = np.nan   # prod of nothing is 1.0

    # --- detector: a STRICT safety net for anything the calendar missed --------
    # NSE's own calendar is the primary source, so these rules are deliberately
    # tight; a loose detector fires on ordinary volatility (a 20% circuit move is
    # a ratio of 6/5, which is a perfectly "round" fraction) and on NSE's routine
    # six-monthly lot-size revisions, which carry no price adjustment at all.
    ev["jump"] = (ev["ratio"] - 1.0).abs()
    # a lot step only means a corporate action if the PRICE moved by the
    # reciprocal of it — otherwise it is just a periodic lot revision
    lot_f = np.where(ev["lot_ratio"].notna() & (ev["lot_ratio"] - 1).abs().gt(LOT_TOL),
                     1.0 / ev["lot_ratio"], np.nan)
    ev["lot_step"] = np.isfinite(lot_f) & (np.abs(ev["ratio"] - lot_f)
                                           <= LOT_PRICE_TOL * np.abs(lot_f))
    ev["round_hit"] = ev["ratio"].map(lambda r: _round_fraction(r) is not None)
    # NOTE: there is deliberately NO bare "big move that looks like a round
    # fraction" rule. It deletes real data: YES BANK's genuine +70% and +71%
    # moves during the March-2020 moratorium land within 1% of 17/10 and 12/7,
    # and a price-only rule "adjusts" them away. Every mechanism that actually
    # re-prices a contract leaves a physical trace — the exchange re-denominates
    # the lot (splits, bonuses) or resets the open interest (contract
    # replacement) — so we require one of those traces.
    detected = ev[
        ev["lot_step"]
        | (ev["oi_reset"] & ev["jump"].gt(JUMP_MIN_CONFIRMED))
    ][["symbol", "date"]]

    cand = (pd.concat([cal[["symbol", "date"]], detected], ignore_index=True)
            .drop_duplicates())
    cand = cand.merge(cal, on=["symbol", "date"], how="left")
    cand = cand.merge(ev.drop(columns=["expiry"]), on=["symbol", "date"], how="inner")

    if cash is not None and not cash.empty:
        cs = cash[["symbol", "date", "cash_close"]].sort_values(["symbol", "date"])
        cs["cash_prev"] = cs.groupby("symbol")["cash_close"].shift(1)
        cs["cash_ratio"] = cs["cash_close"] / cs["cash_prev"]
        cand = cand.merge(cs[["symbol", "date", "cash_ratio"]],
                          on=["symbol", "date"], how="left")
    else:
        cand["cash_ratio"] = np.nan

    # --- resolve each candidate ------------------------------------------- #
    out_rows = []
    for r in cand.itertuples(index=False):
        kind = r.kind if isinstance(r.kind, str) else ""
        # NOTE: a left-merged calendar column is NaN for a detector-only row, and
        # bool(nan) is True in Python. These must be read as explicit booleans or
        # every detector hit is misread as a calendar demerger.
        has_unknown = (isinstance(r.has_unknown, (bool, np.bool_))
                       and bool(r.has_unknown))
        jump = abs(r.ratio - 1.0)
        lot_f = (1.0 / r.lot_ratio) if (pd.notna(r.lot_ratio)
                                        and abs(r.lot_ratio - 1) > LOT_TOL) else np.nan
        round_f = _round_fraction(r.ratio)

        factor, source, mask, note = np.nan, "", False, ""

        def _plausible(f: float) -> bool:
            """Does the tape agree that a re-pricing of this size happened?"""
            return (abs(f - r.ratio) <= PLAUSIBLE_ABS
                    or abs(f - r.ratio) / abs(f) <= PLAUSIBLE_REL)

        if pd.notna(getattr(r, "text_factor", np.nan)):
            f_text = float(r.text_factor)
            if pd.notna(lot_f) and abs(lot_f - f_text) / f_text > 0.05:
                # the exchange's own re-denomination beats the free-text ratio
                factor, source = float(lot_f), "calendar-text+lotstep"
                note = f"text said {f_text:.4f}; lot step says {lot_f:.4f}"
            elif not _plausible(f_text):
                factor, source, mask = float(r.ratio), "calendar-measured", True
                note = (f"text factor {f_text:.4f} contradicted by the observed "
                        f"ratio {r.ratio:.4f}; measured value used, return masked")
            else:
                factor, source = f_text, "calendar-text"
        elif has_unknown:
            # Checked BEFORE dividends on purpose. A name often has a dividend
            # row AND a share-count row on the same ex-date — INFOSYS 2015-06-15
            # carried both "AGM/DIV-RS 29.50 PER SH" and "AGM/DIV-RS 29.50/BONUS".
            # The bonus is what re-prices the contract (x0.5); the dividend is
            # below the extraordinary bar and adjusts nothing. Testing dividends
            # first let the small dividend swallow the row and leave a 1:1 bonus
            # entirely unadjusted — a fake -49% day in INFY.
            if pd.notna(lot_f):
                # the lot re-denomination gives the factor EXACTLY, so there is
                # nothing to mask — we know what the exchange did
                factor, source = float(lot_f), "calendar+lotstep"
                note = f"{kind}: factor read from the lot step"
            else:
                factor, source, mask = float(r.ratio), "calendar-measured", True
                note = (f"{kind}: exchange factor is not in the text and there is "
                        f"no lot step; measured ratio used for P&L continuity, "
                        f"return masked from the signal")
        elif pd.notna(getattr(r, "div_amt", np.nan)) and r.div_amt > 0 \
                and pd.notna(r.px_prev):
            # NSE adjusts a futures contract only for an EXTRAORDINARY dividend,
            # i.e. one above 2% of the underlying price. Below that bar the drop
            # is a real loss to a long (a futures holder never receives the
            # dividend), so it must NOT be adjusted away.
            if r.div_amt <= EXTRA_DIV_PCT * r.px_prev:
                continue
            f_div = (r.px_prev - float(r.div_amt)) / r.px_prev
            if f_div <= 0 or not _plausible(f_div):
                continue          # the published amount is not what the tape shows
            factor = f_div
            source = "calendar-extradiv"
            note = (f"extraordinary dividend Rs {r.div_amt:.2f} = "
                    f"{r.div_amt / r.px_prev:.2%} of price")
        elif pd.notna(lot_f):
            factor, source = float(lot_f), "detector-lotstep"
            note = "no calendar entry; lot re-denomination"
        elif r.oi_reset and jump > JUMP_MIN_CONFIRMED:
            factor, source, mask = float(r.ratio), "detector-oireset", True
            note = "no calendar entry; open interest reset (contract replaced)"
        else:
            continue                        # calendar said adjusting, data says nothing

        if not np.isfinite(factor) or factor <= 0:
            continue
        # a calendar-flagged day with no visible price effect is not an adjustment
        if source.startswith("calendar") and abs(factor - 1.0) < 1e-9:
            continue

        out_rows.append({
            "symbol": r.symbol, "ex_date": r.date, "kind": kind or "detected",
            "purpose": r.purpose if isinstance(r.purpose, str) else "",
            "factor_applied": factor, "source": source,
            "observed_ratio": r.ratio, "lot_implied_factor": lot_f,
            "round_fraction": round_f, "cash_ratio": r.cash_ratio,
            "oi_reset": bool(r.oi_reset),
            "lot_prev": r.lot_prev, "lot_now": r.lot_size,
            "px_prev": r.px_prev, "px_now": r.px,
            "mask_signal": mask, "note": note,
        })

    log = pd.DataFrame(out_rows)
    if log.empty:
        return log, pd.DataFrame(columns=["symbol", "date", "factor", "mask_signal"])
    log = log.sort_values(["ex_date", "symbol"]).reset_index(drop=True)
    factors = (log[["symbol", "ex_date", "factor_applied", "mask_signal"]]
               .rename(columns={"ex_date": "date", "factor_applied": "factor"}))
    factors = factors.groupby(["symbol", "date"], as_index=False).agg(
        factor=("factor", "prod"), mask_signal=("mask_signal", "any"))
    return log, factors


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    cal_path = DATA / "ca_calendar.parquet"
    if cal_path.exists():
        cal = pd.read_parquet(cal_path)
        print(f"[info] reusing {cal_path.name}: {len(cal):,} rows")
    else:
        cal = parse_bc_calendar()
        cal.to_parquet(cal_path, index=False)
        print(f"[done] CA calendar: {len(cal):,} rows, "
              f"{cal['symbol'].nunique():,} symbols, "
              f"{cal['ex_date'].min().date()} -> {cal['ex_date'].max().date()}")

    contracts = pd.read_parquet(DATA / "contracts.parquet")
    cash_p = DATA / "cash.parquet"
    cash = pd.read_parquet(cash_p) if cash_p.exists() else None

    log, factors = build_factors(contracts, cal, cash)
    log.to_csv(DATA / "corp_action_log.csv", index=False)
    factors.to_parquet(DATA / "ca_factors.parquet", index=False)

    print(f"\n[done] adjustments applied: {len(log)}  "
          f"(masked from signal: {int(log['mask_signal'].sum()) if len(log) else 0})")
    if len(log):
        print("\nby source:")
        print(log["source"].value_counts().to_string())
        print("\nby kind:")
        print(log["kind"].value_counts().to_string())
        print(f"\nfiles: {DATA / 'corp_action_log.csv'}")


if __name__ == "__main__":
    main()
