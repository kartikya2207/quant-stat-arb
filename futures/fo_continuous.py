"""
fo_continuous.py — Stage 1d: continuous front-month series per name.

Turns the per-contract panel into one clean return series per symbol — the thing
the PCA actually sees — plus the front-month chain the execution layer trades.

THE ROLL
--------
We hold the nearest expiry until it is 2 TRADING days from expiring, then we are
on the next one. Trading days, not calendar days, so a holiday in expiry week
cannot leave us holding a contract into its last session. Expiry dates are known
when the contract is listed, so the chain uses no future information.

THE ADJUSTMENT — and why it is provably look-ahead-free
-------------------------------------------------------
The prototype used ADDITIVE (Panama) back-adjustment: on each roll, add the
inter-contract gap to every earlier price. That removes the gap, but it also
means the price printed at time t CHANGES every time a later roll happens — so
returns computed from it depend on future rolls. That is look-ahead, and the
brief forbids it.

We use RATIO (proportional) chaining instead. The daily return is the return of
the contract we actually held, on that contract's own prices:

    r(d) = px(d, F(d)) / ( px(d-1, F(d)) * f(d) ) - 1

where F(d) is the front contract on day d and f(d) is the corporate-action factor
from fo_corpactions.py (1.0 on a normal day). On a roll day the return is the NEW
contract's own overnight return, so the inter-contract basis gap never appears as
a return — which is exactly what "remove the roll gap" means for a returns-based
strategy.

The continuous level is then the cumulative product of those returns, anchored at
the FIRST observation. Under ratio chaining returns are invariant to the anchor,
so this is the same series a back-adjusted build would give — but anchoring
forward means the level at time t is a function of data up to t only. Nothing in
this file can see the future.

RE-LISTINGS BREAK THE CHAIN
---------------------------
The F&O universe is point-in-time: names enter and LEAVE it. A name's previous
row in this panel can therefore be years old, and chaining across that gap
invents an enormous fake return — BALRAMCHIN left the universe around Rs 74 and
came back on 2021-12-31 at Rs 369, which naive chaining prints as a +399% day.
A return exists only when the name was also listed on the immediately preceding
exchange session; otherwise the chain breaks and the level re-anchors.

Because the level re-anchors at a break, pct_change() of cont_close is NOT the
return series. cont_ret_wide.parquet is authoritative and is what the harness
consumes; the level is for plotting and validation.

MASKED DAYS
-----------
Where fo_corpactions could not vouch for a factor (demergers, rights, dividends
whose exchange adjustment the tape cannot confirm) the return is set to NaN: the PCA and the OU fit simply
do not see that name-day.

LIQUIDITY
---------
Ranking is by OPEN INTEREST, as specified — expressed in rupees
(oi_shares x price) so it is comparable across names and across the 2024-07-08
format change, and summed over all of a name's listed contracts so the number
does not lurch on roll days. Volume is carried only as a diagnostic.

Outputs (futures/data/):
  continuous.parquet       long: date, symbol, chain + clean return + level + OI
  cont_close_wide.parquet  index=date, cols=symbol — the PCA input
  cont_ret_wide.parquet    index=date, cols=symbol — returns, with masks as NaN
  oi_notional_wide.parquet index=date, cols=symbol — the liquidity ranking input
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ROLL_BEFORE_TD = 2          # roll when the front contract is this close to expiry

# Price basis for RETURNS. The exchange SETTLEMENT price is the official daily
# mark for every listed contract, whereas the CLOSE of a thin next-month contract
# can be a stale last-traded print (they differ on ~30% of contract-days). Both
# bases give the same roll behaviour in practice; settle is used because it is
# the price positions are actually marked and margined at.
# Fills still use the real traded OPEN; this only governs the signal series.
PRICE_BASIS = "settle"      # "settle" (settle, then close) or "close"


def _px(d: pd.DataFrame) -> pd.Series:
    if PRICE_BASIS == "settle":
        return d["settle"].fillna(d["close"])
    return d["close"].fillna(d["settle"])


def extended_calendar(cal: np.ndarray, weeks: int = 12) -> np.ndarray:
    """The observed session calendar extended past its last date with weekdays,
    so that expiries beyond the data are positioned on a real timeline. Weekdays
    over-count sessions by the (few) holidays in that stretch, which can only
    make dte LARGER, i.e. never triggers a roll or an untradable flag early."""
    cal = np.sort(np.asarray(cal, dtype="datetime64[ns]"))
    last = pd.Timestamp(cal[-1])
    ext = pd.bdate_range(last + pd.Timedelta(days=1), last + pd.Timedelta(weeks=weeks))
    return np.concatenate([cal, ext.to_numpy(dtype="datetime64[ns]")])


def build_chain(contracts: pd.DataFrame) -> pd.DataFrame:
    """Front-month chain per (symbol, date), with trading-day distance to expiry."""
    cal = np.sort(pd.DatetimeIndex(contracts["date"].unique()).to_numpy())

    c = contracts.copy()
    c["px"] = _px(c)
    c = c[c["px"].notna() & (c["px"] > 0) & (c["expiry"] >= c["date"])]

    # position of each date / expiry in the exchange trading calendar. An expiry
    # that is not itself a trading day (holiday) maps to the next session, which
    # is the conservative direction: it can only make us roll earlier.
    # The calendar is EXTENDED past the last observed date with weekdays, because
    # a contract expiring after the end of the data (the Jan-2025 contracts in
    # December 2024, or anything in a truncated build) must still be measured
    # against its real expiry session, not against the end of the file. Without
    # this every name looked untradable on the last two sessions of the sample
    # and the truncation test in fut_checks.py failed on the two days before
    # each cutoff.
    cal_ext = extended_calendar(cal)
    d_i = np.searchsorted(cal_ext, c["date"].to_numpy(), side="left")
    e_i = np.searchsorted(cal_ext, c["expiry"].to_numpy(), side="left")
    c["dte"] = (e_i - d_i).astype("int64")      # trading days until expiry

    # front = nearest expiry that is not already inside the roll window;
    # fall back to the nearest available if every listed expiry is inside it.
    ok = c[c["dte"] > ROLL_BEFORE_TD].sort_values(["symbol", "date", "expiry"])
    front = ok.groupby(["symbol", "date"], as_index=False).first()
    missing = (c[["symbol", "date"]].drop_duplicates()
               .merge(front[["symbol", "date"]], on=["symbol", "date"],
                      how="left", indicator=True))
    missing = missing[missing["_merge"] == "left_only"][["symbol", "date"]]
    if len(missing):
        fallback = (c.merge(missing, on=["symbol", "date"])
                    .sort_values(["symbol", "date", "expiry"])
                    .groupby(["symbol", "date"], as_index=False).first())
        front = pd.concat([front, fallback], ignore_index=True)

    front = front.sort_values(["symbol", "date"]).reset_index(drop=True)
    return front[["symbol", "date", "expiry", "px", "open", "close", "settle",
                  "lot_size", "oi_shares", "volume_lots", "dte"]] \
        .rename(columns={"expiry": "front_expiry", "px": "front_px",
                         "open": "front_open", "close": "front_close",
                         "settle": "front_settle", "oi_shares": "front_oi"})


def build_continuous(contracts: pd.DataFrame,
                     ca_factors: pd.DataFrame | None) -> pd.DataFrame:
    front = build_chain(contracts)

    # previous price OF THE CONTRACT WE HOLD TODAY, on the previous trading day
    px_by = (contracts.assign(px=_px)
             .set_index(["symbol", "expiry", "date"])["px"])
    px_by = px_by[~px_by.index.duplicated()]

    front = front.sort_values(["symbol", "date"]).reset_index(drop=True)
    front["prev_date"] = front.groupby("symbol")["date"].shift(1)
    front["prev_expiry"] = front.groupby("symbol")["front_expiry"].shift(1)
    front["is_roll"] = (front["front_expiry"] != front["prev_expiry"]) \
        & front["prev_expiry"].notna()

    # A NAME CAN LEAVE AND RE-ENTER THE F&O UNIVERSE. Its previous row may then be
    # months or years earlier, and chaining across that gap invents an enormous
    # fake return (BALRAMCHIN re-listed 2021-12-31 at 369 after an earlier stint
    # ending at 74 -> a fake +399% day). A return only exists when the name was
    # also listed on the IMMEDIATELY PRECEDING exchange session; otherwise the
    # chain breaks and starts afresh.
    cal = np.sort(pd.DatetimeIndex(contracts["date"].unique()).to_numpy())
    cur_i = np.searchsorted(cal, front["date"].to_numpy(), side="left")
    prev_i = np.searchsorted(cal, front["prev_date"].to_numpy(), side="left")
    front["chain_break"] = front["prev_date"].isna() | (cur_i - prev_i != 1)

    key = pd.MultiIndex.from_arrays(
        [front["symbol"], front["front_expiry"], front["prev_date"]])
    front["px_prev_same"] = px_by.reindex(key).to_numpy()
    # if today's contract had no print yesterday (first day it listed), fall back
    # to yesterday's front contract so the chain does not break
    key_old = pd.MultiIndex.from_arrays(
        [front["symbol"], front["prev_expiry"], front["prev_date"]])
    fallback = px_by.reindex(key_old).to_numpy()
    front["px_prev"] = np.where(np.isfinite(front["px_prev_same"]),
                                front["px_prev_same"], fallback)
    front["ret_basis"] = np.where(np.isfinite(front["px_prev_same"]),
                                  "same-contract", "prev-front")

    # corporate-action factor
    front["ca_factor"] = 1.0
    front["masked"] = False
    if ca_factors is not None and len(ca_factors):
        f = ca_factors.rename(columns={"factor": "_f", "mask_signal": "_m"})
        front = front.merge(f, on=["symbol", "date"], how="left")
        front["ca_factor"] = front["_f"].fillna(1.0)
        front["masked"] = front["_m"].fillna(False).astype(bool)
        front = front.drop(columns=["_f", "_m"])

    with np.errstate(divide="ignore", invalid="ignore"):
        front["ret"] = front["front_px"] / (front["px_prev"] * front["ca_factor"]) - 1.0
    front.loc[front["masked"], "ret"] = np.nan
    front.loc[~np.isfinite(front["ret"]), "ret"] = np.nan
    # a name's first day, or its first day back after dropping out of the F&O
    # universe, has no previous price: no return, not a zero return
    front.loc[front["chain_break"], "ret"] = np.nan

    # continuous level: cumulative product of clean returns, anchored FORWARD at
    # the first observed raw front price of each listing segment (so no future
    # information enters the level at any point, and a re-listing restarts at the
    # real price rather than continuing off a stale one).
    # NOTE: because the level re-anchors at a chain break, pct_change() of this
    # column is NOT the return series. cont_ret_wide.parquet is authoritative;
    # the level is for plotting and validation.
    front["segment"] = front.groupby("symbol")["chain_break"].cumsum()

    def _level(g: pd.DataFrame) -> pd.Series:
        r = g["ret"].fillna(0.0).to_numpy()
        lvl = np.empty(len(g))
        lvl[0] = g["front_px"].iloc[0]
        if len(g) > 1:
            lvl[1:] = lvl[0] * np.cumprod(1.0 + r[1:])
        return pd.Series(lvl, index=g.index)

    front["cont_close"] = (front.groupby(["symbol", "segment"], group_keys=False)
                           .apply(_level, include_groups=False))

    # liquidity: OI notional in rupees, over ALL of a name's listed contracts
    oi_all = contracts.assign(px=_px)
    oi_all["oi_notional"] = oi_all["oi_shares"] * oi_all["px"]
    oi_tot = (oi_all.groupby(["symbol", "date"], as_index=False)["oi_notional"].sum())
    front = front.merge(oi_tot, on=["symbol", "date"], how="left")
    front["front_oi_notional"] = front["front_oi"] * front["front_px"]

    # TRADABILITY. Normally the front contract has more than ROLL_BEFORE_TD days
    # left, because we roll before it does not. The exception is a name being
    # DELISTED from F&O: its last contract expires and no successor is listed, so
    # build_chain's fallback leaves us on an expiring contract (745 symbol-days,
    # 0.17%, all delistings such as JPPOWER, AMTEKAUTO, IOB, UCOBANK). We keep the
    # return — those prices are real and informative for the factor structure —
    # but mark the day untradable so the harness never opens a position in an
    # expiring contract and force-exits anything it still holds.
    front["tradable"] = front["dte"] > ROLL_BEFORE_TD
    # daily liquidity rank, 1 = most open interest. Ranked over EVERY listed name
    # so the rank does not depend on the exclusion list, which is only decided
    # later (fo_validate.py). Stage 2 drops excluded names, then takes the top N.
    front["oi_rank"] = (front.groupby("date")["oi_notional"]
                        .rank(ascending=False, method="first"))

    cols = ["date", "symbol", "front_expiry", "prev_expiry", "is_roll", "dte",
            "chain_break", "segment",
            "front_open", "front_close", "front_settle", "front_px", "px_prev",
            "ret_basis", "ca_factor", "masked", "ret", "cont_close",
            "tradable", "lot_size", "front_oi", "front_oi_notional",
            "oi_notional", "oi_rank",
            "volume_lots"]
    return front[cols].sort_values(["date", "symbol"]).reset_index(drop=True)


def main() -> None:
    contracts = pd.read_parquet(DATA / "contracts.parquet")
    ca_p = DATA / "ca_factors.parquet"
    ca = pd.read_parquet(ca_p) if ca_p.exists() else None
    print(f"[info] contracts: {len(contracts):,} rows; "
          f"CA factors: {0 if ca is None else len(ca)}")

    cont = build_continuous(contracts, ca)
    cont.to_parquet(DATA / "continuous.parquet", index=False)

    close_w = cont.pivot(index="date", columns="symbol", values="cont_close")
    ret_w = cont.pivot(index="date", columns="symbol", values="ret")
    oi_w = cont.pivot(index="date", columns="symbol", values="oi_notional")
    close_w.to_parquet(DATA / "cont_close_wide.parquet")
    ret_w.to_parquet(DATA / "cont_ret_wide.parquet")
    oi_w.to_parquet(DATA / "oi_notional_wide.parquet")

    n_roll = int(cont["is_roll"].sum())
    print(f"\n[done] continuous: {len(cont):,} symbol-days, "
          f"{cont['symbol'].nunique()} symbols, {cont['date'].nunique()} days")
    print(f"       rolls: {n_roll:,}   masked days: {int(cont['masked'].sum())}"
          f"   chain breaks (re-listings): {int(cont['chain_break'].sum())}")
    print(f"       untradable (expiring contract, F&O delisting): "
          f"{int((~cont['tradable']).sum()):,}")
    print(f"       wide frame: {close_w.shape[0]} dates x {close_w.shape[1]} symbols")
    per_year = cont.assign(y=cont["date"].dt.year).groupby("y")["symbol"].nunique()
    print("\n[done] names with a continuous series, per year:")
    print(per_year.to_string())


if __name__ == "__main__":
    main()
