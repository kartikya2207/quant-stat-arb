"""
fut_validation_events.py — Stage 3 (extended): known-answer validation of the
paths the synthetic-price suite cannot reach.

Every one of the eight bugs found in this project lived in code that
fut_validation.py's data never exercises: corporate actions, delistings,
re-listings, vanished contracts and data boundaries. This suite builds a small
synthetic F&O world in which those events are SCRIPTED with known answers, runs
it through the REAL Stage 1 corporate-action and chain code
(fo_corpactions.build_factors, fo_continuous.build_continuous) and the REAL
execution layer (fut_execute.execute_day on a FuturesMarket), and asserts:

  E1  1:1 bonus mid-sample, lot re-denominated ("BONUS 1:1")
  E2  10 -> 5 face-value split, lot re-denominated ("FV SPLT FRM RS 10 TO RS 5")
  E3  extraordinary dividend the exchange adjusts, text "INTDV/SPDV - RS 8 & 4"
      (the abbreviation that caused bug 8)
  E4  ordinary dividend below the bar, text "DIVIDEND- 6% PER ANNUM" (a percent
      of face value, must NOT be read as Rs 6); futures pre-discount it
  E5  a 3% dividend under the 5% regime, text "DIVIDEND - RS 3 PER SH": the
      tape says the exchange did not adjust -> no factor even though 3% > 2%
  E6  demerger with an unknown factor ("DEMERGER"): measured ratio, masked
  E7  a name delisted mid-sample while a position is held
  E8  a name that leaves the F&O universe and re-enters months later at 3x
  E9  a contract that vanishes from the tape before expiry while held
  E10 a position held across the final bar
  E11 a contract whose expiry falls after the data ends (the last two sessions
      must not look like a delisting)
  E12 a routine lot revision with no price effect and no calendar entry
      (must not become a corporate action)
  E13 every newly listed far-month contract has ChgInOI == OI on its first day
      (must not trip the OI-reset detector)
  E14 "INTDIV - RS. 2.50 PER SH" with a tape that says adjusted

Expected answers are stated before the run. Where the code's behaviour is a
documented choice rather than the only right answer (E6: the masked day's real
move is absorbed into the measured factor; E9: closed at the last settle), the
documented behaviour is what is asserted, and the case says so.

Prices are built from a CLEAN return path r_t per name (the economic move) and
the raw tape is derived from it: futures = spot (zero basis), corporate actions
are multiplicative level shifts of the raw tape by the known factor, so the
correct continuous return on every day, including ex-dates, is exactly r_t,
and the correct P&L of a position at zero cost is notional_entry x (cumulative
clean return).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "strategy"))
from fo_corpactions import build_factors                 # noqa: E402
from fo_continuous import build_continuous, ROLL_BEFORE_TD  # noqa: E402
from fut_costs import CostConfig                          # noqa: E402
from fut_execute import ExecConfig, execute_day           # noqa: E402
from fut_ledger import Ledger                             # noqa: E402
from fut_market import FuturesMarket                      # noqa: E402
from harness_decide import Decisions, EntryOrder, ExitOrder  # noqa: E402

CAPITAL = 1e8
RESULTS: list[dict] = []


def check(case: str, what: str, got, expected, tol: float = 1e-9, note: str = "") -> None:
    if isinstance(expected, (float, int, np.floating, np.integer)) and not isinstance(expected, bool):
        ok = bool(np.isfinite(got)) and abs(float(got) - float(expected)) <= tol * max(1.0, abs(float(expected)))
    else:
        ok = got == expected
    RESULTS.append({"case": case, "what": what, "expected": expected, "got": got, "pass": ok, "note": note})


# --------------------------------------------------------------------------- #
# the synthetic world
# --------------------------------------------------------------------------- #
@dataclass
class World:
    dates: pd.DatetimeIndex
    expiries: pd.DatetimeIndex
    clean: pd.DataFrame          # clean price path per name (what a holder economically owns)
    contracts: pd.DataFrame
    cash: pd.DataFrame
    calendar: pd.DataFrame
    events: dict                 # name -> dict(kind, ex_date, factor, ...)


def last_thursdays(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    out = []
    for m in pd.period_range(dates[0], dates[-1] + pd.offsets.MonthEnd(2), freq="M"):
        d = m.to_timestamp(how="end").normalize()
        while d.weekday() != 3:
            d -= pd.Timedelta(days=1)
        out.append(d)
    return pd.DatetimeIndex(out)


def build_world(seed: int = 11) -> World:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", "2022-03-31")          # 2022-03-31 is a Thursday expiry
    n = len(dates)
    expiries = last_thursdays(dates)
    names = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA", "IOTA", "KAPPA", "LAMBDA", "MU"]
    p0 = {"ALPHA": 500.0, "BETA": 1200.0, "GAMMA": 100.0, "DELTA": 100.0, "EPSILON": 100.0, "ZETA": 300.0,
          "ETA": 80.0, "THETA": 50.0, "IOTA": 900.0, "KAPPA": 80.0, "LAMBDA": 250.0, "MU": 40.0}
    lot0 = {k: int(round(5e5 / v / 25) * 25) or 25 for k, v in p0.items()}
    r = pd.DataFrame(rng.normal(0.0002, 0.012, size=(n, len(names))), index=dates, columns=names)
    r.iloc[0] = 0.0
    clean = p0["ALPHA"] * 0 + (1 + r).cumprod() * pd.Series(p0)          # clean price path
    d = lambda s: pd.Timestamp(s)

    # scripted events: (name, kind, ex-date, factor or amount, purpose text)
    events = {
        "ALPHA": dict(kind="bonus", ex=d("2021-06-15"), factor=0.5, purpose="BONUS 1:1", relot=True),
        "BETA": dict(kind="split", ex=d("2021-08-10"), factor=0.5, purpose="FV SPLT FRM RS 10 TO RS 5", relot=True),
        "GAMMA": dict(kind="extradiv", ex=d("2021-09-14"), amount=12.0, purpose="INTDV/SPDV - RS 8 & 4", adjusted=True),
        "DELTA": dict(kind="orddiv", ex=d("2021-05-18"), amount=0.6, purpose="DIVIDEND- 6% PER ANNUM", adjusted=False,
                      lotrev=d("2021-11-01"), lotrev_ratio=1.05),
        "EPSILON": dict(kind="orddiv", ex=d("2021-10-12"), amount=3.0, purpose="DIVIDEND - RS 3 PER SH", adjusted=False),
        "ZETA": dict(kind="demerger", ex=d("2021-07-20"), factor=0.8, purpose="DEMERGER"),
        "ETA": dict(kind="delist", last_expiry=d("2021-10-28")),
        "THETA": dict(kind="gap", gap_from=d("2021-04-01"), gap_to=d("2021-07-15"), jump=3.0),
        "IOTA": dict(kind="vanish", last_row=d("2021-12-10")),
        "KAPPA": dict(kind="extradiv", ex=d("2022-01-18"), amount=2.5, purpose="INTDIV - RS. 2.50 PER SH", adjusted=True),
    }

    # raw tape per name: multiplicative level shift by the event factor from the ex-date on
    raw_spot = clean.copy()
    raw_fut = clean.copy()
    lot = pd.DataFrame({k: float(lot0[k]) for k in names}, index=dates)
    for nm, ev in events.items():
        if ev["kind"] in ("bonus", "split"):
            m = dates >= ev["ex"]
            raw_spot.loc[m, nm] *= ev["factor"]; raw_fut.loc[m, nm] *= ev["factor"]
            lot.loc[m, nm] = lot0[nm] / ev["factor"]
        elif ev["kind"] == "extradiv":
            i = dates.get_loc(ev["ex"]); prev = raw_fut[nm].iloc[i - 1]
            f = 1 - ev["amount"] / prev; ev["factor"] = f
            m = dates >= ev["ex"]
            raw_spot.loc[m, nm] *= f; raw_fut.loc[m, nm] *= f          # both drop: exchange adjusts
        elif ev["kind"] == "orddiv":
            i = dates.get_loc(ev["ex"]); prev = clean[nm].iloc[i - 1]
            f = 1 - ev["amount"] / prev; ev["factor"] = f
            m = dates >= ev["ex"]
            raw_spot.loc[m, nm] *= f
            pre = (dates >= dates[i - 10]) & (dates < ev["ex"])          # futures pre-discount it
            raw_fut.loc[pre, nm] *= f; raw_fut.loc[m, nm] *= f
            if "lotrev" in ev:
                lot.loc[dates >= ev["lotrev"], nm] = lot0[nm] * ev["lotrev_ratio"]
        elif ev["kind"] == "demerger":
            m = dates >= ev["ex"]
            raw_spot.loc[m, nm] *= ev["factor"]; raw_fut.loc[m, nm] *= ev["factor"]
        elif ev["kind"] == "gap":
            m = dates >= ev["gap_to"]
            raw_spot.loc[m, nm] *= ev["jump"]; raw_fut.loc[m, nm] *= ev["jump"]
            clean.loc[m, nm] *= ev["jump"]                               # a genuine re-pricing, not an adjustment

    # contracts: the three nearest expiries per session, price = raw futures (zero basis)
    rows = []
    e_i8, d_i8 = expiries.asi8, dates.asi8
    near = np.searchsorted(e_i8, d_i8, side="left")
    # a contract's TRUE first session: the first day it is among the three nearest
    # expiries (it enters as the far month, three expiries before it is front)
    first_day_of_contract = {}
    for k in range(3):
        exp_k = expiries[np.minimum(near + k, len(expiries) - 1)]
        for j, dt in enumerate(dates):
            for nm in names:
                key = (nm, exp_k[j])
                first_day_of_contract[key] = min(first_day_of_contract.get(key, dt), dt)
    for k in range(3):
        exp_k = expiries[np.minimum(near + k, len(expiries) - 1)]
        for j, dt in enumerate(dates):
            e = exp_k[j]
            for nm in names:
                ev = events.get(nm, {})
                if ev.get("kind") == "delist" and e > ev["last_expiry"]:
                    continue
                if ev.get("kind") == "gap" and ev["gap_from"] <= dt < ev["gap_to"]:
                    continue
                if ev.get("kind") == "vanish" and dt > ev["last_row"]:
                    continue
                px = raw_fut.at[dt, nm]
                key = (nm, e)
                first = first_day_of_contract[key] == dt
                oi = lot.at[dt, nm] * (40 + 3 * k)
                rows.append({"date": dt, "symbol": nm, "expiry": e, "open": px * (1 + 0.0005 * (j % 3 - 1)),
                             "high": px * 1.01, "low": px * 0.99, "close": px, "settle": px,
                             "volume_lots": 500.0, "turnover": 500.0 * lot.at[dt, nm] * px,
                             "oi_shares": oi, "chg_oi_shares": oi if first else lot.at[dt, nm] * 2,
                             "lot_size": lot.at[dt, nm], "underlying": raw_spot.at[dt, nm], "fmt": "udiff"})
    contracts = pd.DataFrame(rows)
    contracts = contracts[contracts["expiry"] >= contracts["date"]].sort_values(["date", "symbol", "expiry"]).reset_index(drop=True)
    cash = raw_spot.stack().rename("cash_close").reset_index().rename(columns={"level_0": "date", "level_1": "symbol"})
    cal = pd.DataFrame([{"symbol": nm, "ex_date": ev["ex"], "purpose": ev["purpose"], "series": "EQ"}
                        for nm, ev in events.items() if "purpose" in ev])
    return World(dates, expiries, clean, contracts, cash, cal, events)


# --------------------------------------------------------------------------- #
def run() -> None:
    w = build_world()
    dates = w.dates
    log, factors = build_factors(w.contracts, w.calendar, w.cash)
    cont = build_continuous(w.contracts, factors)
    c = cont.set_index(["symbol", "date"])
    r = w.clean.pct_change(fill_method=None)
    ev = w.events
    fac = factors.set_index(["symbol", "date"])

    def factor_of(nm, dt):
        return float(fac.loc[(nm, dt), "factor"]) if (nm, dt) in fac.index else 1.0

    def masked_of(nm, dt):
        return bool(fac.loc[(nm, dt), "mask_signal"]) if (nm, dt) in fac.index else False

    # ---- Stage 1 layer ------------------------------------------------------
    check("E1 bonus", "factor from text", factor_of("ALPHA", ev["ALPHA"]["ex"]), 0.5)
    check("E1 bonus", "cont return on ex-date == clean return", c.loc[("ALPHA", ev["ALPHA"]["ex"]), "ret"], r.at[ev["ALPHA"]["ex"], "ALPHA"], 1e-9)
    check("E1 bonus", "not masked", masked_of("ALPHA", ev["ALPHA"]["ex"]), False)
    check("E2 split", "factor from text", factor_of("BETA", ev["BETA"]["ex"]), 0.5)
    check("E2 split", "cont return on ex-date == clean return", c.loc[("BETA", ev["BETA"]["ex"]), "ret"], r.at[ev["BETA"]["ex"], "BETA"], 1e-9)
    check("E3 extradiv INTDV/SPDV", "factor = 1 - 12/prev settle", factor_of("GAMMA", ev["GAMMA"]["ex"]), ev["GAMMA"]["factor"], 1e-9)
    check("E3 extradiv INTDV/SPDV", "source", str(log[(log.symbol == "GAMMA")].source.iloc[0]) if (log.symbol == "GAMMA").any() else "NONE", "calendar-extradiv")
    check("E3 extradiv INTDV/SPDV", "cont return on ex-date == clean return", c.loc[("GAMMA", ev["GAMMA"]["ex"]), "ret"], r.at[ev["GAMMA"]["ex"], "GAMMA"], 1e-9)
    check("E4 ordinary div 6%-of-FV text", "no factor (must not read Rs 6)", factor_of("DELTA", ev["DELTA"]["ex"]), 1.0)
    check("E4 ordinary div 6%-of-FV text", "cont return on ex-date == clean return (futures pre-discounted)", c.loc[("DELTA", ev["DELTA"]["ex"]), "ret"], r.at[ev["DELTA"]["ex"], "DELTA"], 1e-9)
    check("E5 3% div, tape unadjusted", "no factor although 3% > 2% floor", factor_of("EPSILON", ev["EPSILON"]["ex"]), 1.0)
    check("E5 3% div, tape unadjusted", "cont return on ex-date == clean return", c.loc[("EPSILON", ev["EPSILON"]["ex"]), "ret"], r.at[ev["EPSILON"]["ex"], "EPSILON"], 1e-9)
    zx = ev["ZETA"]["ex"]; i = dates.get_loc(zx)
    measured = w.contracts[(w.contracts.symbol == "ZETA") & (w.contracts.date == zx)].sort_values("expiry").settle.iloc[0] / \
        w.contracts[(w.contracts.symbol == "ZETA") & (w.contracts.date == dates[i - 1])].sort_values("expiry").settle.iloc[0]
    check("E6 demerger", "factor = measured same-contract ratio (documented)", factor_of("ZETA", zx), measured, 1e-9)
    check("E6 demerger", "masked", masked_of("ZETA", zx), True)
    check("E6 demerger", "cont return on ex-date is NaN", bool(np.isnan(c.loc[("ZETA", zx), "ret"])), True)
    check("E6 demerger", "cont return the day after == clean return", c.loc[("ZETA", dates[i + 1]), "ret"], r.at[dates[i + 1], "ZETA"], 1e-9)
    # E8 gap / re-listing
    g2 = ev["THETA"]["gap_to"]; i2 = dates.get_loc(g2)
    check("E8 re-listing", "chain break on the first day back", bool(c.loc[("THETA", g2), "chain_break"]), True)
    check("E8 re-listing", "no return on the first day back (no fake +200%)", bool(np.isnan(c.loc[("THETA", g2), "ret"])), True)
    check("E8 re-listing", "return the next day == clean return", c.loc[("THETA", dates[i2 + 1]), "ret"], r.at[dates[i2 + 1], "THETA"], 1e-9)
    check("E8 re-listing", "no rows during the gap", int(((cont.symbol == "THETA") & (cont.date >= ev["THETA"]["gap_from"]) & (cont.date < g2)).sum()), 0)
    # E7 delisting flags
    le = ev["ETA"]["last_expiry"]; il = dates.get_loc(le)
    check("E7 delisting", "untradable on the two sessions before the last expiry", bool(c.loc[("ETA", dates[il - 2]), "tradable"]) or bool(c.loc[("ETA", dates[il - 1]), "tradable"]), False)
    check("E7 delisting", "tradable three sessions before", bool(c.loc[("ETA", dates[il - 3]), "tradable"]), True)
    # E11 expiry after the data ends
    last = dates[-1]
    check("E11 expiry beyond data", "front on the last session is the April contract", pd.Timestamp(c.loc[("LAMBDA", last), "front_expiry"]) > last, True)
    check("E11 expiry beyond data", "last session tradable", bool(c.loc[("LAMBDA", last), "tradable"]), True)
    check("E11 expiry beyond data", "dte on the last session > roll window", int(c.loc[("LAMBDA", last), "dte"]) > ROLL_BEFORE_TD, True)
    # E12 / E13 detector silence
    check("E12 routine lot revision", "no factor on the revision date", factor_of("DELTA", ev["DELTA"]["lotrev"]), 1.0)
    check("E13 far-month OI reset", "no detector-sourced adjustments at all", int(log.source.str.startswith("detector").sum()) if len(log) else 0, 0)
    check("E13 far-month OI reset", "adjustments == scripted adjusting events", int(len(log)), 5, note="ALPHA, BETA, GAMMA, ZETA, KAPPA")
    check("E14 INTDIV - RS. 2.50", "factor = 1 - 2.5/prev settle", factor_of("KAPPA", ev["KAPPA"]["ex"]), ev["KAPPA"]["factor"], 1e-9)
    # the whole return series, every name, every day: equals the clean return except masks and breaks
    cw = cont.pivot(index="date", columns="symbol", values="ret")
    diff = (cw - r.reindex(cw.index)[cw.columns]).abs()
    masked_days = cont[cont.masked | cont.chain_break].set_index(["date", "symbol"]).index
    d_ok = diff.stack().drop(masked_days, errors="ignore")
    for nm in ("DELTA", "EPSILON"):          # the session the futures begin discounting the dividend
        i_ex = dates.get_loc(ev[nm]["ex"]); d0 = dates[i_ex - 10]
        check(f"{'E4' if nm == 'DELTA' else 'E5'} discount start", "cont return == f x (1 + r) - 1 on the day the discount appears (a real futures move)",
              c.loc[(nm, d0), "ret"], ev[nm]["factor"] * (1 + r.at[d0, nm]) - 1, 1e-9)
        d_ok = d_ok.drop((d0, nm), errors="ignore")
    check("ALL", "max |cont return - clean return| over every unmasked, unbroken name-day", float(d_ok.max()), 0.0, 1e-9,
          note=f"{len(d_ok):,} name-days")

    # ---- execution layer ------------------------------------------------------
    ca_log = log.rename(columns={"ex_date": "ex_date"})
    ret_wide = cw
    oi_wide = cont.pivot(index="date", columns="symbol", values="oi_notional")
    market = FuturesMarket.from_frames(cont, w.contracts[["date", "symbol", "expiry", "open", "close", "settle", "lot_size"]],
                                       ret_wide, oi_wide, ca_log)
    zero, full = CostConfig.zero(), CostConfig()
    xcfg = ExecConfig(entry_min_dte=0)          # keep entries on the front so rolls are exercised

    def scripted(name, enter, exit_, cost):
        """Open `name` long 5% at `enter`'s open, run every session, close at `exit_`'s open (or never)."""
        led = Ledger(CAPITAL); fills = []
        i0, i1 = dates.get_loc(enter), (dates.get_loc(exit_) if exit_ is not None else len(dates))
        for i in range(i0, len(dates)):
            # mark at the previous session's close first, as run_backtest does before executing at i
            led.mark(lambda p, ii=i - 1: (market.contract(p.symbol, p.expiry, ii) or type("x", (), {"settle": None})).settle)
            dec = Decisions(exits=[ExitOrder(name, "scripted")] if i == i1 else [],
                            entries=[EntryOrder(name, 0.05, -2.5)] if i == i0 else [])
            rep = execute_day(dec, led, market, i, CAPITAL, i - 1, cost, xcfg)
            fills += rep.fills
            if i > i0 and name not in led:
                break
        f = pd.DataFrame(fills)
        # mark the ledger at the last session it ran
        return led, f

    def raw(name, dt, field="settle"):
        """raw tape price of the (zero-basis) contracts of `name` on session dt."""
        rowz = w.contracts[(w.contracts.symbol == name) & (w.contracts.date == dt)].sort_values("expiry")
        return float(rowz[field].iloc[0])

    def exchange_rule_pnl(f, name, exit_date, ex_date, factor):
        """A price-only adjustment as the exchange books it: the reference price is
        reduced by D = prev_settle x (1 - f) on the ex-date; P&L before the ex-date
        is on the raw price and after it on the raw (lower) price. Zero basis, so
        rolls in between change nothing."""
        o = f[f.action == "OPEN"].iloc[0]
        i = dates.get_loc(ex_date); prev = raw(name, dates[i - 1])
        units = abs(o.lots) * o.lot_size
        return units * ((prev - o.price) + (raw(name, exit_date, "open") - prev * factor))

    def clean_ratio(name, d_from, d_to, open_from=True, open_to=True):
        """clean price ratio between two sessions' opens (opens are settle x a tiny known wobble)."""
        ob = lambda dd: (1 + 0.0005 * (dates.get_loc(dd) % 3 - 1))
        a = w.clean.at[d_from, name] * (ob(d_from) if open_from else 1.0)
        b = w.clean.at[d_to, name] * (ob(d_to) if open_to else 1.0)
        return b / a

    # share-count events and unadjusted dividends: P&L == notional x clean return
    for case, name, enter, exit_, n_adj in [("E1 bonus", "ALPHA", "2021-06-01", "2021-06-25", 1),
                                            ("E2 split", "BETA", "2021-07-27", "2021-08-24", 1),
                                            ("E4 ordinary div", "DELTA", "2021-05-10", "2021-05-26", 0),
                                            ("E5 3% div unadjusted", "EPSILON", "2021-10-05", "2021-10-19", 0)]:
        led, f = scripted(name, pd.Timestamp(enter), pd.Timestamp(exit_), zero)
        o = f[f.action == "OPEN"].iloc[0]
        notional = abs(o.lots) * o.lot_size * o.price
        exp_pnl = notional * (clean_ratio(name, o.date, pd.Timestamp(exit_)) - 1.0)
        check(case, "P&L of a long held across the event == notional x clean return", float(f.realized_pnl.sum()), exp_pnl, 1e-9)
        check(case, "position closed, ledger empty", len(led), 0)
        adj = f[f.action == "CA_ADJUST"]
        check(case, f"adjustment rows == {n_adj}", len(adj), n_adj)
        if ev[name].get("relot"):
            check(case, "lot re-denominated x2", float(adj.lot_size.iloc[0]), o.lot_size / ev[name]["factor"], 1e-9)
    # price-only adjustments: P&L == the exchange's additive rule
    for case, name, enter, exit_ in [("E3 extradiv INTDV/SPDV", "GAMMA", "2021-09-01", "2021-09-28"),
                                     ("E14 INTDIV - RS. 2.50", "KAPPA", "2022-01-05", "2022-01-25")]:
        led, f = scripted(name, pd.Timestamp(enter), pd.Timestamp(exit_), zero)
        exp_pnl = exchange_rule_pnl(f, name, pd.Timestamp(exit_), ev[name]["ex"], ev[name]["factor"])
        check(case, "P&L == exchange rule: reference price reduced by the dividend on the ex-date", float(f.realized_pnl.sum()), exp_pnl, 1e-9)
        adj = f[f.action == "CA_ADJUST"]
        check(case, "exactly one adjustment row", len(adj), 1)
        if len(adj):
            check(case, "lot unchanged", float(adj.lot_size.iloc[0]), float(f[f.action == "OPEN"].lot_size.iloc[0]), 1e-12)
    # E6 demerger: documented behaviour — measured ratio as a price-only adjustment, the masked day's real move absorbed
    led, f = scripted("ZETA", pd.Timestamp("2021-07-06"), pd.Timestamp("2021-08-03"), zero)
    exp_docu = exchange_rule_pnl(f, "ZETA", pd.Timestamp("2021-08-03"), zx, factor_of("ZETA", zx))
    check("E6 demerger", "P&L == price-only adjustment by the measured ratio (documented)", float(f.realized_pnl.sum()), exp_docu, 1e-9,
          note="the ex-date's real move is absorbed into the measured factor; documented choice, not the only right answer")
    # E7 delisting while held
    led, f = scripted("ETA", pd.Timestamp("2021-10-05"), None, zero)
    fc = f[f.action == "FORCE_CLOSE"]
    check("E7 delisting", "force-closed exactly once", len(fc), 1)
    if len(fc):
        check("E7 delisting", "on the session with 2 left on the last contract", pd.Timestamp(fc.date.iloc[0]), dates[il - 2])
        check("E7 delisting", "reason", str(fc.reason.iloc[0]).startswith("delisting"), True)
        o = f[f.action == "OPEN"].iloc[0]; notional = abs(o.lots) * o.lot_size * o.price
        check("E7 delisting", "P&L == notional x clean return to the forced fill", float(f.realized_pnl.sum()), notional * (clean_ratio("ETA", o.date, dates[il - 2]) - 1), 1e-9)
    check("E7 delisting", "no fills after the last expiry", int((pd.to_datetime(f.date) > le).sum()), 0)
    check("E7 delisting", "ledger empty", len(led), 0)
    # E9 vanished contract while held
    led, f = scripted("IOTA", pd.Timestamp("2021-12-01"), None, zero)
    fc = f[f.action == "FORCE_CLOSE"]; v = ev["IOTA"]["last_row"]; iv = dates.get_loc(v)
    check("E9 vanished", "force-closed once, on the first session with no row", pd.Timestamp(fc.date.iloc[0]) if len(fc) else None, dates[iv + 1])
    check("E9 vanished", "fill basis last-mark-NO-ROW (documented)", str(fc.fill_basis.iloc[0]) if len(fc) else "", "last-mark-NO-ROW")
    if len(fc):
        o = f[f.action == "OPEN"].iloc[0]; notional = abs(o.lots) * o.lot_size * o.price
        check("E9 vanished", "P&L == notional x clean return to the last settle", float(f.realized_pnl.sum()),
              notional * (clean_ratio("IOTA", o.date, v, True, False) - 1), 1e-9)
    # E10 / E11 held across the final bar
    led, f = scripted("LAMBDA", pd.Timestamp("2022-03-15"), None, zero)
    pos = led.get("LAMBDA")
    check("E10 held to the end", "still open at the end", pos is not None, True)
    if pos is not None:
        check("E10 held to the end", "rolled once, at 2 sessions to the March expiry", len(f[f.action == "ROLL_CLOSE"]), 1)
        check("E10 held to the end", "roll day", pd.Timestamp(f[f.action == "ROLL_CLOSE"].date.iloc[0]), dates[dates.get_loc(pd.Timestamp("2022-03-31")) - 2])
        check("E10 held to the end", "holding the April contract (expiry after the data)", pos.expiry > last, True)
        check("E10 held to the end", "no forced close", int((f.action == "FORCE_CLOSE").sum()), 0)
        last_settle = market.contract("LAMBDA", pos.expiry, len(dates) - 1).settle
        led.mark(lambda p: market.contract(p.symbol, p.expiry, len(dates) - 1).settle)
        o = f[f.action == "OPEN"].iloc[0]; notional = abs(o.lots) * o.lot_size * o.price
        check("E10 held to the end", "equity == capital + notional x clean return to the final settle", led.equity(),
              CAPITAL + notional * (clean_ratio("LAMBDA", o.date, last, True, False) - 1), 1e-9)
    # E12 lot revision with a position held across it: nothing should happen
    led, f = scripted("DELTA", pd.Timestamp("2021-10-26"), pd.Timestamp("2021-11-09"), zero)
    check("E12 routine lot revision", "no adjustment row while held across it", int((f.action == "CA_ADJUST").sum()), 0)
    o = f[f.action == "OPEN"].iloc[0]; notional = abs(o.lots) * o.lot_size * o.price
    check("E12 routine lot revision", "P&L == notional x clean return", float(f.realized_pnl.sum()), notional * (clean_ratio("DELTA", o.date, pd.Timestamp("2021-11-09")) - 1), 1e-9)
    # costs: every leg charged once, adjustment free, zero mode exactly zero
    led, f = scripted("ALPHA", pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-25"), full)
    check("COSTS", "legs charged: OPEN, ROLL_CLOSE, ROLL_OPEN, CLOSE each > 0", bool((f[f.action != "CA_ADJUST"].cost_total > 0).all()), True)
    check("COSTS", "adjustment row costs 0", float(f[f.action == "CA_ADJUST"].cost_total.sum()), 0.0)
    check("COSTS", "cash == capital + realized - costs", led.cash, CAPITAL + f.realized_pnl.sum() - f.cost_total.sum(), 1e-12)

    # ---- report ----------------------------------------------------------------
    df = pd.DataFrame(RESULTS)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
    def fmt(v):
        return f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)
    df["expected"] = df.expected.map(fmt); df["got"] = df.got.map(fmt)
    print("\n" + "=" * 110 + "\n  EVENT-LEVEL KNOWN-ANSWER VALIDATION (synthetic corporate actions, delistings, boundaries)\n" + "=" * 110)
    print(df[["case", "what", "expected", "got", "pass"]].to_string(index=False))
    n_fail = int((~df["pass"]).sum())
    print("=" * 110)
    print(f"{len(df)} assertions, {len(df) - n_fail} pass, {n_fail} FAIL")
    if n_fail:
        print(df[~df["pass"]][["case", "what", "expected", "got", "note"]].to_string(index=False))
    print(f"adjustments produced by the Stage 1 layer: {len(log)}: " +
          ", ".join(f"{s}/{d.date()} {k} {src} f={fv:.4f}" for s, d, k, src, fv in zip(log.symbol, log.ex_date, log.kind, log.source, log.factor_applied)))
    df.to_csv(HERE.parent / "results_futures_2015_2024" / "diagnostics" / "validation_events.csv", index=False)
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
