"""
fut_checks.py — Stage 3b/3c: integrity checks on the FULL-PERIOD run.

  B  INDEPENDENT EQUITY RECONSTRUCTION. Replay the trade log alone — fills in
     order, realized P&L and costs into cash, open positions marked at each
     session's settlement price from the market — and compare the rebuilt
     DAILY equity curve with the engine's, 2015-2024. Must agree to ~1e-9.
  C  LOOK-AHEAD, stated as assertions over every fill of the full run:
       - no fill on a contract after its expiry
       - every OPEN / ROLL_OPEN is on the chain's front contract for that
         session, and no nearer-expiry contract of that name with more than
         ROLL_BEFORE_TD sessions left was listed (the next month is never
         opened early)
       - every ROLL_CLOSE leaves a contract at exactly ROLL_BEFORE_TD sessions
       - every OPEN fills exactly one session after its decision bar
     and, for the DATA layer, a truncation test: rebuild the corporate-action
     factors and the continuous returns from data ending at a cutoff and
     compare with the full-sample series up to that cutoff. Forward-anchored
     ratio chaining and ex-date-only factor evidence mean they must be
     identical; any difference would be information from after the cutoff.

The P&L of the full run is deliberately NOT printed here — Stage 4 reports it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fut_backtest import FuturesBacktestConfig, run_backtest
from fut_execute import ROLL_BEFORE_TD, ExecConfig
from fut_market import FuturesMarket

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RULE = "=" * 78


def hdr(t):
    print("\n" + RULE + f"\n  {t}\n" + RULE, flush=True)


# --------------------------------------------------------------------------- #
def reconstruct_equity(res, market: FuturesMarket, capital: float) -> pd.DataFrame:
    tr = res.trades.reset_index(drop=True)
    tr["_i"] = tr["date"].map(market.date_index)
    fills_by_day = {i: g for i, g in tr.groupby("_i", sort=True)}
    cash_delta, open_pos = 0.0, {}
    rows = []
    for d in res.equity_curve.index:
        i = market.date_index(d)
        for k in range(i + 1) if not rows else [i]:
            g = fills_by_day.get(k)
            if g is None:
                continue
            for r in g.itertuples(index=False):
                cash_delta += (0.0 if pd.isna(r.realized_pnl) else r.realized_pnl) - r.cost_total
                if r.action in ("OPEN", "ROLL_OPEN"):
                    open_pos[r.symbol] = [r.lots, r.lot_size, r.price, r.expiry, r.price]
                elif r.action == "CA_ADJUST":      # entry / lot re-based, position stays open
                    rec = open_pos[r.symbol]
                    rec[0], rec[1], rec[2] = r.lots, r.lot_size, r.price
                else:
                    open_pos.pop(r.symbol, None)
        unreal = 0.0
        for s, rec in open_pos.items():
            lots, lot, px, exp, last_mark = rec
            cd = market.contract(s, exp, i)
            mark = cd.settle if cd is not None else last_mark   # carry the last settle forward
            rec[4] = mark
            unreal += lots * lot * (mark - px)
        rows.append({"date": d, "recon": capital + cash_delta + unreal, "n_open": len(open_pos)})
    out = pd.DataFrame(rows).set_index("date")
    out["engine"] = res.equity_curve
    out["diff"] = out["engine"] - out["recon"]
    return out


# --------------------------------------------------------------------------- #
def lookahead_fills(res, market: FuturesMarket, contracts: pd.DataFrame) -> dict:
    tr = res.trades
    listed = (contracts.assign(i=contracts["date"].map(market.date_index))
              .groupby(["symbol", "i"])["expiry"].apply(lambda s: sorted(set(s))).to_dict())
    tr = tr[tr["action"] != "CA_ADJUST"]           # adjustments are not fills
    N = ExecConfig().entry_min_dte
    out = {"fills": len(tr), "after_expiry": 0, "no_contract_row": 0,
           "vanished_contract_closes_at_last_settle": 0,
           "back_month_opens": 0, "back_month_open_rule_violated": 0,
           "open_not_front": 0, "open_nearer_month_available": 0,
           "roll_close_not_at_dte2": 0, "open_not_next_session": 0,
           "close_after_expiry": 0}
    for r in tr.itertuples(index=False):
        i = market.date_index(r.date)
        if market.is_expired(r.expiry, i):
            out["after_expiry"] += 1
        if not market.has_contract_row(r.symbol, r.expiry, i):
            if r.action == "FORCE_CLOSE" and r.fill_basis == "last-mark-NO-ROW":
                out["vanished_contract_closes_at_last_settle"] += 1   # tagged, not a look-ahead
            else:
                out["no_contract_row"] += 1
        if r.action == "OPEN" and getattr(r, "contract_choice", "") == "back-month":
            # allowed only when the front has fewer than N sessions left, and only
            # in the very next listed expiry
            f = market.front(i, r.symbol)
            nxt = market.next_expiry(i, r.symbol, f.expiry) if f is not None else None
            out["back_month_opens"] += 1
            if f is None or market.dte(f.expiry, i) >= N or nxt is None \
                    or pd.Timestamp(nxt) != pd.Timestamp(r.expiry):
                out["back_month_open_rule_violated"] += 1
        elif r.action in ("OPEN", "ROLL_OPEN"):
            f = market.front(i, r.symbol)
            if f is None or pd.Timestamp(f.expiry) != pd.Timestamp(r.expiry):
                out["open_not_front"] += 1
            nearer = [e for e in listed.get((r.symbol, i), []) if pd.Timestamp(e) < pd.Timestamp(r.expiry)]
            if any(market.dte(e, i) > ROLL_BEFORE_TD for e in nearer):
                out["open_nearer_month_available"] += 1
        if r.action == "OPEN" and (r.bar - r.entry_bar) != 1:
            out["open_not_next_session"] += 1
        if r.action == "ROLL_CLOSE" and market.dte(r.expiry, i) != ROLL_BEFORE_TD:
            out["roll_close_not_at_dte2"] += 1
        if r.action in ("CLOSE", "FORCE_CLOSE") and market.dte(r.expiry, i) < 0:
            out["close_after_expiry"] += 1
    return out


# --------------------------------------------------------------------------- #
def truncation_test(cutoffs: list[str]) -> pd.DataFrame:
    sys.path.insert(0, str(HERE))
    from fo_corpactions import build_factors
    from fo_continuous import build_continuous
    contracts = pd.read_parquet(DATA / "contracts.parquet")
    cash = pd.read_parquet(DATA / "cash.parquet")
    cal = pd.read_parquet(DATA / "ca_calendar.parquet")
    full_fac = pd.read_parquet(DATA / "ca_factors.parquet")
    full_cont = pd.read_parquet(DATA / "continuous.parquet")[["date", "symbol", "ret", "masked", "chain_break"]]
    rows = []
    for cut in cutoffs:
        t0 = time.time()
        cut_ts = pd.Timestamp(cut)
        ctr = contracts[contracts["date"] <= cut_ts]
        csh = cash[cash["date"] <= cut_ts]
        _log, fac = build_factors(ctr, cal, csh)
        cont = build_continuous(ctr, fac)
        # factors: identical set and values up to the cutoff
        ff = full_fac[full_fac["date"] <= cut_ts].set_index(["symbol", "date"]).sort_index()
        tf = fac.set_index(["symbol", "date"]).sort_index()
        same_keys = ff.index.equals(tf.index)
        fac_diff = float((ff["factor"] - tf["factor"]).abs().max()) if same_keys else np.nan
        mask_same = bool((ff["mask_signal"] == tf["mask_signal"]).all()) if same_keys else False
        # continuous returns: identical (including NaN pattern) up to the cutoff
        fc = full_cont[full_cont["date"] <= cut_ts].set_index(["symbol", "date"]).sort_index()
        tc = cont[["date", "symbol", "ret", "masked", "chain_break"]].set_index(["symbol", "date"]).sort_index()
        rows_same = fc.index.equals(tc.index)
        a, b = fc["ret"].to_numpy(), tc["ret"].to_numpy()
        nan_same = bool(np.array_equal(np.isnan(a), np.isnan(b))) if rows_same else False
        ret_diff = float(np.nanmax(np.abs(a - b))) if rows_same else np.nan
        rows.append({"cutoff": cut, "n_factors_full": len(ff), "n_factors_trunc": len(tf),
                     "factor_keys_identical": same_keys, "max_abs_factor_diff": fac_diff,
                     "masks_identical": mask_same, "n_ret_rows": len(fc),
                     "ret_rows_identical": rows_same, "nan_pattern_identical": nan_same,
                     "max_abs_ret_diff": ret_diff, "secs": round(time.time() - t0)})
        print(f"  cutoff {cut}: factors {len(tf)} vs {len(ff)}, keys identical {same_keys}, "
              f"max|dfactor| {fac_diff:.2e}; returns rows identical {rows_same}, NaN pattern "
              f"identical {nan_same}, max|dret| {ret_diff:.2e}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    market = FuturesMarket()
    contracts = pd.read_parquet(DATA / "contracts.parquet", columns=["date", "symbol", "expiry"])
    print(f"[info] market loaded ({time.time() - t0:.0f}s)", flush=True)

    hdr("FULL-PERIOD RUN (full cost) — P&L withheld until Stage 4")
    cfg = FuturesBacktestConfig(start="2015-01-01", end="2024-12-31", verbose=True)
    t1 = time.time()
    res = run_backtest(market, cfg)
    print(f"[info] run: {len(res.equity_curve)} bars, {len(res.trades)} fills, "
          f"{len(res.skips)} skips, {time.time() - t1:.0f}s", flush=True)

    hdr("B. INDEPENDENT DAILY EQUITY RECONSTRUCTION FROM THE TRADE LOG")
    rc = reconstruct_equity(res, market, cfg.initial_capital)
    print(f"  bars compared            : {len(rc):,}")
    print(f"  max |engine - recon|     : Rs {rc['diff'].abs().max():.3e}")
    print(f"  max relative difference  : {(rc['diff'].abs() / rc['engine']).max():.3e}")
    print(f"  bars differing > 1e-6 Rs : {int((rc['diff'].abs() > 1e-6).sum())}")
    print(f"  -> {'PASS' if (rc['diff'].abs() / rc['engine']).max() < 1e-9 else 'FAIL'} (tolerance 1e-9 relative)")

    hdr("C1. LOOK-AHEAD: EVERY FILL OF THE FULL RUN")
    la = lookahead_fills(res, market, contracts)
    for k, v in la.items():
        print(f"  {k:32s}: {v:,}")
    bad = sum(v for k, v in la.items() if k not in ("fills", "vanished_contract_closes_at_last_settle",
                                                     "back_month_opens"))
    print(f"  -> {'PASS' if bad == 0 else 'FAIL'}")

    hdr("C2. LOOK-AHEAD: STAGE 1 ADJUSTMENTS REBUILT FROM TRUNCATED DATA")
    tt = truncation_test(["2018-06-29", "2021-12-31"])
    ok = bool(tt["factor_keys_identical"].all() and tt["ret_rows_identical"].all()
              and tt["nan_pattern_identical"].all() and (tt["max_abs_factor_diff"] == 0).all()
              and (tt["max_abs_ret_diff"] == 0).all())
    print(f"  -> {'PASS' if ok else 'FAIL'} (series up to a cutoff are byte-identical whether or not "
          f"later data exists)")
    print(f"\n[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
