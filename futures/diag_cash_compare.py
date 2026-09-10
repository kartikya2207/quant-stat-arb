"""
diag_cash_compare.py — is the futures run the same strategy as the cash run?

Per bar, on the cash study's frozen input (yfinance adjusted closes, 435 names)
using the cash code unchanged (strategy/signal.py, harness_backtest._liquid_universe):
  universe A = the cash study's rule, top 200 by 20-day rupee turnover
  universe B = names F&O-listed on that date (Stage 1 membership) that have a
               cash series, top 120 by the same turnover rule — the cash data on
               a point-in-time-ish universe of the futures' size
For each: names scored, complete windows, |s| >= 2 count, OU-gate rejects,
long/short candidate counts. Signal supply only: no book state, no execution.

Then the realised side from the two results folders on common dates:
positions held, gross, hold length, round trips per year.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CASH_DATA = ROOT.parent / "TideTraderBeta" / "results_cash_equity_2015_2024" / "data"
OUT = ROOT / "results_futures_2015_2024" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "futures"))
from fut_backtest import sig                       # strategy/signal.py, loaded once
sys.path.insert(0, str(ROOT / "strategy"))
from harness_backtest import _liquid_universe      # cash universe rule, unchanged

WIN, NF, LB = 60, 5, 20


def signal_supply(rets, px, vol, t, universe):
    win = rets.iloc[t - WIN + 1: t + 1].reindex(columns=universe).dropna(axis=1, how="all")
    obs = win.notna().sum(axis=0)
    n_complete = int(win.notna().all(axis=0).sum())
    if n_complete <= NF + 1:
        return None
    res = sig.compute_signal(win, n_factors=NF, score_symbols=list(win.columns))
    s, k = res.s_score, res.ou_speed
    valid = s.notna()
    return {"n_universe": len(universe), "n_scored": win.shape[1], "n_complete": n_complete,
            "n_lt30": int((obs < 30).sum()), "n_ou_reject": int(((obs >= 30) & s.isna()).sum()),
            "n_valid": int(valid.sum()), "n_ge2": int((s.abs() >= 2).sum()),
            "n_long": int(((s <= -2) & (k > 0)).sum()), "n_short": int(((s >= 2) & (k > 0)).sum()),
            "s_abs_p90": float(s.abs().quantile(.9)) if valid.any() else np.nan}


def main():
    px = pd.read_parquet(CASH_DATA / "prices_wide.parquet")
    vol = pd.read_parquet(CASH_DATA / "volume_wide.parquet")
    good = px.columns[px.notna().any()]
    px, vol = px[good], vol.reindex(columns=good)
    rets = px.pct_change(fill_method=None)
    dates = px.index
    memb = pd.read_parquet(ROOT / "futures" / "data" / "membership.parquet")
    memb_by_date = {d: set(g["symbol"]) for d, g in memb.groupby("date")}
    cash_syms = set(px.columns)
    print(f"cash frame {px.shape}; F&O names with a cash series: {len(cash_syms & set(memb.symbol))} of {memb.symbol.nunique()}")

    rows = []
    t0 = time.time()
    for t in range(WIN, len(dates) - 1):
        d = dates[t]
        uniA = _liquid_universe(vol, px, t, LB, 200)
        fo = memb_by_date.get(d)
        if fo is None:
            continue
        cand = [c for c in _liquid_universe(vol, px, t, LB, 10_000) if c in fo]
        uniB = cand[:120]
        a = signal_supply(rets, px, vol, t, uniA)
        b = signal_supply(rets, px, vol, t, uniB)
        if a is None or b is None:
            continue
        rows.append({"date": d, **{f"A_{k}": v for k, v in a.items()}, **{f"B_{k}": v for k, v in b.items()},
                     "fo_with_cash": len(cand)})
        if (t - WIN) % 500 == 0:
            print(f"  {d.date()} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).set_index("date")
    df.to_parquet(OUT / "cash_signal_supply.parquet")
    print("\nCASH signal supply per bar (A = cash study's top-200 universe; B = F&O members with cash data, top 120):")
    cols = ["n_universe", "n_scored", "n_complete", "n_lt30", "n_ou_reject", "n_valid", "n_ge2", "n_long", "n_short", "s_abs_p90"]
    summ = pd.DataFrame({f"A": df[[f"A_{c}" for c in cols]].mean().values,
                         f"B": df[[f"B_{c}" for c in cols]].mean().values}, index=cols).T
    print(summ.round(2).to_string())
    print(f"F&O members with a cash series per day: mean {df['fo_with_cash'].mean():.1f}")
    print("\nby year (means):")
    yr = df.groupby(df.index.year)[["A_n_ge2", "A_n_ou_reject", "B_n_ge2", "B_n_ou_reject", "fo_with_cash"]].mean().round(1)
    print(yr.to_string())

    # ---- realised side, both studies, common dates --------------------------
    cex = pd.read_csv(ROOT / "results_cash_equity_2015_2024" / "exposure.csv", index_col=0, parse_dates=True)
    ctr = pd.read_csv(ROOT / "results_cash_equity_2015_2024" / "trades.csv", parse_dates=["date"])
    fex = pd.read_csv(ROOT / "results_futures_2015_2024" / "exposure.csv", index_col=0, parse_dates=True)
    ftr = pd.read_csv(ROOT / "results_futures_2015_2024" / "trades.csv", parse_dates=["date"])
    common = cex.index.intersection(fex.index)
    print(f"\nREALISED, common dates {common.min().date()} -> {common.max().date()} ({len(common)} bars):")
    c_open = ctr[ctr.action == "OPEN"].groupby(ctr.date.dt.year).size()
    f_open = ftr[ftr.action == "OPEN"].groupby(ftr.date.dt.year).size()
    # cash hold length: pair OPEN/CLOSE per symbol in order
    def holds(tr, bar_index):
        h = []
        opens = {}
        for r in tr.itertuples(index=False):
            if r.action == "OPEN":
                opens[r.symbol] = r.date
            elif r.action in ("CLOSE", "FORCE_CLOSE") and r.symbol in opens:
                h.append(bar_index.get_loc(r.date) - bar_index.get_loc(opens.pop(r.symbol)))
        return pd.Series(h)
    ch = holds(ctr, cex.index); fh = holds(ftr, fex.index)
    tab = pd.DataFrame({
        "cash": [cex.loc[common, "n_positions"].mean(), cex.loc[common, "gross"].mean(), cex.loc[common, "net"].mean(),
                 ch.mean(), c_open.mean(), len(ctr[ctr.action == "OPEN"])],
        "futures": [fex.loc[common, "n_positions"].mean(), fex.loc[common, "gross_marked"].mean(), fex.loc[common, "net_marked"].mean(),
                    fh.mean(), f_open.mean(), len(ftr[ftr.action == "OPEN"])]},
        index=["avg positions", "avg gross", "avg net", "avg hold (bars, fill to fill)", "entries per year", "entries total"])
    print(tab.round(3).to_string())
    print("\nentries per year:"); print(pd.DataFrame({"cash": c_open, "futures": f_open}).to_string())
    print("\navg positions by year:"); print(pd.DataFrame({"cash": cex.n_positions.groupby(cex.index.year).mean(),
                                                         "futures": fex.n_positions.groupby(fex.index.year).mean()}).round(1).to_string())
    print("\ncash per-name weight at entry (gross / positions): %.4f ; futures achieved weight mean %.4f" %
          ((cex.gross / cex.n_positions.replace(0, np.nan)).mean(), ftr[ftr.action == "OPEN"].achieved_weight.abs().mean()))


if __name__ == "__main__":
    main()
