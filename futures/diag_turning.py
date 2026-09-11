"""
diag_turning.py — does the s-score predict better once the residual has TURNED
than while it is still EXTENDING? (Hypothesis from supervisor discussion:
entering a name whose |s| is still rising is catching a falling knife.)

Signal test only, no backtest. The OOS Fama-MacBeth construction of
fut_oos_test.py (factors / betas / directions frozen at T, strictly forward,
horizons 1/3/5/10) with each date's cross-section split by the recent
direction of |s|: the s-score at T is compared with the s-score the strategy
would have SEEN k sessions earlier (computed on the window ending at T-k, on
the same universe), k in {1, 3, 5} fixed a priori.
    extending : |s_T| > |s_{T-k}|   (moving away from zero)
    turning   : |s_T| < |s_{T-k}|   (moving back toward zero)
Per group, horizon and k: pooled corr(s, fwd), Fama-MacBeth r and t over
per-date cross-sectional correlations (dates with >= MIN_STOCKS names in the
group), hit rate on |s| >= 2, N; a paired t on the per-date difference of
correlations (same dates) and a Welch t on the two sets.
Tail (|s| >= 3): too few names per date for per-date correlations, so the
groups are compared POOLED across dates on the reversion return
-sign(s) * fwd residual (mean, Welch t) and on the hit rate.
Run on futures (PIT top-120 by OI) and on the cash data restricted to F&O
members with a cash series, top 120 by turnover.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CASH_DATA = ROOT.parent / "TideTraderBeta" / "results_cash_equity_2015_2024" / "data"
OUT = ROOT / "results_futures_2015_2024" / "diagnostics"
sys.path.insert(0, str(ROOT / "futures"))
from fut_backtest import sig, _universe
from fut_market import FuturesMarket
sys.path.insert(0, str(ROOT / "strategy"))
from harness_backtest import _liquid_universe

WIN, NF, LB = 60, 5, 120
HORIZONS = [1, 3, 5, 10]
HMAX = 10
ENTRY, TAIL = 2.0, 3.0
KS = [1, 3, 5]
MIN_STOCKS = 10


def s_at(rets, t, cols):
    """s-score per name from the window ending at session t (the trader's view)."""
    win = rets.iloc[t - WIN + 1: t + 1].reindex(columns=cols).dropna(axis=1, how="all")
    if win.notna().all(axis=0).sum() <= NF + 1:
        return None
    return sig.compute_signal(win, n_factors=NF, score_symbols=list(win.columns)).s_score


def collect(rets, dates, sample_T, universe_fn):
    """One row per (date, name, horizon): s_T, fwd residual, |s| change for each k."""
    rows = []
    for T in sample_T:
        uni = universe_fn(T)
        win = rets.iloc[T - WIN + 1: T + 1].reindex(columns=uni).dropna(axis=1, how="all")
        if win.shape[1] <= NF + 1:
            continue
        try:
            factors, loadings, _ = sig.compute_pca_factors(win, NF)
        except Exception:
            continue
        cols = list(loadings.index)
        R = win[cols]
        mu = R.mean(axis=0).values; sigma = R.std(axis=0, ddof=1).replace(0, np.nan).values
        V = loadings.values.T
        Fv = np.column_stack([factors.values, np.ones(len(factors))])
        betas = np.linalg.pinv(Fv) @ R.values
        s_scores, _ = sig.compute_s_scores(sig.compute_residuals(R, factors))
        s = s_scores.reindex(cols)
        prev = {k: s_at(rets, T - k, cols) for k in KS}
        fwd_rets = rets.iloc[T + 1: T + HMAX + 1][cols].values
        resid_fwd = np.full((HMAX, len(cols)), np.nan)
        for d in range(HMAX):
            r_tau = fwd_rets[d]
            z = (r_tau - mu) / sigma
            f_tau = V @ np.nan_to_num(z, nan=0.0)
            resid_fwd[d] = r_tau - np.concatenate([f_tau, [1.0]]) @ betas
        cum = np.nancumsum(resid_fwd, axis=0)
        present = np.cumprod(~np.isnan(fwd_rets), axis=0).astype(bool)
        for j, c in enumerate(cols):
            sv = s.iloc[j]
            if not np.isfinite(sv):
                continue
            base = {"date": dates[T], "symbol": c, "s": sv}
            for k in KS:
                pk = prev[k]
                pv = pk.get(c, np.nan) if pk is not None else np.nan
                base[f"ds{k}"] = abs(sv) - abs(pv) if np.isfinite(pv) else np.nan
            for h in HORIZONS:
                if present[h - 1, j] and np.isfinite(cum[h - 1, j]):
                    rows.append({**base, "h": h, "fwd": cum[h - 1, j]})
    return pd.DataFrame(rows)


def analyse(df, label):
    out_main, out_tail = [], []
    for k in KS:
        col = f"ds{k}"
        d = df[df[col].notna() & (df[col] != 0)].copy()
        d["group"] = np.where(d[col] > 0, "extending", "turning")
        for h in HORIZONS:
            dh = d[d.h == h]
            per_date = {}
            for g in ("extending", "turning"):
                dg = dh[dh.group == g]
                # per-date correlations
                dc = (dg.groupby("date").apply(lambda x: np.corrcoef(x.s, x.fwd)[0, 1]
                                               if len(x) >= MIN_STOCKS and x.s.std() > 0 and x.fwd.std() > 0 else np.nan)
                      .dropna())
                per_date[g] = dc
                ext = dg[dg.s.abs() >= ENTRY]
                out_main.append({"series": label, "k": k, "h": h, "group": g, "n_obs": len(dg),
                                 "share": len(dg) / len(dh), "pooled_r": np.corrcoef(dg.s, dg.fwd)[0, 1],
                                 "n_dates": len(dc), "fm_r": dc.mean(),
                                 "fm_t": dc.mean() / (dc.std(ddof=1) / np.sqrt(len(dc))) if len(dc) > 2 else np.nan,
                                 "hit_ge2": float((np.sign(ext.fwd) == -np.sign(ext.s)).mean()) if len(ext) else np.nan,
                                 "n_ge2": len(ext)})
            a, b = per_date["extending"], per_date["turning"]
            common = a.index.intersection(b.index)
            diff = (b.loc[common] - a.loc[common])                   # turning minus extending (more negative = better)
            se_w = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            out_main.append({"series": label, "k": k, "h": h, "group": "turning - extending",
                             "n_obs": len(common), "share": np.nan, "pooled_r": np.nan, "n_dates": len(common),
                             "fm_r": b.mean() - a.mean(),
                             "fm_t": (b.mean() - a.mean()) / se_w if se_w > 0 else np.nan,      # Welch
                             "hit_ge2": diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 2 else np.nan,  # paired t, stored here
                             "n_ge2": np.nan})
            # tail |s| >= 3, pooled reversion return
            tail = dh[dh.s.abs() >= TAIL].copy()
            tail["rev"] = -np.sign(tail.s) * tail.fwd
            ta, tb = tail[tail.group == "extending"].rev, tail[tail.group == "turning"].rev
            for g, x in (("extending", ta), ("turning", tb)):
                out_tail.append({"series": label, "k": k, "h": h, "group": g, "n": len(x),
                                 "mean_rev_pct": 100 * x.mean() if len(x) else np.nan,
                                 "hit": float((x > 0).mean()) if len(x) else np.nan,
                                 "t_vs_zero": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan})
            se = np.sqrt(ta.var(ddof=1) / max(len(ta), 1) + tb.var(ddof=1) / max(len(tb), 1)) if len(ta) > 2 and len(tb) > 2 else np.nan
            out_tail.append({"series": label, "k": k, "h": h, "group": "turning - extending", "n": len(ta) + len(tb),
                             "mean_rev_pct": 100 * (tb.mean() - ta.mean()) if len(ta) and len(tb) else np.nan,
                             "hit": (tb > 0).mean() - (ta > 0).mean() if len(ta) and len(tb) else np.nan,
                             "t_vs_zero": (tb.mean() - ta.mean()) / se if np.isfinite(se) and se > 0 else np.nan})
    return pd.DataFrame(out_main), pd.DataFrame(out_tail)


def main() -> None:
    pd.set_option("display.width", 250)
    market = FuturesMarket()
    rets, dates = market.ret_wide, market.dates
    pos = pd.Series(range(len(dates)), index=dates)
    month_idx = pos.resample("MS").first().dropna().astype(int)
    sT = [int(i) for i in month_idx.values if i >= WIN + max(KS) and i + HMAX <= len(dates) - 1 and 2015 <= dates[int(i)].year <= 2024]
    fut = collect(rets, dates, sT, lambda T: _universe(market, T, 120))
    fut.to_parquet(OUT / "turning_futures_obs.parquet", index=False)
    del market

    px = pd.read_parquet(CASH_DATA / "prices_wide.parquet"); vol = pd.read_parquet(CASH_DATA / "volume_wide.parquet")
    good = px.columns[px.notna().any()]; px, vol = px[good], vol.reindex(columns=good)
    crets, cdates = px.pct_change(fill_method=None), px.index
    memb = pd.read_parquet(ROOT / "futures" / "data" / "membership.parquet")
    memb_by_date = {d: set(g["symbol"]) for d, g in memb.groupby("date")}
    cpos = pd.Series(range(len(cdates)), index=cdates)
    cmonth = cpos.resample("MS").first().dropna().astype(int)
    cT = [int(i) for i in cmonth.values if i >= WIN + max(KS) and i + HMAX <= len(cdates) - 1 and 2015 <= cdates[int(i)].year <= 2024]

    def fo_names(T):
        fo = memb_by_date.get(cdates[T])
        if fo is None:
            fo = memb_by_date[memb[memb.date <= cdates[T]].date.max()]
        return [c for c in _liquid_universe(vol, px, T, 20, 10_000) if c in fo][:120]

    cash = collect(crets, cdates, cT, fo_names)
    cash.to_parquet(OUT / "turning_cash_pit_obs.parquet", index=False)

    M, Tl = [], []
    for df, label in ((fut, "futures PIT top-120"), (cash, "cash F&O-members top-120")):
        m, t = analyse(df, label); M.append(m); Tl.append(t)
    M, Tl = pd.concat(M), pd.concat(Tl)
    M.to_csv(OUT / "turning_split.csv", index=False); Tl.to_csv(OUT / "turning_tail.csv", index=False)
    fmt = lambda v: f"{v:.3f}"
    for label in M.series.unique():
        print(f"\n=== {label}: full cross-section, by k and horizon (row 'turning - extending': fm_r = difference, fm_t = Welch t, hit_ge2 column = PAIRED t on per-date differences) ===")
        print(M[M.series == label].drop(columns="series").to_string(index=False, float_format=fmt))
        print(f"\n=== {label}: |s| >= 3 tail, pooled reversion return -sign(s) x fwd residual ===")
        print(Tl[Tl.series == label].drop(columns="series").to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
