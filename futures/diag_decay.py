"""
diag_decay.py — is the reversion edge decaying, or was it never reliably there?

The OOS residual-reversion test (factors / betas / directions frozen at T,
strictly forward), exactly as in fut_oos_test.py and diag_cash_oos_pit.py, but
keeping the PER-DATE cross-sectional correlation and hit rate so they can be
aggregated by sub-period:
    futures, top-120-by-OI universe (genuinely point in time)
    cash prices, F&O members with a cash series, top 120 by turnover (PIT selection, survivors only)
    cash prices, the cash study's original top-200 universe (for reference)
Sub-periods: 2015-2018 vs 2019-2024 (Welch t-test on the per-date correlations
between the two), and per calendar year (12 dates each: reported, flagged weak).
Written to results_futures_2015_2024/diagnostics/decay_*.csv.
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

WIN, NF, LB = 60, 5, 20
HORIZONS = [1, 3, 5, 10]
HMAX = 10
ENTRY = 2.0
MIN_STOCKS = 10


def per_date(rets, dates, sample_T, universe_fn):
    """rows: one per (date, horizon): cross-sectional corr, hit rate, n."""
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
        s = s_scores.reindex(cols).values
        valid_s = np.isfinite(s)
        fwd_rets = rets.iloc[T + 1: T + HMAX + 1][cols].values
        resid_fwd = np.full((HMAX, len(cols)), np.nan)
        for d in range(HMAX):
            r_tau = fwd_rets[d]
            z = (r_tau - mu) / sigma
            f_tau = V @ np.nan_to_num(z, nan=0.0)
            resid_fwd[d] = r_tau - np.concatenate([f_tau, [1.0]]) @ betas
        cum = np.nancumsum(resid_fwd, axis=0)
        present = np.cumprod(~np.isnan(fwd_rets), axis=0).astype(bool)
        for h in HORIZONS:
            fwd = cum[h - 1]
            ok = valid_s & present[h - 1] & np.isfinite(fwd)
            sv, fv = s[ok], fwd[ok]
            if sv.size < MIN_STOCKS or sv.std() == 0 or fv.std() == 0:
                continue
            ext = np.abs(sv) >= ENTRY
            hit = float((np.sign(fv[ext]) == -np.sign(sv[ext])).mean()) if ext.any() else np.nan
            rows.append({"date": dates[T], "h": h, "corr": np.corrcoef(sv, fv)[0, 1], "n": int(sv.size),
                         "hit": hit, "n_ext": int(ext.sum())})
    return pd.DataFrame(rows)


def aggregate(df, label):
    def agg(d):
        c = d["corr"].to_numpy()
        w = d["n_ext"].to_numpy(); hit = d["hit"].to_numpy()
        ok = np.isfinite(hit) & (w > 0)
        return pd.Series({"n_dates": len(c), "fm_r": c.mean(),
                          "fm_t": c.mean() / (c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 2 else np.nan,
                          "hit_rate": float(np.average(hit[ok], weights=w[ok])) if ok.any() else np.nan,
                          "n_extremes": int(w.sum())})
    df = df.copy(); df["year"] = df["date"].dt.year
    df["period"] = np.where(df["year"] <= 2018, "2015-2018", "2019-2024")
    halves = df.groupby(["period", "h"]).apply(agg).reset_index()
    years = df.groupby(["year", "h"]).apply(agg).reset_index()
    welch = []
    for h in HORIZONS:
        a = df[(df.h == h) & (df.period == "2015-2018")]["corr"].to_numpy()
        b = df[(df.h == h) & (df.period == "2019-2024")]["corr"].to_numpy()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        welch.append({"h": h, "fm_r_early": a.mean(), "fm_r_late": b.mean(), "diff": a.mean() - b.mean(),
                      "welch_t": (a.mean() - b.mean()) / se, "n_early": len(a), "n_late": len(b)})
    welch = pd.DataFrame(welch)
    for d in (halves, years, welch):
        d.insert(0, "series", label)
    return halves, years, welch


def main() -> None:
    pd.set_option("display.width", 220)
    # ---- futures ------------------------------------------------------------
    market = FuturesMarket()
    rets, dates = market.ret_wide, market.dates
    pos = pd.Series(range(len(dates)), index=dates)
    month_idx = pos.resample("MS").first().dropna().astype(int)
    sT = [int(i) for i in month_idx.values if i >= WIN and i + HMAX <= len(dates) - 1 and 2015 <= dates[int(i)].year <= 2024]
    fut = per_date(rets, dates, sT, lambda T: _universe(market, T, 120))
    fut.to_csv(OUT / "decay_futures_per_date.csv", index=False)
    del market
    # ---- cash ----------------------------------------------------------------
    px = pd.read_parquet(CASH_DATA / "prices_wide.parquet"); vol = pd.read_parquet(CASH_DATA / "volume_wide.parquet")
    good = px.columns[px.notna().any()]; px, vol = px[good], vol.reindex(columns=good)
    crets, cdates = px.pct_change(fill_method=None), px.index
    memb = pd.read_parquet(ROOT / "futures" / "data" / "membership.parquet")
    memb_by_date = {d: set(g["symbol"]) for d, g in memb.groupby("date")}
    cpos = pd.Series(range(len(cdates)), index=cdates)
    cmonth = cpos.resample("MS").first().dropna().astype(int)
    cT = [int(i) for i in cmonth.values if i >= WIN and i + HMAX <= len(cdates) - 1 and 2015 <= cdates[int(i)].year <= 2024]

    def fo_names(T):
        fo = memb_by_date.get(cdates[T])
        if fo is None:
            fo = memb_by_date[memb[memb.date <= cdates[T]].date.max()]
        return [c for c in _liquid_universe(vol, px, T, LB, 10_000) if c in fo]

    cash_pit = per_date(crets, cdates, cT, lambda T: fo_names(T)[:120])
    cash_orig = per_date(crets, cdates, cT, lambda T: _liquid_universe(vol, px, T, LB, 200))
    cash_pit.to_csv(OUT / "decay_cash_pit_per_date.csv", index=False)

    H, Y, W = [], [], []
    for df, label in ((fut, "futures (PIT, top-120 by OI)"), (cash_pit, "cash, F&O members top-120 (survivors)"),
                      (cash_orig, "cash, original top-200 (survivorship-biased)")):
        h, y, w = aggregate(df, label); H.append(h); Y.append(y); W.append(w)
    H, Y, W = pd.concat(H), pd.concat(Y), pd.concat(W)
    H.to_csv(OUT / "decay_halves.csv", index=False); Y.to_csv(OUT / "decay_by_year.csv", index=False)
    W.to_csv(OUT / "decay_welch.csv", index=False)
    fmt = lambda v: f"{v:.3f}"
    print("\n=== HALVES ==="); print(H.to_string(index=False, float_format=fmt))
    print("\n=== EARLY vs LATE, Welch t on per-date correlations ==="); print(W.to_string(index=False, float_format=fmt))
    print("\n=== BY YEAR (12 dates each) ===")
    for lab in Y["series"].unique():
        print(f"\n{lab}:")
        p = Y[Y.series == lab].pivot(index="year", columns="h", values=["fm_t", "hit_rate", "fm_r"])
        print(p.round(2).to_string())


if __name__ == "__main__":
    main()
