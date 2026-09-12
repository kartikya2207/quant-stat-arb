"""
diag_signal_content.py — what the signal knows across the whole cross-section,
scored every name every day, versus what the +-2 tail uses. Read-only.

Every session t from bar 60: the strategy's own signal on the top-120 universe
(compute_pca_factors, frozen betas / directions / standardisation at t, as in
the OOS test), the s-score of every scored name, and the strictly forward
residual return over 1 / 3 / 5 / 10 sessions. Then, by s-score bucket:
  - the reversion return  -sign(s) x forward residual  (mean, hit rate, N)
  - the share of observations, and the share of the cross-section's total
    reversion return that sits in the bucket (where the predictable part is)
and for the whole cross-section: the daily rank IC (Spearman corr of s and
forward residual, sign-flipped so positive = reversion) and decile spreads.
Daily forward windows overlap, so t-statistics on date-level series use a
Newey-West correction with h lags.
Output: diagnostics/signal_content_daily.parquet (obs), signal_content_*.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_futures_2015_2024" / "diagnostics"
sys.path.insert(0, str(ROOT / "futures"))
from fut_backtest import sig, _universe          # noqa: E402
from fut_market import FuturesMarket             # noqa: E402

WIN, NF, SIZE = 60, 5, 120
HORIZONS = [1, 3, 5, 10]
HMAX = 10
BUCKETS = [-99, -3, -2.5, -2, -1, 0, 1, 2, 2.5, 3, 99]
LABELS = ["<-3", "-3..-2.5", "-2.5..-2", "-2..-1", "-1..0", "0..1", "1..2", "2..2.5", "2.5..3", ">3"]


def nw_t(x: np.ndarray, lag: int) -> float:
    x = x[np.isfinite(x)]; n = len(x)
    if n < 10:
        return np.nan
    xc = x - x.mean()
    lrv = xc.var() + 2 * sum((1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean() for k in range(1, lag + 1))
    return x.mean() / np.sqrt(lrv / n) if lrv > 0 else np.nan


def main() -> None:
    pd.set_option("display.width", 230)
    market = FuturesMarket()
    rets, dates = market.ret_wide, market.dates
    t0 = time.time()
    rows = []
    for T in range(WIN, len(dates) - HMAX):
        uni = _universe(market, T, SIZE)
        win = rets.iloc[T - WIN + 1: T + 1].reindex(columns=uni).dropna(axis=1, how="all")
        if win.notna().all(axis=0).sum() <= NF + 1:
            continue
        factors, loadings, _ = sig.compute_pca_factors(win, NF)
        cols = list(loadings.index)
        R = win[cols]
        mu = R.mean(axis=0).values; sigma = R.std(axis=0, ddof=1).replace(0, np.nan).values
        V = loadings.values.T
        Fv = np.column_stack([factors.values, np.ones(len(factors))])
        betas = np.linalg.pinv(Fv) @ R.values
        s_scores, _ = sig.compute_s_scores(sig.compute_residuals(R, factors))
        s = s_scores.reindex(cols).values
        fwd_rets = rets.iloc[T + 1: T + HMAX + 1][cols].values
        resid_fwd = np.full((HMAX, len(cols)), np.nan)
        for d in range(HMAX):
            r_tau = fwd_rets[d]
            z = (r_tau - mu) / sigma
            f_tau = V @ np.nan_to_num(z, nan=0.0)
            resid_fwd[d] = r_tau - np.concatenate([f_tau, [1.0]]) @ betas
        cum = np.nancumsum(resid_fwd, axis=0)
        present = np.cumprod(~np.isnan(fwd_rets), axis=0).astype(bool)
        ok = np.isfinite(s)
        for h in HORIZONS:
            m = ok & present[h - 1] & np.isfinite(cum[h - 1])
            for j in np.where(m)[0]:
                rows.append((dates[T], cols[j], h, s[j], cum[h - 1, j]))
        if (T - WIN) % 500 == 0:
            print(f"  {dates[T].date()} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["date", "symbol", "h", "s", "fwd"])
    df["rev"] = -np.sign(df.s) * df.fwd
    df["bucket"] = pd.cut(df.s, BUCKETS, labels=LABELS)
    df.to_parquet(OUT / "signal_content_daily.parquet", index=False)
    print(f"observations: {len(df):,} over {df.date.nunique()} sessions, {df.symbol.nunique()} names  ({time.time() - t0:.0f}s)")

    # ---- by bucket, per horizon: mean reversion return, NW t on the date-level mean, hit, share of obs and of total content
    tabs = []
    for h in HORIZONS:
        d = df[df.h == h]
        tot_rev = d.rev.sum()
        g = d.groupby("bucket", observed=True)
        daily = d.groupby(["date", "bucket"], observed=True).rev.mean().unstack()
        tab = pd.DataFrame({
            "obs": g.size(), "share_obs": g.size() / len(d),
            "mean_rev_bps": g.rev.mean() * 1e4,
            "nw_t": [nw_t(daily[b].to_numpy(), h) if b in daily else np.nan for b in g.size().index],
            "hit": g.rev.apply(lambda s: (s > 0).mean()),
            "sum_rev_share": g.rev.sum() / tot_rev if tot_rev != 0 else np.nan,
            "sum_rev_pts": g.rev.sum(),
        })
        tab.insert(0, "h", h)
        tabs.append(tab)
        print(f"\n=== horizon {h}d: reversion return -sign(s) x forward residual, by s bucket ===")
        print(tab.round(4).to_string())
    bt = pd.concat(tabs); bt.to_csv(OUT / "signal_content_by_bucket.csv")

    # ---- whole cross-section: daily rank IC and decile spreads
    ic_rows = []
    for h in HORIZONS:
        d = df[df.h == h]
        ic = d.groupby("date").apply(lambda x: -x.s.rank().corr(x.fwd.rank()) if len(x) > 20 else np.nan).dropna()
        # deciles of s each day
        d = d.copy(); d["dec"] = d.groupby("date").s.transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
        dec = d.groupby(["date", "dec"]).fwd.mean().unstack()
        ls = (dec[0] - dec[9]).dropna()                      # long lowest-s decile, short highest
        tail = d[d.s.abs() >= 2].groupby("date").rev.mean().dropna()
        mid = d[(d.s.abs() >= 1) & (d.s.abs() < 2)].groupby("date").rev.mean().dropna()
        ic_rows.append({"h": h, "rank_IC_mean": ic.mean(), "rank_IC_nw_t": nw_t(ic.to_numpy(), h), "IC_positive_share": (ic > 0).mean(),
                        "decile_LS_bps": ls.mean() * 1e4, "decile_LS_nw_t": nw_t(ls.to_numpy(), h),
                        "tail_|s|>=2_bps": tail.mean() * 1e4, "tail_nw_t": nw_t(tail.to_numpy(), h),
                        "mid_1<=|s|<2_bps": mid.mean() * 1e4, "mid_nw_t": nw_t(mid.to_numpy(), h), "n_dates": len(ic)})
    ict = pd.DataFrame(ic_rows); ict.to_csv(OUT / "signal_content_ic.csv", index=False)
    print("\n=== whole cross-section, per horizon (t = Newey-West with h lags on date-level series) ===")
    print(ict.round(4).to_string(index=False))
    # by year, 1d and 5d rank IC
    df1 = df[df.h.isin([1, 5])].copy()
    icy = (df1.groupby(["h", "date"]).apply(lambda x: -x.s.rank().corr(x.fwd.rank()) if len(x) > 20 else np.nan)
           .dropna().reset_index(name="ic"))
    icy["year"] = icy.date.dt.year
    print("\n=== rank IC by year (mean; NW t) ===")
    print(icy.groupby(["h", "year"]).ic.agg(mean="mean", nw_t=lambda x: nw_t(x.to_numpy(), 5), n="size").round(4).unstack(0).to_string())


if __name__ == "__main__":
    main()
