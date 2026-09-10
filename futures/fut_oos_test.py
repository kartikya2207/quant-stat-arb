"""
fut_oos_test.py — Stage 4b: OUT-OF-SAMPLE residual-reversion test on FUTURES.

A faithful port of tests/residual_oos_test.py. The construction is identical:
  At each monthly date T: 60-day window of the continuous futures returns,
  compute_pca_factors (the real signal code) on the top-120-by-OI universe,
  betas / window mu, sigma / factor directions V FROZEN at T, s-score at T from
  compute_s_scores. Forward (T+1..T+10), strictly OOS: factor realisation
  f_tau = V @ z_tau with the frozen directions, residual = r - [f,1].beta,
  cumulative forward residual over h days.
Reported per horizon h in {1,3,5,10}: pooled corr(s, fwd), Fama-MacBeth corr
(mean of per-date cross-sectional corrs) with its t across dates, hit-rate on
tradable extremes (|s| >= 2), mean forward residual for shorts vs longs.

Differences from the cash test are only the data and the universe rule:
returns are cont_ret_wide.parquet (masked days NaN) and the universe is the
top 120 by open-interest notional at T, as in the backtest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fut_backtest import sig, _universe          # the real signal module, loaded once
from fut_market import FuturesMarket

WIN, NF, SIZE = 60, 5, 120
HORIZONS = [1, 3, 5, 10]
HMAX = max(HORIZONS)
ENTRY = 2.0
MIN_STOCKS = 10


def run(start_year: int = 2015, end_year: int = 2024) -> tuple[pd.DataFrame, int]:
    market = FuturesMarket()
    rets, dates = market.ret_wide, market.dates
    print(f"data: {rets.shape[1]} names, {rets.shape[0]} days, {dates.min().date()} -> {dates.max().date()}")
    pos = pd.Series(range(len(dates)), index=dates)
    month_idx = pos.resample("MS").first().dropna().astype(int)
    sample_T = [int(i) for i in month_idx.values
                if i >= WIN and i + HMAX <= len(dates) - 1
                and start_year <= dates[int(i)].year <= end_year]
    print(f"monthly T dates: {len(sample_T)} ({dates[sample_T[0]].date()} -> {dates[sample_T[-1]].date()})")

    pooled_s = {h: [] for h in HORIZONS}; pooled_f = {h: [] for h in HORIZONS}
    date_corr = {h: [] for h in HORIZONS}
    hit = {h: [] for h in HORIZONS}; short_fwd = {h: [] for h in HORIZONS}; long_fwd = {h: [] for h in HORIZONS}

    for T in sample_T:
        window = rets.iloc[T - WIN + 1: T + 1]
        uni = _universe(market, T, SIZE)
        win = window.reindex(columns=uni).dropna(axis=1, how="all")
        if win.shape[1] <= NF + 1:
            continue
        try:
            factors, loadings, _ve = sig.compute_pca_factors(win, NF)
        except Exception:
            continue
        cols = list(loadings.index)
        R = win[cols]
        mu = R.mean(axis=0).values
        sigma = R.std(axis=0, ddof=1).replace(0, np.nan).values
        V = loadings.values.T
        Fv = np.column_stack([factors.values, np.ones(len(factors))])
        betas = np.linalg.pinv(Fv) @ R.values
        resid_in = sig.compute_residuals(R, factors)
        s_scores, _ou = sig.compute_s_scores(resid_in)
        s = s_scores.reindex(cols).values
        valid_s = np.isfinite(s)

        fwd_rets = rets.iloc[T + 1: T + HMAX + 1][cols].values
        resid_fwd = np.full((HMAX, len(cols)), np.nan)
        for d in range(HMAX):
            r_tau = fwd_rets[d]
            z = (r_tau - mu) / sigma
            f_tau = V @ np.nan_to_num(z, nan=0.0)
            fitted = np.concatenate([f_tau, [1.0]]) @ betas
            resid_fwd[d] = r_tau - fitted
        cum = np.nancumsum(resid_fwd, axis=0)
        present = ~np.isnan(fwd_rets)
        allpresent_upto = np.cumprod(present, axis=0).astype(bool)

        for h in HORIZONS:
            fwd = cum[h - 1]
            ok = valid_s & allpresent_upto[h - 1] & np.isfinite(fwd)
            sv, fv = s[ok], fwd[ok]
            if sv.size < MIN_STOCKS:
                continue
            pooled_s[h].append(sv); pooled_f[h].append(fv)
            if sv.std() > 0 and fv.std() > 0:
                date_corr[h].append(np.corrcoef(sv, fv)[0, 1])
            ext = np.abs(sv) >= ENTRY
            if ext.any():
                se, fe = sv[ext], fv[ext]
                hit[h].extend((np.sign(fe) == -np.sign(se)).astype(float).tolist())
                short_fwd[h].extend(fe[se >= ENTRY].tolist())
                long_fwd[h].extend(fe[se <= -ENTRY].tolist())

    rows = []
    print("\n" + "=" * 78)
    print("  OUT-OF-SAMPLE RESIDUAL-REVERSION TEST, FUTURES (factors/betas frozen at T)")
    print("=" * 78)
    print(f"{'horizon':>7} | {'pooled r':>9} | {'FM r':>8} | {'FM t':>7} | "
          f"{'N obs':>8} | {'hit% |s|>=2':>11} | {'short fwd':>10} | {'long fwd':>9}")
    print("-" * 78)
    for h in HORIZONS:
        sv = np.concatenate(pooled_s[h]); fv = np.concatenate(pooled_f[h])
        pooled_r = np.corrcoef(sv, fv)[0, 1]
        dc = np.array(date_corr[h])
        fm_r = dc.mean()
        fm_t = fm_r / (dc.std(ddof=1) / np.sqrt(len(dc))) if len(dc) > 1 else np.nan
        hh = np.array(hit[h]); hitrate = hh.mean() if hh.size else np.nan
        sf = np.mean(short_fwd[h]) if short_fwd[h] else np.nan
        lf = np.mean(long_fwd[h]) if long_fwd[h] else np.nan
        print(f"{h:>6}d | {pooled_r:>+9.4f} | {fm_r:>+8.4f} | {fm_t:>+7.2f} | "
              f"{sv.size:>8,} | {hitrate:>10.1%} | {sf:>+9.4%} | {lf:>+8.4%}")
        rows.append({"horizon": h, "pooled_r": pooled_r, "fm_r": fm_r, "fm_t": fm_t,
                     "n_dates": len(dc), "n_obs": int(sv.size), "hit_rate": hitrate,
                     "short_fwd_mean": sf, "long_fwd_mean": lf, "n_extremes": int(hh.size)})
    print("=" * 78)
    print("reversion edge => negative r, hit% > 50%, short fwd < 0, long fwd > 0")
    print(f"tradable extremes: |s| >= {ENTRY};  N dates = {len(date_corr[HORIZONS[0]])}")
    return pd.DataFrame(rows), len(date_corr[HORIZONS[0]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df, _ = run(a.start_year)
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
