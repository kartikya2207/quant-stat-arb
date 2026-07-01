"""
residual_oos_test.py — OUT-OF-SAMPLE residual-reversion test (the decisive one).

Construction (faithful to signal.py, but factors/betas are FROZEN at T):
  At each monthly date T:
    1. 60-day window of returns ending at T.
    2. compute_pca_factors -> in-sample factor scores F (60 x k) and loadings
       (stock-space eigenvector directions, k x m), via signal.py.
    3. Recompute, with the SAME math compute_residuals uses, each stock's
       betas on [F, const]  -> map factor scores to raw returns.  FROZEN.
       Also freeze window mu/sigma (standardization) and the directions V.
    4. s-score at T from signal.compute_s_scores (the actual trade signal).
  FORWARD (strictly OOS, days T+1..T+10):
    factor realization f_tau = V @ z_tau,  z_tau = (r_tau - mu)/sigma   (fixed)
    residual_{i,tau} = r_{i,tau} - ([f_tau,1] . beta_i)
    cumulative forward residual return over h days = sum_{tau=T+1..T+h} residual.

Tradable question: does s-score at T predict forward residual returns?
  reversion edge => high +s (rich) -> NEGATIVE forward residual => corr < 0.
We report, per horizon h in {1,3,5,10}:
  - pooled corr(s, fwd_resid)
  - Fama-MacBeth corr: mean of per-date cross-sectional corr, t-stat across dates
  - hit-rate on tradable extremes (|s|>=2): sign(fwd_resid) == -sign(s)
  - mean forward residual for short signals (s>=2) vs long signals (s<=-2)
"""
import numpy as np
import pandas as pd

import sys, os  # make the sibling strategy/ package importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "strategy"))
import signal as sig
from harness_backtest import _liquid_universe

WIN, NF, LB, SIZE = 60, 5, 20, 200
HORIZONS = [1, 3, 5, 10]
HMAX = max(HORIZONS)
ENTRY = 2.0          # tradable |s| threshold (matches DecideConfig)
MIN_STOCKS = 10      # min names for a per-date cross-sectional corr
import os
START_YEAR = int(os.environ.get("OOS_START_YEAR", "2019"))  # default 2019; set 2015 for full period

px = pd.read_parquet("./data/prices_wide.parquet")
vol = pd.read_parquet("./data/volume_wide.parquet")
good = px.columns[px.notna().any()]
px = px[good]; vol = vol.reindex(columns=good)
rets = px.pct_change(fill_method=None)
dates = px.index
print(f"data: {px.shape[1]} stocks, {px.shape[0]} days, {dates.min().date()} -> {dates.max().date()}")

pos = pd.Series(range(len(dates)), index=dates)
month_idx = pos.resample("MS").first().dropna().astype(int)
sample_T = [int(i) for i in month_idx.values
            if i >= WIN and i + HMAX <= len(dates) - 1
            and START_YEAR <= dates[int(i)].year <= 2024]
print(f"monthly T dates: {len(sample_T)} ({dates[sample_T[0]].date()} -> {dates[sample_T[-1]].date()})")

# pooled accumulators and per-date corr lists, keyed by horizon
pooled_s = {h: [] for h in HORIZONS}
pooled_f = {h: [] for h in HORIZONS}
date_corr = {h: [] for h in HORIZONS}
# tradable-extreme accumulators
hit = {h: [] for h in HORIZONS}          # 1 if reversion sign correct
short_fwd = {h: [] for h in HORIZONS}    # forward resid where s>=+ENTRY
long_fwd = {h: [] for h in HORIZONS}     # forward resid where s<=-ENTRY

for T in sample_T:
    window = rets.iloc[T - WIN + 1: T + 1]
    uni = _liquid_universe(vol, px, T, LB, SIZE)
    win = window.reindex(columns=uni).dropna(axis=1, how="all")
    if win.shape[1] <= NF + 1:
        continue
    try:
        factors, loadings, _ve = sig.compute_pca_factors(win, NF)
    except Exception:
        continue

    cols = list(loadings.index)               # estimation universe (complete-data names)
    R = win[cols]                             # 60 x m  (same R PCA used)
    mu = R.mean(axis=0).values
    sigma = R.std(axis=0, ddof=1).replace(0, np.nan).values
    V = loadings.values.T                     # k x m  (fixed factor directions)

    # FROZEN betas: regress each raw return on [F1..Fk, const] (matches compute_residuals)
    Fv = np.column_stack([factors.values, np.ones(len(factors))])   # 60 x (k+1)
    betas = np.linalg.pinv(Fv) @ R.values                            # (k+1) x m

    # s-score at T (the actual signal) from the same factors/residuals
    resid_in = sig.compute_residuals(R, factors)
    s_scores, _ou = sig.compute_s_scores(resid_in)
    s = s_scores.reindex(cols).values                                # m
    valid_s = np.isfinite(s)

    # ---- forward residual returns, FROZEN factors ----
    fwd_rets = rets.iloc[T + 1: T + HMAX + 1][cols].values            # HMAX x m
    resid_fwd = np.full((HMAX, len(cols)), np.nan)
    for d in range(HMAX):
        r_tau = fwd_rets[d]                                           # m
        z = (r_tau - mu) / sigma
        f_tau = V @ np.nan_to_num(z, nan=0.0)                         # k  (fixed directions)
        fitted = np.concatenate([f_tau, [1.0]]) @ betas              # m
        resid_fwd[d] = r_tau - fitted
    cum = np.nancumsum(resid_fwd, axis=0)                            # HMAX x m (cumulative)
    # mask: a stock's horizon-h value is valid only if all h forward returns present
    present = ~np.isnan(fwd_rets)
    allpresent_upto = np.cumprod(present, axis=0).astype(bool)        # HMAX x m

    for h in HORIZONS:
        fwd = cum[h - 1]                                              # m
        ok = valid_s & allpresent_upto[h - 1] & np.isfinite(fwd)
        sv, fv = s[ok], fwd[ok]
        if sv.size < MIN_STOCKS:
            continue
        pooled_s[h].append(sv); pooled_f[h].append(fv)
        if sv.std() > 0 and fv.std() > 0:
            date_corr[h].append(np.corrcoef(sv, fv)[0, 1])
        # tradable extremes
        ext = np.abs(sv) >= ENTRY
        if ext.any():
            se, fe = sv[ext], fv[ext]
            # reversion predicts forward residual sign = -sign(s); correct if matches
            hit[h].extend((np.sign(fe) == -np.sign(se)).astype(float).tolist())
            short_fwd[h].extend(fe[se >= ENTRY].tolist())
            long_fwd[h].extend(fe[se <= -ENTRY].tolist())

print("\n" + "=" * 78)
print("  OUT-OF-SAMPLE RESIDUAL-REVERSION TEST  (factors/betas frozen at T)")
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
print("=" * 78)
print("reversion edge => negative r, hit% > 50%, short fwd < 0, long fwd > 0")
print(f"tradable extremes: |s| >= {ENTRY};  N dates = {len(date_corr[HORIZONS[0]])}")
