"""
diag_cash_oos_pit.py — does the cash OOS reversion result survive a
point-in-time universe?

The cash study's OOS test (tests/residual_oos_test.py), code unchanged in its
construction, run three ways on the cash study's frozen input:
  ORIGINAL   top 200 by turnover of the 435 current constituents (as published)
  PIT-120    names F&O-listed on date T (Stage 1 membership, genuinely point in
             time) that have a cash series, top 120 by turnover — the futures
             universe rule applied to cash prices
  PIT-ALL    every F&O-listed name on T with a cash series (~144/day)
Limit of the test: 85 of the 329 F&O names — essentially every delisted or
renamed one — have no cash series at all, so PIT here removes the survivorship
bias of the SELECTION but cannot add the names that died. The futures universe
does contain them.
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
from fut_backtest import sig
sys.path.insert(0, str(ROOT / "strategy"))
from harness_backtest import _liquid_universe

WIN, NF, LB = 60, 5, 20
HORIZONS = [1, 3, 5, 10]
HMAX = 10
ENTRY = 2.0
MIN_STOCKS = 10


def oos(rets, px, vol, dates, sample_T, universe_fn, label):
    pooled_s = {h: [] for h in HORIZONS}; pooled_f = {h: [] for h in HORIZONS}
    date_corr = {h: [] for h in HORIZONS}; hit = {h: [] for h in HORIZONS}
    for T in sample_T:
        uni = universe_fn(T)
        window = rets.iloc[T - WIN + 1: T + 1]
        win = window.reindex(columns=uni).dropna(axis=1, how="all")
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
            if sv.size < MIN_STOCKS:
                continue
            pooled_s[h].append(sv); pooled_f[h].append(fv)
            if sv.std() > 0 and fv.std() > 0:
                date_corr[h].append(np.corrcoef(sv, fv)[0, 1])
            ext = np.abs(sv) >= ENTRY
            if ext.any():
                hit[h].extend((np.sign(fv[ext]) == -np.sign(sv[ext])).astype(float).tolist())
    rows = []
    for h in HORIZONS:
        sv = np.concatenate(pooled_s[h]); fv = np.concatenate(pooled_f[h])
        dc = np.array(date_corr[h])
        rows.append({"universe": label, "horizon": h, "pooled_r": np.corrcoef(sv, fv)[0, 1],
                     "fm_r": dc.mean(), "fm_t": dc.mean() / (dc.std(ddof=1) / np.sqrt(len(dc))),
                     "n_dates": len(dc), "n_obs": int(sv.size), "hit_rate": float(np.mean(hit[h])),
                     "n_extremes": len(hit[h])})
    return pd.DataFrame(rows)


def main():
    px = pd.read_parquet(CASH_DATA / "prices_wide.parquet")
    vol = pd.read_parquet(CASH_DATA / "volume_wide.parquet")
    good = px.columns[px.notna().any()]
    px, vol = px[good], vol.reindex(columns=good)
    rets = px.pct_change(fill_method=None)
    dates = px.index
    memb = pd.read_parquet(ROOT / "futures" / "data" / "membership.parquet")
    memb_by_date = {d: set(g["symbol"]) for d, g in memb.groupby("date")}
    pos = pd.Series(range(len(dates)), index=dates)
    month_idx = pos.resample("MS").first().dropna().astype(int)
    sample_T = [int(i) for i in month_idx.values if i >= WIN and i + HMAX <= len(dates) - 1
                and 2015 <= dates[int(i)].year <= 2024]
    print(f"monthly T dates: {len(sample_T)}")

    def uni_orig(T):
        return _liquid_universe(vol, px, T, LB, 200)

    def fo_names(T):
        fo = memb_by_date.get(dates[T])
        if fo is None:      # cash calendar day not an F&O session: nearest earlier session
            prev = memb[memb.date <= dates[T]].date.max()
            fo = memb_by_date[prev]
        return [c for c in _liquid_universe(vol, px, T, LB, 10_000) if c in fo]

    res = pd.concat([oos(rets, px, vol, dates, sample_T, uni_orig, "ORIGINAL top-200 of 435 (survivorship-biased)"),
                     oos(rets, px, vol, dates, sample_T, lambda T: fo_names(T)[:120], "PIT F&O members with cash data, top 120"),
                     oos(rets, px, vol, dates, sample_T, fo_names, "PIT F&O members with cash data, all")],
                    ignore_index=True)
    res.to_csv(OUT / "cash_oos_pit.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
