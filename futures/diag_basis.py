"""
diag_basis.py — how much of the futures return, and of the PCA residual, is
basis rather than spot.

For every name with both a continuous futures series and a cash close:
    log F_t - log F_{t-1} = (log S_t - log S_{t-1}) + (b_t - b_{t-1}),  b = log(F/S)
on non-roll, non-masked, same-listing days (so the identity is exact; on a roll
day the futures return is the new contract's own return and b jumps by the
inter-contract gap). Reported per name and pooled:
  var(dB) / var(rF), corr(rS, dB), lag-1 autocorr of dB (mean-reverting basis
  noise would manufacture spurious residual reversion).
Then, on the OOS monthly grid, the in-window PCA residual of the FUTURES returns
is regressed on dB (pooled over names and days in the window): R^2 = the share
of residual variance that is basis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fut_backtest import sig, _universe
from fut_market import FuturesMarket

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent.parent / "results_futures_2015_2024" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cont = pd.read_parquet(DATA / "continuous.parquet",
                           columns=["date", "symbol", "front_px", "ret", "is_roll", "masked", "chain_break", "ca_factor"])
    cash = pd.read_parquet(DATA / "cash.parquet", columns=["date", "symbol", "cash_close"])
    m = cont.merge(cash, on=["date", "symbol"], how="inner").sort_values(["symbol", "date"])
    m["b"] = np.log(m["front_px"] / m["cash_close"])
    g = m.groupby("symbol")
    m["dB"] = g["b"].diff()
    m["rS"] = np.log(m["cash_close"]).groupby(m["symbol"]).diff()
    m["rF"] = np.log1p(m["ret"])
    # the identity holds only where nothing else moved: same contract, no CA factor, no break
    clean = (~m["is_roll"]) & (~m["masked"]) & (~m["chain_break"]) & (m["ca_factor"] == 1.0) \
        & m["rF"].notna() & m["rS"].notna() & m["dB"].notna()
    c = m[clean].copy()
    c["resid_identity"] = c["rF"] - c["rS"] - c["dB"]
    print(f"name-days with futures, cash and a clean identity: {len(c):,} "
          f"(identity residual max |.| {c['resid_identity'].abs().max():.2e})")

    rows = []
    for sym, gg in c.groupby("symbol"):
        if len(gg) < 250:
            continue
        rows.append({"symbol": sym, "n": len(gg),
                     "var_rF": gg["rF"].var(), "var_rS": gg["rS"].var(), "var_dB": gg["dB"].var(),
                     "share_dB": gg["dB"].var() / gg["rF"].var(),
                     "corr_rS_dB": gg["rS"].corr(gg["dB"]),
                     "ac1_dB": gg["dB"].autocorr(1), "ac1_rF": gg["rF"].autocorr(1), "ac1_rS": gg["rS"].autocorr(1),
                     "basis_mean_pct": 100 * gg["b"].mean(), "basis_sd_pct": 100 * gg["b"].std()})
    per = pd.DataFrame(rows).set_index("symbol")
    per.to_csv(OUT / "basis_decomposition_by_name.csv")
    print(f"\nper-name ({len(per)} names with >=250 clean days), medians:")
    print(per.median().round(5).to_string())
    print("\nshare of futures-return variance that is basis change (var dB / var rF): "
          f"p10 {per['share_dB'].quantile(.1):.4f}  median {per['share_dB'].median():.4f}  "
          f"p90 {per['share_dB'].quantile(.9):.4f}")
    print(f"lag-1 autocorr: futures ret median {per['ac1_rF'].median():+.4f}, cash ret {per['ac1_rS'].median():+.4f}, "
          f"basis change {per['ac1_dB'].median():+.4f}")
    print(f"corr(cash ret, basis change) median {per['corr_rS_dB'].median():+.4f}")
    # by year, pooled
    c["y"] = c["date"].dt.year
    by = c.groupby("y").apply(lambda d: pd.Series({"var_share_dB": d["dB"].var() / d["rF"].var(),
                                                    "ac1_dB": d["dB"].autocorr(1), "n": len(d)}))
    print("\npooled by year:"); print(by.round(4).to_string())

    # ---- residual variance explained by basis, on the OOS monthly grid --------
    market = FuturesMarket()
    rets = market.ret_wide
    dB_w = c.pivot(index="date", columns="symbol", values="dB").reindex(market.dates)
    dates = market.dates
    pos = pd.Series(range(len(dates)), index=dates)
    month_idx = pos.resample("MS").first().dropna().astype(int)
    sample_T = [int(i) for i in month_idx.values if i >= 60 and dates[int(i)].year >= 2015]
    num = den = 0.0
    r2s, betas = [], []
    for T in sample_T:
        win = rets.iloc[T - 59: T + 1]
        uni = _universe(market, T, 120)
        w = win.reindex(columns=uni).dropna(axis=1, how="any")
        if w.shape[1] < 10:
            continue
        factors, loadings, _ = sig.compute_pca_factors(w, 5)
        resid = sig.compute_residuals(w, factors)
        db = dB_w.loc[resid.index, resid.columns]
        x, y = db.values.ravel(), resid.values.ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 100:
            continue
        beta = np.polyfit(x, y, 1)[0]
        fitted = beta * (x - x.mean())
        ssr = ((y - y.mean() - fitted) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
        r2s.append(1 - ssr / sst); betas.append(beta)
        num += (x.var() * len(x)); den += (y.var() * len(y))
    r2s = np.array(r2s)
    print(f"\nPCA residual of futures returns regressed on basis change, {len(r2s)} monthly 60-day windows:")
    print(f"  R^2 median {np.median(r2s):.4f}  p10 {np.quantile(r2s, .1):.4f}  p90 {np.quantile(r2s, .9):.4f}")
    print(f"  beta median {np.median(betas):+.3f}   var(dB)/var(resid) pooled {num / den:.4f}")
    pd.DataFrame({"T": [dates[t] for t in sample_T[:len(r2s)]], "r2": r2s, "beta": betas}).to_csv(
        OUT / "basis_residual_r2_by_month.csv", index=False)


if __name__ == "__main__":
    main()
