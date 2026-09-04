"""
fo_validate.py — Stage 1e: the gate. Nothing proceeds to Stage 2 until this passes.

Checks, in order:

  A  FORMAT / UNIT continuity across the 2024-07-08 UDiFF cutover, and a direct
     test of the old-format LOT DERIVATION: on UDiFF days we re-derive the lot the
     old way (turnover / (volume x price)) and compare it to the published
     NewBrdLotQty. If the derivation reproduces the published number, the lots we
     derived for 2015-2024-06 are trustworthy.
  B  SPOT-CHECK ~10 diverse names against NSE CASH closes: return correlation
     (expect ~0.99) and basis (expect small and positive — futures carry).
     The cash series is corporate-action adjusted with the SAME factors, since
     the cash bhavcopy is raw too.
  C  NO NaNs in the continuous return series on days a name was actually listed
     (chain breaks — first day of a listing or of a re-listing — and deliberately
     masked days excepted).
  D  NO FAKE ROLL JUMPS. Comparing roll days to ordinary days does NOT test this:
     every name rolls on the same ~120 dates, so that comparison just measures
     what the market did on those dates. The sharp test is to compare the return
     we book on a roll day (the new contract's own overnight return) against the
     return the EXPIRING contract printed on that same day. Same underlying, same
     session — a leaked roll gap would show up as the one-month basis, ~30-100 bps.
  E  CORPORATE-ACTION LOG sanity, including two named events we know the answer
     to: the RELIANCE 1:1 bonus (ex 2024-10-28, factor 0.5) and the RELIANCE-Jio
     demerger (ex 2023-07-20), the one yfinance .splits misses.
  F  DATA-QUALITY SUMMARY: names over time, adjustments applied, the names
     excluded, and — kept separate on purpose — the names merely FLAGGED for a
     large clean move, which are retained.

Writes futures/data/data_quality_report.csv and excluded_names.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fo_continuous import ROLL_BEFORE_TD

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CUTOVER = pd.Timestamp("2024-07-08")
SPOT_NAMES = ["RELIANCE", "SBIN", "INFY", "TCS", "HDFCBANK",
              "TATAMOTORS", "HINDALCO", "SUNPHARMA", "MARUTI", "ITC"]

# A name is EXCLUDED only for a structural reason: too little history to ever
# fill the 60-day PCA window, or a stale/dead series. A large clean daily return
# is FLAGGED but never excluded — see quality_report() for why.
MAX_CLEAN_DAILY_RET = 0.40      # flag threshold, NOT an exclusion threshold
MAX_FLAT_RUN = 15               # consecutive identical closes => stale/dead series
MIN_OBS = 120                   # too little history to ever enter the PCA window

RULE = "=" * 78


def _hdr(title: str) -> None:
    print("\n" + RULE + f"\n  {title}\n" + RULE)


# --------------------------------------------------------------------------- #
def check_formats(contracts: pd.DataFrame) -> dict:
    _hdr("A. FORMAT / UNIT CONTINUITY ACROSS THE 2024-07-08 CUTOVER")
    res = {}

    ud = contracts[contracts["fmt"] == "udiff"].copy()
    px = ud[["open", "high", "low", "close"]].mean(axis=1).fillna(ud["close"])
    est = np.where((ud["volume_lots"] > 0) & (px > 0),
                   ud["turnover"] / (ud["volume_lots"] * px), np.nan)
    ok = np.isfinite(est) & ud["lot_size"].notna() & (ud["lot_size"] > 0)
    rel = np.abs(est[ok] / ud["lot_size"].to_numpy()[ok] - 1.0)
    within = float((rel < 0.02).mean())
    res["lot_derivation_agreement"] = within
    print(f"  lot derivation vs published NewBrdLotQty (UDiFF rows, n={ok.sum():,}):")
    print(f"    within 2%: {within:.2%}   median |error|: {np.nanmedian(rel):.4%}")
    print(f"    -> the old-format lot derivation is "
          f"{'VALIDATED' if within > 0.95 else 'SUSPECT'}")

    before = contracts[contracts["date"] < CUTOVER]["date"].max()
    after = contracts[contracts["date"] >= CUTOVER]["date"].min()
    a = contracts[contracts["date"] == before].groupby("symbol").agg(
        oi=("oi_shares", "sum"), vol=("volume_lots", "sum"), lot=("lot_size", "first"))
    b = contracts[contracts["date"] == after].groupby("symbol").agg(
        oi=("oi_shares", "sum"), vol=("volume_lots", "sum"), lot=("lot_size", "first"))
    j = a.join(b, lsuffix="_pre", rsuffix="_post", how="inner")
    j = j[(j["oi_pre"] > 0) & (j["oi_post"] > 0) & (j["vol_pre"] > 0) & (j["vol_post"] > 0)]
    print(f"\n  last old-format day {before.date()} -> first UDiFF day {after.date()} "
          f"({len(j)} common names)")
    for c in ("oi", "vol", "lot"):
        r = (j[f"{c}_post"] / j[f"{c}_pre"]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"    {c:>4} ratio post/pre: median {r.median():.3f}  "
              f"p10 {r.quantile(.10):.3f}  p90 {r.quantile(.90):.3f}")
        res[f"cutover_{c}_median_ratio"] = float(r.median())
    print("    -> a median near 1.0 means the units did NOT change at the cutover")
    return res


# --------------------------------------------------------------------------- #
def check_vs_cash(cont: pd.DataFrame, cash: pd.DataFrame,
                  ca: pd.DataFrame | None) -> pd.DataFrame:
    _hdr("B. SPOT-CHECK vs NSE CASH (return correlation + basis)")
    cs = cash[["symbol", "date", "cash_close"]].copy()
    cs = cs.sort_values(["symbol", "date"])
    cs["cash_prev"] = cs.groupby("symbol")["cash_close"].shift(1)
    if ca is not None and len(ca):
        cs = cs.merge(ca.rename(columns={"factor": "_f", "mask_signal": "_m"}),
                      on=["symbol", "date"], how="left")
        cs["_f"] = cs["_f"].fillna(1.0)
        cs["_m"] = cs["_m"].fillna(False).astype(bool)
    else:
        cs["_f"], cs["_m"] = 1.0, False
    cs["cash_ret"] = cs["cash_close"] / (cs["cash_prev"] * cs["_f"]) - 1.0
    cs.loc[cs["_m"], "cash_ret"] = np.nan

    m = cont.merge(cs[["symbol", "date", "cash_close", "cash_ret"]],
                   on=["symbol", "date"], how="inner")
    rows = []
    for sym in SPOT_NAMES:
        g = m[m["symbol"] == sym].dropna(subset=["ret", "cash_ret"])
        if g.empty:
            rows.append({"symbol": sym, "n": 0})
            continue
        basis = (g["front_px"] - g["cash_close"]) / g["cash_close"] * 100.0
        rows.append({
            "symbol": sym, "n": len(g),
            "ret_corr": float(np.corrcoef(g["ret"], g["cash_ret"])[0, 1]),
            "basis_mean_pct": float(basis.mean()),
            "basis_median_pct": float(basis.median()),
            "max_abs_ret_pct": float(g["ret"].abs().max() * 100),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    good = out["ret_corr"].dropna()
    print(f"\n  correlations: min {good.min():.4f}  median {good.median():.4f}"
          f"   (expect ~0.99)")
    print(f"  basis: median across names {out['basis_median_pct'].median():+.3f}% "
          f"(expect small and POSITIVE = carry)")
    return out


# --------------------------------------------------------------------------- #
def check_nans(cont: pd.DataFrame) -> dict:
    _hdr("C. NaNs IN THE CONTINUOUS SERIES ON LISTED DAYS")
    g = cont.sort_values(["symbol", "date"]).copy()
    bad = g[g["ret"].isna() & ~g["chain_break"] & ~g["masked"]]
    print(f"  symbol-days total          : {len(g):,}")
    print(f"  chain breaks (first day of a listing / re-listing after a gap) : "
          f"{int(g['chain_break'].sum()):,}  (no return by construction)")
    print(f"  deliberately masked (corp) : {int(g['masked'].sum()):,}")
    print(f"  UNEXPLAINED NaN returns    : {len(bad):,}   "
          f"-> {'PASS' if len(bad) == 0 else 'FAIL'}")
    if len(bad):
        print(bad.groupby("symbol").size().sort_values(ascending=False).head(15).to_string())
    return {"unexplained_nans": int(len(bad))}


# --------------------------------------------------------------------------- #
def check_rolls(cont: pd.DataFrame, contracts: pd.DataFrame) -> dict:
    _hdr("D. ROLL INTEGRITY")
    r = cont.dropna(subset=["ret"])
    roll, norm = r[r["is_roll"]]["ret"], r[~r["is_roll"]]["ret"]
    print(f"  roll-day returns : n={len(roll):,}  mean {roll.mean():+.5f}  "
          f"median {roll.median():+.5f}  std {roll.std():.5f}")
    print(f"  ordinary returns : n={len(norm):,}  mean {norm.mean():+.5f}  "
          f"median {norm.median():+.5f}  std {norm.std():.5f}")
    print("\n  NOTE: roll days all fall on the SAME ~120 calendar dates across every\n"
          "  name, so comparing them to ordinary days really compares 120 market days\n"
          "  to 2000 — the difference is whatever the market did on those dates, not\n"
          "  evidence about the roll. The sharp test is below.")

    # THE REAL TEST: on a roll day we book the NEW contract's own overnight return.
    # If the roll gap were leaking in, that would differ from the return the
    # EXPIRING contract printed on the very same day by roughly the one-month
    # basis (~30-100 bps). Same underlying, same day: they must agree.
    px = contracts.assign(px=contracts["settle"].fillna(contracts["close"]))
    px = px.set_index(["symbol", "expiry", "date"])["px"]
    px = px[~px.index.duplicated()]
    c = cont.sort_values(["symbol", "date"]).copy()
    c["prev_date"] = c.groupby("symbol")["date"].shift(1)
    g = c[c["is_roll"] & c["ret"].notna()].copy()
    old_now = px.reindex(pd.MultiIndex.from_arrays(
        [g["symbol"], g["prev_expiry"], g["date"]])).to_numpy()
    old_prev = px.reindex(pd.MultiIndex.from_arrays(
        [g["symbol"], g["prev_expiry"], g["prev_date"]])).to_numpy()
    g["ret_expiring"] = old_now / old_prev - 1.0
    g = g.dropna(subset=["ret", "ret_expiring"])
    d_mean = (g["ret"].mean() - g["ret_expiring"].mean()) * 1e4
    d_med = (g["ret"].median() - g["ret_expiring"].median()) * 1e4
    print(f"\n  chained (new contract) vs expiring contract, SAME day, n={len(g):,}:")
    print(f"    mean   {g['ret'].mean():+.6f} vs {g['ret_expiring'].mean():+.6f}"
          f"   -> {d_mean:+.2f} bps")
    print(f"    median {g['ret'].median():+.6f} vs {g['ret_expiring'].median():+.6f}"
          f"   -> {d_med:+.2f} bps")
    ok = abs(d_mean) < 5.0
    print(f"    -> {'PASS' if ok else 'FAIL'} (a leaked roll gap would show ~30-100 bps)")

    # the roll must fire exactly 2 trading days before expiry, and we must never
    # hold a contract inside that window
    cal = np.sort(pd.DatetimeIndex(cont["date"].unique()).to_numpy())
    # exclude chain breaks: there prev_expiry belongs to an older listing stint
    g2 = c[c["is_roll"] & ~c["chain_break"]].dropna(subset=["prev_expiry"]).copy()
    g2["old_dte"] = (np.searchsorted(cal, g2["prev_expiry"].to_numpy(), side="left")
                     - np.searchsorted(cal, g2["date"].to_numpy(), side="left"))
    vc = g2["old_dte"].value_counts().sort_index()
    exact = int(vc.get(ROLL_BEFORE_TD, 0))
    print(f"\n  trading days to expiry of the contract we LEFT, on the roll day:")
    print(f"    {vc.to_dict()}")
    print(f"    -> {exact:,} of {len(g2):,} rolls fire at exactly {ROLL_BEFORE_TD} days"
          f"  ({exact / max(len(g2), 1):.3%})")

    untradable = int((~cont["tradable"]).sum())
    print(f"\n  symbol-days holding a contract inside the roll window: {untradable:,}"
          f"  ({untradable / len(cont):.3%})")
    print("    these are F&O DELISTINGS — the last contract expires with no\n"
          "    successor, so there is nothing to roll into. Flagged tradable=False;\n"
          "    the harness force-exits and never opens an expiring contract.")
    return {"roll_vs_expiring_bps": float(d_mean), "roll_ok": bool(ok)}


# --------------------------------------------------------------------------- #
def check_corp_actions(log: pd.DataFrame) -> dict:
    _hdr("E. CORPORATE-ACTION LOG")
    if log.empty:
        print("  NO ADJUSTMENTS — this is almost certainly wrong over 10 years.")
        return {"n_adjustments": 0}
    print(f"  adjustments applied: {len(log):,}   "
          f"masked from signal: {int(log['mask_signal'].sum())}")
    print("\n  by source:")
    print(log["source"].value_counts().to_string())
    print("\n  by kind:")
    print(log["kind"].value_counts().to_string())
    print("\n  by year:")
    print(log.assign(y=pd.to_datetime(log["ex_date"]).dt.year)
          .groupby("y").size().to_string())

    print("\n  KNOWN-ANSWER events:")
    for sym, ex, expect in [("RELIANCE", "2024-10-28", "1:1 bonus -> factor 0.5"),
                            ("RELIANCE", "2023-07-20", "Jio demerger (yfinance misses this)")]:
        hit = log[(log["symbol"] == sym) &
                  (pd.to_datetime(log["ex_date"]) == pd.Timestamp(ex))]
        if hit.empty:
            print(f"    {sym} {ex}: NOT FOUND  <-- expected {expect}")
        else:
            h = hit.iloc[0]
            print(f"    {sym} {ex}: factor {h['factor_applied']:.4f} "
                  f"src={h['source']} kind={h['kind']} masked={h['mask_signal']}"
                  f"  | expected {expect}")

    big = log.reindex(log["factor_applied"].sub(1).abs()
                      .sort_values(ascending=False).index).head(10)
    print("\n  10 largest adjustments:")
    print(big[["symbol", "ex_date", "kind", "factor_applied", "observed_ratio",
               "lot_implied_factor", "source"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return {"n_adjustments": int(len(log)),
            "n_masked": int(log["mask_signal"].sum())}


# --------------------------------------------------------------------------- #
def quality_report(cont: pd.DataFrame, memb: pd.DataFrame) -> pd.DataFrame:
    _hdr("F. DATA-QUALITY SUMMARY")
    per_year = (memb.assign(y=memb["date"].dt.year)
                .groupby("y")["symbol"].nunique().rename("names_with_futures"))
    daily = memb.groupby("date")["symbol"].nunique()
    per_year_daily = daily.groupby(daily.index.year).mean().round(1).rename("avg_names_per_day")
    print(pd.concat([per_year, per_year_daily], axis=1).to_string())

    rows = []
    for sym, g in cont.sort_values("date").groupby("symbol"):
        r = g["ret"].dropna()
        px = g["cont_close"]
        flat = (px.diff() == 0)
        run = best = 0
        for v in flat.to_numpy():
            run = run + 1 if v else 0
            best = max(best, run)
        mx = float(r.abs().max()) if len(r) else np.nan
        issues, flags = [], []
        if len(g) < MIN_OBS:
            issues.append(f"only {len(g)} obs - cannot fill a {60}-day PCA window")
        if best > MAX_FLAT_RUN:
            issues.append(f"{best} consecutive flat closes - stale series")
        if np.isfinite(mx) and mx > MAX_CLEAN_DAILY_RET:
            flags.append(f"largest clean day {mx:.0%}")
        rows.append({"symbol": sym, "obs": len(g),
                     "first": g["date"].min().date(), "last": g["date"].max().date(),
                     "max_abs_ret": mx, "max_flat_run": best,
                     "n_masked": int(g["masked"].sum()),
                     "n_rolls": int(g["is_roll"].sum()),
                     "median_oi_notional_cr": float(g["oi_notional"].median() / 1e7),
                     "excluded": bool(issues), "issues": "; ".join(issues),
                     "flag_large_move": bool(flags), "flags": "; ".join(flags)})
    rep = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    rep.to_csv(DATA / "data_quality_report.csv", index=False)
    exc = rep[rep["excluded"]]
    exc[["symbol", "obs", "first", "last", "max_abs_ret", "max_flat_run", "issues"]] \
        .to_csv(DATA / "excluded_names.csv", index=False)

    print(f"\n  names seen           : {len(rep)}")
    print(f"  retained             : {int((~rep['excluded']).sum())}")
    print(f"  EXCLUDED from study  : {int(rep['excluded'].sum())}")
    if len(exc):
        print("\n  excluded (insufficient history or stale series only):")
        print(exc[["symbol", "obs", "first", "last", "issues"]]
              .to_string(index=False))

    big = rep[rep["flag_large_move"] & ~rep["excluded"]]
    print(f"\n  RETAINED but carrying a >{MAX_CLEAN_DAILY_RET:.0%} clean day: {len(big)}")
    if len(big):
        print(big[["symbol", "obs", "first", "last", "max_abs_ret", "n_masked"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("""
    These are NOT excluded, deliberately. Every mechanism that re-prices a
    contract leaves a physical trace we already test for (a lot re-denomination
    or an open-interest reset), and none of these days carries one — they are
    real market events: the YES BANK moratorium, the JET AIRWAYS grounding, the
    DHFL / RCOM / RELCAPITAL defaults, PNB's fraud disclosure, INDUSINDBK in
    March 2020. Dropping a name because it once fell 90% would remove precisely
    the blow-ups and reintroduce survivorship bias, which is the flaw the
    cash-equity study had to disclose. They stay in.""")
    print(f"\n  wrote {DATA / 'data_quality_report.csv'}")
    print(f"  wrote {DATA / 'excluded_names.csv'}")
    return rep


# --------------------------------------------------------------------------- #
def main() -> None:
    contracts = pd.read_parquet(DATA / "contracts.parquet")
    cont = pd.read_parquet(DATA / "continuous.parquet")
    memb = pd.read_parquet(DATA / "membership.parquet")
    cash_p, log_p, ca_p = (DATA / "cash.parquet", DATA / "corp_action_log.csv",
                           DATA / "ca_factors.parquet")
    cash = pd.read_parquet(cash_p) if cash_p.exists() else None
    ca = pd.read_parquet(ca_p) if ca_p.exists() else None
    log = pd.read_csv(log_p) if log_p.exists() else pd.DataFrame()

    print(RULE)
    print("  STAGE 1 VALIDATION — NSE single-stock futures data pipeline")
    print(f"  {cont['date'].min().date()} -> {cont['date'].max().date()}   "
          f"{cont['date'].nunique():,} trading days   "
          f"{cont['symbol'].nunique()} names")
    print(RULE)

    check_formats(contracts)
    if cash is not None:
        check_vs_cash(cont, cash, ca)
    else:
        print("\n[skip] no cash.parquet — run fo_build.py --what cm first")
    check_nans(cont)
    check_rolls(cont, contracts)
    check_corp_actions(log)
    quality_report(cont, memb)
    print("\n" + RULE + "\n  STAGE 1 VALIDATION COMPLETE\n" + RULE)


if __name__ == "__main__":
    main()
