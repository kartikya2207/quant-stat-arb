"""
diag_power.py — how large a sample would it take to detect an edge of this size?

Two framings, because the strategy's win rate and its mean P&L are not the
same thing (the triple barrier truncates: profit exits are small and frequent,
time and stop exits are larger and rarer, so a 59% win rate coexists with a
per-position mean/sd of ~0.03).
  (a) hit-rate framing: n to reject 50% at 5% two-sided with 80% power for a
      true rate of 52 / 53 / 54 / 55 / 59 %, and n at which the observed rate
      would sit at t = 2.
  (b) mean-P&L framing: with the observed per-position mean/sd ratio, n for
      t = 1.96 and for 80% power; and, at the strategy's realised annual
      Sharpe, the years needed for t = 2.
Then the arithmetic of what would produce such n at the observed entry rate:
entries per year scale with the size of the scored universe (signal density
per name is what it is), so the required universe size and the required years
are computed for each n. The F&O universe is at most ~230 names on any day.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results_futures_2015_2024"


def main() -> None:
    p = pd.read_csv(RES / "diagnostics" / "positions_with_attributes.csv")
    tr = pd.read_csv(RES / "trades.csv")
    ec = pd.read_csv(RES / "equity_curve.csv", index_col=0, parse_dates=True).iloc[:, 0]
    n = len(p); win = (p.pnl > 0).mean(); ms = p.pnl.mean() / p.pnl.std(ddof=1)
    years = (ec.index[-1] - ec.index[0]).days / 365.25
    dr = ec.pct_change().dropna(); sharpe = dr.mean() / dr.std() * np.sqrt(252)
    entries_per_year = n / years
    print(f"observed: {n} closed positions over {years:.2f} years ({entries_per_year:.0f} per year, "
          f"{entries_per_year / 250:.2f} per session); win rate {win:.3f}; per-position mean/sd {ms:.4f} "
          f"(t = {ms * np.sqrt(n):.2f}); annual Sharpe {sharpe:.3f}")
    za, zb = 1.959964, 0.841621          # z(0.975), z(0.80)
    rows = []
    for pr in (0.52, 0.53, 0.54, 0.55, win):
        n80 = ((za * 0.5 + zb * np.sqrt(pr * (1 - pr))) / (pr - 0.5)) ** 2
        n_t2 = (2 * 0.5 / (pr - 0.5)) ** 2
        rows.append({"true_win_rate": round(pr, 3), "n_for_t2": int(np.ceil(n_t2)), "n_for_80pct_power": int(np.ceil(n80)),
                     "universe_for_n80_in_10y": round(120 * n80 / (entries_per_year * 10)),
                     "years_for_n80_at_120": round(n80 / entries_per_year, 1)})
    hit = pd.DataFrame(rows)
    print("\n(a) hit-rate framing, alpha 5% two-sided:"); print(hit.to_string(index=False))
    n196 = (1.96 / ms) ** 2; n80m = ((za + zb) / ms) ** 2
    print(f"\n(b) mean-P&L framing at the observed mean/sd {ms:.4f}: n for t = 1.96: {n196:,.0f}; n for 80% power: {n80m:,.0f} "
          f"-> {n196 / entries_per_year:.0f} and {n80m / entries_per_year:.0f} years at the current entry rate, or a universe of "
          f"{120 * n196 / (entries_per_year * years):,.0f} and {120 * n80m / (entries_per_year * years):,.0f} names over the same ten years")
    for S in (sharpe, 0.3, 0.5):
        print(f"    annual Sharpe {S:.2f}: years for t = 2: {(2 / S) ** 2:.0f}; for t = 2.8 (80% power): {(2.8 / S) ** 2:.0f}")
    # what the arithmetic of the pipeline allows, stated, not proposed
    cand = 4380; paired = 1992
    print(f"\nfor reference (arithmetic only): the funnel produced {cand:,} qualifying candidates in ten years of which "
          f"{paired:,} were paired; the F&O universe peaks near 230 names/day against 120 scored; at 230 names the entry rate "
          f"would be about {entries_per_year * 230 / 120:.0f}/year, i.e. {entries_per_year * 230 / 120 * 10:,.0f} positions in ten years")
    hit.to_csv(RES / "diagnostics" / "power_hit_rate.csv", index=False)


if __name__ == "__main__":
    main()
