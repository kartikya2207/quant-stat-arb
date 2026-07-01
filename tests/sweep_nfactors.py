"""
sweep_nfactors.py — n_factors sweep at TRULY zero cost.

Reuses run_backtest.py's loading pattern and harness_backtest.run_backtest.
NO strategy logic is changed: only n_factors varies; window=60, entry=2.0,
profit=0.5, stop=3.0, hold=15 held at standard via DecideConfig defaults.
Costs forced to exactly zero (statutory fn returns 0, slippage 0).
"""
import time
import numpy as np
import pandas as pd

import sys, os  # make the sibling strategy/ package importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "strategy"))
import harness_backtest as bt
from harness_decide import DecideConfig
from harness_execute import CostConfig

# ---- load data (same pattern as run_backtest.py) ----
print("Loading data...")
prices = pd.read_parquet("./data/prices_wide.parquet")
volume = pd.read_parquet("./data/volume_wide.parquet")
opens = pd.read_parquet("./data/open_wide.parquet")

good = prices.columns[prices.notna().any()]
prices = prices[good]
volume = volume.reindex(columns=good)
opens = opens.reindex(columns=good)
print(f"  prices: {prices.shape}, usable stocks: {len(good)}")

# ---- truly zero cost ----
ZERO_STATUTORY = lambda q, p, b: 0.0
ZERO_SLIPPAGE = CostConfig(slippage_per_side=0.0)

N_FACTORS = [5, 8, 10, 12, 15, 20]
rows = []

for nf in N_FACTORS:
    cfg = bt.BacktestConfig(
        initial_capital=1_000_000.0,
        pca_window=60,
        n_factors=nf,
        universe_size=200,
        liquidity_lookback=20,
        rebalance_every=1,
        decide=DecideConfig(           # standard knobs, explicit
            entry_threshold=2.0,
            profit_barrier=0.5,
            stop_barrier=3.0,
            max_hold_bars=15,
        ),
        cost=ZERO_SLIPPAGE,
    )
    t0 = time.time()
    res = bt.run_backtest(prices, volume, open_wide=opens, cfg=cfg,
                          statutory_cost_fn=ZERO_STATUTORY)
    dt = time.time() - t0

    ec = res.equity_curve
    dr = res.daily_returns
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    peak = ec.cummax()
    max_dd = ((ec - peak) / peak).min()
    total_ret = ec.iloc[-1] / ec.iloc[0] - 1
    cost_paid = res.cost_paid.sum()

    rows.append({
        "n_factors": nf,
        "return": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": len(res.trades),
        "cost": cost_paid,
        "secs": dt,
    })
    print(f"  n_factors={nf:>2}: ret={total_ret:+.1%}  sharpe={sharpe:+.2f}  "
          f"maxDD={max_dd:.1%}  trades={len(res.trades)}  cost={cost_paid:,.0f}  ({dt:.0f}s)")

print("\n" + "=" * 72)
print(f"{'n_factors':>9} | {'return':>9} | {'Sharpe':>7} | {'maxDD':>8} | {'trades':>7} | {'cost':>8}")
print("-" * 72)
for r in rows:
    print(f"{r['n_factors']:>9} | {r['return']:>+8.1%} | {r['sharpe']:>+7.2f} | "
          f"{r['max_dd']:>7.1%} | {r['trades']:>7} | {r['cost']:>8,.0f}")
print("=" * 72)

pd.DataFrame(rows).to_csv("./data/sweep_nfactors.csv", index=False)
print("Saved ./data/sweep_nfactors.csv")
