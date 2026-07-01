"""
run_backtest.py — run the B1 baseline backtest on the real downloaded data.

Usage:
    poetry run python run_backtest.py
"""
import time
import numpy as np
import pandas as pd

import harness_backtest as bt

print("Loading data...")
prices = pd.read_parquet("./data/prices_wide.parquet")
volume = pd.read_parquet("./data/volume_wide.parquet")
opens = pd.read_parquet("./data/open_wide.parquet")
print(f"  prices: {prices.shape}, opens: {opens.shape}")

# drop columns that are entirely NaN (stocks that failed to download) so they
# don't waste cycles; the harness handles them anyway but this is cleaner.
good = prices.columns[prices.notna().any()]
prices = prices[good]
volume = volume.reindex(columns=good)
opens = opens.reindex(columns=good)
print(f"  usable stocks (have some data): {len(good)}")

cfg = bt.BacktestConfig(
    initial_capital=1_000_000.0,
    pca_window=60,
    n_factors=5,
    universe_size=200,
    liquidity_lookback=20,
    rebalance_every=1,      # daily; set to 5 for a faster weekly test
)

print("Running backtest (daily PCA over ~6 years — may take a few minutes)...")
t0 = time.time()
res = bt.run_backtest(prices, volume, open_wide=opens, cfg=cfg)
print(f"Done in {time.time() - t0:.0f}s")

ec = res.equity_curve
dr = res.daily_returns
sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
peak = ec.cummax()
max_dd = ((ec - peak) / peak).min()

print()
print("=" * 55)
print("           B1 BASELINE — REAL DATA RESULT")
print("=" * 55)
print(f"  bars recorded   : {len(ec)}")
print(f"  total trades    : {len(res.trades)}")
print(f"  start equity    : Rs {ec.iloc[0]:,.0f}")
print(f"  end equity      : Rs {ec.iloc[-1]:,.0f}")
print(f"  total return    : {ec.iloc[-1] / ec.iloc[0] - 1:+.1%}")
print(f"  total cost paid : Rs {res.cost_paid.sum():,.0f}")
print(f"  avg gross expo  : {res.exposure['gross'].mean():.2f}")
print(f"  avg net expo    : {res.exposure['net'].mean():+.3f}")
print(f"  avg # positions : {res.exposure['n_positions'].mean():.0f}")
print(f"  net Sharpe (ann): {sharpe:.2f}")
print(f"  max drawdown    : {max_dd:.1%}")
print("=" * 55)

ec.to_csv("./data/equity_curve.csv")
res.trades.to_csv("./data/trades.csv", index=False)
res.exposure.to_csv("./data/exposure.csv")
print("Saved equity_curve.csv, trades.csv, exposure.csv to ./data/")
