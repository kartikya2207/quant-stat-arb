"""
harness_validation.py — known-answer validation of the fixed backtest harness.

Three synthetic datasets (200 stocks x 1000 days), each = common PCA factors +
an idiosyncratic residual with a KNOWN property. Run the real
harness_backtest.run_backtest at ZERO cost and check the verdict.

  Test 1  STRONG reversion  -> MUST profit  (residual = OU level, AR(1) phi=0.4;
                                             lag-1 autocorr of residual returns ~ -0.3)
  Test 2  Random walk       -> MUST be ~0   (iid residual returns, no autocorr)
  Test 3  Momentum (anti)   -> MUST lose normally, and WIN when s-score negated
                               (residual returns AR(1) rho=+0.3)

open[t+1] = close[t] (enter at last observed price; no look-ahead since close[t]
is known at decision time) so a genuine edge is capturable at daily cadence.
"""
import numpy as np
import pandas as pd

import sys, os  # make the sibling strategy/ package importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "strategy"))
import signal as sig
import harness_backtest as hb
from harness_backtest import BacktestConfig
from harness_decide import DecideConfig
from harness_execute import CostConfig

N_STOCKS, N_DAYS, K = 200, 1000, 3
ZERO_STAT = lambda q, p, b: 0.0
ZERO_SLIP = CostConfig(slippage_per_side=0.0)

CFG = BacktestConfig(
    initial_capital=1_000_000.0, pca_window=60, n_factors=5,
    universe_size=N_STOCKS, liquidity_lookback=20, rebalance_every=1,
    decide=DecideConfig(entry_threshold=2.0, profit_barrier=0.5,
                        stop_barrier=3.0, max_hold_bars=15),
    cost=ZERO_SLIP,
)


def make_dataset(kind, seed=0, phi=0.4, rho=0.3):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.015, size=(N_DAYS, K))            # factor returns
    betas = rng.normal(0, 1.0, size=(N_STOCKS, K))        # loadings
    factor_ret = f @ betas.T                              # days x stocks

    e = np.zeros((N_DAYS, N_STOCKS))
    if kind == "reversion":
        eps = rng.normal(0, 0.0125, size=(N_DAYS, N_STOCKS))
        X = np.zeros((N_DAYS, N_STOCKS))                  # OU level
        for t in range(1, N_DAYS):
            X[t] = phi * X[t - 1] + eps[t]
        e[1:] = X[1:] - X[:-1]                            # residual return = dX
    elif kind == "randomwalk":
        e = rng.normal(0, 0.013, size=(N_DAYS, N_STOCKS)) # iid -> level is RW
    elif kind == "momentum":
        nu = rng.normal(0, 0.013, size=(N_DAYS, N_STOCKS))
        for t in range(1, N_DAYS):
            e[t] = rho * e[t - 1] + nu[t]                 # positively autocorrelated
    else:
        raise ValueError(kind)

    total = factor_ret + e
    prices = 100.0 * np.exp(np.cumsum(total, axis=0))
    idx = pd.bdate_range("2019-01-01", periods=N_DAYS)
    cols = [f"S{i:03d}" for i in range(N_STOCKS)]
    prices_df = pd.DataFrame(prices, index=idx, columns=cols)
    opens_df = prices_df.shift(1)                         # open[t]=close[t-1]
    vol_df = pd.DataFrame(rng.uniform(1e5, 2e5, size=(N_DAYS, N_STOCKS)),
                          index=idx, columns=cols)
    # measured lag-1 autocorr of the TRUE residual returns
    ac = np.nanmean([np.corrcoef(e[1:, i], e[:-1, i])[0, 1] for i in range(N_STOCKS)])
    return prices_df, vol_df, opens_df, ac


def run(prices, vol, opens):
    res = hb.run_backtest(prices, vol, open_wide=opens, cfg=CFG,
                          statutory_cost_fn=ZERO_STAT)
    ec, dr = res.equity_curve, res.daily_returns
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    ret = ec.iloc[-1] / ec.iloc[0] - 1
    maxdd = ((ec - ec.cummax()) / ec.cummax()).min()
    return ret, sharpe, maxdd, len(res.trades)


rows = []

# ---- Test 1: strong reversion ----
p, v, o, ac1 = make_dataset("reversion", seed=1)
r1 = run(p, v, o)
rows.append(("1 strong-reversion", ac1, *r1, "PROFIT", r1[0] > 0.0 and r1[1] > 0.3))

# ---- Test 2: random walk ----
p, v, o, ac2 = make_dataset("randomwalk", seed=2)
r2 = run(p, v, o)
rows.append(("2 random-walk", ac2, *r2, "~ZERO", abs(r2[1]) < 0.3))

# ---- Test 3a: momentum, normal signal (should lose) ----
p, v, o, ac3 = make_dataset("momentum", seed=3)
r3 = run(p, v, o)
rows.append(("3 momentum-normal", ac3, *r3, "LOSE", r3[1] < 0.0))

# ---- Test 3b: momentum, s-score NEGATED (should win) ----
_orig = sig.compute_signal
def _flipped(*a, **k):
    out = _orig(*a, **k)
    out.s_score = -out.s_score
    return out
sig.compute_signal = _flipped            # monkeypatch: harness uses sig.compute_signal
try:
    r3f = run(p, v, o)
finally:
    sig.compute_signal = _orig
rows.append(("3 momentum-FLIPPED", ac3, *r3f, "WIN", r3f[1] > 0.0))

# ---- report ----
print("\n" + "=" * 96)
print("  HARNESS VALIDATION — known-answer synthetic data, ZERO cost")
print("=" * 96)
print(f"{'test':>20} | {'resid ac1':>9} | {'return':>9} | {'Sharpe':>7} | "
      f"{'maxDD':>7} | {'trades':>7} | {'expect':>7} | {'PASS?':>5}")
print("-" * 96)
for name, ac, ret, sh, dd, ntr, expect, ok in rows:
    print(f"{name:>20} | {ac:>+9.3f} | {ret:>+8.1%} | {sh:>+7.2f} | "
          f"{dd:>6.1%} | {ntr:>7,} | {expect:>7} | {'PASS' if ok else 'FAIL':>5}")
print("=" * 96)

# overall verdict
t1 = rows[0][-1]; t2 = rows[1][-1]
t3 = rows[2][-1] and rows[3][-1] and (rows[3][3] > rows[2][3])   # flip beats normal
print(f"Test 1 (detect edge)      : {'PASS' if t1 else 'FAIL'}")
print(f"Test 2 (no fake edge)     : {'PASS' if t2 else 'FAIL'}")
print(f"Test 3 (direction handling): {'PASS' if t3 else 'FAIL'}  "
      f"(normal Sharpe {rows[2][3]:+.2f} -> flipped {rows[3][3]:+.2f})")
print(f"OVERALL: {'TRUSTWORTHY' if (t1 and t2 and t3) else 'SUSPECT — investigate'}")
