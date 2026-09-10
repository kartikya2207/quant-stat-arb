"""
fut_validation.py — Stage 3a: known-answer validation of the FUTURES harness.

The same three synthetic worlds as tests/harness_validation.py (200 stocks x
1000 days = common PCA factors + an idiosyncratic residual with a KNOWN
property), pushed through the REAL futures harness — monthly contracts, whole
lots, rolls at 2 sessions to expiry, the ledger, the cost module in its
asserted zero mode — so that what is validated is the code that will produce
the Stage 4 numbers, not a simplified stand-in.

  Test 1  STRONG reversion  (residual return lag-1 autocorr ~ -0.30) -> must PROFIT
  Test 2  Random walk       (autocorr 0)                             -> must be ~ FLAT
  Test 3  Momentum          (autocorr +0.30)                         -> must LOSE
  Test 3f same data, s-score NEGATED                                 -> must WIN

Synthetic futures: every name has a contract for each of the next two monthly
expiries (last Thursday of the month). Contract price = spot (zero basis, so the
roll itself is P&L-neutral and the test isolates the harness mechanics);
open[t] = close[t-1] as in the cash suite, so a genuine daily edge is capturable.
Lot size per name is set so one lot is ~Rs 5 lakh, i.e. the Rs 10 crore book
holds up to 10 lots per name under the 5% cap and lot flooring bites for real.

Pass criteria are the cash suite's: reversion return > 0 and Sharpe > 0.3;
random walk |Sharpe| < 0.3; momentum Sharpe < 0; flipped Sharpe > 0 and above
the un-flipped Sharpe.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import fut_backtest as fb
from fut_backtest import FuturesBacktestConfig, run_backtest
from fut_costs import CostConfig
from fut_market import FuturesMarket

N_STOCKS, N_DAYS, K = 200, 1000, 3
CAPITAL = 100_000_000.0


def make_dataset(kind, seed=0, phi=0.4, rho=0.3):
    """Verbatim from tests/harness_validation.py."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.015, size=(N_DAYS, K))
    betas = rng.normal(0, 1.0, size=(N_STOCKS, K))
    factor_ret = f @ betas.T
    e = np.zeros((N_DAYS, N_STOCKS))
    if kind == "reversion":
        eps = rng.normal(0, 0.0125, size=(N_DAYS, N_STOCKS))
        X = np.zeros((N_DAYS, N_STOCKS))
        for t in range(1, N_DAYS):
            X[t] = phi * X[t - 1] + eps[t]
        e[1:] = X[1:] - X[:-1]
    elif kind == "randomwalk":
        e = rng.normal(0, 0.013, size=(N_DAYS, N_STOCKS))
    elif kind == "momentum":
        nu = rng.normal(0, 0.013, size=(N_DAYS, N_STOCKS))
        for t in range(1, N_DAYS):
            e[t] = rho * e[t - 1] + nu[t]
    else:
        raise ValueError(kind)
    total = factor_ret + e
    prices = 100.0 * np.exp(np.cumsum(total, axis=0))
    idx = pd.bdate_range("2019-01-01", periods=N_DAYS)
    cols = [f"S{i:03d}" for i in range(N_STOCKS)]
    prices_df = pd.DataFrame(prices, index=idx, columns=cols)
    ac = np.nanmean([np.corrcoef(e[1:, i], e[:-1, i])[0, 1] for i in range(N_STOCKS)])
    return prices_df, ac


def _last_thursdays(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    months = pd.period_range(dates[0], dates[-1] + pd.offsets.MonthEnd(3), freq="M")
    out = []
    for m in months:
        d = m.to_timestamp(how="end").normalize()
        while d.weekday() != 3:
            d -= pd.Timedelta(days=1)
        out.append(d)
    return pd.DatetimeIndex(out)


def make_market(prices: pd.DataFrame, seed: int, roll_before_td: int = 2) -> FuturesMarket:
    """Synthetic single-stock futures in the Stage 1 layout."""
    rng = np.random.default_rng(seed + 1000)
    dates, cols = prices.index, list(prices.columns)
    n, m = len(dates), len(cols)
    opens = prices.shift(1)                                   # open[t] = close[t-1]
    expiries = _last_thursdays(dates)
    e_i8, d_i8 = expiries.asi8, dates.asi8
    near_ix = np.searchsorted(e_i8, d_i8, side="left")       # first expiry >= date
    lot = pd.Series(np.maximum(1, np.round(5e5 / prices.iloc[0])).astype(float), index=cols)

    # contracts: near and next expiry for every (date, symbol); price = spot
    frames = []
    for k in (0, 1):
        exp_k = expiries[near_ix + k]
        e_pos = np.searchsorted(d_i8, exp_k.asi8, side="left")
        dte = e_pos - np.arange(n)
        frames.append(pd.DataFrame({
            "date": np.repeat(dates.values, m), "symbol": np.tile(cols, n),
            "expiry": np.repeat(exp_k.values, m),
            "open": opens.values.ravel(), "close": prices.values.ravel(),
            "settle": prices.values.ravel(), "lot_size": np.tile(lot.values, n),
            "dte": np.repeat(dte, m), "k": k}))
    contracts = pd.concat(frames, ignore_index=True)
    contracts = contracts[contracts["open"].notna()]

    # chain: nearest expiry with dte > roll_before_td
    c = contracts.sort_values(["symbol", "date", "expiry"])
    ok = c[c["dte"] > roll_before_td]
    front = ok.groupby(["symbol", "date"], as_index=False).first()
    front = front.sort_values(["symbol", "date"]).reset_index(drop=True)
    prev_exp = front.groupby("symbol")["expiry"].shift(1)
    cont = pd.DataFrame({
        "date": front["date"], "symbol": front["symbol"], "front_expiry": front["expiry"],
        "front_open": front["open"], "front_settle": front["settle"],
        "lot_size": front["lot_size"], "dte": front["dte"], "tradable": True,
        "is_roll": (front["expiry"] != prev_exp) & prev_exp.notna()})
    ret_wide = prices.pct_change(fill_method=None)            # zero basis: chain ret = spot ret
    oi_wide = pd.DataFrame(np.tile(rng.uniform(5e8, 1.5e9, size=m), (n, 1)),
                           index=dates, columns=cols)
    return FuturesMarket.from_frames(cont, contracts[["date", "symbol", "expiry", "open",
                                                      "close", "settle", "lot_size"]],
                                     ret_wide, oi_wide)


CFG = FuturesBacktestConfig(initial_capital=CAPITAL, universe_size=N_STOCKS,
                            cost=CostConfig.zero(), verbose=False)


def run(market: FuturesMarket):
    res = run_backtest(market, CFG)
    ec, dr = res.equity_curve, res.daily_returns
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    ret = ec.iloc[-1] / ec.iloc[0] - 1
    maxdd = ((ec - ec.cummax()) / ec.cummax()).min()
    tr = res.trades
    n_roll = int((tr["action"] == "ROLL_CLOSE").sum()) if len(tr) else 0
    whole = bool((tr["lots"] == tr["lots"].round()).all()) if len(tr) else True
    assert res.cost_paid.sum() == 0.0 and (tr["cost_total"].sum() == 0.0 if len(tr) else True)
    return ret, sharpe, maxdd, len(tr), n_roll, whole, res


def main() -> None:
    rows = []
    t0 = time.time()

    p, ac1 = make_dataset("reversion", seed=1)
    r1 = run(make_market(p, 1))
    rows.append(("1 strong-reversion", ac1, *r1[:6], "PROFIT", r1[0] > 0.0 and r1[1] > 0.3))

    p, ac2 = make_dataset("randomwalk", seed=2)
    r2 = run(make_market(p, 2))
    rows.append(("2 random-walk", ac2, *r2[:6], "~ZERO", abs(r2[1]) < 0.3))

    p, ac3 = make_dataset("momentum", seed=3)
    mk3 = make_market(p, 3)
    r3 = run(mk3)
    rows.append(("3 momentum-normal", ac3, *r3[:6], "LOSE", r3[1] < 0.0))

    _orig = fb.sig.compute_signal

    def _flipped(*a, **k):
        out = _orig(*a, **k)
        out.s_score = -out.s_score
        return out

    fb.sig.compute_signal = _flipped
    try:
        r3f = run(mk3)
    finally:
        fb.sig.compute_signal = _orig
    rows.append(("3 momentum-FLIPPED", ac3, *r3f[:6], "WIN", r3f[1] > 0.0 and r3f[1] > r3[1]))

    print("\n" + "=" * 110)
    print("  FUTURES HARNESS VALIDATION — known-answer synthetic contracts, lots + rolls active, ZERO cost")
    print("=" * 110)
    print(f"{'test':>20} | {'resid ac1':>9} | {'return':>9} | {'Sharpe':>7} | {'maxDD':>7} | "
          f"{'fills':>6} | {'rolls':>5} | {'whole':>5} | {'expect':>6} | {'PASS?':>5}")
    print("-" * 110)
    for name, ac, ret, sh, dd, ntr, nroll, whole, expect, ok in rows:
        print(f"{name:>20} | {ac:>+9.3f} | {ret:>+8.1%} | {sh:>+7.2f} | {dd:>6.1%} | "
              f"{ntr:>6,} | {nroll:>5,} | {str(whole):>5} | {expect:>6} | {'PASS' if ok else 'FAIL':>5}")
    print("=" * 110)
    t1, t2 = rows[0][-1], rows[1][-1]
    t3 = rows[2][-1] and rows[3][-1]
    print(f"Test 1 (detect edge)       : {'PASS' if t1 else 'FAIL'}")
    print(f"Test 2 (no fake edge)      : {'PASS' if t2 else 'FAIL'}")
    print(f"Test 3 (direction handling): {'PASS' if t3 else 'FAIL'}  "
          f"(normal Sharpe {rows[2][3]:+.2f} -> flipped {rows[3][3]:+.2f})")
    print(f"OVERALL: {'TRUSTWORTHY' if (t1 and t2 and t3) else 'SUSPECT - investigate'}   "
          f"({time.time() - t0:.0f}s)")
    # lot-sizing sanity on the reversion run
    op = r1[6].trades.query("action == 'OPEN'")
    print(f"reversion run: lots per entry min {op['lots'].abs().min()} median "
          f"{op['lots'].abs().median():.0f} max {op['lots'].abs().max()}; achieved/target "
          f"{(op['achieved_weight'].abs() / op['target_weight'].abs()).mean():.3f}; "
          f"skips {len(r1[6].skips)}")


if __name__ == "__main__":
    main()
