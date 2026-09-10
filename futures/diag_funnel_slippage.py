"""
diag_funnel_slippage.py — diagnosis, not a study.

  A  full-cost run with the per-bar funnel diagnostic on (funnel.parquet)
  B  slippage sensitivity: 0.02% / 0.05% / 0.08% per side, everything else
     fixed (0.03% is the study run itself)
Outputs go to results_futures_2015_2024/diagnostics/.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fut_backtest import FuturesBacktestConfig, run_backtest
from fut_costs import CostConfig
from fut_market import FuturesMarket

OUT = Path(__file__).resolve().parent.parent / "results_futures_2015_2024" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def summary(res):
    ec, dr = res.equity_curve, res.daily_returns
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    return {"total_return": float(ec.iloc[-1] / ec.iloc[0] - 1), "sharpe": float(sh),
            "max_dd": float(((ec - ec.cummax()) / ec.cummax()).min()),
            "costs": float(res.trades["cost_total"].sum()),
            "fills": int((res.trades["action"] != "CA_ADJUST").sum()),
            "avg_gross": float(res.exposure["gross_marked"].mean())}


def main():
    market = FuturesMarket()
    t0 = time.time()
    cfg = FuturesBacktestConfig(start="2015-01-01", end="2024-12-31", verbose=False, diagnostics=True)
    res = run_backtest(market, cfg)
    res.funnel.to_parquet(OUT / "funnel.parquet", index=False)
    res.trades.to_parquet(OUT / "trades_full.parquet", index=False)
    res.exposure.to_parquet(OUT / "exposure_full.parquet")
    print(f"[A] funnel run {time.time() - t0:.0f}s, {len(res.funnel)} bars", flush=True)
    out = {"0.0003 (study)": summary(res)}
    for slip in (0.0002, 0.0005, 0.0008):
        t0 = time.time()
        c = FuturesBacktestConfig(start="2015-01-01", end="2024-12-31", verbose=False,
                                  cost=replace(CostConfig(), slippage_per_side=slip))
        r = run_backtest(market, c)
        out[f"{slip:.4f}"] = summary(r)
        r.equity_curve.to_csv(OUT / f"equity_slip_{int(slip * 1e4)}bps.csv")
        print(f"[B] slippage {slip:.4%}: {out[f'{slip:.4f}']}  ({time.time() - t0:.0f}s)", flush=True)
    with open(OUT / "slippage_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
