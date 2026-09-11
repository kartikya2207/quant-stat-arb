"""
run_futures_study.py — Stage 4a: the study, 2015-01-01 -> 2024-12-31.

Runs the futures harness at full cost and at (asserted) zero cost and writes
results_futures_2015_2024/ in the layout of the cash results folder:
  equity_curve.csv, equity_curve_zerocost.csv, trades.csv, trades_zerocost.csv,
  exposure.csv, exposure_zerocost.csv, skips.csv, cost_breakdown.json,
  summary.json (every headline metric for both runs).
FINDINGS.md is written by hand from these files; fut_oos_test.py writes
oos_results.csv.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fut_backtest import FuturesBacktestConfig, run_backtest
from fut_costs import CostConfig
from fut_execute import ExecConfig
from fut_market import FuturesMarket

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results_futures_2015_2024"
COMP = ["stt", "exchange", "stamp", "gst", "brokerage", "slippage"]


def metrics(res, capital: float) -> dict:
    ec, dr, ex, sk = res.equity_curve, res.daily_returns, res.exposure, res.skips
    adj = res.trades[res.trades["action"] == "CA_ADJUST"]
    tr = res.trades[res.trades["action"] != "CA_ADJUST"]
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    peak = ec.cummax()
    years = (ec.index[-1] - ec.index[0]).days / 365.25
    op = tr[tr["action"] == "OPEN"]
    closes = tr[tr["action"].isin(["CLOSE", "FORCE_CLOSE"])]
    roll_mask = tr["action"].str.startswith("ROLL")
    costs = {c: float(tr[c].sum()) for c in COMP}
    costs["total"] = float(tr["cost_total"].sum())
    costs["on_rolls"] = float(tr.loc[roll_mask, "cost_total"].sum())
    costs["roll_share"] = costs["on_rolls"] / costs["total"] if costs["total"] > 0 else 0.0
    costs["bps_per_side_notional"] = (costs["total"] / tr["notional"].sum() * 1e4
                                      if tr["notional"].sum() > 0 else 0.0)
    m = {
        "start": str(ec.index[0].date()), "end": str(ec.index[-1].date()), "bars": int(len(ec)),
        "start_equity": float(ec.iloc[0]), "end_equity": float(ec.iloc[-1]),
        "total_return": float(ec.iloc[-1] / ec.iloc[0] - 1),
        "cagr": float((ec.iloc[-1] / ec.iloc[0]) ** (1 / years) - 1),
        "sharpe_ann": float(sharpe),
        "ann_vol": float(dr.std() * np.sqrt(252)),
        "max_drawdown": float(((ec - peak) / peak).min()),
        "costs": costs,
        "fills_total": int(len(tr)),
        "ca_adjustments_on_held_positions": int(len(adj)),
        "back_month_entries": int((op.get("contract_choice", pd.Series(dtype=str)) == "back-month").sum()),
        "back_month_entry_share": float((op.get("contract_choice", pd.Series(dtype=str)) == "back-month").mean()) if len(op) else 0.0,
        "cost_on_back_month_entries": float(op.loc[op.get("contract_choice", pd.Series(dtype=str)) == "back-month", "cost_total"].sum()) if len(op) else 0.0,
        "fills_by_action": {k: int(v) for k, v in tr["action"].value_counts().items()},
        "round_trips": int(len(closes)),
        "rolls": int((tr["action"] == "ROLL_CLOSE").sum()),
        "forced_closes": int((tr["action"] == "FORCE_CLOSE").sum()),
        "exit_reasons": {k: int(v) for k, v in closes["reason"].value_counts().items()},
        "avg_gross_exposure": float(ex["gross_marked"].mean()),
        "avg_net_exposure": float(ex["net_marked"].mean()),
        "avg_gross_target": float(ex["gross_target"].mean()),
        "avg_margin_util": float(ex["margin_util"].mean()),
        "peak_margin_util": float(ex["margin_util"].max()),
        "margin_breaches": int((ex["margin_util"] > 1).sum()),
        "lot_drag_mean_achieved_over_target": float(
            (op["achieved_weight"].abs() / op["target_weight"].abs()).mean()),
        "lots_per_entry": {"min": int(op["lots"].abs().min()), "median": float(op["lots"].abs().median()),
                           "max": int(op["lots"].abs().max())},
        "skipped_entries": int(len(sk)),
        "skipped_by_reason": {k: int(v) for k, v in sk["reason"].value_counts().items()} if len(sk) else {},
        "avg_positions": float(ex["n_positions"].mean()),
        "avg_long": float(ex["n_long"].mean()), "avg_short": float(ex["n_short"].mean()),
        "traded_notional": float(tr["notional"].sum()),
        "fill_basis": {k: int(v) for k, v in tr["fill_basis"].value_counts().items()},
        "win_rate_round_trips": float((closes["realized_pnl"] > 0).mean()) if len(closes) else np.nan,
    }
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-min-dte", type=int, default=ExecConfig().entry_min_dte)
    ap.add_argument("--back-slippage", type=float, default=CostConfig().slippage_back_month_entry)
    ap.add_argument("--tag", default=None, help="write a VARIANT to diagnostics/entry_rule_variants/<tag>/ "
                                                "(full cost only) instead of the headline files")
    a = ap.parse_args()
    out_dir = OUT if a.tag is None else OUT / "diagnostics" / "entry_rule_variants" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    market = FuturesMarket()
    out = {}
    runs = (("full", CostConfig(slippage_back_month_entry=a.back_slippage)), ("zero", CostConfig.zero()))
    if a.tag is not None:
        runs = runs[:1]
    for name, cost in runs:
        t0 = time.time()
        cfg = FuturesBacktestConfig(start="2015-01-01", end="2024-12-31", cost=cost, verbose=True,
                                    exec=replace(ExecConfig(), entry_min_dte=a.entry_min_dte))
        res = run_backtest(market, cfg)
        print(f"[{name}] done in {time.time() - t0:.0f}s")
        suffix = "" if name == "full" else "_zerocost"
        res.equity_curve.to_csv(out_dir / f"equity_curve{suffix}.csv")
        res.trades.to_csv(out_dir / f"trades{suffix}.csv", index=False)
        res.exposure.to_csv(out_dir / f"exposure{suffix}.csv")
        if name == "full":
            res.skips.to_csv(out_dir / "skips.csv", index=False)
            res.signal_stats.to_csv(out_dir / "signal_stats.csv", index=False)
        out[name] = metrics(res, cfg.initial_capital)
        out[name]["config"] = {"initial_capital": cfg.initial_capital, "pca_window": cfg.pca_window,
                               "n_factors": cfg.n_factors, "universe_size": cfg.universe_size,
                               "margin_rate": cfg.margin_rate,
                               "decide": vars(cfg.decide), "cost": vars(cost), "exec": vars(cfg.exec)}
    if a.tag is not None:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(json.dumps({k: v for k, v in out["full"].items() if k != "config"}, indent=2, default=str))
        return
    # lot-count divergence between the two runs
    tf = pd.read_csv(OUT / "trades.csv"); tz = pd.read_csv(OUT / "trades_zerocost.csv")
    key = ["date", "symbol", "action"]
    same_seq = len(tf) == len(tz) and tf[key].equals(tz[key])
    out["divergence"] = {"same_fill_sequence": bool(same_seq),
                         "fills_full": int(len(tf)), "fills_zero": int(len(tz))}
    if same_seq:
        dl = tz["lots"].abs() - tf["lots"].abs()
        out["divergence"].update(fills_with_different_lots=int((dl != 0).sum()),
                                 max_lot_difference=int(dl.abs().max()))
    else:
        m = min(len(tf), len(tz))
        first = next((k for k in range(m) if not tf[key].iloc[k].equals(tz[key].iloc[k])), m)
        out["divergence"]["first_divergent_fill"] = int(first)
        out["divergence"]["first_divergent_date"] = str(tf["date"].iloc[min(first, len(tf) - 1)])
    with open(OUT / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(OUT / "cost_breakdown.json", "w") as f:
        json.dump({k: out[k]["costs"] for k in ("full", "zero")}, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in out[k].items() if kk != "config"} for k in ("full", "zero")}
                     | {"divergence": out["divergence"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
