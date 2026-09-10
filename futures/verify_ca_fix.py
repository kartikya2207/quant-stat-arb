"""
verify_ca_fix.py — replay the six positions that crossed a corporate-action
ex-date in the 2015-2024 run through the execution layer, with the adjustment
in place, and compare the booked P&L with what the pre-fix run booked.
"""
from __future__ import annotations

import pandas as pd

from fut_costs import CostConfig
from fut_execute import ExecConfig, execute_day
from fut_ledger import FuturesPosition, Ledger
from fut_market import FuturesMarket

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strategy"))
from harness_decide import Decisions, ExitOrder

CASES = [  # symbol, ex-date, booked P&L in the pre-fix run (Rs)
    ("IOC", "2018-03-15", 2_680_200.0), ("JUBLFOOD", "2018-06-21", -2_642_962.0),
    ("GAIL", "2020-02-17", 477_926.0), ("NTPC", "2023-02-03", 167_580.0),
    ("IDFC", "2023-02-13", 954_000.0), ("TATASTEEL", "2023-06-22", 185_625.0)]


def main() -> None:
    market = FuturesMarket()
    tr = pd.read_csv("../results_futures_2015_2024/diagnostics/trades_prefix.csv", parse_dates=["date", "expiry"]) \
        if Path("../results_futures_2015_2024/diagnostics/trades_prefix.csv").exists() \
        else pd.read_csv("../results_futures_2015_2024/trades.csv", parse_dates=["date", "expiry"])
    zero = CostConfig.zero()
    rows = []
    for sym, ex, booked in CASES:
        ex = pd.Timestamp(ex)
        t = tr[tr.symbol == sym].sort_values("date")
        o = t[(t.action.isin(["OPEN", "ROLL_OPEN"])) & (t.date < ex)].iloc[-1]
        cl = t[(t.action.isin(["CLOSE", "FORCE_CLOSE", "ROLL_CLOSE"])) & (t.date > o.date)].iloc[0]
        i0 = market.date_index(o.date); i_exit = market.date_index(cl.date)
        led = Ledger(1e8)
        led.open(FuturesPosition(sym, pd.Timestamp(o.expiry), int(o.lots), float(o.lot_size), float(o.price), i0 - 1,
                                 o.date, 0.0, 0.05, 0.05, float(o.price), pd.Timestamp(o.expiry), float(o.price)), 0.0)
        fills = []
        for i in range(i0 + 1, i_exit + 1):
            dec = Decisions(exits=[ExitOrder(sym, cl.reason)] if (i == i_exit and cl.action != "ROLL_CLOSE") else [],
                            entries=[])
            rep = execute_day(dec, led, market, i, 1e8, i0 - 1, zero, ExecConfig())
            fills += rep.fills
            if sym not in led:
                break
        f = pd.DataFrame(fills)
        adj = f[f.action == "CA_ADJUST"].iloc[0]
        realized = f[f.action.isin(["CLOSE", "FORCE_CLOSE", "ROLL_CLOSE"])].realized_pnl.sum()
        rows.append({"symbol": sym, "ex_date": ex.date(), "side": "LONG" if o.lots > 0 else "SHORT",
                     "lots": abs(int(o.lots)), "lot_before": float(o.lot_size), "lot_after": adj.lot_size,
                     "entry_before": float(o.price), "entry_after": adj.price, "exit": float(cl.price),
                     "exit_action": cl.action, "booked_before_fix": booked, "booked_after_fix": realized,
                     "change": realized - booked, "adjust_note": adj.reason})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["adjust_note"]).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    for r in out.itertuples(index=False):
        print(f"  {r.symbol}: {r.adjust_note}")
    print(f"\nTOTAL before {out.booked_before_fix.sum():,.0f}  after {out.booked_after_fix.sum():,.0f}  "
          f"change {out.change.sum():+,.0f}")
    out.to_csv("../results_futures_2015_2024/diagnostics/ca_fix_six_cases.csv", index=False)


if __name__ == "__main__":
    main()
