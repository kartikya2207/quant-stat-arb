"""
diag_stop_reentry.py — after a stop-loss exit, does the strategy re-enter the
same name in the same direction and get stopped again? (Supervisor's concern.)

Measured from the headline trade log (results_futures_2015_2024/trades.csv),
2015-2024. A position = (symbol, entry_bar); its P&L is the sum of realized
P&L over all its legs (entry, rolls, exit), costs summed separately.
  - stop exits, and how many were followed by a re-entry in the same
    direction; sessions from the stop fill to the re-entry fill
  - P&L of the re-entry vs the stopped attempt
  - chains: consecutive same-direction attempts each preceded by a stop
  - share of total position losses from second-and-later attempts
Written to results_futures_2015_2024/diagnostics/stop_reentry*.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results_futures_2015_2024"


def positions(tr: pd.DataFrame) -> pd.DataFrame:
    op = tr[tr.action == "OPEN"][["symbol", "entry_bar", "bar", "date", "lots", "s_score"]] \
        .rename(columns={"bar": "entry_fill_bar", "date": "entry_date"})
    op["side"] = np.where(op.lots > 0, "LONG", "SHORT")
    legs = tr[tr.realized_pnl.notna()].groupby(["symbol", "entry_bar"]).realized_pnl.sum().rename("pnl")
    cost = tr[tr.action != "CA_ADJUST"].groupby(["symbol", "entry_bar"]).cost_total.sum().rename("cost")
    cl = tr[tr.action.isin(["CLOSE", "FORCE_CLOSE"])][["symbol", "entry_bar", "bar", "date", "reason"]] \
        .rename(columns={"bar": "exit_fill_bar", "date": "exit_date"})
    p = op.merge(legs, on=["symbol", "entry_bar"], how="left").merge(cost, on=["symbol", "entry_bar"], how="left") \
        .merge(cl, on=["symbol", "entry_bar"], how="left")
    return p.sort_values(["symbol", "entry_fill_bar"]).reset_index(drop=True)


def main() -> None:
    pd.set_option("display.width", 220)
    tr = pd.read_csv(OUT / "trades.csv", parse_dates=["date", "expiry"])
    p = positions(tr)
    p["net"] = p.pnl - p.cost
    closed = p[p.reason.notna()]
    print(f"positions {len(p)}, closed {len(closed)}; exits: {closed.reason.value_counts().to_dict()}")
    print(f"total P&L before costs Rs {p.pnl.sum():,.0f}; sum of losing positions Rs {p.pnl[p.pnl < 0].sum():,.0f}; "
          f"stop-exit positions Rs {closed[closed.reason == 'stop'].pnl.sum():,.0f}")

    # ---- re-entries after a stop ------------------------------------------
    rows = []
    for r in closed[closed.reason == "stop"].itertuples(index=False):
        later = p[(p.symbol == r.symbol) & (p.entry_fill_bar > r.exit_fill_bar)].sort_values("entry_fill_bar")
        same = later[later.side == r.side]
        nxt = same.iloc[0] if len(same) else None
        rows.append({"symbol": r.symbol, "side": r.side, "stop_date": r.exit_date, "stop_pnl": r.pnl,
                     "reentered_same_dir": nxt is not None,
                     "sessions_to_reentry": (nxt.entry_fill_bar - r.exit_fill_bar) if nxt is not None else np.nan,
                     "reentry_date": nxt.entry_date if nxt is not None else pd.NaT,
                     "reentry_pnl": nxt.pnl if nxt is not None else np.nan,
                     "reentry_reason": nxt.reason if nxt is not None else "",
                     "next_any_dir_sessions": (later.iloc[0].entry_fill_bar - r.exit_fill_bar) if len(later) else np.nan})
    re = pd.DataFrame(rows)
    re.to_csv(OUT / "diagnostics" / "stop_reentry.csv", index=False)
    n_stop = len(re); n_re = int(re.reentered_same_dir.sum())
    print(f"\nstop exits: {n_stop}; re-entered same direction later (any time): {n_re}")
    s = re.sessions_to_reentry.dropna()
    print(f"  sessions from stop fill to re-entry fill: next session {int((s <= 1).sum())}, within 5 {int((s <= 5).sum())}, "
          f"within 15 {int((s <= 15).sum())}, within 60 {int((s <= 60).sum())}, later {int((s > 60).sum())}; median {s.median():.0f}")
    q = re[re.reentered_same_dir & (re.sessions_to_reentry <= 15)]
    print(f"\nre-entries within 15 sessions: {len(q)}")
    if len(q):
        print(q[["symbol", "side", "stop_date", "stop_pnl", "sessions_to_reentry", "reentry_date", "reentry_pnl", "reentry_reason"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
        print(f"  stopped attempts P&L Rs {q.stop_pnl.sum():,.0f}; their re-entries P&L Rs {q.reentry_pnl.sum():,.0f} "
              f"(mean {q.reentry_pnl.mean():,.0f}; {int((q.reentry_pnl < 0).sum())} of {len(q)} lost; "
              f"stopped again {int((q.reentry_reason == 'stop').sum())})")
    a = re[re.reentered_same_dir]
    print(f"\nall same-direction re-entries after a stop ({len(a)}): mean re-entry P&L Rs {a.reentry_pnl.mean():,.0f} vs "
          f"mean of all closed positions Rs {closed.pnl.mean():,.0f}; re-entries stopped again {int((a.reentry_reason == 'stop').sum())}")

    # ---- chains of consecutive same-direction attempts, each preceded by a stop
    chains = []
    for (sym, side), g in p.sort_values("entry_fill_bar").groupby(["symbol", "side"]):
        g = g.reset_index(drop=True)
        run = [0]
        for i in range(1, len(g)):
            if g.loc[i - 1, "reason"] == "stop":
                run.append(i)
            else:
                if len(run) >= 2:
                    chains.append((sym, side, g.loc[run]))
                run = [i]
        if len(run) >= 2:
            chains.append((sym, side, g.loc[run]))
    ch = pd.DataFrame([{"symbol": s_, "side": d, "attempts": len(g), "first": g.entry_date.iloc[0].date(),
                        "last_exit": g.exit_date.iloc[-1].date() if pd.notna(g.exit_date.iloc[-1]) else None,
                        "span_sessions": int(g.exit_fill_bar.max() - g.entry_fill_bar.min()) if g.exit_fill_bar.notna().all() else np.nan,
                        "total_pnl": g.pnl.sum(), "pnl_first": g.pnl.iloc[0], "pnl_later": g.pnl.iloc[1:].sum(),
                        "reasons": ",".join(g.reason.fillna("open").tolist())} for s_, d, g in chains])
    ch.to_csv(OUT / "diagnostics" / "stop_reentry_chains.csv", index=False)
    print(f"\nchains (a stop followed by a same-direction re-entry, consecutively): {len(ch)}; with >= 3 attempts: "
          f"{int((ch.attempts >= 3).sum()) if len(ch) else 0}")
    if len(ch):
        print(ch.sort_values("total_pnl").to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
        later_loss = ch.pnl_later[ch.pnl_later < 0].sum()
        print(f"\nP&L of second-and-later attempts in chains: Rs {ch.pnl_later.sum():,.0f}; their losses Rs {later_loss:,.0f} = "
              f"{later_loss / p.pnl[p.pnl < 0].sum():.1%} of all position losses (Rs {p.pnl[p.pnl < 0].sum():,.0f}); "
              f"{ch.pnl_later.sum() / 1e8:+.3%} of capital")


if __name__ == "__main__":
    main()
