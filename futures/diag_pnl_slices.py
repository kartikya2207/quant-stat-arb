"""
diag_pnl_slices.py — where the realised P&L is and isn't. Read-only.

Every closed position of the headline run (results_futures_2015_2024/trades.csv),
P&L summed over its legs before costs (net of costs also shown), sliced by:
  |s| at entry (2-2.5, 2.5-3, >3), side, hold length, whether it crossed a
  roll, liquidity rank of the name on the decision day (OI rank 1-30, 31-60,
  61-90, 91-120), the equal-weight market move over the holding period (all
  listed names), the exit reason, the calendar year, and industry (NSE's
  Nifty-500 industry list; names that were renamed or delisted are unmapped
  and shown as their own group rather than guessed).
Per slice: n, win rate with its binomial SE, mean P&L per position (Rs lakh)
with a t-statistic, and the slice's total P&L.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_futures_2015_2024"
DATA = ROOT / "futures" / "data"


def positions() -> pd.DataFrame:
    tr = pd.read_csv(RES / "trades.csv", parse_dates=["date", "expiry"])
    op = tr[tr.action == "OPEN"][["symbol", "entry_bar", "bar", "date", "lots", "lot_size", "price", "s_score", "contract_choice"]] \
        .rename(columns={"bar": "entry_fill_bar", "date": "entry_date"})
    legs = tr[tr.realized_pnl.notna()].groupby(["symbol", "entry_bar"]).realized_pnl.sum().rename("pnl")
    cost = tr[tr.action != "CA_ADJUST"].groupby(["symbol", "entry_bar"]).cost_total.sum().rename("cost")
    rolled = tr[tr.action == "ROLL_CLOSE"].groupby(["symbol", "entry_bar"]).size().rename("n_rolls")
    cl = tr[tr.action.isin(["CLOSE", "FORCE_CLOSE"])][["symbol", "entry_bar", "bar", "date", "reason"]] \
        .rename(columns={"bar": "exit_fill_bar", "date": "exit_date"})
    p = (op.merge(legs, on=["symbol", "entry_bar"], how="left").merge(cost, on=["symbol", "entry_bar"], how="left")
         .merge(rolled, on=["symbol", "entry_bar"], how="left").merge(cl, on=["symbol", "entry_bar"], how="inner"))
    p["n_rolls"] = p.n_rolls.fillna(0).astype(int)
    p["net"] = p.pnl - p.cost
    p["side"] = np.where(p.lots > 0, "LONG", "SHORT")
    p["abs_s"] = p.s_score.abs()
    p["s_bucket"] = pd.cut(p.abs_s, [2, 2.5, 3, 99], labels=["2-2.5", "2.5-3", ">3"], right=False)
    p["hold"] = p.exit_fill_bar - p.entry_fill_bar
    p["hold_bucket"] = pd.cut(p.hold, [0, 3, 7, 12, 99], labels=["1-3", "4-7", "8-12", "13-16"])
    p["crossed_roll"] = np.where(p.n_rolls > 0, "rolled", "no roll")
    p["year"] = p.exit_date.dt.year
    p["notional"] = p.lots.abs() * p.lot_size * p.price
    p["ret_pct"] = 100 * p.pnl / p.notional
    return p


def main() -> None:
    pd.set_option("display.width", 220)
    p = positions()
    cont = pd.read_parquet(DATA / "continuous.parquet", columns=["date", "symbol", "oi_rank"])
    rets = pd.read_parquet(DATA / "cont_ret_wide.parquet")
    dates = rets.index
    # liquidity rank on the decision day
    dec_dates = pd.Series(dates[p.entry_bar.values], index=p.index)
    rk = cont.set_index(["date", "symbol"]).oi_rank
    p["oi_rank"] = [rk.get((d, s), np.nan) for d, s in zip(dec_dates, p.symbol)]
    p["liq_bucket"] = pd.cut(p.oi_rank, [0, 30, 60, 90, 120], labels=["rank 1-30", "31-60", "61-90", "91-120"])
    # market move over the holding period: equal-weight mean of all listed names, compounded from entry fill to exit fill
    mkt = rets.mean(axis=1)
    cum = (1 + mkt.fillna(0)).cumprod()
    p["mkt_move"] = 100 * (cum.iloc[p.exit_fill_bar.values].values / cum.iloc[p.entry_fill_bar.values].values - 1)
    p["mkt_bucket"] = pd.cut(p.mkt_move, [-99, -2, 0, 2, 99], labels=["mkt < -2%", "-2..0%", "0..+2%", "> +2%"])
    # industry
    ind = pd.read_csv(ROOT / "universe" / "ind_nifty500list.csv").set_index("Symbol")["Industry"]
    p["industry"] = p.symbol.map(ind).fillna("unmapped (renamed/delisted)")
    p.to_csv(RES / "diagnostics" / "positions_with_attributes.csv", index=False)

    def slice_(col, label):
        g = p.groupby(col, observed=True)
        out = pd.DataFrame({
            "n": g.size(),
            "win_rate": g.pnl.apply(lambda s: (s > 0).mean()),
            "win_se": g.pnl.apply(lambda s: np.sqrt(0.25 / len(s))),
            "mean_pnl_lakh": g.pnl.mean() / 1e5,
            "t_mean": g.pnl.apply(lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan),
            "total_lakh": g.pnl.sum() / 1e5,
            "mean_ret_pct": g.ret_pct.mean(),
            "mean_net_lakh": g.net.mean() / 1e5,
        })
        print(f"\n=== by {label} ===")
        print(out.round(3).to_string())
        return out.assign(dimension=label)

    allp = pd.DataFrame({"n": [len(p)], "win_rate": [(p.pnl > 0).mean()], "win_se": [np.sqrt(0.25 / len(p))],
                         "mean_pnl_lakh": [p.pnl.mean() / 1e5], "t_mean": [p.pnl.mean() / (p.pnl.std(ddof=1) / np.sqrt(len(p)))],
                         "total_lakh": [p.pnl.sum() / 1e5], "mean_ret_pct": [p.ret_pct.mean()], "mean_net_lakh": [p.net.mean() / 1e5]}, index=["ALL"])
    print("=== all closed positions ==="); print(allp.round(3).to_string())
    tabs = [allp.assign(dimension="all")]
    for col, label in [("s_bucket", "|s| at entry"), ("side", "side"), ("hold_bucket", "hold length (bars, fill to fill)"),
                       ("crossed_roll", "crossed a roll"), ("liq_bucket", "OI rank on the decision day"),
                       ("mkt_bucket", "equal-weight market move over the hold"), ("reason", "exit reason"),
                       ("year", "year of exit"), ("industry", "industry (NSE Nifty-500 list)")]:
        tabs.append(slice_(col, label))
    # two-way: side x |s| bucket, and |s| bucket x exit reason
    print("\n=== side x |s| bucket: n / win / mean P&L (lakh) ===")
    two = p.groupby(["side", "s_bucket"], observed=True).pnl.agg(n="size", win=lambda s: (s > 0).mean(), mean_lakh=lambda s: s.mean() / 1e5)
    print(two.round(3).to_string())
    print("\n=== |s| bucket x exit reason: n ===")
    print(pd.crosstab(p.s_bucket, p.reason).to_string())
    # size of win vs loss: the truncation asymmetry behind the 59% win rate
    w, l = p.pnl[p.pnl > 0], p.pnl[p.pnl <= 0]
    print(f"\nwins {len(w)} mean +{w.mean() / 1e5:.2f} lakh; losses {len(l)} mean {l.mean() / 1e5:.2f} lakh; "
          f"win rate {len(w) / len(p):.3f}; mean/sd per position {p.pnl.mean() / p.pnl.std():.4f}")
    pd.concat(tabs).to_csv(RES / "diagnostics" / "pnl_slices.csv")


if __name__ == "__main__":
    main()
