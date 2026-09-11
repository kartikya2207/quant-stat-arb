"""
diag_backmonth_liquidity.py — how thin is the next-month contract relative to
the front, on the sessions where the entry rule would use it?

For every (symbol, session): front = the chain's contract, next = the nearest
later listed expiry. Ratios next/front of open interest (shares) and volume
(lots), overall, by year, and on the rule's window (front dte < 5), for all
names and for the top-120-by-OI universe. Also the share of next-month
contract-days in that window with no trade at all (no fill possible).
Written to results_futures_2015_2024/diagnostics/backmonth_liquidity.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent.parent / "results_futures_2015_2024" / "diagnostics"


def main() -> None:
    cont = pd.read_parquet(DATA / "continuous.parquet",
                           columns=["date", "symbol", "front_expiry", "dte", "oi_rank", "oi_notional"])
    c = pd.read_parquet(DATA / "contracts.parquet",
                        columns=["date", "symbol", "expiry", "open", "oi_shares", "volume_lots", "turnover"])
    c = c.sort_values(["symbol", "date", "expiry"])
    # next expiry after the front, per (symbol, date)
    m = c.merge(cont[["date", "symbol", "front_expiry", "dte", "oi_rank"]], on=["date", "symbol"])
    front = m[m["expiry"] == m["front_expiry"]]
    later = m[m["expiry"] > m["front_expiry"]].groupby(["symbol", "date"], as_index=False).first()
    j = front.merge(later, on=["symbol", "date"], suffixes=("_f", "_n"))
    j["oi_ratio"] = j["oi_shares_n"] / j["oi_shares_f"].replace(0, np.nan)
    j["vol_ratio"] = j["volume_lots_n"] / j["volume_lots_f"].replace(0, np.nan)
    j["next_no_trade"] = (j["volume_lots_n"].fillna(0) == 0) | j["open_n"].isna()
    j["window"] = j["dte_f"] < 5
    j["top120"] = j["oi_rank_f"] <= 120
    j["year"] = j["date"].dt.year

    def summ(d):
        return pd.Series({"n": len(d), "oi_ratio_med": d["oi_ratio"].median(), "oi_ratio_p25": d["oi_ratio"].quantile(.25),
                          "vol_ratio_med": d["vol_ratio"].median(), "vol_ratio_p25": d["vol_ratio"].quantile(.25),
                          "next_no_trade_share": d["next_no_trade"].mean(),
                          "next_vol_lots_med": d["volume_lots_n"].median(),
                          "next_turnover_cr_med": d["turnover_n"].median() / 1e7})
    rows = {
        "all name-days": summ(j),
        "all name-days, rule window (front dte<5)": summ(j[j["window"]]),
        "top-120 names, all days": summ(j[j["top120"]]),
        "top-120 names, rule window": summ(j[j["top120"] & j["window"]]),
    }
    tab = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print("next-month / front-month liquidity:")
    print(tab.round(3).to_string())
    by = j[j["top120"] & j["window"]].groupby("year").apply(summ)
    print("\ntop-120 names, rule window, by year:")
    print(by.round(3).to_string())
    tab.to_csv(OUT / "backmonth_liquidity.csv")
    by.to_csv(OUT / "backmonth_liquidity_by_year.csv")


if __name__ == "__main__":
    main()
