"""
run_futures_dryrun.py — Stage 2 dry run on a few months of data.

Not a backtest result. It proves the plumbing before the full run:
  1. every fill is on a real contract: the (symbol, expiry, date) row exists in
     contracts.parquet, the fill price is that row's open (or a tagged fallback),
     the contract has not expired, an OPEN is on the chain's front contract with
     more than 2 sessions left, a ROLL leaves a contract at exactly 2 sessions.
  2. lot counts are whole and sane; target vs achieved weight (rounding drag);
     every skipped entry has a reason.
  3. rolls: the ledger carries the position across as two charged trades.
  4. zero-cost mode charges exactly 0.0 and produces the same fills.
  5. P&L reconstruction: equity change == realized + unrealized - costs, from
     the trade log alone.
  6. margin utilisation.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from fut_backtest import FuturesBacktestConfig, run_backtest
from fut_costs import CostConfig
from fut_market import FuturesMarket

pd.set_option("display.width", 220)
RULE = "=" * 78


def hdr(t):
    print("\n" + RULE + f"\n  {t}\n" + RULE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2019-06-28")
    ap.add_argument("--capital", type=float, default=100_000_000.0)
    a = ap.parse_args()

    t0 = time.time()
    market = FuturesMarket()
    print(f"[info] market loaded in {time.time() - t0:.0f}s: {len(market)} sessions, "
          f"{len(market.symbols)} names")
    contracts = pd.read_parquet("data/contracts.parquet",
                                columns=["date", "symbol", "expiry", "open", "close", "settle"])
    ckey = contracts.set_index(["symbol", "expiry", "date"])

    cfg = FuturesBacktestConfig(initial_capital=a.capital, start=a.start, end=a.end, verbose=False)
    t0 = time.time()
    res = run_backtest(market, cfg)
    print(f"[info] full-cost dry run {a.start} -> {a.end}: {time.time() - t0:.0f}s")
    tr, sk, ex = res.trades, res.skips, res.exposure

    # ------------------------------------------------------------------ 1
    hdr("1. FILLS LAND ON REAL CONTRACTS")
    print(f"fills: {len(tr)}   by action: {tr['action'].value_counts().to_dict()}")
    print(f"fill basis: {tr['fill_basis'].value_counts().to_dict()}")
    exists, px_ok, not_expired, open_ok, roll_ok = 0, 0, 0, 0, 0
    n_open, n_roll = 0, 0
    for r in tr.itertuples(index=False):
        k = (r.symbol, r.expiry, r.date)
        if k in ckey.index:
            exists += 1
            row = ckey.loc[k]
            ref = row[r.fill_basis] if r.fill_basis in ("open", "close", "settle") else np.nan
            px_ok += int(np.isclose(ref, r.price))
        i = market.date_index(r.date)
        not_expired += int(not market.is_expired(r.expiry, i))
        if r.action == "OPEN":
            n_open += 1
            f = market.front(i, r.symbol)
            if getattr(r, "contract_choice", "") == "back-month":
                open_ok += int(f is not None and market.dte(f.expiry, i) < 5
                               and market.next_expiry(i, r.symbol, f.expiry) == r.expiry)
            else:
                open_ok += int(f is not None and f.expiry == r.expiry
                               and market.dte(r.expiry, i) > 2)
        if r.action == "ROLL_CLOSE":
            n_roll += 1
            roll_ok += int(market.dte(r.expiry, i) == 2)
    print(f"contract row exists in contracts.parquet : {exists}/{len(tr)}")
    print(f"fill price == that row's {'{open|close|settle}'} : {px_ok}/{len(tr)}")
    print(f"contract not expired on fill date         : {not_expired}/{len(tr)}")
    print(f"OPEN on the chain's front with dte > 2    : {open_ok}/{n_open}")
    print(f"ROLL_CLOSE leaves a contract at dte == 2  : {roll_ok}/{n_roll}")
    print("\nsample fills:")
    cols = ["date", "symbol", "expiry", "action", "side", "lots", "lot_size", "price",
            "notional", "cost_total", "reason", "target_weight", "achieved_weight"]
    print(tr[cols].head(12).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ------------------------------------------------------------------ 2
    hdr("2. LOT SIZING")
    op = tr[tr["action"] == "OPEN"]
    print(f"OPEN fills: {len(op)}   lots: min {op['lots'].abs().min()}  median "
          f"{op['lots'].abs().median():.0f}  max {op['lots'].abs().max()}   "
          f"all integers: {bool((op['lots'] == op['lots'].round()).all())}")
    print(f"one-lot notional at fill (Rs lakh): median {(op['lot_size'] * op['price']).median() / 1e5:.1f}")
    print(f"fill notional (Rs lakh): median {op['notional'].median() / 1e5:.1f}   "
          f"max {op['notional'].max() / 1e5:.1f}   cap = {cfg.decide.max_weight_per_name * a.capital / 1e5:.0f}")
    drag = (op["achieved_weight"].abs() / op["target_weight"].abs())
    print(f"achieved / target weight: mean {drag.mean():.3f}  median {drag.median():.3f}  "
          f"min {drag.min():.3f}   (1.0 = no rounding drag)")
    print(f"sum |target| {op['target_weight'].abs().sum():.3f}  sum |achieved| "
          f"{op['achieved_weight'].abs().sum():.3f}")
    print(f"skipped entries: {len(sk)}")
    if len(sk):
        print(sk["reason"].value_counts().to_string())
    ne = (tr[tr.action == "OPEN"].groupby("date").apply(
        lambda g: pd.Series({"n_long": int((g.lots > 0).sum()), "n_short": int((g.lots < 0).sum()),
                             "w_long": g.achieved_weight[g.lots > 0].sum(),
                             "w_short": g.achieved_weight[g.lots < 0].sum()})))
    print("\nachieved long vs short weight on entry days (pairing after flooring):")
    print(ne.describe().loc[["mean", "min", "max"]].round(4).to_string())

    # ------------------------------------------------------------------ 3
    hdr("3. ROLLS IN THE LEDGER")
    rc = tr[tr["action"].isin(["ROLL_CLOSE", "ROLL_OPEN"])]
    print(f"roll legs: {len(rc)}  on dates: {sorted(rc['date'].dt.date.unique())}")
    if len(rc):
        sym = rc.iloc[0]["symbol"]
        h = tr[tr["symbol"] == sym][cols + ["fill_basis", "realized_pnl"]]
        print(f"\nfull life of {sym} (entry -> roll -> exit):")
        print(h.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    fc = tr[tr["action"] == "FORCE_CLOSE"]
    print(f"\nforced closes: {len(fc)}  reasons: {fc['reason'].value_counts().to_dict()}")

    # ------------------------------------------------------------------ 4
    hdr("4. ZERO-COST MODE")
    zcfg = FuturesBacktestConfig(initial_capital=a.capital, start=a.start, end=a.end,
                                 cost=CostConfig.zero(), verbose=False)
    zres = run_backtest(market, zcfg)
    zt = zres.trades
    comp = ["stt", "exchange", "stamp", "brokerage", "gst", "slippage", "cost_total"]
    print(f"zero-cost fills: {len(zt)}   sum of every cost component: "
          f"{zt[comp].sum().sum():.10f}   cost_paid.sum(): {zres.cost_paid.sum():.10f}")
    assert zt[comp].abs().sum().sum() == 0.0 and zres.cost_paid.sum() == 0.0, "zero-cost mode charged"
    key = ["date", "symbol", "action"]
    same = len(zt) == len(tr) and zt[key].reset_index(drop=True).equals(tr[key].reset_index(drop=True))
    print(f"same fills as the full-cost run (date/symbol/action): {same}")
    if same:
        dl = (zt["lots"].abs() - tr["lots"].abs())
        print(f"  lot-count differences: {int((dl != 0).sum())} fills, max |diff| {int(dl.abs().max())} lot(s) "
              f"- the sizing base is equity, and equity differs by the costs paid")
    else:
        m = min(len(zt), len(tr))
        first = next((k for k in range(m) if not zt[key].iloc[k].equals(tr[key].iloc[k])), m)
        print(f"  first divergence at fill #{first} of {len(tr)} (full) / {len(zt)} (zero): paths separate once "
              f"a lot count differs and a later decision sees a different book")
        print(pd.concat([tr.iloc[max(0, first - 1): first + 2][key + ["lots"]].reset_index(drop=True),
                         zt.iloc[max(0, first - 1): first + 2][key + ["lots"]].reset_index(drop=True)],
                        axis=1, keys=["full", "zero"]).to_string())
    print(f"full-cost total: Rs {res.cost_paid.sum():,.0f} on Rs {tr['notional'].sum() / 1e7:.1f} cr traded "
          f"= {res.cost_paid.sum() / tr['notional'].sum() * 1e4:.2f} bps per side-notional")
    print("cost components (full-cost run):")
    print((tr[comp].sum()).round(0).to_string())

    # ------------------------------------------------------------------ 5
    hdr("5. P&L RECONSTRUCTION FROM THE TRADE LOG")
    for name, r_, t_ in [("full cost", res, tr), ("zero cost", zres, zt)]:
        realized = t_["realized_pnl"].sum()
        costs = t_["cost_total"].sum()
        # unrealized at the end: rebuild open positions from the log
        open_pos = {}
        for row in t_.itertuples(index=False):
            if row.action in ("OPEN", "ROLL_OPEN"):
                open_pos[row.symbol] = (row.lots, row.lot_size, row.price, row.expiry)
            else:
                open_pos.pop(row.symbol, None)
        last_i = market.date_index(r_.equity_curve.index[-1])
        unreal = 0.0
        for s, (lots, lot, px, expv) in open_pos.items():
            cd = market.contract(s, expv, last_i)
            unreal += lots * lot * (cd.settle - px)
        recon = a.capital + realized - costs + unreal
        print(f"{name:9s}: equity_end {r_.equity_curve.iloc[-1]:,.2f}   reconstructed "
              f"{recon:,.2f}   diff {r_.equity_curve.iloc[-1] - recon:+.6f}   "
              f"(realized {realized:,.0f}, costs {costs:,.0f}, unrealized {unreal:,.0f}, "
              f"open positions {len(open_pos)})")

    # ------------------------------------------------------------------ 6
    hdr("6. EXPOSURE, MARGIN, SIGNAL COVERAGE")
    print(ex[["n_positions", "n_long", "n_short", "gross_target", "gross_entry",
              "gross_marked", "net_marked", "margin_util"]].describe().loc[["mean", "min", "max"]]
          .round(4).to_string())
    print(f"margin breaches (utilisation > 1): {int((ex['margin_util'] > 1).sum())}")
    ss = res.signal_stats
    print("\nsignal coverage per bar:")
    print(ss[["n_universe", "n_scored", "n_complete_window", "n_held", "n_signal",
              "n_exits", "n_entries"]].describe().loc[["mean", "min", "max"]].round(1).to_string())
    # ------------------------------------------------------------------ 7
    hdr("7. FORCED EXIT AND LOT-CHANGE ROLL, ON REAL CONTRACTS")
    from fut_execute import ExecConfig, execute_day
    from fut_ledger import FuturesPosition, Ledger
    from harness_decide import Decisions
    nothing = Decisions(exits=[], entries=[])

    def synthetic(sym, expiry, entry_date, lots, days):
        i0 = market.date_index(entry_date)
        cd = market.contract(sym, pd.Timestamp(expiry), i0)
        led = Ledger(a.capital)
        led.open(FuturesPosition(sym, pd.Timestamp(expiry), lots, cd.lot_size, cd.open, i0,
                                 market.dates[i0], 0.0, 0.05, lots * cd.lot_size * cd.open / a.capital,
                                 cd.open, pd.Timestamp(expiry), cd.open), 0.0)
        rows = []
        for i in range(i0 + 1, i0 + 1 + days):
            rep = execute_day(nothing, led, market, i, a.capital, i0, CostConfig(), ExecConfig())
            rows += rep.fills
        return led, pd.DataFrame(rows)

    led, f = synthetic("CHENNPETRO", "2019-06-27", "2019-06-20", 2, 8)   # last contract, no successor
    print("CHENNPETRO, last F&O contract (expiry 2019-06-27), 2 lots bought 2019-06-20:")
    print(f[["date", "expiry", "action", "lots", "price", "fill_basis", "reason"]].to_string(index=False))
    i_fc = market.date_index(f.iloc[0]["date"])
    print(f"  -> force-closed with dte = {market.dte(pd.Timestamp('2019-06-27'), i_fc)}, "
          f"position left: {len(led)}, later fills: {len(f) - 1}")
    assert len(f) == 1 and f.iloc[0]["action"] == "FORCE_CLOSE" and len(led) == 0

    led, f = synthetic("DLF", "2019-06-27", "2019-06-20", 4, 4)          # lot 2600 -> 2800 at the roll
    print("\nDLF, 4 lots of the June-2019 contract bought 2019-06-20 (lot 2600; July lot 2800):")
    print(f[["date", "expiry", "action", "lots", "lot_size", "price", "notional", "reason"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    p = led.get("DLF")
    print(f"  -> ledger now: expiry {p.expiry.date()}, {p.lots} lots x {p.lot_size:.0f}, entry {p.entry_price}, "
          f"realized so far Rs {p.realized_pnl:,.0f}, rolls {p.n_rolls}; notional kept: "
          f"{p.lots * p.lot_size * p.entry_price / f.iloc[0]['notional']:.3f} of the pre-roll notional")
    assert p.n_rolls == 1 and p.lot_size == 2800 and p.lots == 3

    ec = res.equity_curve
    print(f"\n(dry-run P&L, NOT a result) equity {ec.iloc[0]:,.0f} -> {ec.iloc[-1]:,.0f}  "
          f"({ec.iloc[-1] / ec.iloc[0] - 1:+.2%}); zero-cost {zres.equity_curve.iloc[-1] / a.capital - 1:+.2%}")
    print(RULE)


if __name__ == "__main__":
    main()
