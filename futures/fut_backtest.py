"""
fut_backtest.py — Stage 2: the daily loop for the futures version.

The strategy is the cash strategy, unchanged, on Stage 1 data:

  SIGNAL   strategy/signal.py, loaded as-is. 60-day window of the continuous
           front-month returns, correlation PCA via SVD, 5 factors, OU fit with
           the 0<b<1 gate, s-score. PCA is estimated on the names with a
           complete 60-day window (compute_pca_factors drops any column with a
           NaN); every scored name — including partial histories with >=30
           observations and the held names outside today's universe — is then
           projected onto those frozen factors, exactly as the cash harness
           does. A masked name-day is a NaN in the window and simply drops out
           of that name's regression and OU fit.
  DECIDE   strategy/harness_decide.py, unchanged: +-2 entry, triple barrier
           (profit |s|<=0.5, stop |s|>=3.0, 15-bar time stop), OU-speed gate,
           min(longs, shorts) pairing, inverse-vol sizing, 5% per-name cap,
           gross cap 1.0. The ledger presents the Book interface it expects.
  UNIVERSE top 120 by open-interest notional at T's close, among names listed
           on T, union the names currently held (orphan rule: a held name is
           always scored so it can always exit).

Only EXECUTION differs from cash (fut_execute.py): fills on the real front
contract at T+1 open, whole lots floored, monthly rolls as two charged trades,
futures cost schedule, SPAN-proxy margin.

LOOK-AHEAD DISCIPLINE, bar by bar
  close of T   : equity marked at T's settle; universe from T's OI; signal from
                 returns up to and including T; decisions. Sizing base = equity
                 at T's close.
  open of T+1  : fills at T+1 opens on front(T+1). front(T+1) depends only on
                 expiry dates and the exchange calendar.
  close of T+1 : marked at T+1's settle (the next iteration's first step).
  Nothing from T+1 reaches the signal or the sizing; the continuous series is
  forward-anchored ratio chaining (Stage 1), so no price at T changes later.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STRATEGY = HERE.parent / "strategy"


def _load_strategy_signal():
    """strategy/signal.py shadows the stdlib `signal` module when imported by
    name, so load it from its path under a private name."""
    spec = importlib.util.spec_from_file_location("qsa_signal", STRATEGY / "signal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qsa_signal"] = mod        # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


sig = _load_strategy_signal()
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))
from harness_decide import DecideConfig, decide          # noqa: E402  (strategy/, unchanged)

from fut_costs import CostConfig, assert_zero            # noqa: E402
from fut_execute import ExecConfig, execute_day          # noqa: E402
from fut_ledger import Ledger                            # noqa: E402
from fut_market import FuturesMarket                     # noqa: E402


@dataclass
class FuturesBacktestConfig:
    initial_capital: float = 100_000_000.0   # Rs 10 crore: one lot is Rs 3-8 lakh,
                                             # the 5% cap must hold several lots
    pca_window: int = 60
    n_factors: int = 5
    universe_size: int = 120
    margin_rate: float = 0.25
    start: str | None = None                 # first DECISION date (inclusive)
    end: str | None = None                   # last date marked (inclusive)
    decide: DecideConfig = field(default_factory=DecideConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    verbose: bool = True


@dataclass
class FuturesBacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    skips: pd.DataFrame
    exposure: pd.DataFrame
    cost_paid: pd.Series
    config: FuturesBacktestConfig
    signal_stats: pd.DataFrame


def _universe(market: FuturesMarket, t: int, size: int) -> list[str]:
    """Top-`size` names by OI notional at session t among names listed on t."""
    oi = market.oi_wide.iloc[t].dropna()
    oi = oi[oi > 0]
    return oi.sort_values(ascending=False).head(size).index.tolist()


def run_backtest(market: FuturesMarket, cfg: FuturesBacktestConfig) -> FuturesBacktestResult:
    if cfg.cost.is_zero:
        assert_zero(cfg.cost)
    dec_cfg = cfg.decide
    assert abs(dec_cfg.max_weight_per_name - cfg.exec.max_weight_per_name) < 1e-12
    assert abs(dec_cfg.target_gross - cfg.exec.target_gross) < 1e-12

    n = len(market)
    t0 = market.date_index(cfg.start) if cfg.start else 0
    t0 = max(t0, cfg.pca_window)                 # need a full window ending at t
    t_last = market.date_index(cfg.end) if cfg.end else n - 1
    t_last = min(t_last, n - 1)

    ledger = Ledger(cfg.initial_capital)
    rets = market.ret_wide
    eq_hist, dates_hist, expo_hist, cost_hist, sig_hist = [], [], [], [], []
    fills, skips = [], []
    wall = time.time()

    def settle_of(pos):
        cd = market.contract(pos.symbol, pos.expiry, cur_t)
        return None if cd is None else cd.settle

    for t in range(t0, t_last):                  # t+1 must exist for execution
        cur_t = t
        # ---- close of T: mark ------------------------------------------------ #
        ledger.mark(settle_of)
        equity = ledger.equity()
        eq_hist.append(equity); dates_hist.append(market.dates[t])
        expo_hist.append(ledger.summary(equity, cfg.margin_rate))

        # ---- close of T: universe, signal, decide ---------------------------- #
        universe = _universe(market, t, cfg.universe_size)
        score_syms = list(dict.fromkeys(universe + ledger.symbols))
        win = rets.iloc[t - cfg.pca_window + 1: t + 1].reindex(columns=score_syms)
        win = win.dropna(axis=1, how="all")
        n_complete = int(win.notna().all(axis=0).sum())
        stats = {"date": market.dates[t], "n_universe": len(universe),
                 "n_scored": win.shape[1], "n_complete_window": n_complete,
                 "n_held": len(ledger), "n_signal": 0, "n_exits": 0, "n_entries": 0}
        if n_complete <= cfg.n_factors + 1:
            cost_hist.append(0.0); sig_hist.append(stats); continue
        res = sig.compute_signal(win, n_factors=cfg.n_factors, score_symbols=list(win.columns))
        s_scores, ou_speed = res.s_score, res.ou_speed
        vols = win.std(axis=0)
        tradable = {u for u in universe
                    if (f := market.front(t, u)) is not None and f.tradable}
        decisions = decide(ledger, s_scores, ou_speed, vols, current_bar=t,
                           cfg=dec_cfg, tradable=tradable)
        stats.update(n_signal=int(s_scores.notna().sum()), n_exits=len(decisions.exits),
                     n_entries=len(decisions.entries))
        sig_hist.append(stats)

        # ---- open of T+1: execute ------------------------------------------- #
        rep = execute_day(decisions, ledger, market, t + 1, equity, t, cfg.cost, cfg.exec)
        fills.extend(rep.fills); skips.extend(rep.skips)
        cost_hist.append(rep.cost_total)

        if cfg.verbose and (t - t0) % 250 == 0:
            print(f"  [{market.dates[t].date()}] equity {equity:,.0f}  pos {len(ledger)}  "
                  f"fills {len(fills)}  {time.time() - wall:.0f}s", flush=True)

    # final mark at t_last close (positions left open are marked, not closed)
    cur_t = t_last
    ledger.mark(settle_of)
    equity = ledger.equity()
    eq_hist.append(equity); dates_hist.append(market.dates[t_last])
    expo_hist.append(ledger.summary(equity, cfg.margin_rate)); cost_hist.append(0.0)

    idx = pd.DatetimeIndex(dates_hist)
    equity_curve = pd.Series(eq_hist, index=idx, name="equity")
    return FuturesBacktestResult(
        equity_curve=equity_curve,
        daily_returns=equity_curve.pct_change().fillna(0.0),
        trades=pd.DataFrame(fills),
        skips=pd.DataFrame(skips),
        exposure=pd.DataFrame(expo_hist, index=idx),
        cost_paid=pd.Series(cost_hist, index=idx, name="cost"),
        config=cfg,
        signal_stats=pd.DataFrame(sig_hist),
    )
