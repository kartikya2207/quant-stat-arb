"""
harness/backtest.py — Piece 4 of the B1 harness: THE MAIN LOOP.

Ties everything together into a running cross-sectional backtest. Walks through
time one bar at a time. The look-ahead-safe rhythm (GUIDELINES.md §1.4) is:

    at the close of bar T:
        - compute signal from returns up to and INCLUDING bar T's close
        - decide exits/entries
    at the open of bar T+1:
        - execute those decisions (fills at T+1 open)
    at the close of bar T+1:
        - mark every open position to market -> update equity

So a decision made on T's close is acted on at T+1's open and first marked at
T+1's close. We NEVER use information from a bar to trade within that same bar.

Marking (the mark-to-market decision):
    Each bar, portfolio equity = cash + sum of position values, where a position's
    value moves with price since entry. We track equity as a daily series so
    Sharpe / drawdown are meaningful.

Universe / orphan handling (§5):
    Each bar we score (today's liquid universe) ∪ (current holdings). The liquid
    universe is the top-N by recent average turnover among names with data. Held
    names always get scored so they can always exit (orphan rule).

This module is the orchestrator. It calls signal.py and Pieces 1-3. It does not
itself reimplement any of their logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import signal as sig
from harness_book import Book
from harness_decide import DecideConfig, decide
from harness_execute import CostConfig, ExecResult, make_cost_fn, execute


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    pca_window: int = 60               # trading days for the rolling PCA (slow clock)
    n_factors: int = 5
    universe_size: int = 250           # top-N liquid names scored each bar
    liquidity_lookback: int = 20       # days of turnover to rank liquidity
    rebalance_every: int = 1           # run decisions every N bars (1 = daily)
    decide: DecideConfig = field(default_factory=DecideConfig)
    cost: CostConfig = field(default_factory=CostConfig)


@dataclass
class BacktestResult:
    equity_curve: pd.Series            # index=date -> portfolio equity
    daily_returns: pd.Series           # equity.pct_change()
    trades: pd.DataFrame               # one row per fill
    exposure: pd.DataFrame             # daily gross/net/n_positions
    cost_paid: pd.Series               # daily ₹ cost
    config: BacktestConfig


def _liquid_universe(volume_wide: pd.DataFrame, prices_wide: pd.DataFrame,
                     upto_idx: int, lookback: int, size: int) -> list:
    """Top-`size` symbols by average rupee turnover over the trailing `lookback`
    bars, among names that have valid (non-NaN, positive) price at `upto_idx`."""
    lo = max(0, upto_idx - lookback + 1)
    vol_win = volume_wide.iloc[lo:upto_idx + 1]
    px_win = prices_wide.iloc[lo:upto_idx + 1]
    turnover = (vol_win * px_win).mean(axis=0)          # avg ₹ turnover per name
    # require a valid current price
    cur_px = prices_wide.iloc[upto_idx]
    valid = cur_px.notna() & (cur_px > 0)
    turnover = turnover[valid].dropna()
    return turnover.sort_values(ascending=False).head(size).index.tolist()


def run_backtest(
    prices_wide: pd.DataFrame,          # index=date, cols=symbols (ADJUSTED close)
    volume_wide: pd.DataFrame,          # same shape, volumes (for liquidity + turnover)
    open_wide: pd.DataFrame | None = None,  # next-open prices; if None, use close (approx)
    cfg: BacktestConfig | None = None,
    statutory_cost_fn: Callable[[int, float, bool], float] | None = None,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    cost_fn = make_cost_fn(statutory_cost_fn, cfg.cost)

    # align everything on the same dates/cols
    prices_wide = prices_wide.sort_index()
    volume_wide = volume_wide.reindex_like(prices_wide)
    if open_wide is None:
        # fallback: approximate next-open with next-close-shifted. NOTE: using the
        # actual open is better; pass open_wide when available.
        open_wide = prices_wide
    else:
        open_wide = open_wide.reindex_like(prices_wide)

    returns_wide = prices_wide.pct_change(fill_method=None)
    dates = prices_wide.index
    n = len(dates)

    book = Book()
    cash = cfg.initial_capital
    equity_hist, exposure_hist, cost_hist, eq_dates = [], [], [], []
    all_fills: list = []

    # --- ACCOUNTING MODEL ---------------------------------------------------
    # cash is the single source of truth for realized money. While a position is
    # open, equity = cash + marked unrealized P&L of open positions. When a
    # position closes, its realized P&L (exit vs entry, on its share basis) is
    # credited into cash so the gain/loss PERSISTS (this is the fix for the
    # telescoping bug). Each position's share basis is fixed at OPEN and stored,
    # so marking and realizing use the SAME shares (no inconsistency).
    shares_held: dict[str, float] = {}     # symbol -> signed share count fixed at entry

    def mark_unrealized(mark_idx: int) -> float:
        """Sum of unrealized P&L of currently-open positions at bar close."""
        cur_px = prices_wide.iloc[mark_idx]
        upnl = 0.0
        for pos in book.positions():
            p_now = cur_px.get(pos.symbol, np.nan)
            sh = shares_held.get(pos.symbol, 0.0)
            if pd.isna(p_now) or sh == 0.0:
                continue
            upnl += sh * (p_now - pos.entry_price)
        return upnl

    def portfolio_equity(mark_idx: int) -> float:
        return cash + mark_unrealized(mark_idx)

    equity_at_entry: dict[str, float] = {}   # equity level when each position opened

    # we need at least pca_window+1 bars of returns before the first decision
    start = cfg.pca_window + 1

    last_equity = cfg.initial_capital
    for t in range(start, n - 1):            # stop at n-1 so t+1 (execution) exists
        # ---- mark equity at bar t close (before today's decisions execute) ----
        last_equity = portfolio_equity(t)

        # only act every `rebalance_every` bars
        if (t - start) % cfg.rebalance_every != 0:
            equity_hist.append(last_equity); eq_dates.append(dates[t])
            exposure_hist.append(book.summary()); cost_hist.append(0.0)
            continue

        # ---- SEE: signal from returns up to and including bar t ----
        window = returns_wide.iloc[t - cfg.pca_window + 1: t + 1]
        universe = _liquid_universe(volume_wide, prices_wide, t,
                                    cfg.liquidity_lookback, cfg.universe_size)
        # score universe ∪ holdings (orphan rule)
        score_syms = list(dict.fromkeys(universe + book.symbols))
        # window restricted to scored names (drop all-NaN cols)
        win = window.reindex(columns=score_syms).dropna(axis=1, how="all")
        if win.shape[1] <= cfg.n_factors + 1:
            # not enough names this bar; just mark and continue
            equity_hist.append(last_equity); eq_dates.append(dates[t])
            exposure_hist.append(book.summary()); cost_hist.append(0.0)
            continue

        try:
            res = sig.compute_signal(win, n_factors=cfg.n_factors,
                                     score_symbols=score_syms)
        except Exception:
            equity_hist.append(last_equity); eq_dates.append(dates[t])
            exposure_hist.append(book.summary()); cost_hist.append(0.0)
            continue

        s_scores = res.s_score
        ou_speed = res.ou_speed
        # per-name vol for inverse-vol sizing = residual std over the window
        vols = win.std(axis=0)
        # liquidity gate: only universe names are 'tradable' for ENTRY
        tradable = set(universe)

        # ---- DECIDE (on bar t's info) ----
        # capture entry prices of positions BEFORE execute mutates the book,
        # so we can realize their P&L when execute closes them.
        entry_px_before = {p.symbol: p.entry_price for p in book.positions()}

        decisions = decide(book, s_scores, ou_speed, vols,
                           current_bar=t, cfg=cfg.decide, tradable=tradable)

        # ---- ACT: execute at bar t+1's OPEN ----
        exec_open = open_wide.iloc[t + 1]
        # capital base for sizing = current equity (compounding)
        exec_res: ExecResult = execute(
            decisions, book, exec_open, capital=last_equity,
            current_bar=t, cost_fn=cost_fn, s_scores=s_scores,
        )
        # process fills: realize P&L on closes, register share basis on opens
        for f in exec_res.fills:
            if f.action == "OPEN":
                equity_at_entry[f.symbol] = last_equity
                shares_held[f.symbol] = float(f.shares)   # signed shares fixed at entry
            elif f.action == "CLOSE":
                # realize P&L into cash: shares * (exit_price - entry_price)
                sh = shares_held.pop(f.symbol, 0.0)
                entry_px = entry_px_before.get(f.symbol, f.price)
                realized = sh * (f.price - entry_px)
                cash += realized
                equity_at_entry.pop(f.symbol, None)
            f_row = {"date": dates[t + 1], "symbol": f.symbol, "action": f.action,
                     "shares": f.shares, "price": f.price, "cost": f.cost,
                     "reason": f.reason}
            all_fills.append(f_row)

        # costs reduce cash
        cash -= exec_res.total_cost

        # ---- record (mark at bar t close already computed) ----
        equity_hist.append(last_equity); eq_dates.append(dates[t])
        exposure_hist.append(book.summary()); cost_hist.append(exec_res.total_cost)

    equity_curve = pd.Series(equity_hist, index=pd.DatetimeIndex(eq_dates), name="equity")
    daily_returns = equity_curve.pct_change().fillna(0.0)
    trades = pd.DataFrame(all_fills)
    exposure = pd.DataFrame(exposure_hist, index=equity_curve.index)
    cost_paid = pd.Series(cost_hist, index=equity_curve.index, name="cost")

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        trades=trades,
        exposure=exposure,
        cost_paid=cost_paid,
        config=cfg,
    )
