"""
harness/execute.py — Piece 3 of the B1 harness.

Turns Decisions (Piece 2) into actual fills and updated book state, and returns
the cash cost of trading. Two things happen here that did not happen before:

  1. WEIGHTS -> SHARES at a real price (the next bar's OPEN — look-ahead-safe).
  2. COSTS are charged: delivery STT + statutory charges (from the framework's
     cost model) PLUS a flat slippage buffer (spread + impact).

COST MODEL (read GUIDELINES.md §1.9 and the spread/impact decision):
  - Statutory + STT: reuse tidetrader's costs.py (the `nse_equity_delivery` preset)
    — exact Indian charges with correct buy/sell asymmetry. We call it as a
    (quantity, price)->₹ callable. A faithful standalone fallback is included so
    this file is testable in isolation; on Kartikya's machine, pass the real one.
  - Slippage buffer: a FLAT per-side haircut (spread + impact), slightly
    conservative so real costs are <= backtest costs (real perf >= test perf).
    Default 0.10% per side (≈0.05% spread + 0.05% impact).

This module is intentionally mechanical. It does NOT decide trades (Piece 2) and
does NOT compute P&L over time (Piece 4) — it only executes one bar's decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from harness_book import Book
from harness_decide import Decisions


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
# Faithful standalone delivery-cost fallback (used for testing this file alone).
# Mirrors the structure of NSE equity DELIVERY charges. On the real machine,
# replace with tidetrader's costs.make_commission_func("nse_equity_delivery").
def _fallback_delivery_cost(quantity: int, price: float, is_buy: bool) -> float:
    """Approximate NSE delivery charges for one side. ₹ for `quantity` shares."""
    turnover = abs(quantity) * price
    # STT: 0.1% on BOTH buy and sell for delivery
    stt = 0.001 * turnover
    # exchange transaction charge (~0.00297% NSE) + SEBI (~0.0001%)
    exch = 0.0000297 * turnover
    sebi = 0.000001 * turnover
    # stamp duty: 0.015% on BUY only
    stamp = 0.00015 * turnover if is_buy else 0.0
    # brokerage: assume discount-broker delivery = 0 (many are free on delivery)
    brokerage = 0.0
    # GST 18% on (brokerage + exchange txn)
    gst = 0.18 * (brokerage + exch)
    return stt + exch + sebi + stamp + brokerage + gst


@dataclass
class CostConfig:
    slippage_per_side: float = 0.0010      # flat spread+impact buffer, per side (0.10%)


def make_cost_fn(
    statutory_fn: Callable[[int, float, bool], float] | None = None,
    cfg: CostConfig | None = None,
) -> Callable[[int, float, bool], float]:
    """Return a (quantity, price, is_buy) -> ₹total_cost callable.

    statutory_fn : the real framework cost callable. If None, uses the faithful
                   standalone fallback above. The slippage buffer is ALWAYS added
                   on top.
    """
    cfg = cfg or CostConfig()
    stat = statutory_fn or _fallback_delivery_cost

    def cost_fn(quantity: int, price: float, is_buy: bool) -> float:
        if quantity == 0:
            return 0.0
        statutory = stat(abs(quantity), price, is_buy)
        slippage = cfg.slippage_per_side * abs(quantity) * price
        return statutory + slippage

    return cost_fn


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
@dataclass
class Fill:
    symbol: str
    action: str            # 'OPEN' | 'CLOSE' | 'RESIZE'
    shares: int            # signed: + buy, - sell (the trade, not the position)
    price: float
    cost: float            # ₹ cost of this fill
    reason: str = ""       # exit reason if a close


@dataclass
class ExecResult:
    fills: list            # list[Fill]
    total_cost: float      # ₹ summed over all fills this bar


def _shares_for_weight(weight: float, capital: float, price: float) -> int:
    """Convert a signed target weight to a signed whole-share count at `price`."""
    if price <= 0 or capital <= 0:
        return 0
    rupee_target = weight * capital
    return int(rupee_target / price)        # truncates toward zero; sign preserved


def execute(
    decisions: Decisions,
    book: Book,
    open_prices: pd.Series,     # next bar's OPEN price per symbol (the fill price)
    capital: float,             # current capital (for weight->shares)
    current_bar: int,
    cost_fn: Callable[[int, float, bool], float],
    s_scores: pd.Series | None = None,
) -> ExecResult:
    """Execute one bar's decisions at the given open prices. Mutates `book`.

    Order: exits first (free capital/positions), then entries.
    Any symbol without a valid open price is skipped (can't fill what has no price).
    """
    fills: list = []
    total_cost = 0.0

    # ---- EXITS ----
    for ex in decisions.exits:
        sym = ex.symbol
        pos = book.get(sym)
        if pos is None:
            continue
        px = open_prices.get(sym, np.nan)
        if pd.isna(px) or px <= 0:
            # no price to exit at this bar; leave the position, try again next bar
            continue
        # closing a position: trade direction is OPPOSITE the position side
        cur_shares = _shares_for_weight(pos.weight, capital, pos.entry_price)
        trade_shares = -cur_shares           # buy back shorts / sell longs
        is_buy = trade_shares > 0
        c = cost_fn(abs(trade_shares), px, is_buy)
        total_cost += c
        fills.append(Fill(sym, "CLOSE", trade_shares, float(px), c, ex.reason))
        book.close(sym)

    # ---- ENTRIES ----
    for en in decisions.entries:
        sym = en.symbol
        if sym in book:                      # safety: never double-open
            continue
        px = open_prices.get(sym, np.nan)
        if pd.isna(px) or px <= 0:
            continue
        shares = _shares_for_weight(en.weight, capital, px)
        if shares == 0:                      # weight too small to buy 1 share
            continue
        is_buy = shares > 0
        c = cost_fn(abs(shares), px, is_buy)
        total_cost += c
        fills.append(Fill(sym, "OPEN", shares, float(px), c, ""))
        book.open(sym, weight=en.weight, bar=current_bar,
                  s_score=en.s_score, price=float(px))

    return ExecResult(fills=fills, total_cost=total_cost)
