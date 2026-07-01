"""
harness/decide.py — Piece 2 of the B1 harness.

Turns one bar's s-scores into INTENDED TRADES, given the current book.
Two jobs, in order (exits before entries — frees capital/risk budget first):

  1. EXITS  : for every HELD position, check triple-barrier (profit/stop/time).
  2. ENTRIES: for every UNHELD universe name that passes the gates, propose a new
              position; then balance long vs short and size by inverse-vol.

This module DECIDES only. It returns plain instructions; Piece 3 executes them
(fills at next open, applies costs). No prices are touched here except the entry
s-score thresholds — sizing uses volatility passed in.

Implements:
  - entry threshold ±2.0, OU-speed gate, (liquidity gate hook)            (§1.4)
  - triple barrier: profit |s|<=0.5, stop |s|>=3.0, time>=max_hold_bars   (§1.5)
  - inverse-vol sizing, dollar-neutral, per-name cap                       (§1.6)
  - orphaned positions: exits are checked on the BOOK, not the universe    (§5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from harness_book import Book


# ---- instruction objects ------------------------------------------------- #
@dataclass
class ExitOrder:
    symbol: str
    reason: str            # 'profit' | 'stop' | 'time'


@dataclass
class EntryOrder:
    symbol: str
    weight: float          # signed target weight (+long / -short)
    s_score: float


@dataclass
class Decisions:
    exits: list            # list[ExitOrder]
    entries: list          # list[EntryOrder]


# ---- config (defaults match GUIDELINES) ---------------------------------- #
@dataclass
class DecideConfig:
    entry_threshold: float = 2.0      # |s| must exceed this to enter
    profit_barrier: float = 0.5       # |s| <= this -> take profit
    stop_barrier: float = 3.0         # |s| >= this -> cut loss
    max_hold_bars: int = 15           # time stop (in bars)
    min_ou_speed: float = 0.0         # OU kappa must exceed this to enter (>0 = reverts)
    max_weight_per_name: float = 0.05 # per-name cap (|weight|)
    target_gross: float = 1.0         # total gross exposure target (sum |w|); 1.0 = 100%
    max_positions: int = 40           # cap on number of positions


# ---- EXITS --------------------------------------------------------------- #
def decide_exits(book: Book, s_scores: pd.Series, current_bar: int,
                 cfg: DecideConfig) -> list:
    """Check triple-barrier on every HELD position.

    NOTE (orphan rule §5): we iterate over book.symbols, NOT the universe. A held
    name that dropped out of the universe is still scored (its s-score is passed
    in by the caller, who scores universe ∪ holdings) and thus can still exit.
    If a held name has no s-score available (NaN/missing), only the TIME stop can
    act on it — it can never get stuck.
    """
    exits: list = []
    for pos in book.positions():
        s = s_scores.get(pos.symbol, np.nan)
        held_bars = current_bar - pos.entry_bar

        # 1) time stop always works, even if s is missing
        if held_bars >= cfg.max_hold_bars:
            exits.append(ExitOrder(pos.symbol, "time"))
            continue

        if pd.isna(s):
            # no signal this bar; rely on time stop (handled above). hold for now.
            continue

        # 2) profit: residual reverted toward equilibrium
        if abs(s) <= cfg.profit_barrier:
            exits.append(ExitOrder(pos.symbol, "profit"))
            continue

        # 3) stop: residual diverged further (thesis wrong). HARD — always on.
        if abs(s) >= cfg.stop_barrier:
            exits.append(ExitOrder(pos.symbol, "stop"))
            continue

    return exits


# ---- ENTRIES ------------------------------------------------------------- #
def decide_entries(book: Book, s_scores: pd.Series, ou_speed: pd.Series,
                   vols: pd.Series, cfg: DecideConfig,
                   exiting: set | None = None,
                   tradable: set | None = None) -> list:
    """Propose new entries from universe names that pass the gates.

    Parameters
    ----------
    s_scores : s-score per symbol (the universe names live here).
    ou_speed : OU kappa per symbol (reversion-speed gate).
    vols     : per-name volatility (e.g. residual std or return std) for inverse-vol sizing.
    exiting  : symbols being exited THIS bar — don't re-enter them same bar.
    tradable : optional set of symbols passing the liquidity gate. If None, all pass.
               (Liquidity gate is a measurable fact; wire real data later — §1.4.)

    Returns a balanced, inverse-vol-sized, dollar-neutral list of EntryOrders.
    """
    exiting = exiting or set()

    # --- BOOK-LEVEL GROSS CAP -------------------------------------------------
    # max_positions caps NEW entries per bar, but total book gross can still
    # drift above target as positions accumulate across bars. Pin leverage to
    # target_gross by only deploying the gross budget left AFTER this bar's exits
    # free up room. Existing positions keep their weights (§6) — we just never
    # add beyond the cap. Sizing below uses `available_gross`, not target_gross.
    committed_gross = sum(abs(p.weight) for p in book.positions()
                          if p.symbol not in exiting)
    available_gross = cfg.target_gross - committed_gross
    if available_gross <= 1e-9:
        return []                       # already at/above the gross cap; add nothing

    # 1) candidates: universe names not held, not exiting, past threshold, that revert
    longs, shorts = [], []
    for sym, s in s_scores.items():
        if pd.isna(s):
            continue
        if sym in book or sym in exiting:
            continue
        if tradable is not None and sym not in tradable:
            continue                      # liquidity gate
        k = ou_speed.get(sym, np.nan)
        if pd.isna(k) or k <= cfg.min_ou_speed:
            continue                      # OU-speed gate: only trade what reverts
        if s <= -cfg.entry_threshold:
            longs.append(sym)             # too cheap -> long
        elif s >= cfg.entry_threshold:
            shorts.append(sym)            # too rich -> short

    if not longs and not shorts:
        return []

    # 2) balance count: market-neutral wants roughly equal long & short slots.
    #    take the most-extreme |s| on each side up to a shared cap.
    slots_each = cfg.max_positions // 2
    longs = sorted(longs, key=lambda x: s_scores[x])[:slots_each]            # most negative first
    shorts = sorted(shorts, key=lambda x: -s_scores[x])[:slots_each]        # most positive first
    n = min(len(longs), len(shorts))      # dollar-neutral: equal numbers each side
    longs, shorts = longs[:n], shorts[:n]
    if n == 0:
        return []

    # 3) inverse-vol sizing within each side, then scale to gross target,
    #    split equally long vs short so net ~ 0.
    chosen = longs + shorts
    inv_vol = {}
    for sym in chosen:
        v = vols.get(sym, np.nan)
        inv_vol[sym] = (1.0 / v) if (pd.notna(v) and v > 0) else 0.0

    # normalize inverse-vol weights WITHIN each side
    def side_weights(side_syms):
        tot = sum(inv_vol[s] for s in side_syms)
        if tot <= 0:
            # fallback: equal weight
            return {s: 1.0 / len(side_syms) for s in side_syms}
        return {s: inv_vol[s] / tot for s in side_syms}

    lw = side_weights(longs)
    sw = side_weights(shorts)

    # each side gets half the AVAILABLE gross budget (capped at target_gross
    # across the whole book, not just this bar's entries); longs +, shorts -
    half_gross = available_gross / 2.0
    entries: list = []
    for sym in longs:
        w = +half_gross * lw[sym]
        w = float(np.clip(w, -cfg.max_weight_per_name, cfg.max_weight_per_name))
        entries.append(EntryOrder(sym, w, float(s_scores[sym])))
    for sym in shorts:
        w = -half_gross * sw[sym]
        w = float(np.clip(w, -cfg.max_weight_per_name, cfg.max_weight_per_name))
        entries.append(EntryOrder(sym, w, float(s_scores[sym])))

    return entries


def decide(book: Book, s_scores: pd.Series, ou_speed: pd.Series,
           vols: pd.Series, current_bar: int, cfg: DecideConfig,
           tradable: set | None = None) -> Decisions:
    """Full daily decision: exits first, then entries on what's left."""
    exits = decide_exits(book, s_scores, current_bar, cfg)
    exiting = {e.symbol for e in exits}
    entries = decide_entries(book, s_scores, ou_speed, vols, cfg,
                             exiting=exiting, tradable=tradable)
    return Decisions(exits=exits, entries=entries)
