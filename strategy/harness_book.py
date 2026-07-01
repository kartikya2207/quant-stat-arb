"""
harness/book.py — Piece 1 of the B1 cross-sectional backtest harness.

THE POSITIONS BOOK
------------------
Tracks what the strategy currently holds, as TARGET WEIGHTS (signed fractions of
capital). Long = positive weight, short = negative weight. Dollar-neutral means
the longs and shorts roughly cancel (sum of weights ~ 0).

WHY WEIGHTS (read GUIDELINES.md §6 decision):
- "equal risk, vol-scaled, longs ~= shorts" all express naturally as weights.
- dollar-neutrality is just "sum of weights ~ 0"; gross exposure is "sum of |weights|".
- conversion to actual share counts happens later, at EXECUTION, using the price.
  This keeps the strategy logic (weights) separate from fill mechanics (shares).

This Piece does NOT decide trades or compute P&L — it is only the record of
positions and the helpers to reason about them. Later pieces (decision, execution,
loop) use it.

A "position" here also remembers the context needed by the EXIT logic later:
the s-score at entry, the bar index when opened (for the time-stop), and the side.
That is what makes the orphaned-position rule (GUIDELINES.md §5) implementable —
the book knows everything it holds, independent of today's universe.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    """One open position. Weight is signed: + long, - short."""
    symbol: str
    weight: float                 # signed target weight (fraction of capital)
    entry_bar: int                # bar index when opened (for time-stop)
    entry_s_score: float          # s-score at entry (context / diagnostics)
    entry_price: float            # fill price at entry (for P&L + return calc)

    @property
    def side(self) -> str:
        return "LONG" if self.weight > 0 else "SHORT"


class Book:
    """The collection of open positions. Pure bookkeeping."""

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    # ---- basic queries ---------------------------------------------------- #
    def __contains__(self, symbol: str) -> bool:
        return symbol in self._positions

    def __len__(self) -> int:
        return len(self._positions)

    def get(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    @property
    def symbols(self) -> list[str]:
        """Everything currently held — the set the EXIT logic must score
        (the 'holdings' half of universe ∪ holdings, GUIDELINES.md §5)."""
        return list(self._positions.keys())

    def positions(self) -> list[Position]:
        return list(self._positions.values())

    # ---- mutations -------------------------------------------------------- #
    def open(self, symbol: str, weight: float, bar: int,
             s_score: float, price: float) -> None:
        if symbol in self._positions:
            raise ValueError(f"already holding {symbol}; close before reopening")
        if weight == 0:
            raise ValueError(f"refusing to open {symbol} with zero weight")
        self._positions[symbol] = Position(
            symbol=symbol, weight=weight, entry_bar=bar,
            entry_s_score=s_score, entry_price=price,
        )

    def close(self, symbol: str) -> Position:
        if symbol not in self._positions:
            raise KeyError(f"not holding {symbol}; cannot close")
        return self._positions.pop(symbol)

    def update_weight(self, symbol: str, new_weight: float) -> None:
        """Resize an existing position (used when rebalancing toward a new target)."""
        if symbol not in self._positions:
            raise KeyError(f"not holding {symbol}; cannot resize")
        if new_weight == 0:
            raise ValueError("use close() to flatten, not update_weight(0)")
        self._positions[symbol].weight = new_weight

    # ---- exposure diagnostics -------------------------------------------- #
    def gross_exposure(self) -> float:
        """Sum of |weights| — total capital at work (long + short). Caps live here
        in later risk logic (e.g. <= 2.0 means 2x gross)."""
        return sum(abs(p.weight) for p in self._positions.values())

    def net_exposure(self) -> float:
        """Sum of signed weights — directional tilt. Dollar-neutral targets ~0."""
        return sum(p.weight for p in self._positions.values())

    def long_exposure(self) -> float:
        return sum(p.weight for p in self._positions.values() if p.weight > 0)

    def short_exposure(self) -> float:
        return sum(p.weight for p in self._positions.values() if p.weight < 0)

    def n_long(self) -> int:
        return sum(1 for p in self._positions.values() if p.weight > 0)

    def n_short(self) -> int:
        return sum(1 for p in self._positions.values() if p.weight < 0)

    def summary(self) -> dict:
        return {
            "n_positions": len(self),
            "n_long": self.n_long(),
            "n_short": self.n_short(),
            "gross": round(self.gross_exposure(), 4),
            "net": round(self.net_exposure(), 4),
            "long_exp": round(self.long_exposure(), 4),
            "short_exp": round(self.short_exposure(), 4),
        }
