"""
fut_ledger.py — Stage 2: the position ledger, in whole lots on real contracts.

The cash harness kept a Book of TARGET WEIGHTS and converted to shares at fill
time. A futures position cannot be a weight: it is an integer number of lots of
one specific contract, and it changes contract every month. So the ledger holds

    FuturesPosition(symbol, expiry, lots, lot_size, entry_price, ...)

and exposes exactly the interface harness_decide.decide() needs from a Book —
`sym in ledger`, `ledger.positions()`, `.symbol`, `.weight`, `.entry_bar` —
so the decision logic from strategy/harness_decide.py runs UNCHANGED on top of
it. `weight` is the ACHIEVED weight at entry (after lot flooring), so the
book-level gross cap in decide() sees what is really on, not what was wanted.
`target_weight` is kept beside it so the lot-rounding drag can be reported.

ACCOUNTING (mirrors the cash harness fix: realized P&L persists in cash)
  equity = cash + sum over open positions of lots x lot_size x (mark - entry)
  close  : cash += lots x lot_size x (fill - entry) - cost
  roll   : the old contract is closed at its fill (P&L realized into cash) and
           the new one opened at ITS fill with entry_price = that fill. The
           position keeps its entry_bar (the time-stop counts from the original
           entry), its s-score at entry and its weights; it changes expiry,
           lot count if the lot size changed, and entry price. That is what
           "carry across the roll with P&L continuity" means in a ledger: no
           P&L is lost at the roll, and none is invented — the basis between
           the two contracts is paid for real at the two fills.
  mark   : at the exchange settlement price of the held contract.

MARGIN
  A SPAN proxy: margin_rate x gross notional at the current mark. Reported as
  utilisation = margin / equity every bar; a breach (utilisation > 1) is
  flagged, never acted on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FuturesPosition:
    symbol: str
    expiry: pd.Timestamp
    lots: int                    # signed: + long, - short
    lot_size: float
    entry_price: float           # of the CURRENT contract (reset at each roll)
    entry_bar: int               # bar of the ORIGINAL entry; time-stop counts from here
    entry_date: pd.Timestamp
    entry_s_score: float
    target_weight: float         # what decide() asked for (signed)
    weight: float                # what lot flooring achieved at entry (signed)
    mark_price: float
    orig_expiry: pd.Timestamp
    orig_entry_price: float
    realized_pnl: float = 0.0    # Rs, accumulated across rolls, before costs
    cost_paid: float = 0.0       # Rs, entry + rolls (exit cost is added at close)
    n_rolls: int = 0

    @property
    def side(self) -> str:
        return "LONG" if self.lots > 0 else "SHORT"

    def notional(self, price: float | None = None) -> float:
        p = self.mark_price if price is None else price
        return abs(self.lots) * self.lot_size * p

    def unrealized(self) -> float:
        return self.lots * self.lot_size * (self.mark_price - self.entry_price)


class Ledger:
    """Book-compatible collection of futures positions plus the cash account."""

    def __init__(self, initial_cash: float) -> None:
        self.cash = float(initial_cash)
        self._pos: dict[str, FuturesPosition] = {}

    # ---- Book interface used by harness_decide ------------------------------ #
    def __contains__(self, symbol: str) -> bool:
        return symbol in self._pos

    def __len__(self) -> int:
        return len(self._pos)

    def get(self, symbol: str) -> FuturesPosition | None:
        return self._pos.get(symbol)

    @property
    def symbols(self) -> list[str]:
        return list(self._pos.keys())

    def positions(self) -> list[FuturesPosition]:
        return list(self._pos.values())

    # ---- mutations ----------------------------------------------------------- #
    def open(self, pos: FuturesPosition, cost: float) -> None:
        if pos.symbol in self._pos:
            raise ValueError(f"already holding {pos.symbol}")
        if pos.lots == 0:
            raise ValueError(f"refusing to open {pos.symbol} with zero lots")
        pos.cost_paid += cost
        self.cash -= cost
        self._pos[pos.symbol] = pos

    def close(self, symbol: str, fill_price: float, cost: float) -> tuple[FuturesPosition, float]:
        """Close at `fill_price`. Returns (position, realized P&L of this leg)."""
        pos = self._pos.pop(symbol)
        realized = pos.lots * pos.lot_size * (fill_price - pos.entry_price)
        pos.realized_pnl += realized
        pos.cost_paid += cost
        pos.mark_price = fill_price
        self.cash += realized - cost
        return pos, realized

    def roll(self, symbol: str, old_fill: float, cost_close: float,
             new_expiry: pd.Timestamp, new_fill: float, new_lot_size: float,
             new_lots: int, cost_open: float) -> float:
        """Move a position to the next contract. Returns realized P&L of the leg closed."""
        pos = self._pos[symbol]
        realized = pos.lots * pos.lot_size * (old_fill - pos.entry_price)
        self.cash += realized - cost_close - cost_open
        pos.realized_pnl += realized
        pos.cost_paid += cost_close + cost_open
        pos.expiry = pd.Timestamp(new_expiry)
        pos.lots = int(new_lots)
        pos.lot_size = float(new_lot_size)
        pos.entry_price = float(new_fill)
        pos.mark_price = float(new_fill)
        pos.n_rolls += 1
        return realized

    def adjust(self, symbol: str, factor: float, relot: bool,
               published_lot: float | None = None) -> tuple[float, float, float]:
        """Apply a corporate-action adjustment to a HELD position at the start of
        the ex-date, exactly as the exchange does to a holder's reference price,
        so that unrealized P&L is continuous through the event.
          re-denominating event (bonus / split / rights): price x f, lot / f
              (the published lot is used when the data carries one), lots unchanged
          price-only event (dividend / demerger): reference price reduced by the
              rupee amount D = mark x (1 - f); lot and lots unchanged.
        Returns (old_entry, new_entry, new_lot_size)."""
        pos = self._pos[symbol]
        old_entry, before = pos.entry_price, pos.unrealized()
        if relot:
            new_lot = float(published_lot) if published_lot else pos.lot_size / factor
            pos.entry_price *= factor
            pos.mark_price *= factor
            pos.lot_size = new_lot
        else:
            d = pos.mark_price * (1.0 - factor)
            pos.entry_price -= d
            pos.mark_price -= d
        assert abs(pos.unrealized() - before) < 1e-6 * max(1.0, abs(before)) + 1e-3, \
            (symbol, before, pos.unrealized())
        return old_entry, pos.entry_price, pos.lot_size

    # ---- marking / diagnostics ---------------------------------------------- #
    def mark(self, price_of) -> float:
        """price_of(pos) -> settle price or None (keep last mark). Returns unrealized."""
        upnl = 0.0
        for pos in self._pos.values():
            p = price_of(pos)
            if p is not None and p > 0:
                pos.mark_price = float(p)
            upnl += pos.unrealized()
        return upnl

    def unrealized(self) -> float:
        return sum(p.unrealized() for p in self._pos.values())

    def equity(self) -> float:
        return self.cash + self.unrealized()

    def gross_notional(self) -> float:
        return sum(p.notional() for p in self._pos.values())

    def net_notional(self) -> float:
        return sum(p.lots * p.lot_size * p.mark_price for p in self._pos.values())

    def margin_required(self, rate: float) -> float:
        return rate * self.gross_notional()

    def summary(self, equity: float | None = None, margin_rate: float = 0.25) -> dict:
        eq = self.equity() if equity is None else equity
        ps = self._pos.values()
        g = self.gross_notional()
        return {
            "n_positions": len(self._pos),
            "n_long": sum(1 for p in ps if p.lots > 0),
            "n_short": sum(1 for p in ps if p.lots < 0),
            "gross_target": round(sum(abs(p.target_weight) for p in ps), 4),
            "gross_entry": round(sum(abs(p.weight) for p in ps), 4),
            "gross_marked": round(g / eq, 4) if eq > 0 else float("nan"),
            "net_marked": round(self.net_notional() / eq, 4) if eq > 0 else float("nan"),
            "margin_required": round(rate_margin(g, margin_rate), 2),
            "margin_util": round(rate_margin(g, margin_rate) / eq, 4) if eq > 0 else float("nan"),
        }


def rate_margin(gross_notional: float, rate: float) -> float:
    return rate * gross_notional
