"""
fut_execute.py — Stage 2: turn one bar's Decisions into fills on real contracts.

Runs at the OPEN of session i (= T+1) on decisions made at the close of T.
Order of operations, and why:

  1. EXITS     close what decide() said to close, on the contract actually held,
               at that contract's open. Frees gross budget first.
  2. FORCED    for every remaining position, look at the held contract's
               trading days to expiry on session i:
                 dte > ROLL_BEFORE_TD  -> nothing to do
                 dte <= ROLL_BEFORE_TD -> the chain has moved on. If the name has
                     a tradable successor on session i (front(i) is a later
                     expiry), ROLL: close the near contract at its open, open the
                     next at ITS open — two real trades, both charged. If there is
                     no successor (an F&O delisting) or the chain is untradable,
                     FORCE-CLOSE. A contract is never held into its last two
                     sessions and never traded after expiry.
  3. ENTRIES   on front(i) — the chain's contract for session i, which is the
               near month until the roll date and the next month from the roll
               date on; the next month is never opened early.

LOT SIZING (floor, never round up)
  one_lot   = lot_size x open
  cap       = max_weight_per_name x equity
  budget    = (target_gross - gross already on the book) x equity, decremented
              as this bar's entries fill
  target    = |weight| x equity
  If one lot exceeds the cap, the remaining budget, or the target, the name is
  SKIPPED and logged with the reason. Otherwise lots = floor(target / one_lot).
  achieved weight = lots x one_lot / equity is what the ledger records; the
  target is kept beside it for the drag report.

FILL PRICES
  Entries fill only at a real opening print (no open, no entry). Exits, forced
  exits and rolls must happen, so they fall back to close, then settle, and the
  fill is tagged with its basis so the report can count how often that was
  needed (3 front-contract opens are missing in 439,005 listed days).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fut_costs import CostConfig, charge
from fut_ledger import FuturesPosition, Ledger
from fut_market import FuturesMarket

ROLL_BEFORE_TD = 2        # must equal fo_continuous.ROLL_BEFORE_TD


@dataclass
class ExecConfig:
    max_weight_per_name: float = 0.05
    target_gross: float = 1.0
    roll_before_td: int = ROLL_BEFORE_TD


@dataclass
class ExecReport:
    fills: list = field(default_factory=list)   # dict rows for the trade log
    skips: list = field(default_factory=list)   # dict rows: entries not filled
    cost_total: float = 0.0
    realized_total: float = 0.0


def _fill_price(cd) -> tuple[float, str] | None:
    """(price, basis) for a must-fill order: open, else close, else settle."""
    if cd is None:
        return None
    for basis in ("open", "close", "settle"):
        p = getattr(cd, basis)
        if np.isfinite(p) and p > 0:
            return float(p), basis
    return None


def _row(date, bar, pos: FuturesPosition | None, symbol, expiry, action, trade_lots,
         lot_size, price, basis, cb, reason="", target_w=np.nan, achieved_w=np.nan,
         realized=np.nan, s_score=np.nan) -> dict:
    r = {"date": date, "bar": bar, "symbol": symbol, "expiry": pd.Timestamp(expiry),
         "action": action, "side": "BUY" if trade_lots > 0 else "SELL",
         "lots": int(trade_lots), "lot_size": float(lot_size), "price": float(price),
         "fill_basis": basis, "reason": reason,
         "target_weight": target_w, "achieved_weight": achieved_w,
         "realized_pnl": realized, "s_score": s_score,
         "entry_bar": pos.entry_bar if pos is not None else bar}
    r.update(cb.as_dict())
    return r


def execute_day(decisions, ledger: Ledger, market: FuturesMarket, i: int,
                equity: float, decision_bar: int, cost_cfg: CostConfig,
                cfg: ExecConfig) -> ExecReport:
    """Execute at the open of session i. `equity` is the mark at the decision bar
    (T close), the sizing base — nothing from session i is used for sizing."""
    rep = ExecReport()
    date = market.dates[i]

    def book_cost(cb) -> float:
        rep.cost_total += cb.total
        return cb.total

    # ---- 0. CORPORATE ACTIONS ON HELD POSITIONS ---------------------------- #
    # Applied before anything trades on session i, as the exchange does: a
    # holder's reference price (and, for share-count events, the lot) is
    # adjusted at the start of the ex-date. Without this the raw price move of
    # a bonus, split or extraordinary dividend was booked against the position
    # while the signal series had it adjusted away (found in diagnosis: six
    # positions in 2015-2024, errors of up to Rs 31 lakh each).
    for pos in ledger.positions():
        ca = market.corporate_action(i, pos.symbol)
        if ca is None:
            continue
        factor, relot = ca
        cd = market.contract(pos.symbol, pos.expiry, i)
        published = (cd.lot_size if (relot and cd is not None and np.isfinite(cd.lot_size)
                                     and abs(cd.lot_size - pos.lot_size) > 1e-9) else None)
        old_entry, new_entry, new_lot = ledger.adjust(pos.symbol, factor, relot, published)
        from fut_costs import CostBreakdown
        rep.fills.append(_row(date, i, pos, pos.symbol, pos.expiry, "CA_ADJUST", pos.lots,
                              new_lot, new_entry, "adjustment",
                              CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                              f"factor {factor:.6f}: " + ("lot re-denominated" if relot else
                                                          "reference price reduced")
                              + f"; entry {old_entry:.4f} -> {new_entry:.4f}",
                              pos.target_weight, pos.weight))

    # ---- 1. EXITS ---------------------------------------------------------- #
    for ex in decisions.exits:
        pos = ledger.get(ex.symbol)
        if pos is None:
            continue
        fp = _fill_price(market.contract(pos.symbol, pos.expiry, i))
        if fp is None:
            if market.is_expired(pos.expiry, i):
                # should be unreachable: forced exits below close before expiry
                fp = (pos.mark_price, "last-mark-EXPIRED")
            else:
                continue                      # no print today; try next session
        px, basis = fp
        cb = charge(cost_cfg, pos.lots, pos.lot_size, px, is_buy=pos.lots < 0)
        c = book_cost(cb)
        closed, realized = ledger.close(pos.symbol, px, c)
        rep.realized_total += realized
        rep.fills.append(_row(date, i, closed, closed.symbol, closed.expiry, "CLOSE",
                              -closed.lots, closed.lot_size, px, basis, cb, ex.reason,
                              closed.target_weight, closed.weight, realized))

    # ---- 2. FORCED EXITS AND ROLLS ---------------------------------------- #
    for pos in ledger.positions():
        dte = market.dte(pos.expiry, i)
        old = market.contract(pos.symbol, pos.expiry, i)
        if old is None:
            # The held contract has no row today. Contracts never have internal
            # gaps in this data, so this is a contract that VANISHED before its
            # expiry: a symbol change (INFRATEL -> INDUSTOWER, Dec 2020), a
            # merger (HDFC, Jul 2023) or a delisting without a final print. It
            # cannot be marked or traded any more, so it is closed at once at
            # its last settlement price and the fill is tagged; carrying it
            # would only book stale marks. In reality the position would have
            # converted into the successor name; not modelled.
            cb = charge(cost_cfg, pos.lots, pos.lot_size, pos.mark_price, is_buy=pos.lots < 0)
            c = book_cost(cb)
            closed, realized = ledger.close(pos.symbol, pos.mark_price, c)
            rep.realized_total += realized
            rep.fills.append(_row(date, i, closed, closed.symbol, closed.expiry, "FORCE_CLOSE",
                                  -closed.lots, closed.lot_size, pos.mark_price, "last-mark-NO-ROW",
                                  cb, "contract vanished from the tape (symbol change / "
                                  "merger / delisting): closed at its last settle",
                                  closed.target_weight, closed.weight, realized))
            continue
        if dte > cfg.roll_before_td:
            continue
        old_px, old_basis = _fill_price(old)
        front = market.front(i, pos.symbol)
        can_roll = (front is not None and front.tradable
                    and pd.Timestamp(front.expiry) > pd.Timestamp(pos.expiry)
                    and market.dte(front.expiry, i) > cfg.roll_before_td)
        fp_new = _fill_price(market.contract(pos.symbol, front.expiry, i)) if can_roll else None
        if can_roll and fp_new is not None:
            new_px, new_basis = fp_new
            new_lot = float(front.lot_size)
            if new_lot == pos.lot_size:
                new_lots = pos.lots
            else:   # lot revision between contracts: keep the notional, floor
                n = math.floor(abs(pos.lots) * pos.lot_size * old_px / (new_lot * new_px))
                new_lots = int(math.copysign(n, pos.lots))
            if new_lots != 0:
                cb_c = charge(cost_cfg, pos.lots, pos.lot_size, old_px, is_buy=pos.lots < 0)
                cb_o = charge(cost_cfg, new_lots, new_lot, new_px, is_buy=new_lots > 0)
                c_c, c_o = book_cost(cb_c), book_cost(cb_o)
                old_expiry, old_lots, old_lotsz = pos.expiry, pos.lots, pos.lot_size
                realized = ledger.roll(pos.symbol, old_px, c_c, front.expiry, new_px,
                                       new_lot, new_lots, c_o)
                rep.realized_total += realized
                rep.fills.append(_row(date, i, pos, pos.symbol, old_expiry, "ROLL_CLOSE",
                                      -old_lots, old_lotsz, old_px, old_basis, cb_c,
                                      f"dte={dte}", pos.target_weight, pos.weight, realized))
                rep.fills.append(_row(date, i, pos, pos.symbol, front.expiry, "ROLL_OPEN",
                                      new_lots, new_lot, new_px, new_basis, cb_o,
                                      f"dte={market.dte(front.expiry, i)}",
                                      pos.target_weight, pos.weight))
                continue
            reason = "roll: one lot of the next contract exceeds the position"
        else:
            reason = ("delisting: no successor contract" if not can_roll
                      else "roll: successor has no print")
        cb = charge(cost_cfg, pos.lots, pos.lot_size, old_px, is_buy=pos.lots < 0)
        c = book_cost(cb)
        closed, realized = ledger.close(pos.symbol, old_px, c)
        rep.realized_total += realized
        rep.fills.append(_row(date, i, closed, closed.symbol, closed.expiry, "FORCE_CLOSE",
                              -closed.lots, closed.lot_size, old_px, old_basis, cb, reason,
                              closed.target_weight, closed.weight, realized))

    # ---- 3. ENTRIES -------------------------------------------------------- #
    committed = sum(abs(p.weight) for p in ledger.positions())
    budget_w = cfg.target_gross - committed
    for en in decisions.entries:
        sym = en.symbol
        if sym in ledger:
            continue
        front = market.front(i, sym)

        def skip(reason, one_lot=np.nan, lot_size=np.nan, px=np.nan):
            rep.skips.append({"date": date, "bar": i, "symbol": sym, "reason": reason,
                              "target_weight": en.weight, "one_lot_notional": one_lot,
                              "lot_size": lot_size, "price": px, "equity": equity,
                              "budget_weight": budget_w})

        if front is None:
            skip("not listed on the fill date"); continue
        if not front.tradable or market.dte(front.expiry, i) <= cfg.roll_before_td:
            skip("front contract inside the roll window (delisting)"); continue
        px = front.open
        if not (np.isfinite(px) and px > 0):
            skip("no opening print"); continue
        lot_size = float(front.lot_size)
        one_lot = lot_size * px
        target_rs = abs(en.weight) * equity
        cap_rs = cfg.max_weight_per_name * equity
        budget_rs = budget_w * equity
        if one_lot > cap_rs + 1e-9:
            skip("one lot > per-name cap", one_lot, lot_size, px); continue
        if one_lot > budget_rs + 1e-9:
            skip("one lot > remaining gross budget", one_lot, lot_size, px); continue
        if one_lot > target_rs + 1e-9:
            skip("one lot > target weight", one_lot, lot_size, px); continue
        lots = math.floor(min(target_rs, budget_rs) / one_lot)
        if lots < 1:
            skip("floor gave zero lots", one_lot, lot_size, px); continue
        lots = int(math.copysign(lots, en.weight))
        achieved = lots * one_lot / equity
        cb = charge(cost_cfg, lots, lot_size, px, is_buy=lots > 0)
        c = book_cost(cb)
        pos = FuturesPosition(
            symbol=sym, expiry=pd.Timestamp(front.expiry), lots=lots, lot_size=lot_size,
            entry_price=float(px), entry_bar=decision_bar, entry_date=date,
            entry_s_score=float(en.s_score), target_weight=float(en.weight),
            weight=float(achieved), mark_price=float(px),
            orig_expiry=pd.Timestamp(front.expiry), orig_entry_price=float(px))
        ledger.open(pos, c)
        budget_w -= abs(achieved)
        rep.fills.append(_row(date, i, pos, sym, front.expiry, "OPEN", lots, lot_size, px,
                              "open", cb, "", en.weight, achieved, s_score=en.s_score))
    return rep
