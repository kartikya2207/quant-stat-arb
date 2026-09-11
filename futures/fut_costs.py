"""
fut_costs.py — Stage 2: transaction costs for NSE single-stock futures.

One order = one contract, one side, `lots` lots at `price` per share. Every
component is a named field so a run can be audited component by component, and
so a TRUE zero-cost mode exists: `CostConfig.zero()` sets every component to 0
and `charge()` then returns exactly 0.0 — not "approximately", exactly. The
cash-equity study had a bug where a fallback cost function still charged STT
in the "zero-cost" run; `assert_zero()` guards against that class of bug.

Schedule (as specified for this study; all rates on rupee notional):
  STT        0.02%  on the SELL side only (futures STT is sell-side only)
  exchange   0.0019% both sides (NSE transaction charge)
  stamp duty 0.002%  on the BUY side only
  brokerage  Rs 20 flat per order
  GST        18% on (brokerage + exchange charge)
  slippage   0.03% per side (spread + impact buffer) on FRONT-month fills;
             0.05% per side on a BACK-month ENTRY (the next-month contract is
             thinner). Both are assumptions, not measurements.
SEBI turnover fee (Rs 10 / crore) and clearing charges are not in the
specification and are not charged; both are below 0.0002% and immaterial.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CostConfig:
    stt_sell: float = 0.0002          # 0.02% of notional, sell side
    exchange: float = 0.000019        # 0.0019% of notional, both sides
    stamp_buy: float = 0.00002        # 0.002% of notional, buy side
    brokerage_per_order: float = 20.0 # Rs, flat
    gst: float = 0.18                 # on brokerage + exchange
    slippage_per_side: float = 0.0003 # 0.03% of notional, each side, front month
    slippage_back_month_entry: float = 0.0005  # 0.05% on an entry in the next-month contract

    @classmethod
    def zero(cls) -> "CostConfig":
        """Every component zero. Nothing else changes."""
        return cls(stt_sell=0.0, exchange=0.0, stamp_buy=0.0,
                   brokerage_per_order=0.0, gst=0.0, slippage_per_side=0.0,
                   slippage_back_month_entry=0.0)

    @property
    def is_zero(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())


@dataclass(frozen=True)
class CostBreakdown:
    notional: float
    stt: float
    exchange: float
    stamp: float
    brokerage: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return self.stt + self.exchange + self.stamp + self.brokerage + self.gst \
            + self.slippage

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cost_total"] = self.total
        return d


def charge(cfg: CostConfig, lots: int, lot_size: float, price: float,
           is_buy: bool, back_month: bool = False) -> CostBreakdown:
    """Cost of ONE order of `lots` lots (unsigned) at `price`. `back_month`
    selects the higher slippage assumed for an entry in the next-month contract."""
    lots = abs(int(lots))
    if lots == 0 or price <= 0 or lot_size <= 0:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    notional = lots * lot_size * price
    stt = 0.0 if is_buy else cfg.stt_sell * notional
    exch = cfg.exchange * notional
    stamp = cfg.stamp_buy * notional if is_buy else 0.0
    brok = cfg.brokerage_per_order
    gst = cfg.gst * (brok + exch)
    slip = (cfg.slippage_back_month_entry if back_month else cfg.slippage_per_side) * notional
    return CostBreakdown(notional, stt, exch, stamp, brok, gst, slip)


def assert_zero(cfg: CostConfig) -> None:
    """Prove the zero-cost mode charges exactly nothing on a spread of orders."""
    assert cfg.is_zero, "CostConfig is not the zero configuration"
    for lots, lot, px, buy in [(1, 250, 2500.0, True), (7, 1000, 99.5, False),
                               (100, 8800, 12.35, True), (3, 550, 1300.25, False)]:
        c = charge(cfg, lots, lot, px, buy)
        assert c.total == 0.0, c
        assert all(v == 0.0 for k, v in c.as_dict().items() if k != "notional"), c
