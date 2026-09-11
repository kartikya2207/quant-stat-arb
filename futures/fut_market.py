"""
fut_market.py — Stage 2: read-only access to the Stage 1 futures data.

Two views of the same market, and the loop must never confuse them:

  THE SIGNAL VIEW   cont_ret_wide.parquet — the continuous front-month return
                    series per name (corporate-action clean, roll gaps removed,
                    masked days NaN). This is what the PCA and the OU fit see.
                    It is a SERIES, not a tradable instrument.

  THE TRADING VIEW  contracts.parquet — one row per (date, symbol, expiry) with
                    the real open / settle / lot size of that specific contract.
                    Every fill, every mark and every roll happens here, on a
                    named contract with a named expiry.

The chain (continuous.parquet) is the bridge: for each (date, symbol) it says
which contract the strategy holds that day — the nearest expiry with more than
ROLL_BEFORE_TD trading days left — and whether the name is tradable (False only
when a name is being delisted and its last contract is expiring with no
successor). Expiry dates are known at listing, so "which contract is front on
day D" uses no information beyond the exchange calendar.

Trading-day arithmetic: `dte(expiry, i)` is the number of exchange sessions
between session i and the expiry session; an expiry that falls on a holiday
maps to the next session, which can only make us roll earlier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


@dataclass(frozen=True)
class FrontInfo:
    expiry: pd.Timestamp
    open: float          # NaN when the contract had no opening print
    settle: float
    lot_size: float
    dte: int
    tradable: bool
    is_roll: bool


@dataclass(frozen=True)
class ContractDay:
    open: float
    close: float
    settle: float
    lot_size: float


class FuturesMarket:
    def __init__(self, data_dir: Path | None = None) -> None:
        data_dir = Path(data_dir) if data_dir else DATA
        cont = pd.read_parquet(data_dir / "continuous.parquet")
        contracts = pd.read_parquet(
            data_dir / "contracts.parquet",
            columns=["date", "symbol", "expiry", "open", "close", "settle", "lot_size"])
        log_p = data_dir / "corp_action_log.csv"
        ca_log = pd.read_csv(log_p, parse_dates=["ex_date"]) if log_p.exists() else None
        self._ingest(cont, contracts,
                     pd.read_parquet(data_dir / "cont_ret_wide.parquet"),
                     pd.read_parquet(data_dir / "oi_notional_wide.parquet"), ca_log)

    @classmethod
    def from_frames(cls, cont: pd.DataFrame, contracts: pd.DataFrame,
                    ret_wide: pd.DataFrame, oi_wide: pd.DataFrame) -> "FuturesMarket":
        """Build a market from in-memory frames in the Stage 1 layout (used by the
        known-answer validation to run the REAL harness on synthetic contracts)."""
        m = cls.__new__(cls)
        m._ingest(cont, contracts, ret_wide, oi_wide)
        return m

    def _ingest(self, cont: pd.DataFrame, contracts: pd.DataFrame,
                ret_wide: pd.DataFrame, oi_wide: pd.DataFrame,
                ca_log: pd.DataFrame | None = None) -> None:
        cont, contracts = cont.copy(), contracts.copy()
        for df, cols in ((cont, ["date", "front_expiry"]), (contracts, ["date", "expiry"])):
            for c in cols:
                df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
        self.dates = pd.DatetimeIndex(np.sort(cont["date"].unique()))
        self._date_pos = {d.value: i for i, d in enumerate(self.dates)}
        self._dates_i8 = self.dates.asi8
        # calendar extended past the data with weekdays, so a contract expiring
        # after the last session is measured against its real expiry, not the
        # end of the file (see fo_continuous.extended_calendar)
        last = self.dates[-1]
        ext = pd.bdate_range(last + pd.Timedelta(days=1), last + pd.Timedelta(weeks=12))
        self._cal_ext_i8 = np.concatenate([self._dates_i8, ext.asi8])

        # signal view, aligned to the exchange calendar
        self.ret_wide = ret_wide.reindex(self.dates)
        self.oi_wide = oi_wide.reindex(self.dates)
        self.symbols = list(self.ret_wide.columns)

        # the chain: (date_i, symbol) -> FrontInfo
        di = cont["date"].astype("int64").map(self._date_pos).to_numpy()
        self._front: dict[tuple[int, str], FrontInfo] = {}
        for i, sym, exp, op, st, lot, dte, trd, roll in zip(
                di, cont["symbol"], cont["front_expiry"], cont["front_open"],
                cont["front_settle"], cont["lot_size"], cont["dte"],
                cont["tradable"], cont["is_roll"]):
            self._front[(int(i), sym)] = FrontInfo(
                pd.Timestamp(exp), float(op), float(st), float(lot), int(dte),
                bool(trd), bool(roll))

        # corporate actions on HELD positions: (date_i, symbol) -> (factor, re-denominates lot?)
        # The factor is the one the signal series uses (fo_corpactions). Whether
        # the exchange re-denominates the lot is a property of the event kind:
        # bonus / split / rights / consolidation change the share count, so the
        # lot scales by 1/f and the price by f; a dividend or demerger adjustment
        # lowers the reference price by a rupee amount and leaves the lot alone.
        self._ca: dict[tuple[int, str], tuple[float, bool]] = {}
        if "ca_factor" in cont.columns:
            relot_kinds = {}
            if ca_log is not None and len(ca_log):
                for sym, d, kind in zip(ca_log["symbol"], pd.to_datetime(ca_log["ex_date"]), ca_log["kind"].astype(str)):
                    relot_kinds[(sym, pd.Timestamp(d).value)] = any(
                        k in kind for k in ("bonus", "split", "rights", "consolidation"))
            ev = cont[cont["ca_factor"] != 1.0]
            for i, sym, d, f in zip(ev["date"].astype("int64").map(self._date_pos), ev["symbol"],
                                    ev["date"].astype("int64"), ev["ca_factor"]):
                self._ca[(int(i), sym)] = (float(f), relot_kinds.get((sym, int(d)), False))

        # the trading view: (symbol, expiry_i8, date_i) -> ContractDay
        contracts = contracts[contracts["date"].isin(self.dates)]
        cdi = contracts["date"].astype("int64").map(self._date_pos).to_numpy()
        # listed expiries per (session, symbol), ascending — for the entry rule
        # that opens in the NEXT month when the front is about to roll
        ex = (pd.DataFrame({"i": cdi, "symbol": contracts["symbol"].to_numpy(),
                            "e": contracts["expiry"].astype("int64").to_numpy()})
              .sort_values(["symbol", "i", "e"]))
        self._expiries: dict[tuple[int, str], tuple[int, ...]] = {
            (int(i), sym): tuple(int(v) for v in g["e"].to_numpy())
            for (sym, i), g in ex.groupby(["symbol", "i"], sort=False)}
        self._px: dict[tuple[str, int, int], ContractDay] = {}
        for sym, exp, i, op, cl, st, lot in zip(
                contracts["symbol"], contracts["expiry"].astype("int64"), cdi,
                contracts["open"], contracts["close"], contracts["settle"],
                contracts["lot_size"]):
            self._px[(sym, int(exp), int(i))] = ContractDay(
                float(op), float(cl), float(st), float(lot))

    # ---- calendar ----------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.dates)

    def date_index(self, date) -> int:
        """Position of the first session on or after `date`."""
        return int(np.searchsorted(self._dates_i8, pd.Timestamp(date).value, side="left"))

    def dte(self, expiry: pd.Timestamp, i: int) -> int:
        """Trading days from session i to the expiry session (negative = expired)."""
        e = int(np.searchsorted(self._cal_ext_i8, pd.Timestamp(expiry).value, side="left"))
        return e - i

    def is_expired(self, expiry: pd.Timestamp, i: int) -> bool:
        return pd.Timestamp(expiry).value < self._dates_i8[i]

    # ---- chain -------------------------------------------------------------- #
    def front(self, i: int, symbol: str) -> FrontInfo | None:
        return self._front.get((i, symbol))

    def listed(self, i: int, symbol: str) -> bool:
        return (i, symbol) in self._front

    def next_expiry(self, i: int, symbol: str, after: pd.Timestamp) -> pd.Timestamp | None:
        """The nearest listed expiry of `symbol` on session i strictly after `after`."""
        a = pd.Timestamp(after).value
        for e in self._expiries.get((i, symbol), ()):
            if e > a:
                return pd.Timestamp(e)
        return None

    def corporate_action(self, i: int, symbol: str) -> tuple[float, bool] | None:
        """(factor, lot re-denominated?) if `symbol` has an adjustment on session i."""
        return self._ca.get((i, symbol))

    # ---- contracts ---------------------------------------------------------- #
    def contract(self, symbol: str, expiry: pd.Timestamp, i: int) -> ContractDay | None:
        return self._px.get((symbol, pd.Timestamp(expiry).value, i))

    def has_contract_row(self, symbol: str, expiry: pd.Timestamp, i: int) -> bool:
        return (symbol, pd.Timestamp(expiry).value, i) in self._px
