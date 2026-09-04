# Futures data pipeline — NSE single-stock futures, 2015–2024

Data layer for the futures version of the PCA-residual stat-arb strategy, built on the
**free NSE bhavcopy archive**. Status: **Stage 1 complete and validated at full scale**
(329 names, 2,467 trading days). The 2-name prototype (`futures_proto_v2.py`) is kept for
reference; everything below supersedes it.

## Pipeline

| step | script | output |
|---|---|---|
| 1a download | `fo_download.py` | raw `.zip` cache of three NSE archives |
| 1b parse | `fo_build.py` | `contracts.parquet`, `membership.parquet`, `cash.parquet` |
| 1c corp actions | `fo_corpactions.py` | `ca_calendar.parquet`, `ca_factors.parquet`, **`corp_action_log.csv`** |
| 1d continuous | `fo_continuous.py` | `continuous.parquet` + wide frames for the harness |
| 1e validate | `fo_validate.py` | `data_quality_report.csv`, `excluded_names.csv` |

```bash
python fo_download.py --what fo      # then --what cm, --what pr
python fo_build.py --what fo         # then --what cm
python fo_corpactions.py && python fo_continuous.py && python fo_validate.py
```

The raw cache is ~2.4 GB, so it is written **outside the repo** by default
(`%LOCALAPPDATA%/quant-stat-arb/fo_cache`) rather than into this OneDrive-synced folder.
Override with `QSA_CACHE_DIR`. Downloads: 2,468 trading days per archive, 141 holidays,
**0 errors**; a 404 writes a `.missing` marker so holidays are never re-requested.

## Three archives, and why the third one matters

- **`fo`** F&O bhavcopy → single-stock futures per contract per day. Both layouts handled:
  the old per-month CSV before 2024-07-08 and UDiFF after.
- **`cm`** cash bhavcopy → the independent spot reference for basis and validation.
- **`pr`** the daily "price report" zip, which contains **`Bc<ddmmyy>.csv` — NSE's own
  corporate-action calendar** (SYMBOL, EX_DT, PURPOSE). This is the real corp-action source
  the prototype's checklist asked for. It carries `RELIANCE / 20-07-2023 / DEMERGER`, the
  event `yfinance .splits` does not report at all. The Bc file is only a rolling window of
  *upcoming* ex-dates and drops an entry once the ex-date arrives, so every trading day's
  file is downloaded and unioned — sampling weekly would silently lose corporate actions.

## Corporate actions — four signals, and a plausibility guard

The calendar is the primary source (31,551 rows). Splits and bonuses also give the exact
factor in the text (`BONUS 1:1` → 0.5, `FV SPLT FRM RS 10 TO RS 2` → 0.2). Three physical
detectors back it up, all deliberately strict because they exist to catch calendar gaps,
not to second-guess the calendar:

1. **lot step** — the exchange re-denominates the contract, so `lot(D)/lot(D-1) = 1/f`
   exactly. Only counts when the price *also* moved by the reciprocal: NSE revises lot
   sizes every six months with no price adjustment at all.
2. **OI reset** — a contract-replacing action resets the whole open interest
   (`ChgInOI == OI` for a contract that existed the previous session). Tested on the front
   contract only; a newly listed far month trivially satisfies it on its first day.
3. ~~bare round-fraction price jump~~ — **deliberately removed.** It deletes real data:
   YES BANK's genuine +70% and +71% moves during the March-2020 moratorium land within 1%
   of 17/10 and 12/7, and a price-only rule "adjusts" them away.

Every calendar-derived factor is then checked against what the price actually did
overnight, because NSE's free text is sometimes wrong: it published NHPC's 2016-02-18
interim dividend as **both** `RS 9.20` and `RE 0.92` on a ₹19 stock. Summing those invented
a 53% adjustment and turned an ordinary day into a fake **+120%**. When the text and the
tape disagree, the adjustment is refused rather than trusted.

**Dividends are deliberately not adjusted.** Unlike the cash study, a futures holder never
receives the dividend, so the ex-date drop is a real loss to a long. Only dividends above
NSE's 2%-of-price "extraordinary" bar — the exact rule the exchange applies — are adjusted.

**524 adjustments applied** over 10 years: 318 extraordinary dividends, 105 from calendar
text (77 bonus, 28 split), 50 measured, 49 detector-only, 2 lot-step-resolved. Every one is
logged to `corp_action_log.csv` with its source, the observed ratio, the lot-implied
factor, and whether the day was masked.

Where the factor **cannot** be known — a demerger's size is not in the text, and the
observed jump mixes the exchange's adjustment with that day's real trading — we use the
measured ratio for P&L continuity (what the exchange's own adjustment achieves for a
position holder) and **mask that name-day out of the signal**. 50 name-days are masked. We
give up one name-day rather than trade a number we cannot vouch for.

## The continuous series is look-ahead-free by construction

The prototype used **additive (Panama)** back-adjustment: on each roll, add the gap to
every earlier price. That removes the gap but makes the price at time *t* change every time
a later roll happens — so returns depend on future rolls. That is look-ahead.

We use **ratio (proportional) chaining**: the daily return is the return of the contract we
actually held, on that contract's own prices,
`r(d) = px(d, F(d)) / (px(d-1, F(d)) · f(d)) − 1`.
Under ratio chaining returns are invariant to the anchor, so this is the same series a
back-adjusted build gives — but anchoring *forward* means the level at time *t* depends
only on data up to *t*.

**Roll:** 2 **trading** days before expiry (not calendar days, so a holiday in expiry week
cannot leave us holding into the last session). Expiry dates are known at listing.

**Re-listings break the chain.** The F&O universe is point-in-time and names *leave* it.
BALRAMCHIN left around ₹74 and came back on 2021-12-31 at ₹369 — naive per-symbol chaining
prints that as a **+399% day**. A return exists only when the name was also listed on the
immediately preceding session.

Because the level re-anchors at a break, `pct_change()` of `cont_close` is **not** the
return series. **`cont_ret_wide.parquet` is authoritative** and is what the harness consumes.

## Units, lot sizes, liquidity

Open interest is in **shares** in both formats; volume is in **contracts** in both
(verified across the cutover — the prototype's "volume units changed" worry does not bite,
median post/pre ratios: OI 1.005, lot 1.000). Turnover is in rupees.

**Liquidity ranking is by open interest**, expressed in **rupees** (`oi_shares × price`) so
it is comparable across names and across the format change, and summed over all of a name's
contracts so it does not lurch on roll days.

**Lot size** is published in UDiFF (`NewBrdLotQty`) and derived for the old format **per
contract (symbol, expiry)** — not per symbol-day. The lot belongs to the contract, and two
expiries of one name genuinely differ while a revision phases in: on 2023-05-03 AARTIIND's
May and June contracts were on 850 while the new July contract had OI of 7,000, which 850
does not divide. Deriving per symbol-day mixed them and produced a series that jittered
every day (851, 850, 852, 849, 857, 832…), which then tripped the corp-action detector
constantly. Per contract we take the divisor of `gcd(OI, ΔOI)` closest to the median of
`turnover/(volume × price)`.

## Stage 1 validation results

| check | result |
|---|---|
| lot derivation vs published `NewBrdLotQty` (66,207 UDiFF rows) | **99.75% within 2%**, median error 0.20% → derivation VALIDATED |
| units across the 2024-07-08 cutover (181 common names) | OI ×1.005, lot ×1.000 → unchanged |
| return correlation vs NSE cash, 10 diverse names | min **0.9869**, median **0.9937** (expect ~0.99) |
| basis vs cash | median **+0.295%** — small and positive = carry |
| unexplained NaNs in the return series | **0** |
| roll integrity: chained vs expiring contract, same day | **+0.39 bps** (a leaked roll gap would be 30–100 bps) |
| roll timing | **21,243 of 21,244** rolls fire at exactly 2 trading days (99.995%) |
| known-answer: RELIANCE 1:1 bonus 2024-10-28 | factor **0.5000** from calendar text ✅ |
| known-answer: RELIANCE–Jio demerger 2023-07-20 | caught, factor 0.9242, masked ✅ (yfinance misses it entirely) |

Comparing roll days to ordinary days is *not* a valid roll test — every name rolls on the
same ~120 dates, so it only measures what the market did on those dates. The sharp test is
the chained-vs-expiring comparison above.

### Coverage and exclusions

| year | names with futures | avg names/day |
|---|---|---|
| 2015 | 166 | 153.8 |
| 2016 | 179 | 172.2 |
| 2017 | 218 | 200.3 |
| 2018 | 211 | 206.4 |
| 2019 | 202 | 175.7 |
| 2020 | 153 | 140.2 |
| 2021 | 197 | 163.5 |
| 2022 | 203 | 196.2 |
| 2023 | 192 | 186.8 |
| 2024 | 231 | 184.9 |

**329 names seen, 292 retained, 37 excluded** — all excluded for insufficient history only
(34 are the Nov-2024 F&O expansion with 22 sessions; RANBAXY, RDEL, SSLT are short-lived
old listings). None is excluded for a data-quality failure. After exclusions the tradable
universe is **136–213 names per day**, so the top-120 selection is achievable on every one
of the 2,467 days.

11 names are **flagged but retained** for carrying a >40% clean daily move: AMTEKAUTO,
CONCOR, DHFL, INDUSINDBK, INFIBEAM, JETAIRWAYS, PCJEWELLER, PNB, RCOM, RELCAPITAL, YESBANK.
These are real market events — the YES BANK moratorium, the JET AIRWAYS grounding, the
DHFL/RCOM/RELCAPITAL defaults, PNB's fraud disclosure — and none carries the physical trace
of a re-pricing. **Dropping a name because it once fell 90% would remove precisely the
blow-ups and reintroduce survivorship bias**, the flaw the cash-equity study had to
disclose. They stay in.

745 symbol-days (0.17%) hold a contract inside the roll window. These are F&O **delistings**
— the last contract expires with no successor, so there is nothing to roll into. They are
flagged `tradable=False`; the harness must force-exit and never open an expiring contract.

## Notes

- Spot reference is the NSE **cash** bhavcopy, not yfinance `Close` (which is
  dividend-adjusted for old dates and distorts the basis level).
- NSE's cash bhavcopy `PREVCLOSE` is the **raw** prior close, *not* corporate-action
  adjusted — checked directly, so it cannot be used to recover adjustment factors.
- `./cache/` and `./data/` are gitignored (regenerable). The three audit CSVs
  (`corp_action_log.csv`, `data_quality_report.csv`, `excluded_names.csv`) are committed
  deliberately as the audit trail.
