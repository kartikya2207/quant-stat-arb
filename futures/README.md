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

## Corporate actions — three signals, and a plausibility guard

The calendar is the primary source (31,551 rows). Splits and bonuses also give the exact
factor in the text (`BONUS 1:1` → 0.5, `FV SPLT FRM RS 10 TO RS 2` → 0.2). Two physical
detectors back it up, both deliberately strict because they exist to catch calendar gaps,
not to second-guess the calendar:

1. **lot step** — the exchange re-denominates the contract, so `lot(D)/lot(D-1) = 1/f`
   exactly. Only counts when the price *also* moved by the reciprocal — within 10% of
   `1/lot ratio` **and closer to it than to 1.0**. The second half matters: NSE revises
   lot sizes every six months with no price adjustment at all, and for a 5–10% revision a
   plain 10% band around `1/lot ratio` contains 1.0, so every routine revision date
   (2016-07-01, 2018-12-28, 2019-06-28, 2020-06-26, …) passed it and produced a fake
   2–11% adjustment — 49 of them, every one with the cash price moving exactly as the
   futures did. Old-format lots are *derived* and carry a few percent of noise, so a
   derived lot must step by more than 5%; a published UDiFF lot by 0.5%.
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

**Dividends: only what the exchange actually adjusted.** A futures holder never receives
the dividend; the futures discount it ahead of the ex-date, which is a real loss to a long
and must not be adjusted away. The exchange adjusts only an "extraordinary" dividend, and
that bar has moved: **10%** of the underlying's market value, cut to **5%** by SEBI circular
CIR/MRD/DoP-1/P/00108/2018 (5 Jul 2018) and to **2%** by SEBI/HO/MRD2/MRD2_DCAP/P/CIR/2022/90
(28 Jun 2022), which the tape shows the exchange applying from October 2022. A fixed 2% rule
over 2015–2024 therefore adjusted away ~220 real losses. Worse, the bar is measured at the
price on the day before the *announcement*, which the bhavcopy does not carry, so no fixed
rule classifies events near a bar (TATASTEEL's Rs 51 of June 2022 was 5.4% of the ex-date
price but under 5% when announced, and was not adjusted).

So the exchange's decision is read off the tape, per event. An unadjusted dividend is
already discounted by the futures, so on the ex-date the **futures/cash basis jumps by the
dividend yield**; an adjusted one leaves the basis unchanged. Over 1,144 dividend events
that jump divided by the yield is ~1 for every sub-bar dividend in every year and ~0 for
every adjusted one (median by year × yield bucket: 0.9–1.0 below 2% throughout; 0.99 → 0.01
in the 2–5% bucket across the 2022 change; 1.0 → −0.03 in the 5–10% bucket across 2019/20).
A factor is applied only when that ratio is ≤ 0.35 (adjusted); ≥ 0.65 is an ordinary day;
in between (8 events) the day is masked.

**260 adjustments applied** over 10 years: 105 from calendar text (77 bonus, 28 split),
96 extraordinary dividends confirmed by the tape, 57 measured-and-masked (24 demergers,
22 rights, 1 split with its ratio cut off by NSE's field width, 10 dividend/bonus rows the
tape cannot classify), 2 rights issues resolved from a published lot step, **0 detector-only**.
Every one is logged to `corp_action_log.csv` with its source, the observed ratio, the
lot-implied factor, the basis jump / yield, and whether the day was masked.

Where the factor **cannot** be known — a demerger's size is not in the text, and the
observed jump mixes the exchange's adjustment with that day's real trading — we use the
measured ratio for P&L continuity (what the exchange's own adjustment achieves for a
position holder) and **mask that name-day out of the signal**. 57 name-days are masked. We
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
| return correlation vs NSE cash, 10 diverse names | min **0.9869**, median **0.9936** (expect ~0.99); all 292 retained names: p5 0.964, median 0.990 |
| basis vs cash | median **+0.295%** — small and positive = carry |
| unexplained NaNs in the return series | **0** |
| roll integrity: chained vs expiring contract, same day | **+0.31 bps** mean, +0.42 bps median over 21,240 rolls; the gap a leak would inject is +45 bps median |
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

10 names are **flagged but retained** for carrying a >40% clean daily move: AMTEKAUTO,
DHFL, INDUSINDBK, INFIBEAM, JETAIRWAYS, PCJEWELLER, PNB, RCOM, RELCAPITAL, YESBANK. On
every one of those days the NSE cash close moved the same way (YESBANK and RELCAPITAL by
less, because their futures were pricing a resolution the cash market's circuit filters
could not), so they are real market events — the YES BANK moratorium, the JET AIRWAYS
grounding, the DHFL/RCOM/RELCAPITAL defaults, PNB's fraud disclosure — and none carries the
physical trace of a re-pricing. (CONCOR was on this list until its 2018-06-26 "−49.8%" was
traced to a 10→5 face-value split whose new value NSE's field width had cut off:
`FV SPLT FRM RS 10/- TO RS`. It is now measured and masked.) **Dropping a name because it once fell 90% would remove precisely the
blow-ups and reintroduce survivorship bias**, the flaw the cash-equity study had to
disclose. They stay in.

299 symbol-days (0.07%) hold a contract inside the roll window. These are F&O **delistings**
— the last contract expires with no successor, so there is nothing to roll into. They are
flagged `tradable=False`; the harness must force-exit and never open an expiring contract.

## Stage 2 — the strategy on futures (built, dry-run only; full backtest not yet run)

| module | role |
|---|---|
| `fut_market.py` | read-only access to Stage 1: the **signal view** (`cont_ret_wide`), the **chain** (which contract is front on each day, tradable flag) and the **trading view** (real open/settle/lot per contract-day). Trading-day arithmetic for expiry. |
| `fut_costs.py` | futures cost schedule per order (STT sell 0.02%, exchange 0.0019%, stamp buy 0.002%, ₹20 brokerage, GST 18% on brokerage+exchange, slippage 0.03%/side) and a true zero-cost mode that is asserted to charge exactly 0. |
| `fut_ledger.py` | positions in **whole lots on named contracts**; exposes the `Book` interface so `strategy/harness_decide.py` runs unchanged. Cash + realized P&L persist; rolls change the contract, keep the entry bar and the weights. SPAN-proxy margin at 25% of gross notional. |
| `fut_execute.py` | at T+1 open, in order: **corporate-action adjustment of held positions** on an ex-date (bonus/split/rights: entry price × f, lot ÷ f; dividend/demerger: reference price reduced by the rupee amount — unrealised P&L continuous either way; found missing in diagnosis), exits on the held contract, forced exits (delisting, vanished contract) and **rolls** at 2 trading days to expiry as two charged trades, entries on the chain's front contract with **floored** lot counts and skip-with-reason when one lot exceeds the cap / budget / target. |
| `fut_backtest.py` | the daily loop. Signal = `strategy/signal.py` loaded as-is (60-day window, correlation PCA/SVD, 5 factors, OU gate, s-score); decisions = `strategy/harness_decide.py` unchanged; universe = top 120 by OI notional ∪ held. Sizing base = equity at T close. |
| `run_futures_dryrun.py` | proves the plumbing on a few months: every fill is on a real contract row at its open, OPENs are on the front with dte>2, rolls leave at dte==2, lots are whole, zero-cost charges 0, trade-log P&L reconstruction matches equity exactly, forced exit and lot-change roll exercised on real contracts. |

Default capital is **₹10 crore**: one lot is ₹3–8 lakh on the top-120 names, so the 5% per-name cap
(₹50 lakh) holds 6–17 lots and lot flooring costs ~6% of target weight rather than making every
name untradable. Dry run Jan–Jun 2019: 116 entries, 48 rolls, 0 skips, 0 forced closes, costs
4.4 bps per side of notional, margin utilisation ≤ 23%.

## Stage 3 — harness validation (passed 2026-09-10)

| check | result |
|---|---|
| known-answer suite (`fut_validation.py`), synthetic monthly contracts, lots + rolls active, zero cost | strong reversion (ac −0.30) **+117%, Sharpe +4.2**; random walk **+1%, Sharpe +0.08**; momentum (ac +0.30) **−41%, Sharpe −1.8**; same data with the s-score negated **+62%, Sharpe +1.7** → 4/4 PASS |
| independent daily equity reconstruction from the trade log, 2015–2024, 2,407 bars (`fut_checks.py`) | max difference **₹2.7e-7**, relative **2.3e-15** |
| look-ahead over every fill of the full run | 0 fills after expiry, 0 next-month contracts opened before the roll date, every OPEN one session after its decision bar, every ROLL_CLOSE at exactly 2 sessions to expiry |
| Stage 1 adjustments rebuilt from data truncated at 2018-06-29 and 2021-12-31 | corporate-action factors and continuous returns **byte-identical** to the full-sample series up to each cutoff |

Two things the checks caught and fixed: (1) days-to-expiry for a contract expiring after the end
of the data was measured against the end of the file, which made every name look untradable on the
last two sessions of the sample and broke the truncation test — the session calendar is now
extended past the data with weekdays (`fo_continuous.extended_calendar`); (2) a contract that
vanishes from the tape before expiry (a symbol change such as INFRATEL → INDUSTOWER, or a merger)
was carried at a stale mark for a week — it is now closed at once at its last settlement price and
the fill is tagged `last-mark-NO-ROW` (one such fill in the 2015–2024 run).

## Diagnosis scripts (post-Stage-4)

`diag_funnel_slippage.py` (per-bar signal funnel via the read-only `diagnostics` flag in
`fut_backtest.py`, plus the slippage sweep), `diag_cash_compare.py` (same-date signal supply on the
cash data, two universe rules, and realised-side comparison), `diag_cash_oos_pit.py` (the cash OOS
test on a point-in-time F&O universe), `diag_basis.py` (basis / carry decomposition),
`verify_ca_fix.py` (the six held-position corporate-action cases before and after the fix). Outputs
in `results_futures_2015_2024/diagnostics/`.

## Notes

- Spot reference is the NSE **cash** bhavcopy, not yfinance `Close` (which is
  dividend-adjusted for old dates and distorts the basis level).
- NSE's cash bhavcopy `PREVCLOSE` is the **raw** prior close, *not* corporate-action
  adjusted — checked directly, so it cannot be used to recover adjustment factors.
- **Limitation to carry into the final findings:** the Bc corporate-action file is a rolling
  window of *upcoming* ex-dates, and our PR archive starts on 2015-01-01. Corporate actions
  with ex-dates in the first weeks of January 2015 were announced in December 2014 files we do
  not have, so early-January-2015 events can be missing from the calendar; the physical
  detectors (lot step, OI reset) are the only cover for that window.
- `./cache/` and `./data/` are gitignored (regenerable). The three audit CSVs
  (`corp_action_log.csv`, `data_quality_report.csv`, `excluded_names.csv`) are committed
  deliberately as the audit trail.
