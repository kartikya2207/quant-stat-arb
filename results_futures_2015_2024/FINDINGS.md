# PCA-Residual Statistical Arbitrage — Single-Stock Futures Study (NSE, 2015–2024)

**STATUS: complete, 2026-09-12.** The cash-equity strategy (frozen in `results_cash_equity_2015_2024/`)
ported unchanged in signal and decision logic to NSE single-stock futures, on a point-in-time F&O
universe built from the free NSE bhavcopy archive, with fills on real contracts in whole lots. The
first numbers (2026-09-10) were withheld pending an adversarial diagnosis, which found one accounting
bug (§9, fixed) and changed how the cash study should be read (§7). A final execution change — not
opening a position in a contract about to roll (§1, §4b) — and a decay analysis (§12) were added
afterwards. Every number below is from the current code.

---

## 1. Strategy (unchanged from cash) and execution
Per bar: 60-day window of daily returns → correlation-form PCA (z-score, SVD), 5 factors → residuals
(intercept included) → OU fit on the cumulative residual with the 0<b<1 gate → s-score.
`s ≤ −2` long, `s ≥ +2` short. Triple-barrier exit: profit `|s| ≤ 0.5`, stop `|s| ≥ 3.0`, time stop
15 bars. Inverse-vol sizing, dollar-neutral by `min(longs, shorts)` pairing, per-name cap 5%, book
gross cap 1.0×, at most 40 positions. Signal from T close → fill at T+1 open → mark at T+1 close.

Universe each day: **top 120 by open-interest notional** among names with a listed single-stock
future that day, ∪ names currently held. PCA is estimated on names with a complete 60-day window
(118.7 on average); partial histories with ≥30 observations and held names outside the universe are
projected onto those frozen factors. Masked name-days are absent from the window.

Execution: fills on real contracts at the next open; **whole lots, floored** (a name is skipped when
one lot exceeds the 5% cap, the remaining gross budget, or the target); **rolls 2 trading days
before expiry** as two real, charged trades; **held positions adjusted on corporate-action ex-dates**
as the exchange adjusts a holder's reference price (§9); a name whose last contract expires with no
successor is force-closed.

**Entry contract (added after the diagnosis).** A new position opens in the chain's front contract
unless that contract has fewer than `entry_min_dte = 5` sessions left, in which case it opens
directly in the **next-month contract** rather than pay a roll (two charged legs) within days. Since
the chain rolls at 2 sessions, the rule bites on the two sessions before each roll date. N = 5 was
chosen a priori (one trading week) and is not tuned; N ≤ 3 disables the rule. Existing rolls are
unchanged. **A back-month entry is charged 0.05% per side of slippage against 0.03% on the front
month. This is an assumption, not a measurement**; §4b reports the measured liquidity of the
next-month contract on those sessions so the reader can judge it.

Costs per order: STT 0.02% sell-side, exchange 0.0019%, stamp 0.002% buy-side, GST 18% on
brokerage + exchange, brokerage ₹20 flat, slippage 0.03% per side (0.05% on a back-month entry).
SPAN proxy 25% of gross notional, reported not enforced. Config: `initial_capital = ₹10 crore`,
`pca_window 60, n_factors 5, universe_size 120, entry_min_dte 5`, daily.

## 2. Data
- NSE F&O bhavcopy (old CSV to 2024-07-05, UDiFF from 2024-07-08), cash bhavcopy, and the daily PR
  zip's `Bc` corporate-action calendar, every trading day 2015-01-01 → 2024-12-31 (2,467 sessions).
- **329 names ever listed**, **292 retained**, 37 excluded for insufficient history (34 are the
  Nov-2024 F&O expansion with 22 sessions; RANBAXY, RDEL, SSLT). The universe is read off each day's
  file, so there is **no survivorship bias**: names enter and leave, 23 re-listings break the chain.
- Corporate actions: 264 adjustments (105 from calendar text, 100 extraordinary dividends confirmed
  by the ex-date basis jump, 57 measured-and-masked, 2 rights from a published lot step), 57
  name-days masked. Ordinary dividends are deliberately not adjusted.
- Stage 1 gate: return correlation with cash 0.990 median across names; basis +0.31% median; roll
  test +0.31 bps; 0 unexplained NaNs. See `futures/README.md`.

## 3. Headline result — 2015-03-31 → 2024-12-31, 2,407 bars, ₹10 crore
| metric | **Full cost** | **Zero cost** |
|---|---|---|
| total return | **+13.7%** | **+27.0%** |
| CAGR | +1.32% | +2.48% |
| annualised Sharpe | **+0.32** | **+0.57** |
| Sharpe standard error (Lo 2002, √((1+s_d²/2)/n)·√252) | ±0.32 | ±0.32 |
| t-stat of mean daily return | 0.98 | 1.75 |
| annualised volatility | 4.6% | 4.6% |
| max drawdown | **−13.0%** | **−10.9%** |
| total costs | **₹1,23,61,335** | ₹0.00 |
| costs per year, % of capital | 1.27% | 0 |
| fills (round trips + roll legs) | 5,231 (1,988 + 2×625, 1 forced) | 5,231 |
| back-month entries (rule, §1) | 180 of 1,993 (9.0%) | 180 |
| corporate-action adjustments of held positions | 14 | 14 |
| average gross exposure (marked) | 0.376 | 0.377 |
| average net exposure | −0.006 | −0.006 |
| average / peak margin utilisation | 9.4% / 24.8% | 9.4% / 24.8% |
| margin breaches | 0 | 0 |
| lot-rounding drag (mean achieved / target weight) | 0.937 | 0.941 |
| lots per entry (min / median / max) | 1 / 8 / 139 | 1 / 8 / 152 |
| skipped entries | 1 (MRF 2015-08-10: one lot ₹56.5 lakh > 5% cap) | 1 |
| average positions held (long / short) | 8.0 (4.0 / 4.1) | 8.0 |
| round-trip win rate | 58.8% | 58.8% |

**Neither Sharpe is statistically distinguishable from zero over ten years** (t 1.0 and 1.8).

Yearly returns (full / zero cost), costs, average gross and positions:

| year | full | zero | cost ₹ lakh | avg gross | avg positions |
|---|---|---|---|---|---|
| 2015 (Apr–Dec) | −0.0% | +0.9% | 8.9 | 0.39 | 8.2 |
| 2016 | +4.4% | +5.6% | 12.9 | 0.42 | 9.1 |
| 2017 | +4.6% | +5.7% | 11.6 | 0.35 | 7.7 |
| 2018 | **+11.1%** | **+12.8%** | 14.4 | 0.39 | 8.4 |
| 2019 | +0.4% | +1.2% | 14.2 | 0.40 | 8.6 |
| 2020 | +1.6% | +2.9% | 12.4 | 0.35 | 7.4 |
| 2021 | −2.2% | −1.4% | 11.8 | 0.34 | 7.2 |
| 2022 | −0.5% | +0.4% | 11.6 | 0.33 | 7.0 |
| 2023 | −6.1% | −4.5% | 13.6 | 0.41 | 8.7 |
| 2024 | +0.6% | +1.6% | 12.1 | 0.38 | 8.2 |

**2018 alone is most of the result** (+11.1 of +13.7 points at full cost); 2019–2024 sum to −6.2
points at full cost and +0.2 before costs. §12 examines whether that is decay or noise.

## 4. Costs, and how fragile the positive return is
Headline run, ₹2,786 crore traded notional, **4.44 bps per side of notional**:

| component | ₹ | share |
|---|---|---|
| slippage (0.03% front / 0.05% back-month entry) | 85,46,610 | 69.1% |
| STT (0.02% sell side) | 27,88,407 | 22.6% |
| exchange (0.0019%) | 5,29,288 | 4.3% |
| stamp (0.002% buy side) | 2,78,305 | 2.3% |
| GST | 1,14,104 | 0.9% |
| brokerage (₹20/order) | 1,04,620 | 0.8% |
| **total** | **1,23,61,335** | |
| of which on the 625 rolls (1,250 legs) | 28,98,561 | **23.4%** |
| of which on the 180 back-month entries | 6,03,158 | 4.9% |

Slippage sensitivity, run on the front-only baseline (before the entry rule; the rule adds about
+1.8 points at every level, see §4b):

| slippage per side | total return | Sharpe | max drawdown |
|---|---|---|---|
| 0.02% | +14.6% | +0.34 | −12.8% |
| **0.03% (assumed)** | **+11.8%** | **+0.28** | **−13.3%** |
| 0.05% | +6.5% | +0.17 | −14.8% |
| 0.08% | −1.6% | −0.01 | −16.2% |

**The return dies between 5 and 8 bps of slippage per side against 3 assumed**; a linear
extrapolation puts break-even at ~8 bps per side (~16 bps round trip). Slippage is 69% of all costs
and is a flat assumption with no size or liquidity dependence, so the sign of the headline rests on
that one number.

### 4b. The entry rule: what moved, and how much of it is real
Same signals, same decisions; only the contract a new position opens in changes when the front has
fewer than N sessions left (`diagnostics/baseline_front_only/`, `diagnostics/entry_rule_variants/`):

| run | return | Sharpe | costs ₹ lakh | roll share | rolls | back-month entries | back-month cost ₹ lakh |
|---|---|---|---|---|---|---|---|
| baseline, front only (N ≤ 3) | +11.8% | +0.28 | 128.5 | 28.5% | 798 | 0 | 0 |
| **N = 5, back-month slippage 0.05% (headline, post bug-8 fix)** | **+13.7%** | **+0.32** | **123.6** | **23.4%** | **625** | **180** | **6.0** |
| N = 5, back-month slippage 0.03% (same as front) | +14.1% | +0.32 | 121.8 | 23.8% | 622 | 184 | 4.2 |
| N = 8, 0.05% (sensitivity) | +14.0% | +0.32 | 116.5 | 16.5% | 418 | 437 | 14.9 |
| N = 12, 0.05% (sensitivity) | +13.2% | +0.31 | 109.5 | 7.3% | 170 | 836 | 28.3 |
| baseline, zero cost | +25.6% | +0.54 | 0 | — | 798 | 0 | 0 |
| N = 5, zero cost (post bug-8 fix) | +27.0% | +0.57 | 0 | — | 625 | 180 | 0 |

Three things to read off this. **The cost saving is small**: at N = 5 the rule avoids 176 rolls and
saves ₹5.0 lakh, 4% of total costs, 0.05% of capital; the higher back-month slippage assumption
takes back ₹2.0 lakh of that (the 0.03% vs 0.05% rows differ by 0.44 points). **Most of the +1.8
point change in the headline is not cost**: the zero-cost run moved by +1.7 points with identical
signals, because a position opened in the next month follows a different price path (the basis
between the two contracts, and no roll-day basis jump) — that is variance in the outcome, not
evidence the rule is better. **N is not monotone** (+13.6, +14.0, +13.2 at 5, 8, 12) and the
headline stays at the a-priori N = 5; picking N = 8 because it prints higher would be tuning to
noise and is not done. Stage 3 checks were re-run under the rule: reconstruction to 2e-15, 184
back-month opens all on the next listed expiry with the front under 5 sessions, 0 violations.

Measured liquidity of the next-month contract, so the 0.05% assumption can be judged
(`diag_backmonth_liquidity.py`): on the rule's window (front dte < 5), top-120 names, the next-month
contract carries a median **25% of the front's open interest and 41% of its volume** (25th
percentile 15% and 25%), median daily volume 1,797 lots and turnover ₹114 crore against a typical
₹47 lakh fill, and has no trade at all on 0.1% of those sessions. The ratio rose over the sample
(volume ratio 0.31 in 2015–2016, 0.59–0.63 in 2022–2024). Against 2.4× lower volume, a 1.67× slippage
multiplier is **reasonable under a square-root impact model and optimistic under a linear one**.

## 5. Why gross is 0.38 with 8 positions: the funnel
Per-name weight is the 5% cap on every entry (4.7% achieved after lot flooring) in both studies, so
gross is 0.047 × positions held, and positions held is entries per day × hold length. Per bar,
averaged over 2,406 bars (read-only diagnostic in `fut_backtest.py`, `diagnostics/funnel.parquet`):

| step | names per bar |
|---|---|
| universe | 120.0 |
| complete 60-day window | 118.7 |
| fewer than 30 observations | 0.3 |
| OU gate reject (b outside 0–1) | 0.00 |
| valid s-score | 119.7 |
| **\|s\| ≥ 2** | **2.93** |
| of which already held or exiting | 1.11 |
| long candidates / short candidates | 0.88 / 0.94 |
| **pairs = min(L, S)** | **0.41** |
| lost to pairing | 0.99 |
| lost to the 20-per-side slot cap | 0.00 |
| entries decided / opened | 0.83 / 0.83 |
| skipped for lot reasons | 0.00 |
| bars with the gross budget exhausted | 0 |

Two constraints bind and nothing else does. **Signal supply**: only 2.4% of names sit beyond ±2 on
a given day (the 90th percentile of |s| is 1.5). **Pairing**: of 4,380 candidates over ten years,
**2,388 (54.5%) were discarded by min(longs, shorts)** — one-sided bars are the norm (572 bars with
shorts and no longs, 552 with longs and no shorts, 845 with both). The discard is symmetric (1,123
longs, 1,265 shorts), so it does not bias the side. The OU gate rejects nothing, lot limits skipped
one entry in ten years, the gross budget was never exhausted, and inverse-vol sizing never operates
because the cap binds first. The low gross measures the pairing rule, not the opportunity set.

## 6. Is it the same strategy as cash? Yes, at a different universe size
The cash signal code run on the cash data (`diag_cash_compare.py`):

| | cash data, top 200 of 435 (the cash study) | cash data, F&O members, top 120 | futures |
|---|---|---|---|
| \|s\| ≥ 2 per bar | 4.87 | 2.97 | 2.93 |
| OU-gate rejects per bar | 0.01 | 0.00 | 0.00 |
| per-name weight at entry | 0.047 | — | 0.047 |
| hold, bars fill to fill | 9.76 | — | 9.74 |
| entries per year | 361 | — | 199 |
| average positions | 14.6 | — | 8.0 |
| average gross | 0.67 | — | 0.38 |

Signal density is identical per name; positions and gross scale with universe size (8.0/14.6 = 0.55
≈ 120/200). The comparison between the two studies is clean up to that scaling.

## 7. Out-of-sample residual-reversion test, and what it does to the cash claim
Same construction as the cash study (factors, betas, standardisation and directions frozen at T,
strictly forward), on the futures returns with the top-120-by-OI universe. **2015–2024, 117 monthly
dates, 13,868 obs** (`fut_oos_test.py`, `oos_results.csv`):

| horizon | pooled r | Fama–MacBeth r | Fama–MacBeth t | hit% (\|s\|≥2) | short fwd | long fwd |
|---|---|---|---|---|---|---|
| 1d | −0.0144 | −0.0146 | −1.41 | 51.9% | +0.056% | +0.116% |
| 3d | −0.0042 | −0.0065 | −0.60 | 47.9% | +0.263% | −0.056% |
| 5d | +0.0055 | +0.0050 | +0.46 | 49.9% | +0.175% | +0.193% |
| 10d | −0.0016 | −0.0034 | −0.32 | 54.0% | −0.101% | +0.530% |

Insignificant at every horizon; 2019–2024 (72 dates) is the same: 1d t −1.48, others ≈ 0.

The cash study reported FM t of −2.3 to −2.8 at every horizon on a universe of 435 *current*
constituents applied backwards. The same test, code unchanged, re-run on the cash data with each
date's cross-section restricted to names **F&O-listed on that date** (`diag_cash_oos_pit.py`):

| universe | 1d | 3d | 5d | 10d |
|---|---|---|---|---|
| cash, original: top 200 of 435 (survivorship-biased) | −2.35 | −2.75 | −2.25 | −2.71 |
| cash, F&O members with cash data, top 120 by turnover | −2.52 | −1.84 | −1.18 | −1.72 |
| cash, F&O members with cash data, all (~145/day) | −2.18 | −1.10 | −1.00 | −1.70 |
| futures, genuinely point-in-time | −1.41 | −0.60 | +0.46 | −0.32 |

**The cash study's multi-horizon significance does not survive a point-in-time universe. Only the
1-day horizon holds, and only among surviving names**: 85 of the 329 F&O names — essentially every
one that was delisted or renamed (DHFL, JETAIRWAYS, INFIBEAM, PCJEWELLER, HDFC, CADILAHC, …) — have
no cash series at all, so the restriction removes the selection bias but cannot add the names that
died. The futures universe contains them, and its numbers are consistent with what remains once
they are in. The cash FINDINGS has been annotated accordingly.

## 8. Basis and carry are not the explanation
Over 417,156 clean name-days the identity futures return = cash return + basis change holds to
2e-15 (`diag_basis.py`). The basis change is **1.9% of futures-return variance** for the median name
(p90 5.1%), and regressing the in-window PCA residual of the futures returns on it gives a median
**R² of 0.2%** across 117 monthly windows. The basis change is itself strongly mean-reverting
(lag-1 autocorrelation −0.33), which if anything would manufacture a little spurious reversion.
Basis contamination should not be offered as a reason the signal is weaker on futures.

## 9. The corporate-action bug (found in diagnosis, fixed, numbers above are post-fix)
The signal series adjusts for corporate actions and masks the ex-date; the position ledger did not.
A held position crossing an ex-date had the raw contract price move booked against it: a short in
IOC across the 1:1 bonus of 2018-03-15 booked +₹26.8 lakh, a long in JUBLFOOD across the split of
2018-06-21 booked −₹26.4 lakh, a long in NMDC across the 2:1 bonus of 2024-12-27 carried a −67%
mark into the final equity, and shorts across exchange-adjusted dividends and demergers (CESC,
MOTHERSUMI, PEL, GAIL, NTPC, IDFC, TATASTEEL, OFSS) booked the adjustment as profit. Fourteen
positions were affected in ten years. It was found by checking, day by day, that the realised
position return equals the signal return on every held name-day (it now does on all 13,955
non-roll, non-ex-date days).

Fix (`fut_ledger.adjust`, applied as step 0 of `fut_execute.execute_day`, before anything trades on
the ex-date): a share-count event (bonus, split, rights, consolidation) scales the entry and mark
price by the factor and the lot by its reciprocal, using the published lot where the data carries one
and lot/f for the derived pre-2024 lots; a dividend or demerger adjustment reduces the reference
price by the rupee amount the exchange subtracts and leaves the lot alone. Both branches keep
unrealised P&L continuous at the moment of adjustment (asserted in code). `verify_ca_fix.py` replays
the cases (`diagnostics/ca_fix_six_cases.csv`). Effect on the front-only study: full-cost total
+10.7% → +11.8%, Sharpe +0.24 → +0.28, 2022 +2.3% → −0.8%, 2024 −3.2% → +0.4%. Pre-fix outputs are
kept in `diagnostics/*_prefix.*`.

**Bug 8 (final audit, fixed).** The dividend-text parser matched only a `DIV` token followed within
six characters by an amount, so NSE's newer abbreviations `INTDV`/`SPDV` (no `DIV` token) and long
forms such as `INTDIV - RS. 2.50 PER SH` were classified as non-dividends. Of 33 such texts on F&O
names, two were at or above the bar in force and the tape shows the exchange adjusted them —
NATIONALUM 2023-03-21 (3.1%) and HEROMOTOCO 2024-02-21 (2.1%) — and the signal series booked the raw
drop (−0.5% and −2.8% where +2.7% and −0.6% were right). Two more were recovered because the parser
now sums every amount in the dividend segments of a text (ITC 2023-05-30 `DIV- 6.75 SPLDV- 2.75`,
TCS 2023-01-16 `INTDIV- RS 8 SPLDV RS 67`, both 2.3%, previously read as the first amount only). The
other 31 recovered texts are below the bar in force (one, ITC 2016, is above the 2% floor but under
the 10% bar of its time). No position was held across any of the four. A first version of the fix
over-read a consolidation ratio (`DIV-0.06/CON RE 1TO RS10` → ₹10) and produced two fake ZEEL
adjustments before amounts were restricted to the `/`-separated segments that carry a dividend token;
the event-level suite (§11) now covers both text forms. Effect on the study: full-cost +13.62% →
+13.65%, Sharpe unchanged at +0.32; zero-cost +27.31% → +27.02% through a slightly different fill
path (5,221 → 5,231 fills; only 2023 and 2024 change). Pre-fix outputs in `diagnostics/pre_bug8/`.

**Ninth audit (final verification pass): two corrections applied.** (i) The Sharpe standard error had
been computed as √((1+S²/2)/years) with the *annualised* Sharpe in Lo's correction term, where the
per-period Sharpe belongs; the record showed ±0.33 (full) and ±0.35 (zero), the correct figures are
±0.32 and ±0.32. Direction conservative, t-statistics unaffected; the number is now produced by
`run_futures_study.py` (`sharpe_se_lo2002`). (ii) One F&O session, **2021-03-30**, was silently
missing from the whole study: both NSE bhavcopy URLs return 404 for that day although the exchange
was open (the cash file exists), the downloader wrote a holiday marker, and because the calendar is
defined by the files that parsed, no downstream check could see that every name's 2021-03-31 return
spanned two sessions. The day is recovered from NSE's market-activity archive
(`archives/fo/mkt/fo30032021.zip`, per-contract OHLC and open interest, no settlement price — the
close stands in — and no change in OI), `fo_build` now picks the CSV in a zip by its layout rather than
by position, and `fo_validate` has a calendar gate: every cash session must have an F&O session or an
explicit whitelisted reason. 2,468 sessions in both calendars after the fix.

**Documented, not changed: multi-row dividend aggregation.** When one ex-date carries several
dividend rows with different amounts, the total is taken as the maximum across rows, not the sum,
because NHPC's 2016-02-18 text printed one dividend as both 9.20 and 0.92 and a sum would double it.
Eight such dates have a summed yield above 2%; seven were under the bar in force and rightly left
alone; one (PETRONET 2017-09-07) was adjusted at 0.978 where the sum implies 0.966. Genuinely
ambiguous, exposure one adjustment of 1.2%, left as is.

## 10. Cash vs futures
| | Cash equity (frozen study) | Single-stock futures (this study) |
|---|---|---|
| capital | ₹10 lakh | ₹10 crore |
| universe | top 200 by turnover of 435 current constituents (survivorship-biased) | top 120 by OI of a point-in-time F&O universe (329 names, 292 retained) |
| instrument | adjusted close series, integer shares | real front/next-month contracts, whole lots, monthly rolls |
| costs assumed | STT 0.1%/side delivery + 0.10%/side slippage (~42 bps RT) | futures schedule + 0.03%/side slippage (4.4 bps/side, ~9 bps RT) |
| total return, full cost | **−32.9%** | **+13.7%** |
| Sharpe, full cost | **−0.56** | **+0.32 ± 0.32** |
| max drawdown, full cost | −40.8% | −13.0% |
| total return, zero cost | +32.9% | +27.0% |
| Sharpe, zero cost | +0.46 | +0.57 |
| average gross exposure | 0.67 | 0.38 (universe 120 vs 200; §6) |
| trades / fills | 7,205 | 5,231 (1,988 round trips + 625 rolls) |
| OOS Fama–MacBeth t, 1d / 3d / 5d / 10d | −2.35 / −2.75 / −2.25 / −2.71 (survivorship-biased; §7) | −1.41 / −0.60 / +0.46 / −0.32 |
| break-even cost | ~18–20 bps RT (below the 22 bps STT floor) | ~16 bps RT (above the ~9 bps assumed) |

## 11. Harness validation (Stage 3, all passed, re-run after each change)
**Synthetic-price suite** (`fut_validation.py`, monthly contracts with lots and rolls active, zero
cost): strong reversion +117% / Sharpe +4.2; random walk +1% / +0.08; momentum −41% / −1.8; momentum
with the signal negated +62% / +1.7.

**Event-level suite** (`fut_validation_events.py`, added after the final audit). Every one of the
eight bugs found in this project lived in code the synthetic-price suite structurally cannot reach,
because that data has no corporate actions, delistings, re-listings, vanished contracts or data
boundaries. This suite builds a small synthetic F&O world from a clean return path per name, derives
a raw tape with scripted events, runs it through the REAL Stage 1 corporate-action and chain code and
the REAL execution layer, and asserts 73 known answers stated in advance: a 1:1 bonus and a 10→5
split with lot re-denomination (factor from text, return on the ex-date equals the clean return, a
held position's P&L equals notional × clean return, lot doubled); an extraordinary dividend written
`INTDV/SPDV - RS 8 & 4` and another written `INTDIV - RS. 2.50 PER SH` (the bug-8 forms) with a tape
that says the exchange adjusted (factor 1 − D/prev, P&L equals the exchange's additive rule); an
ordinary dividend written `DIVIDEND- 6% PER ANNUM` (must not be read as ₹6; no factor; futures
pre-discount it) and a 3% dividend whose tape says unadjusted (no factor although 3% > 2%); a
demerger with unknown factor (measured ratio, masked, NaN return, documented P&L treatment); a name
delisted while held (force-closed at exactly 2 sessions to its last expiry, nothing traded after); a
name that leaves the universe and re-enters at 3× (chain break, no fake +200%); a contract that
vanishes before expiry while held (closed on the first missing session at the last settle, tagged);
a position held across the final bar with its contract expiring after the data (rolled at 2 sessions,
never force-closed, tradable on the last session); a routine 5% lot revision with no price effect (no
adjustment); every far-month listing showing ChgInOI == OI on its first day (no detector hit); and
costs charged once per leg with adjustments free. 73 of 73 pass; the Stage 1 layer produces exactly
the five scripted adjustments and nothing else. Two cases assert documented choices rather than
unique right answers and say so (the demerger's masked-day move is absorbed into the measured factor;
the vanished contract is closed at its last settle). Building the suite exposed one implicit
dependency worth recording: the ex-date adjustment and the vanished-contract close read the
position's `mark_price` as "the previous settle", which holds only because the loop marks every bar;
a driver that skipped marking got wrong rupee amounts. Not changed, but a robustness improvement
would fetch the previous settle from the market explicitly. Independent daily equity reconstruction from the trade log, 2015–2024:
relative difference 2e-15. Look-ahead over all 5,231 fills: none after expiry, none on a non-front
contract except the 180 rule-conforming back-month entries, no next month opened early otherwise,
every OPEN one session after its decision bar, every roll at exactly 2 sessions to expiry. Stage 1
adjustments rebuilt from data truncated at 2018-06-29 and 2021-12-31 are byte-identical to the
full-sample series. Zero-cost mode asserted to charge exactly 0.0.

## 12. Decay or variance? The sub-period evidence
The question: is the concentration of profit in 2018 and the losses of 2021–2024 a genuine edge that
has decayed, or noise around an edge that was never reliably there? Three pieces of evidence
(`diag_decay.py`, `diagnostics/decay_*.csv`).

**(a) OOS Fama–MacBeth by half.** Per-date cross-sectional correlations aggregated within
2015–2018 (45 monthly dates) and 2019–2024 (72), with a Welch t-test on the difference:

| series | horizon | 2015–2018 r / t / hit | 2019–2024 r / t / hit | Welch t (early − late) |
|---|---|---|---|---|
| futures, PIT top-120 | 1d | −0.009 / −0.47 / 57% | −0.018 / −1.48 / 49% | +0.44 |
| | 3d | +0.012 / +0.65 / 47% | −0.018 / −1.33 / 48% | +1.32 |
| | 5d | +0.025 / +1.50 / 53% | −0.008 / −0.56 / 48% | +1.51 |
| | 10d | +0.002 / +0.12 / 58% | −0.007 / −0.50 / 52% | +0.41 |
| cash, F&O members top-120 (survivors) | 1d | −0.025 / −1.59 / 59% | −0.028 / −1.95 / 53% | +0.14 |
| | 3d | −0.008 / −0.49 / 54% | −0.025 / −1.92 / 48% | +0.87 |
| | 5d | +0.001 / +0.04 / 54% | −0.020 / −1.44 / 50% | +1.02 |
| | 10d | −0.013 / −0.68 / 58% | −0.023 / −1.65 / 55% | +0.46 |
| cash, original top-200 (biased) | 1d | −0.019 / −1.57 / 55% | −0.022 / −1.77 / 54% | +0.13 |
| | 3d | −0.025 / −1.89 / 62% | −0.022 / −2.01 / 51% | −0.21 |
| | 5d | −0.022 / −1.66 / 61% | −0.018 / −1.57 / 52% | −0.21 |
| | 10d | −0.037 / −2.52 / 60% | −0.018 / −1.49 / 53% | −1.01 |

On futures the reversion signal is **absent in 2015–2018** — the 3- and 5-day correlations have the
wrong sign — and **weakly present in 2019–2024** (t −1.5 and −1.3 at 1 and 3 days, still not
significant). The point-in-time cash series shows the same direction. Only the survivorship-biased
original cash series has any early-period signal, and even there the halves do not differ. No Welch
test reaches |t| = 1.6. The shape Avellaneda–Lee documented (a stronger early sample decaying into a
weaker late one) is not in this data; the OOS signal is, if anything, stronger late.

**(b) By calendar year** (12 dates each — reported for completeness, too few dates to be
meaningful individually): on futures the 1-day FM t ranges from −2.7 (2019) to +0.9 (2017) with no
trend; **2018, the backtest's best year, shows no OOS reversion at all** (t −0.3 / −0.6 / −0.7 /
−1.4 at 1 / 3 / 5 / 10 days), while 2019, a flat backtest year, is the strongest reversion year in
every series (futures t −2.7 at 1d; cash PIT −3.9). The backtest's yearly P&L does not track the
yearly strength of the signal it trades.

**(c) What 2018 actually was.** Per closed position, zero cost, front-only baseline: 2018 had 230
positions averaging +₹0.65 lakh (t 2.0 within the year); every other year's mean is between
−₹0.28 and +₹0.29 lakh (|t| ≤ 1.4), and pooled over 1,989 positions the mean is +₹0.13 lakh with
t 1.4. **One position — a short in INFIBEAM through its −71% day of 2018-09-28 — contributed
₹45.6 lakh, 30% of the year; the ten best positions were 77% of it.** Assigning the 1,989 positions
to years at random, the best year's total reaches 2018's in 14% of draws.

**Conclusion: the evidence supports "the edge was never reliably there and 2018 is variance", not
"genuinely present early and decayed".** The sub-samples are too small to *prove* a difference
between halves in either direction (that part of the question is a null answer), but the sign of
the evidence is the opposite of decay: the OOS signal is weaker early than late, 2018's profit is
not accompanied by a 2018 signal, and it is concentrated in a handful of positions including one
crash. Nothing here should be read as an edge that used to work.

### 12b. Does the s-score predict better once the residual has turned? (supervisor's hypothesis)
**Hypothesis** (from supervisor discussion): entries at |s| ≥ 2 include names whose residual is
still extending away from zero — catching a falling knife — and the signal might be more reliable
once the residual has turned. **Method** (`diag_turning.py`, signal test only, no backtest): the OOS
Fama–MacBeth construction of §7, with each date's cross-section split by the change in |s| over the
previous k sessions, k ∈ {1, 3, 5} fixed a priori, using the s-score the strategy would have seen
k sessions earlier (its own window): *extending* if |s| rose, *turning* if it fell. Per group,
horizon and k: Fama–MacBeth r and t over per-date correlations (dates with ≥10 names in the group),
hit rate on |s| ≥ 2, Welch and paired t on the difference. Run on futures (PIT top-120) and on the
cash data restricted to F&O members (top 120). Note that these are the same names on the same
dates, so agreement between them is one piece of evidence, not two.

Full cross-section, Fama–MacBeth t (turning / extending), Welch t of the difference:

| k | h | futures: turning | extending | Welch | cash PIT: turning | extending | Welch |
|---|---|---|---|---|---|---|---|
| 1 | 1d | −1.77 | −0.30 | −1.00 | −1.68 | −2.01 | +0.28 |
| 1 | 10d | −0.95 | +0.22 | −0.82 | −1.99 | −1.08 | −0.66 |
| 3 | 1d | −1.25 | −0.86 | −0.35 | −2.22 | −1.88 | −0.29 |
| 3 | 3d | −1.76 | +0.30 | −1.45 | −2.52 | −0.64 | −1.18 |
| 3 | 10d | −2.32 | +0.70 | **−2.18** | −3.04 | −0.49 | −1.76 |
| 5 | 1d | −1.63 | −0.54 | −0.70 | −1.69 | −2.41 | +0.58 |
| 5 | 3d | −2.34 | +1.01 | **−2.31** | −2.25 | −1.12 | −0.51 |
| 5 | 5d | −1.71 | +1.73 | **−2.44** | −2.36 | −0.22 | −1.41 |
| 5 | 10d | −3.40 | +1.38 | **−3.39** | −4.74 | −0.10 | **−2.94** |

(full table: `diagnostics/turning_split.csv`; groups are ~50/50 of the cross-section, 116 dates.)
Across the whole cross-section the pattern is real-looking: at k = 5 the turning half carries all
of the reversion (futures t −3.4 at 10 days, cash −4.7) and the extending half none or the wrong
sign, and the gap widens with k and horizon; at k = 1 and at 1 day there is no difference.

At the entry threshold, where a filter would act, the picture changes. Names beyond |s| ≥ 2 that
are *turning* are rare, because a name that has been falling toward zero for k sessions seldom
still exceeds 2:

| k | futures: turning n / hit (1d, 3d, 5d, 10d) | extending n / hit | cash PIT: turning n / hit | extending n / hit |
|---|---|---|---|---|
| 1 | 76 / 59%, 45%, 49%, 54% | 272 / 50%, 49%, 50%, 54% | 88 / 55%, 50%, 51%, 57% | 280 / 55%, 50%, 51%, 56% |
| 3 | 36 / 50%, 47%, 61%, 53% | 312 / 52%, 48%, 48%, 54% | 34 / 71%, 68%, 62%, 65% | 334 / 53%, 49%, 50%, 55% |
| 5 | **13** / 46%, 69%, 62%, 77% | 335 / 52%, 47%, 49%, 53% | **19** / 79%, 74%, 63%, 74% | 349 / 53%, 49%, 51%, 55% |

The |s| ≥ 3 tail, where the falling-knife concern is sharpest, contains 0–2 turning names per k in
ten years on either series; nothing can be said about it (`diagnostics/turning_tail.csv`).

**Conclusion.** In the full cross-section the evidence does support the hypothesis: the s-score's
forward predictive power sits in names whose residual has already turned, and is absent in names
still extending. But at the ±2 entry threshold the split samples are too small to distinguish —
13 to 76 turning entries in ten years on futures, 0–2 in the tail — and the extending entries, which
are 90–96% of what the strategy actually trades, show hit rates of ~50%, i.e. no edge. So the
practical content is negative twice over: the existing entries are not predictive, and the subset
the hypothesis would keep is too small to trade or to test (one to eight entries a year). **No
backtest variant was run**: at the k the signal test favours (5) the filtered strategy would make
13 trades in ten years, and choosing a smaller k to get a tradable count would be exactly the
variant-hunting this test was designed to exclude. Verdict unchanged.

### 12c. Re-entry after a stop-loss: does the strategy get stopped repeatedly on a collapsing name? (supervisor's concern)
**Concern** (from supervisor discussion): a name crosses −2, we go long, it keeps falling to −3, the
stop fires, but s is still beyond −2 so the entry gate lets us straight back in, and we are stopped
again on the same collapsing name. **Measured** from the headline trade log, 2015–2024
(`diag_stop_reentry.py`, `diagnostics/stop_reentry*.csv`); a position's P&L is summed over all its
legs, before costs.

- **65 stop exits** among 1,986 closed positions (3.3%). 61 of the 65 names were re-entered in the
  same direction at some later date; **28 on the very next session** (the gate let them straight
  back in), 34 within 5 sessions, 35 within 15, 36 within 60, 25 more than 60 sessions later.
- **The repeated-stop scenario never occurred.** Not one of the 61 re-entries was stopped again
  (they exited on profit or time), and there is no name with three or more consecutive
  same-direction attempts. Every chain is exactly two attempts long.
- The 35 re-entries within 15 sessions: their stopped first attempts lost ₹64.5 lakh; the
  re-entries themselves lost **₹19.5 lakh in total** (mean −₹0.56 lakh, 18 of 35 negative). Over
  all 61 same-direction re-entries the mean P&L is −₹0.35 lakh against +₹0.13 lakh for the average
  closed position: the second attempt is worse on average, but on 61 draws with a per-position
  standard deviation of ₹4.2 lakh the difference is within noise (t ≈ −0.9).
- **Money involved:** second-and-later attempts sum to **−₹21.2 lakh over ten years, −0.21% of
  capital**; their gross losses are ₹92.6 lakh, **3.6% of all position losses**. A lockout that
  removed all 35 quick re-entries would have added at most +0.2% of capital before costs over the
  decade, and those slots would have been filled by other entries of unknown sign.
- Incidental observation: the |s| ≥ 3 "stop" barrier is symmetric, so it also fires on a position
  whose s has moved *through* zero to the far side — 12 of the 65 "stops" closed at a profit
  (L&TFH +₹2.5 lakh, TRENT +₹1.4 lakh). It is a barrier on |s|, not a loss limit.

**Conclusion: this barely happens.** The gate does let stopped names straight back in (28 times in
ten years), but the re-entries were never stopped again and the whole bucket is worth −0.2% of
capital. **The lockout variant (block re-entry until s re-crosses the threshold or M = 15 sessions
pass) was not run**, per the pre-agreed rule that it runs only if the measurement shows a material
pattern. Verdict unchanged.

## 13. Caveats
- **Capital.** ₹10 crore was chosen because one lot of a top-120 name is ₹3–8 lakh: under the 5%
  per-name cap, ₹10 lakh of capital would skip nearly the whole universe. **Not a retail-accessible
  result.** At ₹10 crore lot flooring still costs 6% of target weight on average.
- **Back-month slippage is an assumption** (0.05% vs 0.03%), judged against measured liquidity in
  §4b: reasonable under square-root impact, optimistic under linear. Removing the assumption
  entirely changes the headline by 0.4 points.
- **Basis and carry are not modelled in the signal** (the execution layer pays the basis for real
  at every roll); §8 shows this is immaterial to the residual.
- **Corporate-action calendar gap.** The `Bc` file is a rolling window of upcoming ex-dates and the
  archive starts 2015-01-01, so actions with ex-dates in the first weeks of January 2015 announced
  in December 2014 can be missing; only the physical detectors cover that window.
- **Names excluded for insufficient history:** 37 (34 from the Nov-2024 F&O expansion, plus
  RANBAXY, RDEL, SSLT). Ten names carrying a >40% clean daily move were kept: their cash prices
  moved the same way.
- **Vanished contracts.** A contract that disappears from the tape before expiry (symbol change,
  merger) is closed at its last settlement price and tagged; in reality the position would convert
  into the successor name. One instance: INFRATEL, 2020-12-18, −₹3.88 lakh.
- **Zero-cost and full-cost runs differ in lot counts** because sizing is equity-based and equity
  differs by the costs paid: 2,442 of 5,221 fills differ by up to 12 lots, the sequence of fills is
  identical, daily returns correlate 0.997.
- **Cost model.** Slippage is flat with no size or liquidity dependence and is 69% of all costs;
  §4 shows the sign of the result depends on it. STT is applied at 0.02% throughout although it was
  0.01% before October 2024 (conservative). SEBI turnover fee and clearing charges (each
  < 0.0002%) are not charged. Fills assume executability at the printed open in the size traded.
- **Lot sizes before 2024-07-08 are derived** per contract from turnover and OI (99.75% within 2% of
  the published UDiFF lot where both exist); the estimate uses the contract's whole life and does
  not change on an ex-date, which is why the corporate-action re-denomination is inferred from the
  factor for those contracts.
- **Margin** is a 25%-of-notional proxy, not SPAN + exposure margin, and is not charged interest.
- **Pairing.** The min(longs, shorts) rule is the cash study's and was kept deliberately; §5 shows
  it discards more than half the qualifying signals. Relaxing it would be a strategy change, not
  tested here.
- Same limitations as the cash study on dynamics: daily marks, no intraday, no impact curve.

## 13b. Known limitations recorded by the final audits (not fixed)
- **Corporate-action price basis across the seam.** `fo_corpactions` measures event ratios on the
  close (settle as fallback) while the chain and the ledger work on the settlement price. For text
  and dividend factors the two agree exactly (all 100 extraordinary dividends: 0.0000 difference in
  the rupee amount). For the 57 measured-ratio events the close and settle ratios differ by up to
  1.2% (ADANIENT 2018-04-05); five such events were crossed while held, and the P&L-continuity error
  from the mismatch totals **₹12,004** (largest ₹6,646, MOTHERSUMI 2022-01-14). Fix would be to
  measure the ratio on the chain's basis.
- **Implicit previous-settle dependency.** `Ledger.adjust` and the vanished-contract close read the
  position's `mark_price` as the previous settlement price, which holds only because the loop marks
  every bar (a scripted driver that skipped marking produced wrong rupee amounts while the event
  suite was being built). Fix would be to fetch the previous session's settle from the market
  explicitly in both paths. No effect on current results.
- **Event cases the suite does not cover** (present in the real data, untested): rights issues with
  a published lot change; a bonus with no ratio in the text; an expiry moved off Thursday by a
  holiday; the 2024-07-08 format cutover with derived versus published lots; a symbol rename with a
  successor; a corporate action on a roll day or an entry day; two calendar rows on one ex-date; a
  text factor contradicted by the tape; a listed contract that never trades; a partial-history
  entry; the 20-per-side slot cap and gross budget binding; a chain break on the first day of the
  sample; multiple simultaneous positions and the pairing rule; and the OOS test itself, which has no
  known-answer test.

## 14. Verdict
Unchanged by the entry rule. The port succeeds as an engineering result and fails as an investment
result. Futures remove the cost floor that killed the cash version: the identical strategy that
returned −32.9% (Sharpe −0.56) under cash-delivery costs returns +13.7% (Sharpe +0.32) under a
realistic futures cost schedule with the entry rule. But +0.32 ± 0.32 over ten years is not
distinguishable from zero; the return is one good year (2018), concentrated in a handful of
positions, and dies between 5 and 8 bps of slippage against 3 assumed; the bare signal reaches only
t 1.8; the out-of-sample Fama–MacBeth test is insignificant at every horizon on a point-in-time
universe and was, if anything, weaker in the early years than the late ones; and the same re-run
shows the cash study's own multi-horizon significance was carried by survivorship. **There is no
statistically supported evidence of a tradable residual-reversion edge in NSE single-stock futures
with this signal, and the earlier cash-equity evidence for the edge is weaker than that study
stated.**

---

### Files in this folder
- `FINDINGS.md` — this record
- `equity_curve.csv`, `equity_curve_zerocost.csv` — daily equity, ₹ (headline: entry rule N = 5)
- `trades.csv`, `trades_zerocost.csv` — every fill and corporate-action adjustment: contract
  expiry, contract choice (front / back-month), sessions to expiry at fill, lots, lot size, price,
  fill basis, cost by component, target vs achieved weight, realized P&L, reason
- `exposure.csv`, `exposure_zerocost.csv` — daily positions, gross target/entry/marked, net, margin
- `skips.csv` · `signal_stats.csv` · `cost_breakdown.json` · `summary.json`
- `oos_results.csv` (2015–2024), `oos_results_2019_2024.csv`
- `diagnostics/` — `baseline_front_only/` (the run before the entry rule), `pre_bug8/` (before the parser fix),
  `entry_rule_variants/{same_slippage,N8,N12}/`, `backmonth_liquidity*.csv`, `decay_*.csv`, `turning_*.csv`, `stop_reentry*.csv`,
  `slippage_sensitivity.json`, `equity_slip_*bps.csv`, `cash_oos_pit.csv`,
  `basis_decomposition_by_name.csv`, `basis_residual_r2_by_month.csv`, `ca_fix_six_cases.csv`,
  pre-fix outputs `*_prefix.*`; per-bar tables are parquet (gitignored, regenerated by the scripts)

### Reproduce (from `futures/`, poetry Python 3.11 env)
- Data: `python fo_download.py --what {fo,cm,pr}` → `fo_build.py --what {fo,cm}` → `fo_corpactions.py`
  → `fo_continuous.py` → `fo_validate.py`
- Validation: `python fut_validation.py` · `python fut_validation_events.py` · `python fut_checks.py`
  (`fo_validate.py` now also gates the F&O calendar against the cash calendar)
- Study: `python run_futures_study.py` (headline) · `--tag X --entry-min-dte N --back-slippage s`
  for a variant · OOS: `python fut_oos_test.py --start-year 2015 --out ../results_futures_2015_2024/oos_results.csv`
- Diagnosis: `diag_funnel_slippage.py`, `diag_cash_compare.py`, `diag_cash_oos_pit.py`,
  `diag_basis.py`, `verify_ca_fix.py`, `diag_backmonth_liquidity.py`, `diag_decay.py`, `diag_turning.py`,
  `diag_stop_reentry.py`
