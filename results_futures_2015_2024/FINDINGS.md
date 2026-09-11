# PCA-Residual Statistical Arbitrage — Single-Stock Futures Study (NSE, 2015–2024)

**STATUS: complete, 2026-09-11.** The cash-equity strategy (frozen in `results_cash_equity_2015_2024/`)
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
- Corporate actions: 260 adjustments (105 from calendar text, 96 extraordinary dividends confirmed
  by the ex-date basis jump, 57 measured-and-masked, 2 rights from a published lot step), 57
  name-days masked. Ordinary dividends are deliberately not adjusted.
- Stage 1 gate: return correlation with cash 0.990 median across names; basis +0.31% median; roll
  test +0.31 bps; 0 unexplained NaNs. See `futures/README.md`.

## 3. Headline result — 2015-03-31 → 2024-12-31, 2,407 bars, ₹10 crore
| metric | **Full cost** | **Zero cost** |
|---|---|---|
| total return | **+13.6%** | **+27.3%** |
| CAGR | +1.32% | +2.51% |
| annualised Sharpe | **+0.32** | **+0.57** |
| Sharpe standard error (≈√((1+S²/2)/years)) | ±0.33 | ±0.35 |
| t-stat of mean daily return | 0.98 | 1.77 |
| annualised volatility | 4.6% | 4.6% |
| max drawdown | **−12.9%** | **−10.7%** |
| total costs | **₹1,23,44,940** | ₹0.00 |
| costs per year, % of capital | 1.27% | 0 |
| fills (round trips + roll legs) | 5,221 (1,986 + 2×622, 1 forced) | 5,221 |
| back-month entries (rule, §1) | 184 of 1,991 (9.2%) | 184 |
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
| 2023 | −6.0% | −4.3% | 13.6 | 0.41 | 8.7 |
| 2024 | +0.4% | +1.6% | 12.1 | 0.38 | 8.2 |

**2018 alone is most of the result** (+11.1 of +13.6 points at full cost); 2019–2024 sum to −6.3
points at full cost and +0.4 before costs. §12 examines whether that is decay or noise.

## 4. Costs, and how fragile the positive return is
Headline run, ₹2,900 crore traded notional, **4.44 bps per side of notional**:

| component | ₹ | share |
|---|---|---|
| slippage (0.03% front / 0.05% back-month entry) | 85,36,730 | 69.2% |
| STT (0.02% sell side) | 27,83,986 | 22.6% |
| exchange (0.0019%) | 5,27,853 | 4.3% |
| stamp (0.002% buy side) | 2,77,780 | 2.3% |
| GST | 1,14,155 | 0.9% |
| brokerage (₹20/order) | 1,04,420 | 0.8% |
| **total** | **1,23,44,940** | |
| of which on the 622 rolls (1,244 legs) | 28,86,000 | **23.4%** |
| of which on the 184 back-month entries | 6,16,970 | 5.0% |

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
| **N = 5, back-month slippage 0.05% (headline)** | **+13.6%** | **+0.32** | **123.4** | **23.4%** | **622** | **184** | **6.2** |
| N = 5, back-month slippage 0.03% (same as front) | +14.1% | +0.32 | 121.8 | 23.8% | 622 | 184 | 4.2 |
| N = 8, 0.05% (sensitivity) | +14.0% | +0.32 | 116.5 | 16.5% | 418 | 437 | 14.9 |
| N = 12, 0.05% (sensitivity) | +13.2% | +0.31 | 109.5 | 7.3% | 170 | 836 | 28.3 |
| baseline, zero cost | +25.6% | +0.54 | 0 | — | 798 | 0 | 0 |
| N = 5, zero cost | +27.3% | +0.57 | 0 | — | 622 | 184 | 0 |

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

## 10. Cash vs futures
| | Cash equity (frozen study) | Single-stock futures (this study) |
|---|---|---|
| capital | ₹10 lakh | ₹10 crore |
| universe | top 200 by turnover of 435 current constituents (survivorship-biased) | top 120 by OI of a point-in-time F&O universe (329 names, 292 retained) |
| instrument | adjusted close series, integer shares | real front/next-month contracts, whole lots, monthly rolls |
| costs assumed | STT 0.1%/side delivery + 0.10%/side slippage (~42 bps RT) | futures schedule + 0.03%/side slippage (4.4 bps/side, ~9 bps RT) |
| total return, full cost | **−32.9%** | **+13.6%** |
| Sharpe, full cost | **−0.56** | **+0.32 ± 0.33** |
| max drawdown, full cost | −40.8% | −12.9% |
| total return, zero cost | +32.9% | +27.3% |
| Sharpe, zero cost | +0.46 | +0.57 |
| average gross exposure | 0.67 | 0.38 (universe 120 vs 200; §6) |
| trades / fills | 7,205 | 5,221 (1,986 round trips + 622 rolls) |
| OOS Fama–MacBeth t, 1d / 3d / 5d / 10d | −2.35 / −2.75 / −2.25 / −2.71 (survivorship-biased; §7) | −1.41 / −0.60 / +0.46 / −0.32 |
| break-even cost | ~18–20 bps RT (below the 22 bps STT floor) | ~16 bps RT (above the ~9 bps assumed) |

## 11. Harness validation (Stage 3, all passed, re-run after each change)
Known-answer suite on synthetic monthly contracts with lots and rolls active, zero cost: strong
reversion +117% / Sharpe +4.2; random walk +1% / +0.08; momentum −41% / −1.8; momentum with the
signal negated +62% / +1.7. Independent daily equity reconstruction from the trade log, 2015–2024:
relative difference 2e-15. Look-ahead over all 5,221 fills: none after expiry, none on a non-front
contract except the 184 rule-conforming back-month entries, no next month opened early otherwise,
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

## 14. Verdict
Unchanged by the entry rule. The port succeeds as an engineering result and fails as an investment
result. Futures remove the cost floor that killed the cash version: the identical strategy that
returned −32.9% (Sharpe −0.56) under cash-delivery costs returns +13.6% (Sharpe +0.32) under a
realistic futures cost schedule with the entry rule. But +0.32 ± 0.33 over ten years is not
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
- `diagnostics/` — `baseline_front_only/` (the run before the entry rule),
  `entry_rule_variants/{same_slippage,N8,N12}/`, `backmonth_liquidity*.csv`, `decay_*.csv`,
  `slippage_sensitivity.json`, `equity_slip_*bps.csv`, `cash_oos_pit.csv`,
  `basis_decomposition_by_name.csv`, `basis_residual_r2_by_month.csv`, `ca_fix_six_cases.csv`,
  pre-fix outputs `*_prefix.*`; per-bar tables are parquet (gitignored, regenerated by the scripts)

### Reproduce (from `futures/`, poetry Python 3.11 env)
- Data: `python fo_download.py --what {fo,cm,pr}` → `fo_build.py --what {fo,cm}` → `fo_corpactions.py`
  → `fo_continuous.py` → `fo_validate.py`
- Validation: `python fut_validation.py` · `python fut_checks.py`
- Study: `python run_futures_study.py` (headline) · `--tag X --entry-min-dte N --back-slippage s`
  for a variant · OOS: `python fut_oos_test.py --start-year 2015 --out ../results_futures_2015_2024/oos_results.csv`
- Diagnosis: `diag_funnel_slippage.py`, `diag_cash_compare.py`, `diag_cash_oos_pit.py`,
  `diag_basis.py`, `verify_ca_fix.py`, `diag_backmonth_liquidity.py`, `diag_decay.py`
