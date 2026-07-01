# PCA-Residual Statistical Arbitrage — Cash-Equity Study (NSE, 2015–2024)

**STATUS: FROZEN CHECKPOINT** — complete and final cash-equity result, preserved before pivoting to a futures version. Frozen 2026-06-30. Do not edit; this folder is the immutable record of the cash-equity study.

---

## 1. Strategy
PCA-residual cross-sectional mean reversion (Avellaneda–Lee style), market-neutral, NSE equities.

Pipeline (per bar): daily returns → **correlation-form PCA** (z-score each name, then SVD) → **residuals** (each return minus its factor-fitted part, intercept included) → **OU fit** on the cumulative residual → **s-score** (deviation in its own equilibrium sigmas).

Trade rule: `s ≤ −2` → **long** (cheap), `s ≥ +2` → **short** (rich). Triple-barrier exit: profit `|s| ≤ 0.5`, stop `|s| ≥ 3.0`, time stop 15 bars. Inverse-vol sizing, dollar-neutral, per-name cap 5%, **book gross capped at 1.0×**.

Timing (look-ahead-safe): signal from returns up to & including bar **T** close → execute at **T+1 open** → first mark at **T+1 close**.

Definitive config: `pca_window=60, n_factors=5, universe_size=200 (top liquid), liquidity_lookback=20, rebalance daily, initial capital ₹1,000,000`.

## 2. Data
- yfinance adjusted OHLCV, **435 NSE names** (`universe_clean.txt`), **2015-01-01 → 2024-12-30**.
- Frame: **2466 trading days × 435 stocks**. Download outcome: **310 clean / 125 flagged / 0 failed** (124 of 125 flags = a single zero-volume day over 10 yrs — benign).
- Coverage: **303 full-history** (data by 2015-01-31), **132 later IPOs** (leading NaNs, handled by the harness), **0 delistings / early-ending**.
- Frozen input copies live in `./data/` here (prices/volume/open_wide.parquet + data_quality_report.csv). **yfinance is not reproducible** — these parquets are the canonical input for this study.

## 3. Headline result — definitive backtest (2015-04 → 2024-12, 2404 bars)
| metric | **Full cost** (realistic) | **Zero cost** (bare signal) |
|---|---|---|
| total return | **−32.9%** | **+32.9%** |
| net Sharpe (ann) | **−0.56** | **+0.46** |
| max drawdown | −40.8% | −17.6% |
| total cost paid | ₹615,445 | ₹0.00 |
| avg gross exposure | 0.67 | 0.67 |
| avg net exposure | −0.016 | −0.016 |
| total trades | 7,205 | 7,229 |

Files: `equity_curve.csv` (full), `equity_curve_zerocost.csv`, `trades.csv`, `exposure.csv`.

**A real but thin reversion edge (~+3%/yr gross, Sharpe +0.46) that does NOT survive realistic NSE cash-delivery costs.**

## 4. Out-of-sample residual-reversion test (factors/betas FROZEN at T, strictly forward)
**2015–2024** — 117 monthly dates, 23,167 obs:
| horizon | pooled r | Fama-MacBeth t | hit% (\|s\|≥2) |
|---|---|---|---|
| 1d | −0.0192 | **−2.35** | 54.1% |
| 3d | −0.0211 | **−2.75** | 55.1% |
| 5d | −0.0187 | **−2.25** | 54.9% |
| 10d | −0.0276 | **−2.71** | 55.9% |

2019–2024 baseline (72 dates): 1d r = −0.0177, FM t = −1.77 (borderline / insignificant).

**Finding:** reversion is **weak (r ≈ −0.02)** but, over the full 10 years, **statistically significant at every horizon** (FM t −2.3…−2.8, hit 54–56%, direction correct: rich→down, cheap→up). The longer sample turned the earlier borderline result significant. *Real edge, small magnitude.*

## 5. Cost-sensitivity break-even (BASE config; only the cost function varied)
| cost level | ~RT bps | return | Sharpe | cost ₹ | trades |
|---|---|---|---|---|---|
| STT + 0.10%/side (full realistic) | ~42 | −32.9% | −0.56 | 615,445 | 7,205 |
| STT + 0.05%/side | ~32 | −21.1% | −0.32 | 508,657 | 7,207 |
| STT + 0.02%/side | ~26 | −12.6% | −0.16 | 436,820 | 7,213 |
| STT only (no slippage) | ~22 | −6.6% | −0.07 | 383,439 | 7,215 |
| **── break-even ≈ 18–20 bps RT ──** | | | **0.00** | | |
| Half-STT + 0.02%/side | ~15 | +2.8% | +0.08 | 293,748 | 7,223 |
| 0.05%/side only, no STT (futures-like) | ~10 | +13.3% | +0.22 | 190,668 | 7,225 |
| 0.02%/side only, no STT | ~4 | +24.6% | +0.36 | 80,029 | 7,229 |
| Truly zero (reference) | 0 | +32.9% | +0.46 | 0 | 7,229 |

**Break-even ≈ 18–20 bps all-in round-trip.** Decisive point: **NSE cash-delivery STT alone ≈ 22 bps**, and "STT only / zero slippage" is already Sharpe −0.07 — so the statutory tax floor by itself exceeds break-even. **The strategy cannot profit in cash delivery under any slippage assumption.** A futures-like regime (~10 bps) yields Sharpe +0.22. File: `cost_sensitivity_sweep.json`.

## 6. Turnover sweep (full cost; only turnover knobs varied) — no setting beats costs
12 configs (entry threshold, rebalance frequency, max-hold, profit barrier, + combos). Key rows:
| config | return | Sharpe | cost ₹ | trades | gross |
|---|---|---|---|---|---|
| profit 0.75 (best genuine trader) | −24.0% | −0.39 | 692,962 | 7,322 | 0.60 |
| entry 2.5 | −12.0% | −0.56 | 75,554 | 749 | 0.08 |
| base (e2.0/r1/h15/p0.5) | −32.9% | −0.56 | 615,445 | 7,205 | 0.67 |
| rebal 10d | −20.5% | −0.55 | 150,418 | 1,412 | 0.25 |
| hold 30d | −45.1% | −0.80 | 524,531 | 6,974 | 0.74 |
| combo e2.5/r5/h30 (degenerate) | +0.5% | +0.04 | 20,944 | 198 | 0.04 |
| combo e3.0/r10/h60/p1 (degenerate) | +0.5% | +0.49 | 434 | **4** | 0.00 |

**No configuration that stays meaningfully invested clears costs** (best genuine trader = profit 0.75 at Sharpe −0.39). The only "Sharpe>0" configs are **degenerate** — they barely trade (4 trades / 0.00 gross; 198 trades / +0.04). Reducing turnover walks the loss toward 0 *by trading less*, never producing a tradable positive. Two mechanics worth noting: a **higher** profit barrier exits *sooner* → *more* trades (not fewer); longer holds make it worse (held losers, higher gross). Files: `turnover_sweep_part0..3.json`.

## 7. Known-answer validation (`harness_validation.py`) — ALL PASS → harness TRUSTWORTHY
| test | return | Sharpe | expected | result |
|---|---|---|---|---|
| strong reversion | +129.6% | +4.28 | PROFIT | ✅ |
| random walk | +0.4% | +0.05 | ~ZERO | ✅ |
| momentum (normal) | −42.2% | −1.69 | LOSE | ✅ |
| momentum (s-score flipped) | +67.1% | +1.68 | WIN | ✅ |

Confirms: detects real edge, invents none on noise, correct direction/sign handling — with all fixes in place.

## 8. n_factors sweep — PROVENANCE CAVEAT
`sweep_nfactors.csv` was run on the **earlier 2019–2024 dataset at ZERO cost** (not the 2015–2024 frame). Results: nf=5 → Sharpe 0.04, 8 → −0.15, 10 → 0.39, 12 → −0.16, 15 → 0.15, 20 → −0.32. **Non-monotonic noise** — see §10.

## 9. P&L bug found & fixed (and gross-cap fix)
- **Realized-P&L-on-close (telescoping) bug:** closed positions' P&L was not persisted to cash, so equity "snapped back" and distorted results. **Fix** (`harness_backtest.py`): each position's signed share basis is fixed at entry (`shares_held`); on close, `realized = shares_held × (exit − entry)` is credited to `cash`; open positions are marked on the *same* share basis; `pct_change(fill_method=None)` prevents fake post-IPO returns. **Verified** by an independent equity reconstruction matching the harness to **0.000** under zero cost.
- **Gross-cap fix** (added this study, `harness_decide.decide_entries`): new entries are sized to the *remaining* budget = `target_gross − committed gross of non-exiting positions`, pinning book leverage to ≤ target (previously drifted to 3–4×). Confirmed live: avg gross 0.67 ≤ 1.0, net ≈ 0.

## 10. Overfitting traps avoided
1. **n_factors not tuned to the in-sample best.** The sweep is non-monotonic noise (nf=10 looked best at Sharpe 0.39 vs nf=5 at 0.04). Selecting nf=10 would be curve-fitting noise; the principled **nf=5** was kept.
2. **Sweep "winners" not cherry-picked.** The turnover sweep's only positive-Sharpe configs were **degenerate** (4 trades / 0 gross). We required configs to actually trade and reported the honest negative rather than dressing an artifact as a profitable strategy.

*(Methodology guard: the edge was validated strictly OUT-OF-SAMPLE with factors/betas frozen at T — guarding against finite-sample spurious mean reversion that an in-sample test would reward; ~8% of pure random walks spuriously fit as reverting in a unit check.)*

## 11. Caveats / assumptions
- **Survivorship bias:** universe is a *current* constituent list applied backward (0 delistings in the data confirms this). Inflates results; market-neutrality mitigates but does not remove it. No point-in-time index membership.
- **Free-shorting assumption:** shorts pay **no borrow fee** and face **no locate / short-availability** constraint — every name is assumed freely shortable daily. Real NSE shorting (SLB limits, small/mid-caps) would worsen results.
- **Cost-model approximations:** statutory charges via a faithful-but-approximate NSE *delivery* schedule (STT 0.1%/side + stamp + exch + GST); slippage modeled as a **flat 0.10%/side** proxy for spread+impact (no size/liquidity dependence, no impact curve). The ~18–20 bps break-even is therefore approximate.
- **Other:** integer-share truncation; daily marks; next-open fills assume executability at the printed open; no intraday dynamics.

## 12. Verdict
Harness is **correct and trustworthy** (validated). The signal has a **genuine but thin** reversion edge (significant OOS; zero-cost Sharpe +0.46) that is **NOT tradable in the NSE cash-delivery regime** — STT alone (~22 bps) exceeds the ~18–20 bps break-even, and no turnover setting escapes it. Viability requires a **structurally lower-cost regime** (futures-like ≤10–15 bps → Sharpe ~0.2). → This motivates the **futures version**.

---

### Files in this folder
- `FINDINGS.md` — this record
- `equity_curve.csv`, `equity_curve_zerocost.csv` — daily equity (full vs zero cost)
- `trades.csv` — every fill (full-cost run) · `exposure.csv` — daily gross/net/#positions
- `cost_sensitivity_sweep.json` — 8 cost levels (§5)
- `turnover_sweep_part0..3.json` — 12-config turnover sweep, round-robin split (§6)
- `sweep_nfactors.csv` — n_factors sweep (2019–2024, zero cost — provenance caveat, §8)
- `data/` — frozen input: prices/volume/open_wide.parquet + data_quality_report.csv

### Working code version (cash-equity)
`signal.py`, `harness_book.py`, `harness_decide.py` (incl. gross cap), `harness_execute.py`, `harness_backtest.py` (incl. realized-P&L fix), `data_prep.py`, `run_backtest.py`, `residual_oos_test.py` (start year via `OOS_START_YEAR`, default 2019), `harness_validation.py`.

### Reproduce
- Data: `poetry run python data_prep.py --start 2015-01-01 --end 2024-12-31 --symbols-file universe_clean.txt --out-dir ./data --sleep 0.5` (or use the frozen `data/` here).
- Full cost: `poetry run python run_backtest.py` · Zero cost: pass `statutory_cost_fn=lambda q,p,b:0.0` + `CostConfig(slippage_per_side=0.0)`.
- OOS: `OOS_START_YEAR=2015 poetry run python residual_oos_test.py` · Validation: `poetry run python harness_validation.py`.
