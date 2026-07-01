# PCA-Residual Statistical Arbitrage on NSE Equities

A cross-sectional, market-neutral **statistical-arbitrage** research project on Indian (NSE) equities,
using **PCA-residual mean reversion** (Avellaneda–Lee style), with a deliberately rigorous and
*honest* evaluation of whether the edge survives real transaction costs.

> This is a standalone research repository. It builds on the **TideTrader** trading engine
> (by Siddhesh Sali), which is declared as an optional dependency in `pyproject.toml`
> (`git+https://github.com/sid4sal/TideTraderBeta`) **for future integration** — the pipeline here
> is self-contained and does not import the engine yet.

---

## TL;DR — the headline result

The signal has a **real but thin** mean-reversion edge that is **statistically significant
out-of-sample**, but it **does not survive realistic Indian cash-equity transaction costs.**

| Cash-equity backtest, 2015–2024 (daily) | Zero cost (bare signal) | Full realistic cost |
|---|---|---|
| Total return | **+32.9%** | **−32.9%** |
| Annualised Sharpe | **+0.46** | **−0.56** |
| Max drawdown | −17.6% | −40.8% |

STT alone (~22 bps round-trip) already exceeds the ~18–20 bps break-even, so the strategy cannot be
profitable in the cash-delivery regime under any turnover setting. → Current work ports it to
**single-stock futures** (a structurally lower-cost regime).

---

## The strategy

Per trading day, over a rolling 60-day window of daily returns:
1. **PCA** (correlation form — z-score each name, then SVD) extracts common factors.
2. **Residuals**: each stock's return minus its factor-fitted part (idiosyncratic move).
3. **s-score**: an Ornstein–Uhlenbeck fit on the cumulative residual → standardized mean-reversion
   signal. Rich (`s ≥ +2`) → short; cheap (`s ≤ −2`) → long.
4. **Portfolio**: triple-barrier exits (profit/stop/time), inverse-vol sizing, dollar-neutral,
   per-name cap, book gross capped at 1.0×.
5. **Look-ahead-safe timing**: signal from returns up to bar *T* close → execute at *T+1* open →
   mark at *T+1* close.

---

## Cash-equity study (2015–2024) — complete

Universe: 435 NSE names, 2,466 trading days. Full writeup:
**[`results_cash_equity_2015_2024/FINDINGS.md`](results_cash_equity_2015_2024/FINDINGS.md)**.

**Key findings**
- **OOS reversion is statistically significant** — out-of-sample test with factors/betas frozen at
  *T* and strictly-forward returns: Fama–MacBeth *t* = **−2.3 to −2.8** across 1/3/5/10-day horizons,
  correlation ≈ **−0.02** (weak but real), hit-rate 54–56%.
- **Cost break-even ≈ 18–20 bps round-trip**, **below the unavoidable ~22 bps STT floor** of NSE cash
  delivery → not tradable in cash equities.
- **Turnover sweep**: no configuration that stays meaningfully invested beats costs.
- **Known-answer validation**: the harness correctly detects a planted edge (Sharpe +4.3), finds
  ~zero on a random walk, and loses on momentum but wins when the signal is flipped — the
  backtester is trustworthy.
- **P&L integrity**: equity independently reconstructed from the trade log matches the harness to
  floating-point zero.

**Caveats (documented):** survivorship bias, a no-borrow-cost free-shorting assumption, and
cost-model approximations.

---

## Futures version — current work

Porting the same daily strategy to **single-stock futures** (F&O-eligible ~180–220 names), where
the cost structure (no STT delivery tax, ~10–15 bps all-in) is where this thin edge could become
tradable. Data comes from the **free NSE F&O bhavcopy** archive (2015–2024).

The data pipeline is prototyped and validated on 2 names (see [`futures/`](futures/)): both bhavcopy
formats parse, front-month continuous series with roll back-adjustment works, and a
**corporate-action (split/bonus) adjustment** was added (RELIANCE's 1:1 bonus had shown as a fake
−50% jump). See [`futures/README.md`](futures/README.md) for details and the scaling checklist.

---

## Repository layout

```
strategy/    # the pipeline — modules + runners (co-located so their bare imports resolve)
             #   signal.py  harness_book.py  harness_decide.py  harness_execute.py
             #   harness_backtest.py  data_prep.py  run_backtest.py  make_clean_universe.py
tests/       # validation & analysis (residual_oos_test, harness_validation, sweep_nfactors)
universe/    # universe.txt  universe_clean.txt  ind_nifty500list.csv
docs/        # GUIDELINES.md (design spec)
results_cash_equity_2015_2024/   # frozen cash-equity results + FINDINGS.md
futures/     # futures data pipeline (+ its README)
data/        # gitignored — regenerable downloads (parquets/cache never committed)
```

The strategy modules import each other by bare name, so they're kept together in `strategy/`; run
scripts **from the repo root** so `./data` resolves. The `tests/` scripts add `strategy/` to the
path automatically.

## Setup

```bash
poetry install          # installs numpy/pandas/pyarrow/yfinance/requests
                        # (the TideTrader engine is an optional group — poetry install --with engine)
```

## Reproduce (run from the repo root)

```bash
# cash data (yfinance, split/dividend-adjusted) -> ./data
poetry run python strategy/data_prep.py --start 2015-01-01 --end 2024-12-31 \
    --symbols-file universe/universe_clean.txt --out-dir ./data

poetry run python strategy/run_backtest.py         # full-cost cash-equity backtest
OOS_START_YEAR=2015 poetry run python tests/residual_oos_test.py   # OOS reversion test
poetry run python tests/harness_validation.py      # known-answer validation (synthetic; no data)
```

**Data note:** cash prices via yfinance; futures via NSE bhavcopy. Large parquet data and download
caches are gitignored (regenerable) — only code and result summaries are committed.
