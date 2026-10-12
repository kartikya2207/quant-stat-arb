# PCA-Residual Statistical Arbitrage on NSE Equities

Testing whether PCA-residual mean reversion (Avellaneda–Lee style) makes money on Indian stocks once you
charge realistic costs. Spoiler: it doesn't, and most of this repo is me working out *why* and making sure the
"no" is trustworthy.

> This is a standalone research repo. It builds on the **TideTrader** engine (by Siddhesh Sali), which
> `pyproject.toml` declares as an optional dependency (`git+https://github.com/sid4sal/TideTraderBeta`) for
> later integration. The pipeline here is self-contained and doesn't import it yet.

## What it is

Stat-arb strategies bet that a stock's *idiosyncratic* move (its return after removing what the market and
sector factors explain) tends to reverse. The signal is well known from US data. I wanted to see if it
survives on NSE, first in cash equities and then in single-stock futures, with honest costs and no look-ahead.

The strategy, per trading day on a rolling 60-day window of daily returns:

1. **PCA** (correlation form: z-score, then SVD) with 5 factors.
2. **Residuals**: each stock's return minus its factor-fitted part.
3. **s-score**: an Ornstein–Uhlenbeck fit on the cumulative residual. `s ≥ +2` → short, `s ≤ −2` → long.
4. **Portfolio**: triple-barrier exits (profit / stop / 15-bar time stop), inverse-vol sizing, dollar-neutral,
   per-name cap, book gross capped at 1.0×.
5. **Timing**: signal from returns up to the close of bar *T*, execute at the *T+1* open, mark at the *T+1* close.

## What I did

**Cash-equity study, 2015–2024** (435 names, 2,466 trading days;
[`FINDINGS.md`](results_cash_equity_2015_2024/FINDINGS.md)):

| Cash equity, daily | Zero cost | Full realistic cost |
|---|---|---|
| Total return | +32.9% | **−32.9%** |
| Annualised Sharpe | +0.46 | **−0.56** |
| Max drawdown | −17.6% | −40.8% |

Break-even is about 18–20 bps round trip. NSE delivery STT alone is ~22 bps, so no turnover setting can save it
(the turnover sweep's only "positive" configs barely traded at all).

**Futures study, 2015–2024** ([`FINDINGS.md`](results_futures_2015_2024/FINDINGS.md), code in
[`futures/`](futures/)). Futures have a much lower cost floor, so I ported the identical signal to NSE
single-stock futures. I built a data pipeline from the free NSE bhavcopy archive: a point-in-time F&O universe
(329 names seen, 292 kept, no survivorship bias), real contracts in whole lots, monthly rolls, 264
corporate-action adjustments, and ₹10 crore of capital.

| Futures, daily | Zero cost | Full futures cost (~9 bps RT) |
|---|---|---|
| Total return | +23.5% | **+10.8%** |
| Annualised Sharpe | +0.50 | **+0.26 (± 0.32)** |
| Max drawdown | −11.1% | −13.5% |

Positive, but not distinguishable from zero. Most of it is one year (2018) and a handful of positions. It
dies between 5 and 8 bps of slippage per side, against the 3 bps I assumed. The out-of-sample Fama–MacBeth
test is insignificant at every horizon on the point-in-time universe. **Verdict: no statistically supported
tradable edge.**

**Checking the machinery itself.** Before I believe any result, the harness has to pass known-answer tests. It
earns +129.6% (Sharpe +4.28) on a planted reversion edge, ~0 on a random walk, loses on momentum, and wins
once the signal is flipped. Equity rebuilt independently from the trade log matches the harness to
floating-point zero.

## What I learnt

- **Survivorship bias can manufacture significance.** My cash study first reported OOS reversion significant
  at every horizon (Fama–MacBeth t −2.3 to −2.8). Re-running the same code on a *point-in-time* universe
  during the futures work left only the 1-day horizon standing, and only among surviving names. I added that
  correction to the cash FINDINGS instead of quietly replacing the number.
- **Accounting bugs hide in corporate actions.** The signal adjusted for splits, bonuses and dividends, but
  the position ledger didn't. Fourteen positions over ten years booked fake P&L (e.g. +₹26.8 lakh on an IOC
  short across a 1:1 bonus). I found it by checking that the realised position return equals the signal
  return on every held name-day. A later audit found a dividend-text parser that missed NSE's `INTDV`/`SPDV`
  abbreviations ("Bug 8").
- **A small calendar gap moves a "result".** Recovering one missing session (2021-03-30) shifted the headline
  by about three points. For a strategy this thin, that is a warning in itself.
- **Don't tune to noise.** The `n_factors` sweep was non-monotonic (nf=10 looked best at Sharpe 0.39, nf=5 at
  0.04), so I kept the principled nf=5 instead of picking the winner.
- **Under-powered isn't the same as no edge.** A rank IC of 0.012 (Newey–West t 4.7) does exist across the
  whole futures cross-section. But the strategy only trades the |s| ≥ 2 tail, and 1,974 positions is well
  short of the ~4,200 it would need for t = 2. Trading the 1–2 band is untested, and it would be a different
  strategy.
- **Costs are the strategy.** Slippage is 69% of the futures cost bill and it's a flat assumption. The sign
  of the headline depends on that one number.

Known caveats (all in the FINDINGS): free shorting with no borrow cost in cash, flat slippage with no impact
model, next-open fills assumed executable, and ordinary dividends deliberately not adjusted in futures.

## Repo structure

```
strategy/    # signal.py, harness_{book,decide,execute,backtest}.py, data_prep.py,
             # run_backtest.py, make_clean_universe.py (bare imports, so they live together)
tests/       # residual_oos_test.py, harness_validation.py, sweep_nfactors.py
universe/    # universe.txt, universe_clean.txt, ind_nifty500list.csv
futures/     # bhavcopy pipeline (fo_*), futures harness (fut_*), diagnostics (diag_*), README
docs/        # GUIDELINES.md (design spec)
results_cash_equity_2015_2024/   # frozen cash results + FINDINGS.md
results_futures_2015_2024/       # futures results, diagnostics/ + FINDINGS.md
data/        # gitignored: regenerable downloads, never committed
```

## How to run

```bash
poetry install            # numpy / pandas / pyarrow / yfinance / requests
                          # (TideTrader engine is optional: poetry install --with engine)

# Cash equity, run from the repo root so ./data resolves
poetry run python strategy/data_prep.py --start 2015-01-01 --end 2024-12-31 \
    --symbols-file universe/universe_clean.txt --out-dir ./data
poetry run python strategy/run_backtest.py                          # full-cost backtest
OOS_START_YEAR=2015 poetry run python tests/residual_oos_test.py   # OOS reversion test
poetry run python tests/harness_validation.py                      # known-answer checks (no data needed)

# Futures, run from futures/ (see futures/README.md for the full pipeline)
python fo_download.py --what fo   # then fo_build.py, fo_corpactions.py, fo_continuous.py, fo_validate.py
python run_futures_study.py       # headline run
```

Cash prices come from yfinance (split/dividend-adjusted). Futures data comes from the NSE bhavcopy archive.
Parquets and download caches are gitignored. Only code and result summaries are committed.
