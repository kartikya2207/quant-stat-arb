# PROJECT GUIDELINES — PCA-Residual Statistical Arbitrage

**Owner:** Kartikya (Team A, Strategy Design) — Prof. Sandip Mondal, Quantitative Research & Trading Framework
**Audience:** This file is the single source of truth for both the human (Kartikya) and any agentic coding assistant (Claude Code, etc.) working on this strategy.
**Status:** Phase 3 (implementation) starting. Strategy design is COMPLETE and frozen. Build order is FIXED (see §2).

> **READ THIS FIRST (for any coding assistant):** The strategy below is fully designed. Do not redesign it. Do not "improve" the signal, add ML where it isn't specified, or change the risk rules. Your job is to implement what is written here, verify correctness, and do the wiring. When in doubt, ask — do not silently deviate.

---

## 0. THE ONE-LINE STRATEGY

Market-neutral, mid-frequency **PCA-residual statistical arbitrage** on liquid Indian equities (~250–300 names from Nifty 500). Strip out common market/sector factors via PCA; trade the mean-reversion of the leftover residual (the s-score); hold intraday to ~2–3 weeks; dollar-neutral; built classical-first, then an ML exit layer, then an intraday extension.

**HONEST FRAMING (never drop this):** This is a HYPOTHESIS TEST, not a guaranteed-return system. There is NO mathematical proof residuals revert — OU mean-reversion is a modeling assumption validated empirically. The reported edge in the literature is modest and decaying (Avellaneda-Lee PCA Sharpe ~0.9, already decayed). Indian costs (STT especially) may kill it. **A clean negative result, properly analyzed, is a VALID and valuable result** per the project brief. Do not oversell. Model costs pessimistically. Never tune to make the backtest look good.

---

## 1. THE STRATEGY MECHANICS (what's being built)

### 1.1 The signal (pure math — NO ML here, ever)
- **PCA** on a rolling 60-day window of daily returns of the liquid universe → top ~5 principal components = the common forces (market, sectors).
- **Residual** for each stock = its return − the common part. Computed by regressing the stock's returns on the PCA factors; the regression coefficients ARE the exposures (betas), and the leftover IS the residual. (Because PCA components are orthogonal, projection = regression here.)
- **Residual modeled as Ornstein-Uhlenbeck (OU)** mean-reverting process — this is WHY it oscillates rather than exploding (a naive cumulative sum of residuals would explode; OU has a pullback term). This is an ASSUMPTION, validated empirically, not a theorem.
- **s-score** = (current residual − equilibrium) / (residual's own volatility). Measured in the stock's OWN standard deviations, not a fixed %.
- **PCA DECOMPOSES, it does not PREDICT.** The reversion bet is OUR hypothesis, not a PCA output.

### 1.2 Universe (A2)
- Start from Nifty 500 → filter for liquidity (volume, spread), drop penny stocks, <2–3yr history, pending M&A → ~250–300 names → 20–40 positions/day.
- **v1 = FIXED universe** (one list for the whole backtest). Dynamic universe (survivorship handling) is a LATER upgrade, not v1.
- PCA uses ALL universe names; the s-score threshold selects which to actually trade.

### 1.3 Data (A3)
- **Daily ADJUSTED OHLCV** (splits/dividends adjusted — critical) + volume, 5+ years, 60-day rolling PCA window.
- Source decision: **yfinance with `auto_adjust=True`** (`.NS` suffix) → download once → save to local parquet → custom data source reads parquet. (See §4 for why, and the data-prep script.)
- Intraday version (later) needs 1–5 min OHLCV bars (NOT tick/LOB). Even the pure-intraday version STILL needs 60 days of DAILY data for the PCA structure + residual normal range (slow clock), PLUS live minute prices for the current residual (fast clock) — two frequencies, two jobs.

### 1.4 Entry (A4) — ALL hard gates must pass (pass/fail), THEN ML sizing
- s-score past **±2.0** (cost-driven choice vs literature's ±1.25 — fewer, higher-conviction trades cut turnover/cost. A reasoned starting point, to be confirmed empirically, NOT backtest-fished).
- **OU mean-reversion-speed filter**: only trade stocks whose residual actually reverts (and fast enough). OU speed from regressing today's residual on yesterday's; speed = −ln(b). Extreme s-score + no reversion = probably real news → skip. (Filter measurable facts; don't ML them.)
- **Liquidity check**: tradeable now (not halted, normal volume, reasonable spread).
- **Market-neutral**: balance longs ≈ shorts; prioritize most extreme signals up to position cap.
- **TIMING (look-ahead rule — CRITICAL):** signal computed from bar T's CLOSE → execute at bar T+1's OPEN. NEVER trade on the bar that generated the signal. Same for daily (today's close → tomorrow's open) and intraday (this minute's close → next minute's open).

### 1.5 Exit (A5) — triple barrier, whichever hits first
- **Profit**: s-score reverts to **±0.5** → take profit.
- **Stop**: s-score diverges to **±3.0** → cut loss (thesis was wrong). HARD, always active.
- **Time**: held ~**15 days** without reverting → free the capital.
- Same next-open timing as entry.

### 1.6 Sizing (A6)
- **Equal RISK, not equal rupees** — size inversely to volatility (calm stock bigger, wild stock smaller).
- Total portfolio ~1% daily vol target. Dollar-neutral. Hard cap ~5%/stock.
- Final size = base risk (vol-scaled) × ML multiplier (B2) × regime multiplier (A7), bounded by hard caps (A8).

### 1.7 Regime filter (A7)
- **India VIX graduated throttle**: <18 full, 18–25 half, 25–35 quarter, >35 stop. Applies an overall multiplier to ALL positions.
- Do NOT change the s-score threshold by regime (overfitting trap). Use the regime filter (scale down/off in crises); the threshold stays fixed.

### 1.8 Risk management (A8) — layered, HARD, never ML-controlled
- Per-trade stop (s ±3.0). Per-stock cap ~5%. Portfolio: 2× gross max, net ≈ 0, ≤40 positions, ≤30%/sector.
- **Drawdown circuit breaker**: halve size @5%, no-new @10%, close-all @15%.
- **Crowding risk** (2007 Quant Quake, Khandani-Lo): "diversified" positions can become ONE correlated bet in stress. Defenses soften, do NOT eliminate. Keep sizing conservative.

### 1.9 Costs (A9) — THE BINDING CONSTRAINT, model PESSIMISTICALLY
- ~0.35–0.50% round trip. **STT dominates.**
- **CRITICAL STT FACT:** delivery STT ~0.1% per side (both sides); intraday STT much lower (~0.025%, sell side). The intraday-vs-delivery classification LOCKS based on whether you hold past the ENTRY-DAY close.
- Gross edge per trade is small (~0.3–0.8% avg over winners+losers) → net is slightly-negative to modestly-positive → costs may kill it. This is the central empirical question.
- The framework's `costs.py` ALREADY models the full Indian stack (STT with buy/sell asymmetry, brokerage, exchange charges, GST, stamp duty) with `nse_equity_intraday` and `nse_equity_delivery` presets. REUSE IT. (See §4.)
- **Slippage is NOT modeled by the framework** — fold a slippage/impact estimate into the cost callable (see §3 gap 4).

### 1.10 Evaluation (A10) — honest, no self-deception
- **Walk-forward**: train past / test future / roll forward. One honest look at held-back data, never tweak after.
- Report a PANEL: net (after-cost) Sharpe & Sortino, max drawdown & duration, win rate, avg win/loss ratio, turnover.
- **Deflated Sharpe Ratio** (Bailey-López de Prado 2014): correct for the number of variations tried. MUST honestly count every variation. It's the overfitting catch — but only works if you don't hide trials.
- Regime breakdown (calm/crisis/choppy — check COVID). Report failures honestly.

---

## 2. BUILD ORDER (FIXED — do not reorder)

The brief mandates classical-first, measured, THEN ML, THEN complexity. Each layer must be testable against the one below. Building everything at once = undebuggable tangle.

1. **Daily classical baseline** — PCA → residual → s-score, fixed entry/exit rules, dollar-neutral, regime filter, hard risk limits. Get a clean walk-forward backtest. **This is the moment of truth: does the pure baseline have any edge after costs?**
2. **ML sizing layer (B2)** — meta-labelling for position size. Must BEAT the baseline.
3. **ML exit model (B3)** — context-aware exits incl. the STT day-0 factor. Hard stop ALWAYS remains. Must beat fixed barriers.
4. **Intraday extension (B4)** — live minute s-score, low-STT scalping of fast reversions.
5. **Iterate** — backtest → analyze → refine. Document every change.

> **PRESENTATION NOTE:** Kartikya will PRESENT the full version (with the real ML exit model that decides intraday-exit vs hold-a-few-days). That is the GOAL/showcase. But the BASELINE is built first as the foundation the ML exit trains on — even though it may not be spotlighted in the final presentation. The baseline must exist and produce trades before the ML exit can be trained or validated.

---

## 3. ML LAYER DETAILS (Phase B — built AFTER baseline works)

**PRINCIPLE: ML refines SIZING and TIMING only. NEVER direction (that's price prediction = fails) and NEVER risk limits (must stay hard). Gates verify measurable facts; ML scores quality of survivors.**

### 3.1 Where ML adds value — ONLY these
- Entry filtering + sizing (meta-labelling) — highest value.
- Exit decisions (Kartikya's idea).
- (Optional, low priority) regime classification.
- NOT the signal, NOT the risk limits.

### 3.2 Meta-labelling sizing model (B2)
- **Features**: s-score, OU speed, residual age, stock vol, volume, sector, PCA fit quality, India VIX, market trend, (optional) news sentiment.
- **Labels**: triple-barrier outcome (using the FIXED exit rules, NOT the ML exit model — avoids circularity).
- **Model**: gradient boosting (XGBoost) → probability(trade works) → scales position.
- **Label fix (Kartikya's catch):** binary 1/0 labels are unfair (₹100 win = ₹10000 win). Keep 1/0 but **SAMPLE-WEIGHT by P&L magnitude** (López de Prado's prescription) + uniqueness weighting (overlapping holding periods leak — weight by uniqueness).
- Probability → size: threshold gate OR probability-scaled.

### 3.3 ML exit model (B3 — Kartikya's idea, the presentation centerpiece)
- **Inputs**: current s-score, days held, profit-AFTER-cost (cost-aware), s-score trajectory, vol, regime.
- **Output**: exit-or-hold, AND for the intraday version: exit-now-intraday vs hold-N-days.
- **GUARDRAIL (non-negotiable):** the hard stop loss is ALWAYS active. The model can exit EARLY but can NEVER hold a runaway loser. Guards against disposition effect ("cut winners, hold losers" = small wins + catastrophic losses).
- **Trained on**: what historical price paths reveal as the better action. NO circularity — sizing model trains on fixed-rule outcomes; exit model trains on historical paths; neither trains on itself.
- **STT day-0 factor (Kartikya's idea — CONFIRMED CORRECT FORM):**
  - It is a **DAY-0 (entry-day) ONLY** decision: "is this trade already profitable enough after the LOW intraday STT to scalp out TODAY, vs carry overnight into the HIGHER delivery STT?"
  - The low-STT window exists ONLY on the entry day. Once held overnight, it's already delivery — the factor switches OFF from day 1.
  - Implement as a FEATURE ("is this entry day?" + "intraday-STT-adjusted profit now") that the model learns from — it discovers when same-day exit historically beat holding.
  - Must bias toward TAKING cheap profitable exits, NEVER toward HOLDING losers. Hard stop always overrides. It's a refinement, not a game-changer (many reversions take >1 day and become delivery regardless).

### 3.4 Intraday version (B4)
- **Two clocks**: SLOW (once each morning) = PCA common forces + residual normal mean/std from 60 days DAILY data. FAST (every minute) = current residual from live minute price + s-score + trade decisions.
- **Timing**: compute residual/s-score from bar-T CLOSE → execute at bar-T+1 OPEN. Cannot trade on the bar that generated the signal (need its close first). SAME for exits.
- **KEEP BAR-BASED LOGIC — do NOT trade on continuous live price.** Reason: backtest only has bars; trading on continuous live prices means the backtest can't reproduce live behavior (manufactures the backtest-live gap that kills strategies). If faster reaction needed → use smaller bars (15s/30s), don't abandon bars.
- **ONLY exception:** the hard stop loss may fire intra-bar live for safety (it's protection, not signal; modeled conservatively in backtest). The signal (entry, profit exit, ML decisions) stays bar-based.
- Computation per minute is milliseconds for ~250 stocks — time is NOT a constraint at minute/5-min frequency.

### 3.5 Deferred / optional (NOT yet designed)
- Signal combination (only relevant once multiple signals exist).
- News-sentiment / agentic layer (GDELT + FinBERT). Exploratory, future phases only.

---

## 4. THE FRAMEWORK (Team B's TideTraderBeta) — what to reuse, what's missing

Repo: github.com/sid4sal/TideTraderBeta (package `tidetrader`). Kartikya's fork: github.com/kartikya2207/TideTraderBeta. Local: `C:\Users\iitia\OneDrive\Documents\SURP\CODE\TideTraderBeta`. Env: poetry (installed, `poetry install` done).

### 4.1 THE BIG ARCHITECTURAL FACT: the framework is SINGLE-SYMBOL
- One strategy instance trades ONE stock. The underlying `backtesting.py` library is single-instrument.
- In backtest, every order goes all-in on full equity (`size=_FULL_EQUITY`) and `exclusive_orders=True` auto-closes the prior position. NO native notion of fractional weights or a multi-name basket.
- `backtest_multi.py` does NOT solve this — it runs INDEPENDENT per-stock backtests in parallel (each own ₹100k, no shared book), for per-stock parameter tuning. It is NOT a portfolio backtester.
- **PCA stat-arb is inherently CROSS-SECTIONAL** (needs ~250 stocks in ONE shared book on each bar). **This mismatch is the core Phase-3 challenge.**
- **DECISION: Kartikya builds his own cross-sectional backtest harness** (does NOT wait on Team B), reusing the framework's components where they fit. A short heads-up message was sent to Siddesh flagging the gap (not blocking on him).

### 4.2 What to REUSE from the framework
- **Cost model (`costs.py`)** — the full Indian cost stack incl. STT with correct buy/sell asymmetry, `nse_equity_intraday` and `nse_equity_delivery` presets. It's a `(quantity, price) → ₹charge` callable. REUSE this directly in the harness. (`nse_equity_delivery`: STT ~0.1% both sides — the relevant preset for overnight holds.)
- **Data-source interface (`BaseDataSource`)** — write a custom daily-adjusted source FOLLOWING this interface so it stays compatible. Interface: `get_ohlcv(symbol, start, end, interval) -> DataFrame` with tz-naive DatetimeIndex and columns ['Open','High','Low','Close','Volume'].
- **Metrics/reporting** — reuse `backtesting.py`'s standard set where possible; compute custom metrics (Deflated Sharpe, residual half-life, turnover) ourselves from trades + equity curve.
- **Strategy interface** (if/where single-symbol pieces are reused): subclass `TradingStrategy`; implement `tick()` (required), optionally `set_defaults`, `initial_setup`, `compute_stats` (indicators go in `compute_stats`, returning {name: Series}; read in `tick` as `self.stats.<name>[-1]`). Handle the `'Close'`/`'close'` column-case split. Act via `self.buy()/self.sell()/self.square_off()`.

### 4.3 Framework GAPS Kartikya must fill (the build list)
1. **Cross-sectional / portfolio backtest harness** (THE BIG ONE). Each bar: compute residuals → target weights → simulate a weighted, dollar-neutral book with the existing cost callable. The single-symbol adapter cannot express this. Build it ourselves.
2. **Daily adjusted data**. No daily-bar fetch path and no split/dividend adjustment in the framework (Dhan path only pulls intraday minute bars; cleaning only fixes timestamps). Fix = custom `BaseDataSource` over pre-adjusted daily parquet (from yfinance). UNADJUSTED PRICES WOULD INJECT FAKE SPLIT/DIVIDEND JUMPS THAT CORRUPT THE PCA COVARIANCE MATRIX — this is why adjustment is non-negotiable.
3. **Universe management** for ~250–300 names (list + symbol resolution). v1 fixed list.
4. **Slippage / market impact** — NOT modeled by the framework (fills at bar open/close, no spread/impact). For a thin-edge, high-turnover strategy this makes the backtest falsely optimistic. Fold an impact estimate into the cost callable (e.g. ~0.05–0.10% spread + size/liquidity-scaled impact).
5. **Position sizing by weight** — the order API takes integer shares (ignored in the current backtest adapter anyway); weighted dollar-neutral sizing is part of the harness we write.

### 4.4 Timing in the framework (GOOD — matches our design)
- Default `trade_on_close=False` → a market order placed on bar T executes at bar T+1's OPEN. Exactly our look-ahead rule. Controllable (`trade_on_close=True` for current-close fills). Limit/stop/target supported.

---

## 5. THE ORPHANED-POSITION RULE (Kartikya's catch — must implement)

**Problem:** if you only compute s-scores for today's liquid universe, a stock you HOLD that later drops OUT of the universe becomes invisible — no s-score → no exit signal → stuck position bleeding forever.

**Root cause:** conflating two different sets:
- "Which stocks can I ENTER today?" = the liquidity-filtered universe.
- "Which stocks must I monitor for EXIT?" = EVERYTHING I currently hold.

**The fix:** the set of stocks you compute s-scores for each day =
**(today's liquid universe) ∪ (everything currently held)**.
Always score your holdings, even ones kicked out of the universe — so exit logic keeps working on them. You just don't OPEN new positions outside the universe.

**Mechanics for a dropped-out holding:** to get its s-score you still need its residual = its exposures to the current PCA factors. You can still PROJECT a dropped stock's returns onto the factors computed from the liquid universe — a stock doesn't need to be IN the PCA estimation to get a residual FROM it. (Open sub-decision for v1: project onto today's factors vs. freeze last-in-universe exposures — default to projecting onto today's factors.)

---

## 6. OPEN DECISIONS still to make during the build

(These were flagged; resolve as we implement — do NOT guess silently.)
1. **Backtest engine style**: bar-by-bar loop (matches live, slower) vs vectorized (faster, look-ahead risk). For ~250 stocks × 5yr daily both are feasible.
2. **Position representation**: target weights vs rupee notional vs share counts in the book.
3. **Rebalancing frequency**: full rebalance to target each day (high turnover/cost) vs trade only the changes (lower turnover) vs threshold rebalance. Directly hits the cost constraint.
4. **Orphan residual**: project dropped names onto today's factors vs freeze last exposures (default: project onto today's).
5. **Slippage form**: flat per-trade haircut vs size/liquidity-scaled.
6. **Cost callable integration**: confirm `costs.py` callable slots cleanly into the portfolio loop or needs a thin wrapper.

---

## 7. WORKING AGREEMENT (human + coding assistant)

- **Kartikya teaching/working style:** ONE concept per turn, intuition + concrete numeric example BEFORE formula, short responses, no info-dumps, no trivial check-questions whose answer was just given. Blunt, accurate, calibrated feedback (not blanket praise). Push back honestly when something is wrong.
- **Division of labor:** the strategy reasoning / heavy harness logic is designed WITH Kartikya in the chat (Claude.ai, holds full strategy context). Reference code is produced there. Kartikya pastes files into the repo; **Claude Code verifies, wires into the framework, and does the lighter implementation work.**
- **For the coding assistant:** implement what's in this file. Verify correctness against it. Do NOT redesign the strategy, change risk rules, ML-the-signal, or tune for backtest appearance. Surface problems; don't paper over them. Respect the honest-framing mandate (§0).
- **Deployment:** PAPER TRADING only (no real money). Note for any future real-money step: overnight shorting of cash equities is restricted in India — would force the short leg toward intraday or futures. Not a blocker for paper trading.

---

## 7b. DEFERRED ITEMS — things we said "later" (DO NOT FORGET)

Running list of work consciously postponed. Each must be done before the relevant stage is "complete." Check off / update as done.

- [ ] **Scale the universe to full ~250–300 names.** Currently using a 10-stock demo for fast signal-logic testing. The signal code is identical for 250; only the data changes. NEXT data task: assemble the Nifty 500 liquid universe → `universe.txt` → full download. yfinance is flaky under bulk load (intermittent "no data" that retries fix) — when scaling, slow the pacing and raise retries.
- [ ] **PCA on 10 stocks is degenerate** (can't extract 5 clean factors from 10 names). The demo run validates LOGIC ONLY, not realistic factor structure. Re-validate signal behavior once the full universe is in.
- [ ] **Orphaned-position handling (§5).** Must implement: score = (today's universe) ∪ (everything held). Not yet in code.
- [ ] **Slippage / market impact in the cost model (§4.3 gap 4).** yfinance/framework model neither; fold an estimate into the cost callable. Not yet done.
- [ ] **The 6 open design decisions in §6** — resolve each during the build, don't guess silently.
- [ ] **Custom `BaseDataSource`** wrapping the parquet files, conforming to the framework interface (§4.2).
- [ ] **Reuse the framework cost model (`costs.py`)** rather than rewriting STT — confirm it slots into the harness.
- [ ] **The cross-sectional harness itself (§4.3 gap 1)** — the big build.
- [ ] **Then, in order:** baseline backtest → ML sizing (B2) → ML exit incl. STT day-0 factor (B3) → intraday (B4). Build order is FIXED (§2).
- [ ] **Data refresh cadence** — yfinance data is a snapshot; decide when/how to refresh for paper trading.
- [ ] **Universe survivorship** — v1 is a fixed list (includes survivorship bias). Note it as a known limitation in results; dynamic universe is a later upgrade.

---

## 8. KEY LITERATURE (verified citations)

- Avellaneda & Lee (2010), *Quant. Finance* 10(7):761–782 — PCA stat-arb, OU residual, the core method. Sharpe ~0.9 (PCA), ~1.1 (ETF), ~1.51 (ETF+volume), decaying.
- Khandani & Lo (2007), *J. Investment Mgmt* 5(4) — Aug 2007 Quant Quake, crowding risk.
- Bailey & López de Prado (2014), *J. Portfolio Mgmt* 40(5):94–107 — Deflated Sharpe Ratio.
- López de Prado (2018), *Advances in Financial Machine Learning*, Wiley — meta-labelling, triple-barrier, purged CV, sample weighting.
- Gatev, Goetzmann & Rouwenhorst (2006), *RFS* 19(3):797–827 — pairs trading lineage.
- (Full set in the Phase 1 / Phase 2 literature review documents.)

---

*End of guidelines. This file is the contract. If reality diverges from it, update the file — don't let the code and the plan silently drift apart.*
