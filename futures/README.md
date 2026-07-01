# Futures data pipeline (prototype)

Single-stock-futures data pipeline for the futures version of the PCA-residual stat-arb
strategy (daily, NSE, 2015–2024), built on the **free NSE F&O bhavcopy** archive.
Status: **prototype validated on 2 names (RELIANCE, SBIN)** — not yet scaled to the full
~190-name F&O universe, and the strategy is not yet ported onto it.

## `futures_proto_v2.py` — what it does (read-only smoke test)
1. **Downloads NSE bhavcopy** from `archives.nseindia.com` (browser headers), handling
   **both formats**: the old CSV (pre-2024-07-08) and the new UDiFF (post). Futures
   (`FUTSTK` / `STF`) and cash equity (`EQ`) segments. Files cached under `./cache/`
   (gitignored, regenerable).
2. **Extracts the front-month** contract per day (OHLC / settle / volume / OI).
3. **Builds a continuous series**: rolls to the next contract ~2 calendar days before
   expiry, with **additive (Panama) roll back-adjustment** to remove the inter-contract
   basis gap.
4. **Corporate-action adjustment (the key fix)**: NSE futures bhavcopy prices are RAW —
   splits/bonuses cause fake price jumps (RELIANCE's 1:1 bonus, ex-2024-10-28, showed as a
   fake −50% day). Factors come from `yfinance .splits`, back-adjusted to current scale and
   applied multiplicatively to the price panel **before** the roll adjustment.
5. **Validates**: no fake roll/corp jumps, no NaNs/gaps on market days, basis vs raw cash
   ≈ carry, future-vs-cash return correlation ≈ 0.99.

### Validated results (2 names)
| series | max \|adj ret\| | basis (mean) | ret corr vs cash |
|---|---|---|---|
| RELIANCE 2019 (old fmt) | 2.6% | +0.21% | 0.993 |
| RELIANCE 2024 (new fmt) | 3.8% (was 49.6% before corp-fix) | +0.44% | 0.992 |
| SBIN 2019 / 2024 | 4.5% / 5.2% | +0.32% / +0.39% | 0.997 / 0.994 |

## Before scaling to the full universe — data checklist
1. **Corporate actions**: `yfinance .splits` covers splits/bonuses only — **NOT**
   demergers/spinoffs (e.g. RIL→Jio Financial, Aug-2023) or special dividends. Add NSE
   corporate-actions coverage (or detect integer-ratio overnight jump + OI step); verify
   the adjustment date matches the bhavcopy jump per name.
2. **Volume units** differ across the 2024-07-08 format change (`CONTRACTS` vs
   `TtlTradgVol`) — prefer OI / turnover value for liquidity ranking; normalize if used.
3. **Point-in-time F&O membership**: the ~180–220 single-stock-futures universe changes
   over 2015–2024 — use a historical F&O-eligible list per date (futures analogue of
   survivorship).

## Notes
- Spot reference = NSE **cash** bhavcopy (raw); do not use yfinance `Close` for basis
  (it is dividend-adjusted for old dates, which distorts the level).
- `./cache/` and `./data/` are gitignored (regenerable downloads; do not commit).
