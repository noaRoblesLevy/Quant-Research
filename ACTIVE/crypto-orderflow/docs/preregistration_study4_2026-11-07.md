# Pre-registration, study 4: short-horizon crypto signals with order flow (2026-11-07)

Written and committed **before** any test of this study was run. Nothing here may change
after results exist; deviations go in the results report, labelled.

## Why

Studies 1 to 3 found no robust edge in slow strategies (monthly to weekly, few
instruments). A higher Sharpe needs many independent bets: short holding periods across
many instruments. Binance futures data contains, per bar, the volume bought by
aggressive takers, which gives signed order flow. This study tests a small, fixed set of
short-horizon cross-sectional signals, with realistic costs.

## Data

Binance USD-M perpetuals, 1-hour bars with taker-buy volume, and funding, for the same
23 coins as the funding studies (BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LINK, DOT, LTC, BCH,
FIL, NEAR, AVAX, XLM, AAVE, UNI, TRX, ATOM, ETC, SAND, AXS, COMP).

| Period | Dates | Use |
|---|---|---|
| Selection | 2021-02 to 2023-12 | |
| Confirmation | 2024-01 to 2025-05 | |
| Holdout | 2025-06 to 2026-10 | downloaded for this study, used **once** |

Order-flow imbalance of coin i over a window: OFI = (2 x taker-buy volume - volume) /
volume, summed over the window's bars and divided by the window's total volume.

## Signals (7 trials, fixed directions)

Every signal ranks the 23 coins; the book is long the 5 highest scores and short the
5 lowest, equal weight (+-0.1 each, gross 1, dollar-neutral).

| Code | Score | Rebalance | Rationale |
|---|---|---|---|
| REV24_8 | minus the 24h return | every 8h | short-term reversal |
| REV24_24 | minus the 24h return | every 24h | same, lower turnover |
| OFIC24_8 | 24h OFI | every 8h | order-flow continuation |
| OFIC24_24 | 24h OFI | every 24h | same, lower turnover |
| OFIR24_8 | minus 24h OFI | every 8h | order-flow reversal (crowded aggressive buying unwinds) |
| OFIR24_24 | minus 24h OFI | every 24h | same, lower turnover |
| OFIC1_1 | 1h OFI | every hour | highest breadth, continuation |

The direction of order flow is genuinely uncertain in the literature, so both are
tested and both count as trials.

## Execution and costs

- Signal at the close of hour t; trade at the close of hour t+1 (one-bar lag); positions
  earn from t+1 to the next rebalance.
- Cost: **7 bps per unit of traded notional** (taker fee about 5 bps plus 2 bps
  slippage). Sensitivity reported at 0 (gross), 2 (maker), 7 and 12 bps.
- Funding is paid or received on the perp positions.
- No leverage financing (perpetuals are margined); returns are per unit of gross notional.

## Metrics

Net Sharpe on daily returns, gross Sharpe, CAGR and maximum drawdown with the book
scaled to 10% annual volatility (60-day trailing, scale capped at 5x), annual turnover,
annual cost drag, correlation with BTC and with SPY.

## Decision rules

- **F1 feasibility:** net Sharpe (at 7 bps) of at least **1.0** in both Selection and
  Confirmation, and maximum drawdown at most 20% at the 10% volatility target in both.
  (A market-neutral book with Sharpe 1 at 10% volatility earns roughly cash + 10%, which
  would beat the S&P on a risk basis.)
- **F2 selection:** the feasible signal with the highest min(Sharpe Selection, Sharpe
  Confirmation).
- **F3 holdout:** every signal is reported on the holdout once.
- **F4 deployment:** a feasible candidate becomes a live shadow ledger; otherwise the
  conclusion is "no short-horizon edge after costs".

## Expectation written down beforehand

Gross short-term reversal and order-flow effects are likely visible. At 7 bps per side,
hourly and 8-hourly rebalancing are expected to lose most or all of the edge to costs;
24-hour rebalancing is the most likely to survive, if anything does.
