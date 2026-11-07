# Pre-registration: funding-carry improvement study (2026-11-07)

Written and committed **before** any of the tests below were run. The commit time is the
evidence that the configurations, metrics and decision rules were fixed in advance.
Nothing in this file may be changed after results exist; deviations go in the results
report, labelled as deviations.

## Why

The live forward test and the paper "Carry or Dispersion?" showed three problems:

1. The book is a cross-sectional **dispersion** bet; the carry is a small, steady tailwind.
2. Ranking on a **single funding snapshot** makes the book very sensitive to the rebalance
   day (the same rules gave -4.8% or -15.3% over five months).
3. Nearly all carry comes from the **short leg**; the long leg earns nothing and hedges badly.

Having already tested five variants (A to D), the risk of finding something that looks
good by chance is real. This protocol limits that risk.

## Configurations (24 per venue, all reported)

Three factors, fully crossed:

| Factor | Levels |
|---|---|
| Signal `L` (funding used for ranking) | latest print (L=1); mean of last 3 days (L=9); mean of last 7 days (L=21) |
| Tranches `K` | 1 (one book, rebalanced every 7 days); 7 (seven sub-books of 1/7 capital, rebalanced on 7 different days, one day apart) |
| Construction | **XS0** baseline: 1/N legs, BTC beta hedge. **XSC**: inverse-vol legs, BTC+ETH hedge (variant C). **PAIR**: each short matched to the remaining coin with the highest trailing correlation, long weight = pair hedge ratio / N, plus a BTC beta hedge on the residual. **CC**: cash-and-carry, short the N highest-funding perps and long the same coins on spot; slots whose smoothed funding is not positive stay in cash |

Fixed for all: N = 11 per leg, hold = 7 days (21 periods of 8h), 8 bps cost on the
turnover of every leg (perp, spot and hedge), 35-day warm-up before any trading,
90-period beta and correlation windows, 30-period volatility window, 8h funding grid.

## Data

- **Selection sample**: Binance USD-M perps and spot, Feb 2021 to May 2025.
- **Confirmation sample**: Bybit linear perps and spot, Aug 2022 to May 2025.
- **Holdout**: Bybit, 6 June to 6 November 2026 (the live forward-test period). Used
  **once**, after the candidate is chosen, and never for tuning.

## Metrics

- Primary: annualised Sharpe ratio of net 8h returns.
- Secondary: annualised return and volatility, maximum drawdown, carry and price split.
- **Schedule sensitivity** (K = 1 only): spread of the final cumulative return across the
  7 possible start days (standard deviation and range).
- CC returns are quoted per unit of spot notional. That equals return on capital if spot
  can serve as collateral for the perp short (as in a Bybit unified account); with
  separate accounts capital roughly doubles and returns halve. Sharpe is unaffected.

## Decision rules

- **R1, separate effects.** Each single change (L, K, or construction) is reported against
  the baseline (XS0, L=1, K=1) on both venues.
- **R2, selection.** The candidate is the configuration with the highest
  `min(Sharpe Binance, Sharpe Bybit)` (maximin). Ties go to lower schedule sensitivity.
- **R3, robustness.** The candidate counts as an improvement only if its Sharpe beats the
  baseline on **both** venues. If not, the conclusion is "no robust improvement found".
- **R4, holdout.** Baseline and candidate (and, for information, the best configuration of
  each construction) are evaluated once on the holdout. Results are reported whatever
  they are; nothing is re-tuned afterwards.
- **R5, deflation.** The deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) of the
  candidate is computed on the Bybit sample with 48 trials (24 configurations x 2 venues).
- **R6, deployment.** If the candidate passes R3 it starts as an additional live shadow
  ledger next to the existing ones (baseline and variant C stay unchanged). The holdout
  is reported but does not block deployment, because five months cannot settle it and the
  shadow ledger is itself the real out-of-sample test.

## Validation before any result is read

The new engine must reproduce the existing baseline (scripts/eval_hedge_variants.py:
Sharpe 1.32 Binance, 1.24 Bybit) within 0.05 before the study is run.
