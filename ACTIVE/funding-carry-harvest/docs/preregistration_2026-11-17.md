# Pre-registration: regime filter for cash-and-carry (2026-11-17)

Written and committed **before** any of the tests below were run. The commit time is the
evidence that the configurations, metrics and decision rules were fixed in advance.
Nothing in this file may be changed after results exist; deviations go in the results
report, labelled as deviations.

## Why

The improvement study of 2026-11-07 (`docs/improvement_results/README.md`) selected
cash-and-carry CC L=21 K=7: short the 11 perps with the highest 7-day mean funding, long
the same coins on spot, seven tranches. Its return is a funding regime: 40% carry in
2021, 14 to 15% in 2024, but 4% in 2025 (below the cash yield) and 2% in the 2026 holdout
(below costs). CC already leaves a slot empty when the smoothed funding is not positive,
but a coin earning 2% a year still costs turnover and ties up capital that could earn
the cash yield.

Hypothesis: raising the entry threshold from "positive" to a hurdle that covers costs and
the cash yield keeps the good regimes and sits in cash in the bad ones, so the excess
return over cash improves.

## Configurations (4 per venue, 8 trials, all reported)

CC L=21 K=7 exactly as in the 2026-11-07 study (N = 11 slots, hold 7 days, 8 bps per leg
turnover, 35-day warm-up), with one change: a slot is filled only if the coin's 7-day
mean funding, annualised (x 1095), exceeds a hurdle `tau`:

| tau (per year) | meaning |
|---|---|
| 0% | the current live configuration (control) |
| 5% | roughly costs (about 2 to 3.5% a year in the study) plus a margin |
| 10% | roughly costs plus the cash yield; just below the Bybit default rate (10.95%) |
| 15% | only coins clearly above the default rate |

Nothing else changes. Coins are still chosen among the 11 highest smoothed funding rates.

## Cash

Capital not used for a spot leg earns the cash yield. The engine reports, per period,
the gross spot weight `g_t` (share of capital in use). Excess return over cash:

    excess_t = net_t - g_t * cash_t

`cash_t` is the approximate annual average 3-month US T-bill yield used in
`scripts/eval_cc_robustness.py` (2021 0.05%, 2022 2.0%, 2023 5.1%, 2024 5.0%, 2025 4.2%),
and an assumed 3.7% for 2026, converted per 8h period. Because an empty slot simply
earns cash, `excess_t` is the return the strategy adds over holding cash.

## Metrics

- Primary: annualised Sharpe ratio of `excess_t` (8h returns).
- Secondary: annualised excess return, net return, volatility, max drawdown, average
  `g_t`, Sharpe on monthly excess returns, excess return per calendar year.

## Data

Identical to the 2026-11-07 study: Binance perps and spot Feb 2021 to May 2025
(selection), Bybit Aug 2022 to May 2025 (confirmation), Bybit 6 June to 6 November 2026
(holdout, used once).

## Decision rules

- **R1, validation.** With tau = 0 the modified engine must reproduce the CC L=21 K=7 row
  of `docs/improvement_results/grid.csv` (Sharpe on both venues) within 0.01, and the
  baseline XS0 L=1 K=1 rows must be unchanged. If not, the study is not run.
- **R2, selection.** The candidate is the tau with the highest
  `min(excess Sharpe Binance, excess Sharpe Bybit)`. Ties go to the lower tau.
- **R3, robustness.** The candidate counts as an improvement only if its excess Sharpe
  beats tau = 0 on **both** venues. Otherwise the conclusion is "no robust improvement"
  and the live configuration stays as it is.
- **R4, holdout.** All four tau are evaluated once on the holdout and reported whatever
  they show. Nothing is re-tuned afterwards.
- **R5, deployment.** If the candidate passes R3, it starts as an extra live shadow ledger
  next to the unchanged CC ledger. If it does not, nothing is deployed.

## Separate decision taken on the same day (not part of this test)

The cross-sectional ledgers (baseline since 2026-06-06, variant C since 2026-11-06) are
frozen: they stop advancing from 2026-11-17 16:00 UTC, and their files are kept as they
are. Reasons, recorded before this study is run so that its outcome cannot influence
them:

1. Baseline after 164 days: -21.7% (carry +2.5%, price -25.1%), live Sharpe about -1.8
   against a backtest of 1.34; that is about two standard errors below the backtest.
2. The ledger counts no costs. The 2026 holdout costs of the baseline (5.5% a year)
   equal its live carry (5.6% a year), so the net carry is about zero.
3. Variant C's 8h returns correlate 0.96 with the baseline and are slightly more
   volatile (0.72% against 0.65% per step), so its purpose, lower risk, failed live.
4. The 2026-11-07 study already showed that no change of weights, hedge, signal or
   tranches fixes the cross-sectional book. Further live observation cannot change a
   decision, only add noise to it.
