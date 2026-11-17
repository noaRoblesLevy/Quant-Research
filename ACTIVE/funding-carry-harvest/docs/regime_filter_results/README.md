# Regime filter for cash-and-carry: results (2026-11-17)

Protocol: [`../preregistration_2026-11-17.md`](../preregistration_2026-11-17.md), committed
and pushed (525bb95, 19:05 CET) **before** the study was run. Code:
`scripts/eval_regime_filter.py`; engine change: `Config.tau` and a `gross` column in
`quantresearch/carry_lab.py`.

## R1: validation

With tau = 0 the engine reproduces the 2026-11-07 grid exactly (CC L=21 K=7: 5.9593
Binance, 6.3891 Bybit; baseline XS0: 1.3197, 1.2375). The live replay with tau = 0 also
reproduces the live CC ledger to the cent ($99,883.15 after 31 steps).

## Results (excess over cash; idle capital earns the T-bill yield)

| tau | venue | excess Sharpe (8h) | excess Sharpe (monthly) | excess %/yr | net %/yr | cost %/yr | capital in use |
|---|---|---:|---:|---:|---:|---:|---:|
| 0% | Binance | 4.62 | 1.65 | 10.9 | 14.1 | 2.2 | 96% |
| 5% | Binance | 4.77 | 1.65 | 10.9 | 13.5 | 2.3 | 81% |
| 10% | Binance | 4.87 | 1.67 | 10.9 | 12.3 | 1.6 | 47% |
| 15% | Binance | **6.74** | 1.42 | 9.0 | 9.5 | 0.8 | 22% |
| 0% | Bybit | 2.79 | 1.31 | 3.4 | 7.9 | 2.7 | 98% |
| 5% | Bybit | 3.27 | 1.44 | 3.7 | 7.5 | 2.6 | 84% |
| 10% | Bybit | 4.62 | 1.69 | 4.1 | 6.5 | 1.8 | 52% |
| 15% | Bybit | **5.76** | 1.44 | 3.1 | 3.9 | 0.6 | 15% |

Excess return per calendar year (%):

| | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|
| Binance tau 0% | 38.2 | 4.1 | 2.4 | 7.0 | -1.7 |
| Binance tau 10% | 37.4 | 4.2 | 2.1 | 7.2 | -0.3 |
| Binance tau 15% | 34.9 | 0.0 | 1.3 | 6.1 | 0.0 |
| Bybit tau 0% | | 1.1 | 3.5 | 6.9 | -3.0 |
| Bybit tau 10% | | 1.7 | 3.5 | 7.7 | -0.7 |
| Bybit tau 15% | | 0.0 | 2.2 | 6.6 | 0.0 |

## R2 and R3

The maximin excess Sharpe selects **tau = 15%** (5.76), which beats tau = 0 on both
venues (6.74 > 4.62, 5.76 > 2.79). It passes R3.

## R4: holdout (Bybit, 6 June to 6 November 2026, evaluated once)

| tau | excess %/yr | net %/yr | carry %/yr | cost %/yr | capital in use |
|---|---:|---:|---:|---:|---:|
| 0% | -5.3 | -1.8 | 2.0 | 3.5 | 96% |
| 5% | -3.6 | -1.9 | 0.5 | 2.2 | 45% |
| 10% | -0.1 | +0.2 | 0.4 | 0.3 | 6% |
| 15% | 0.0 | 0.0 | 0.0 | 0.0 | 0% |

With tau = 15% no coin cleared the hurdle during the holdout: the book was in cash for
the whole five months, which beat the control by 5.3 points a year of excess return.

## R5: deployment

A fourth live shadow ledger, `forward_test_cc.py --tau 0.15` (state
`forward_state_cc_tau15.json`), starts at 2026-11-18 00:00 UTC next to the unchanged
tau = 0 ledger. A replay of 7 to 17 November shows it would have been in cash for the
whole period.

## Reading the result honestly

- **The primary metric flatters a book that is mostly in cash.** Periods in cash have an
  excess return of exactly zero, which lowers the volatility of 8h returns and raises
  their Sharpe ratio. On monthly returns, and on excess return per year, tau = 10% is
  slightly better than tau = 15% on both venues (monthly 1.67 / 1.69 against 1.42 / 1.44).
  The protocol picked tau = 15% and that choice stands; tau = 10% is the more balanced
  setting and is reported here for that reason, not deployed.
- **The filter does not create return, it avoids losing it.** Every tau earns about the
  same in good years. The gain is in the bad years (2025, the 2026 holdout), where the
  filter sits in cash instead of paying costs for carry below the cash yield.
- **Four trials, one obvious direction.** The hurdle is one parameter with a clear
  economic meaning, so the multiple-testing risk is small, but the samples still hold
  only one or two bad-carry regimes.
