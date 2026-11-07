# Pre-registration: multi-asset portfolio study (2026-11-07)

Written and committed **before** any test was run. Nothing here may change after results
exist; deviations go in the results report, labelled as deviations.

## Goal (set by the owner)

The highest annual return subject to a **maximum drawdown of about 20%**, consistent
across market regimes, compared with the S&P 500. A 30%/yr return was discussed and
judged unrealistic at that drawdown (it needs a sustained Sharpe ratio near 2.6); it is
not a target of this study.

## Principle against overfitting

Only standard parameters from the literature are used. No signal parameter is
optimised. Exactly one quantity is calibrated per portfolio, by a rule fixed here: the
volatility target (see "Risk scaling").

## Data

Daily adjusted closes (dividends reinvested) from Yahoo Finance; cash yield from the
13-week US T-bill (^IRX).

| Group | Instruments |
|---|---|
| Equities | SPY, QQQ, IWM, EFA, EEM |
| Bonds | TLT, IEF, LQD, TIP |
| Real assets | GLD, DBC, VNQ |
| Currency | UUP (trend-following only) |
| Crypto sleeve | BTC-USD (from 2015, only in portfolio P3) |

An instrument enters once it has 12 months of history.

| Period | Dates | Use |
|---|---|---|
| Selection | 2007-07 to 2016-12 | includes 2008; calibration of the volatility target |
| Confirmation | 2017-01 to 2022-12 | includes 2018 Q4, 2020, 2022 |
| Holdout | 2023-01 to 2026-11-06 | used **once**, after the candidate is chosen |

## Building blocks (monthly rebalance)

Signals at the month-end close, trades at the next day's close (one-day lag).

| Code | Block | Rule |
|---|---|---|
| BENCH | S&P 500 | buy and hold SPY |
| SMA | trend-filtered S&P | SPY when its close is above its 10-month moving average, else cash (Faber 2007) |
| RP | risk parity | long-only, weights proportional to 1 / (60-day volatility) over the 12 non-currency ETFs, gross 100% |
| TF | trend following | per instrument, signal = mean of sign(1, 3, 6, 12-month return), in [-1, 1]; weight proportional to signal / (60-day volatility), all 13 ETFs, long and short (Moskowitz, Ooi and Pedersen 2012; Hurst, Ooi and Pedersen 2017) |
| TFL | long-only trend | as TF with negative signals set to zero (cash) |

## Portfolios

| Code | Composition (equal risk budgets: each block scaled to the same volatility first) |
|---|---|
| P1 | RP + TF |
| P2 | SPY + TF |
| P3 | RP + TF + BTC trend sleeve (BTC uses the TF signal; 10% of the risk budget) |
| P4 | SMA + TF |
| P5 | RP + TFL (long-only version of P1, for accounts that cannot short) |

The blocks SMA, RP, TF and TFL are also evaluated on their own. Total: BENCH plus
9 risk-scaled strategies.

## Risk scaling (the single calibrated quantity)

Every strategy except BENCH is scaled monthly to a target annual volatility using the
60-day realised volatility of its unscaled returns, with leverage capped at 3x. The
target is the highest value on the grid 4%, 5%, ..., 25% for which the **maximum
drawdown in the Selection period is at most 15%** (a 5-point buffer below the 20% limit).
It is then frozen for Confirmation and Holdout.

## Costs and financing

5 bps per unit of turnover. Unused capital earns the T-bill yield. Gross exposure above
100% is financed at T-bill + 1%. Short positions pay an extra 0.5% a year borrow fee.

## Metrics

CAGR, annualised volatility, Sharpe ratio on excess returns over T-bills, maximum
drawdown, worst calendar year, share of positive calendar years, correlation with SPY.

## Decision rules

- **D1, feasibility.** A strategy qualifies only if its maximum drawdown is at most 20%
  in **both** Selection and Confirmation, and its excess Sharpe beats SPY's in both.
- **D2, selection.** Among qualifying strategies, the candidate has the highest
  `min(CAGR Selection, CAGR Confirmation)`.
- **D3, holdout.** Candidate, SPY and all qualifying strategies are evaluated once on the
  Holdout. Reported whatever the result; nothing is re-tuned afterwards.
- **D4, deployment.** The candidate starts as a monthly paper-trading ledger. Real money
  is the owner's decision after at least 6 months of paper results.

## Honest expectation, written down beforehand

A diversified trend plus risk-parity portfolio has historically reached an excess Sharpe
ratio of about 0.7 to 1.0. At a 15 to 20% drawdown that implies roughly 10 to 15% a
year, not 30%.
