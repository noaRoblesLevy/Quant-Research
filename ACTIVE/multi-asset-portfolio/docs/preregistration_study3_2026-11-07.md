# Pre-registration, study 3: a portfolio of uncorrelated strategies (2026-11-07)

Written and committed **before** any test of this study was run. Nothing here may change
after results exist; deviations go in the results report, labelled.

## Why

Study 2 showed that single building blocks reach a Sharpe ratio of about 0.35 to 0.65.
The Sharpe ratio of a portfolio grows with the number of **independent** return streams
("fundamental law": Sharpe ~ edge per bet x sqrt(independent bets)). This study combines
several low-correlation strategies across asset classes and also tests the owner's idea
of using whatever worked recently and replacing it when it stops working.

## Owner's constraints

Maximum drawdown about 20%; higher return is preferred over long-lived strategies.

## Sleeves (daily return streams, literature-standard rules, nothing optimised)

| Code | Sleeve | Rule |
|---|---|---|
| EQ | US equities | SPY, buy and hold |
| ETF_TF | multi-asset trend | the TF block of study 2 (13 ETFs, signal = mean sign of 1/3/6/12-month returns, weight ~ signal / 60-day vol) |
| FX_CARRY | G10 currency carry | 9 currencies vs USD (EUR, JPY, GBP, CHF, AUD, NZD, CAD, SEK, NOK). Monthly: long the 3 with the highest central-bank policy rate, short the 3 lowest, each weight ~ 1 / 60-day vol. Rates: BIS policy rates (WS_CBPOL, monthly), using the previous month's value. Returns include the daily interest differential |
| FX_TF | currency trend | TF rule on the same 9 currencies, returns including the interest differential |
| CRYPTO_TF | crypto trend | TF rule on BTC and ETH (ETH from 2018) |
| CRYPTO_CC | crypto cash-and-carry | the candidate of the 2026-11-07 funding study (CC L=21 K=7) on Bybit (the venue of the live ledger), from 2022-08; absent before. Its return is measured in excess of the T-bill rate (the spot capital could have earned cash) |

A sleeve enters once its rule has the history it needs. All sleeves are expressed as
daily **excess returns over the T-bill rate**; the portfolio earns the T-bill rate on its
capital plus the scaled sum of sleeve excess returns.

Data sources were fixed after checking availability and before any test: FRED refused
connections, so policy rates come from the BIS; Binance data stops in May 2025, so
cash-and-carry uses Bybit, extended to October 2026.

## Combinations (monthly rebalance, signals at month end, trades next day)

| Code | Rule |
|---|---|
| EQRISK | equal risk: each available sleeve weighted by 1 / its 60-day volatility |
| WF_SHARPE | the owner's idea, as a rule: at each quarter end, weight each sleeve by max(trailing 252-day Sharpe, 0) / 60-day volatility; if every sleeve is non-positive, fall back to EQRISK |
| WF_TOP2 | concentrated version: equal risk in only the 2 sleeves with the best trailing 252-day Sharpe, re-chosen each quarter |
| EX_EQ | EQRISK without the equity sleeve (a market-neutral portfolio) |

Each sleeve on its own is reported too. Every strategy is scaled to a volatility target
calibrated exactly as in study 2: the highest of 4%, 5%, ..., 25% with maximum drawdown
at most 15% in the Selection period, leverage capped at 3x, then frozen. Costs: 5 bps per
unit turnover (ETFs, FX), the funding study's 8 bps per leg for CRYPTO_CC; cash at the
T-bill rate, leverage at T-bill + 1%, short borrow fee 0.5% a year on ETFs.

## Periods

| Period | Dates |
|---|---|
| Selection (calibration) | 2007-07 to 2016-12 |
| Confirmation | 2017-01 to 2022-12 |
| Holdout (once) | 2023-01 to 2026-10 |

## Metrics

CAGR, volatility, Sharpe on excess returns, maximum drawdown, worst year, share of
positive years, correlation with SPY; also reported (not used to decide): profit factor
on monthly returns, Sortino, Calmar. Plus the correlation matrix of the sleeves.

## Decision rules

- **D1 feasibility:** max drawdown at most 20% and Sharpe above SPY's in both Selection
  and Confirmation.
- **D2 selection:** among feasible strategies, the highest min(CAGR Selection, CAGR
  Confirmation).
- **D3 holdout:** the candidate, SPY and all feasible strategies, evaluated once.
- **H1, the owner's hypothesis ("use what worked recently"):** counts as supported only
  if WF_SHARPE's Sharpe exceeds EQRISK's in **both** Confirmation and Holdout. WF_TOP2 is
  reported alongside.
- **D4 deployment:** the candidate becomes a monthly paper ledger; real money is the
  owner's decision after at least 6 months.

## Expectation written down beforehand

If the sleeves are close to uncorrelated with Sharpe ratios of 0.3 to 0.6 each, an
equal-risk portfolio could reach 0.7 to 1.0. Chasing recent winners has a poor record
in the literature (performance chasing), so H1 is expected to fail, but it is tested.
