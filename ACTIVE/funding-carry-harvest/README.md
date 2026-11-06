# FTMO Edge Research

A small, honest strategy-research harness for hunting **FTMO-survivable edges** on
forex/CFD data. Built around one principle learned the hard way: *judge every idea
by whether it would pass and survive the actual challenge — not by raw profit.*

## Why this exists

The MT5 Strategy Tester optimises for return. FTMO doesn't pay for return — it pays
for **staying under a 5% daily loss and a 10% max drawdown while reaching +10%.**
Those are different objectives. This harness measures the FTMO one directly,
including a Monte-Carlo **probability of passing**.

## Install

```bash
cd C:\Users\noaro\QUANT\Quant-Research\FTMO_Edge_Research
pip install -r requirements.txt
```

`MetaTrader5` is Windows-only and needs the MT5 terminal installed + logged in
(a free FTMO demo works). Everything except live data works without it.

## Verify it works (no data, no MT5 needed)

```bash
python tests/test_engine.py                 # pins down the engine accounting
python scripts/run_example.py --synthetic   # full pipeline on random-walk data
```

The synthetic run should land near break-even minus costs — proof the engine
isn't inventing P&L.

## Use real FTMO data

```bash
# 1. find your broker's exact symbol names
python scripts/download_data.py --list us100
python scripts/download_data.py --list xau

# 2. download + cache
python scripts/download_data.py --symbols US100.cash EURUSD XAUUSD --timeframe H1 --start 2015-01-01

# 3. run a strategy with metrics + FTMO pass-probability
python scripts/run_example.py --symbol US100.cash --timeframe H1 --start 2018-01-01
```

## Package layout

```
quantresearch/
  data/         mt5_loader.py (real feed)  ·  synthetic.py (no-data testing)
  indicators.py EMA / SMA / ATR / RSI / Bollinger
  backtest/     engine.py — event-driven, scale-out exits, bar-level equity (incl. floating)
  evaluation/   metrics.py (Sharpe/Sortino/MAE-MFE/…)  ·  ftmo.py (challenge simulator + Monte Carlo)
  validation/   walkforward.py — OOS & rolling walk-forward splitters
  strategies/   base.py (interface)  ·  example_meanreversion.py (DEMO template)
scripts/        download_data.py  ·  run_example.py
tests/          test_engine.py (standalone self-test)
data_cache/     parquet cache (auto-filled)
```

## How to research an edge here (the discipline)

1. **Hypothesis first.** Write down *who's on the other side and why they lose*
   before testing. No economic story → probably data-mining.
2. **Write a `Strategy`** (entry signal + stop distance only). Keep exits separate
   so you can A/B them independently — exits can't rescue a bad entry.
3. **Backtest with realistic costs** (`spread_price` comes through from MT5).
4. **Read the FTMO evaluator, not the profit.** Look at `P(pass)`, `P(daily breach)`,
   max DD, worst day. A 25%-return/12%-DD system *fails*; an 8%/2% system *passes*.
5. **Validate out-of-sample** with `validation/` before believing anything.
6. **Discount for the number of ideas you tried** — if you test 50 things, a few
   will look good by luck.

## Key FTMO numbers baked into `evaluation/ftmo.py`

| Rule (2-step) | Phase 1 | Phase 2 |
|---|---|---|
| Profit target | +10% | +5% |
| Max daily loss | -5% (static, vs day start) | -5% |
| Max total loss | -10% (static floor) | -10% |
| Min trading days | 4 | 4 |

Edit `FTMORules(...)` to model the 1-step (−3% daily, −10% trailing) or other firms.

## Status

Scaffold complete and self-tested. The included strategy is a **template, not an
edge** — the real work (finding edges) starts next.
