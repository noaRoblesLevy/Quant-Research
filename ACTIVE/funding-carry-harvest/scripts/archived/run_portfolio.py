"""
Portfolio backtest: diversified H4 mean reversion across all NON-JPY FX pairs.

The basket is defined by a STRUCTURAL RULE (non-JPY FX), not by cherry-picking
the pairs that backtested well — that avoids selection bias. Per-market risk is
small so the worst-case simultaneous exposure stays within FTMO limits.

    python scripts/run_portfolio.py
    python scripts/run_portfolio.py --tf 1D --risk 0.004
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantresearch.backtest import ExitConfig
from quantresearch.evaluation.metrics import print_summary
from quantresearch.evaluation.ftmo import FTMORules, print_ftmo
from quantresearch.portfolio import run_portfolio
from quantresearch.strategies import BollingerReversionExitAtMean

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
CCYS = {"USD", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD"}  # NOTE: JPY excluded by rule


def is_non_jpy_fx(sym: str) -> bool:
    return len(sym) == 6 and sym[:3] in CCYS and sym[3:] in CCYS


def resample(df, rule):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread_price": "median"}
    return df.resample(rule).agg(agg).dropna()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="4h", help="resample rule, e.g. 4h or 1D")
    p.add_argument("--risk", type=float, default=0.0025, help="risk per market per trade")
    p.add_argument("--account", type=float, default=100000.0)
    p.add_argument("--half", default="all", choices=["all", "first", "second"],
                   help="evaluate on a time half (quasi-OOS)")
    args = p.parse_args()

    files = sorted(CACHE.glob("*_H1.parquet"))
    data = {}
    for f in files:
        sym = f.stem[:-3]
        if not is_non_jpy_fx(sym):
            continue
        d = resample(pd.read_parquet(f), args.tf)
        if args.half != "all":
            mid = d.index[len(d) // 2]
            d = d.loc[d.index < mid] if args.half == "first" else d.loc[d.index >= mid]
        data[sym] = d

    print(f"Basket: {len(data)} non-JPY FX pairs @ {args.tf}  | per-market risk {args.risk:.2%}  | half={args.half}")
    print("  " + ", ".join(data))

    strat = BollingerReversionExitAtMean(bb_period=20, bb_std=2.0, atr_period=14, atr_mult=2.0)
    res = run_portfolio(data, strat.generate_signals, ExitConfig(tp_ladder=[]),
                        account=args.account, per_market_risk=args.risk)

    eq, trades = res["equity"], res["trades"]
    years = (eq.index.max() - eq.index.min()).days / 365.25
    print(f"  Aggregate trades: {len(trades)}  (~{len(trades)/years/52:.1f}/week over {years:.1f}y)")

    print_summary(eq, trades, title=f"Portfolio: non-JPY FX mean reversion @ {args.tf}")
    print_ftmo(eq, FTMORules(), initial_balance=args.account)


if __name__ == "__main__":
    main()
