"""
Does a trend-regime filter rescue the mean-reversion portfolio?

Runs the non-JPY FX H4 portfolio at several ADX_max thresholds (None = unfiltered
baseline). MR should improve as we cut out trending regimes — if it doesn't, that's
our signal to pivot to a different edge type.

    python scripts/test_regime_filter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantresearch.backtest import ExitConfig
from quantresearch.evaluation.metrics import summarize
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo
from quantresearch.portfolio import run_portfolio
from quantresearch.strategies import BollingerReversionExitAtMean

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
CCYS = {"USD", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD"}


def is_non_jpy_fx(s):
    return len(s) == 6 and s[:3] in CCYS and s[3:] in CCYS


def resample(df, rule="4h"):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread_price": "median"}
    return df.resample(rule).agg(agg).dropna()


def main():
    data = {f.stem[:-3]: resample(pd.read_parquet(f))
            for f in sorted(CACHE.glob("*_H1.parquet")) if is_non_jpy_fx(f.stem[:-3])}
    years = max((d.index.max() - d.index.min()).days for d in data.values()) / 365.25
    rules = FTMORules()

    rows = []
    for adx_max in [None, 35, 30, 25, 20, 15]:
        strat = BollingerReversionExitAtMean(bb_std=2.0, atr_mult=2.0, adx_max=adx_max)
        res = run_portfolio(data, strat.generate_signals, ExitConfig(tp_ladder=[]),
                            per_market_risk=0.0025)
        s = summarize(res["equity"], res["trades"])
        mc = monte_carlo(res["equity"], rules)
        rows.append({
            "ADX_max": "none" if adx_max is None else adx_max,
            "trades/wk": round(len(res["trades"]) / years / 52, 1),
            "ret%": round(s["total_return_pct"], 1),
            "maxDD%": round(s["max_dd_pct"], 1),
            "Sharpe": round(s["sharpe"], 2),
            "worstDay%": round(s["worst_day_pct"], 1),
            "win%": round(s.get("win_rate_pct", 0), 1),
            "exp_R": round(s.get("expectancy_r", 0), 3),
            "P(pass)%": round(mc.get("p_pass", 0) * 100, 1),
        })

    print("\n=== Regime filter sweep: non-JPY FX mean reversion @ H4 (risk 0.25%/mkt) ===")
    print("    ADX_max = upper bound on trend strength to allow an entry (lower = only quiet ranges)\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nGood result = maxDD collapses and Sharpe rises as ADX_max tightens.")


if __name__ == "__main__":
    main()
