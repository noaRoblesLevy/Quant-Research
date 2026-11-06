"""
Universe survivor filter for Daily/H4 mean reversion.

For every cached FX pair, at each timeframe:
  * split history in half
  * a market SURVIVES only if expectancy is positive in BOTH halves AND the
    full-history net return (after spread) is positive
This is the temporal-robustness gate that killed EURUSD/USDJPY earlier.

Output: survivors per timeframe, their per-year trade count, and the AGGREGATE
frequency of a portfolio that trades all survivors (does it clear ~3 trades/week?).

    python scripts/scan_universe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantresearch.backtest import BacktestConfig, ExitConfig, run_backtest
from quantresearch.evaluation.metrics import summarize
from quantresearch.strategies import BollingerReversionExitAtMean

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
CCYS = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
TFS = {"H4": "4h", "D1": "1D"}


def is_fx(symbol: str) -> bool:
    return len(symbol) == 6 and symbol[:3] in CCYS and symbol[3:] in CCYS


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread_price": "median"}
    return df.resample(rule).agg(agg).dropna()


def run(d: pd.DataFrame) -> dict:
    strat = BollingerReversionExitAtMean(bb_period=20, bb_std=2.0, atr_period=14, atr_mult=2.0)
    sig = strat.generate_signals(d)
    res = run_backtest(d, sig, ExitConfig(tp_ladder=[]), BacktestConfig(risk_pct=0.005))
    return summarize(res["equity"], res["trades"])


def main():
    files = sorted(CACHE.glob("*_H1.parquet"))
    fx = [(f, f.stem[:-3]) for f in files if is_fx(f.stem[:-3])]
    print(f"FX pairs found: {len(fx)}")

    rows = []
    for f, sym in fx:
        df = pd.read_parquet(f)
        years = (df.index.max() - df.index.min()).days / 365.25
        for tf, rule in TFS.items():
            d = resample(df, rule)
            mid = d.index[len(d) // 2]
            a = run(d.loc[d.index < mid])
            b = run(d.loc[d.index >= mid])
            full = run(d)
            survive = (a.get("expectancy_r", 0) > 0 and b.get("expectancy_r", 0) > 0
                       and full["total_return_pct"] > 0)
            rows.append({
                "market": sym, "tf": tf,
                "trades/yr": round(full.get("n_trades", 0) / years, 1),
                "expH1": round(a.get("expectancy_r", 0), 3),
                "expH2": round(b.get("expectancy_r", 0), 3),
                "net%": round(full["total_return_pct"], 1),
                "maxDD%": round(full["max_dd_pct"], 1),
                "win%": round(full.get("win_rate_pct", 0), 1),
                "SURVIVE": "YES" if survive else "",
            })

    table = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None, "display.width", 160)

    for tf in TFS:
        sub = table[table.tf == tf].sort_values("expH2", ascending=False)
        surv = sub[sub.SURVIVE == "YES"]
        print(f"\n=== {tf} — {len(surv)} survivors of {len(sub)} ===")
        print(sub.to_string(index=False))
        if len(surv):
            agg = surv["trades/yr"].sum()
            print(f"  Portfolio of {tf} survivors: {agg:.0f} trades/yr  (~{agg/52:.1f}/week)")
            print(f"  Survivors: {', '.join(surv['market'])}")


if __name__ == "__main__":
    main()
