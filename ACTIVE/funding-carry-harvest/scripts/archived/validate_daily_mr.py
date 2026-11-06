"""
Validate the Daily mean-reversion edge before building portfolio infrastructure.

Two questions, both designed to catch curve-fitting / luck:
  1. TEMPORAL: does the edge appear in BOTH halves of history (not just one period)?
  2. PARAMETER: does it hold across a neighbourhood of settings (a plateau, not a spike)?

An edge that only lives in one half, or at one magic parameter, is noise.

    python scripts/validate_daily_mr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantresearch.backtest import BacktestConfig, ExitConfig, run_backtest
from quantresearch.evaluation.metrics import summarize
from quantresearch.strategies import BollingerReversionExitAtMean

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
FX = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread_price": "median"}
    return df.resample("1D").agg(agg).dropna()


def run(d: pd.DataFrame, bb_std=2.0, atr_mult=2.0) -> dict:
    strat = BollingerReversionExitAtMean(bb_period=20, bb_std=bb_std, atr_period=14, atr_mult=atr_mult)
    sig = strat.generate_signals(d)
    res = run_backtest(d, sig, ExitConfig(tp_ladder=[]), BacktestConfig(risk_pct=0.005))
    return summarize(res["equity"], res["trades"])


def main():
    daily = {s: to_daily(pd.read_parquet(CACHE / f"{s}_H1.parquet")) for s in FX}

    # ---------- 1. Temporal split-half ----------
    print("\n=== 1. TEMPORAL ROBUSTNESS (default params bb_std=2.0, atr_mult=2.0) ===")
    print("    Edge must be positive in BOTH halves to be trusted.\n")
    rows = []
    for s, d in daily.items():
        mid = d.index[len(d) // 2]
        first, second = d.loc[d.index < mid], d.loc[d.index >= mid]
        a, b = run(first), run(second)
        rows.append({
            "market": s,
            "H1_trades": a.get("n_trades", 0), "H1_exp_R": round(a.get("expectancy_r", 0), 3),
            "H1_ret%": round(a["total_return_pct"], 1),
            "H2_trades": b.get("n_trades", 0), "H2_exp_R": round(b.get("expectancy_r", 0), 3),
            "H2_ret%": round(b["total_return_pct"], 1),
            "both_pos": "YES" if (a.get("expectancy_r", 0) > 0 and b.get("expectancy_r", 0) > 0) else "no",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 2. Parameter robustness ----------
    print("\n\n=== 2. PARAMETER ROBUSTNESS (full history, expectancy_R) ===")
    print("    Want a PLATEAU of positive values, not a single lucky spike.\n")
    for s in ["EURUSD", "GBPUSD"]:
        d = daily[s]
        grid = []
        for bb in [1.5, 2.0, 2.5, 3.0]:
            row = {"bb_std": bb}
            for am in [1.5, 2.0, 3.0]:
                r = run(d, bb_std=bb, atr_mult=am)
                row[f"atr{am}"] = round(r.get("expectancy_r", 0), 3)
            grid.append(row)
        print(f"-- {s} (expectancy_R by bb_std rows x atr_mult cols) --")
        print(pd.DataFrame(grid).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
