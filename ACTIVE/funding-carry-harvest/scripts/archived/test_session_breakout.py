"""
Test the opening-range breakout across the non-JPY FX universe (H1).

Runs several session configurations as a diversified portfolio, with split-half
robustness on each. Decisive test of whether a session breakout edge exists.

    python scripts/test_session_breakout.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantresearch.backtest import ExitConfig
from quantresearch.evaluation.metrics import summarize
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo
from quantresearch.portfolio import run_portfolio
from quantresearch.strategies import SessionBreakout

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
CCYS = {"USD", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD"}


def is_non_jpy_fx(s):
    return len(s) == 6 and s[:3] in CCYS and s[3:] in CCYS


# (label, open_hour, range_hours, close_hour) — server time is EET
CONFIGS = [
    ("London 9-2-18", 9, 2, 18),
    ("London 9-1-16", 9, 1, 16),
    ("NY 15-1-22", 15, 1, 22),
    ("LonOpen 9-3-17", 9, 3, 17),
]


def metrics(eq, trades, n_years):
    s = summarize(eq, trades)
    return {
        "trades/wk": round(len(trades) / n_years / 52, 1),
        "ret%": round(s["total_return_pct"], 1),
        "maxDD%": round(s["max_dd_pct"], 1),
        "Sharpe": round(s["sharpe"], 2),
        "win%": round(s.get("win_rate_pct", 0), 1),
        "exp_R": round(s.get("expectancy_r", 0), 3),
    }


def main():
    data = {f.stem[:-3]: pd.read_parquet(f)
            for f in sorted(CACHE.glob("*_H1.parquet")) if is_non_jpy_fx(f.stem[:-3])}
    years = max((d.index.max() - d.index.min()).days for d in data.values()) / 365.25
    rules = FTMORules()
    print(f"Universe: {len(data)} non-JPY FX pairs, H1, {years:.1f} years\n")

    rows = []
    for label, oh, rh, ch in CONFIGS:
        strat = SessionBreakout(open_hour=oh, range_hours=rh, close_hour=ch, atr_mult=1.5)
        port = run_portfolio(data, strat.generate_signals, ExitConfig(tp_ladder=[]),
                             per_market_risk=0.0025)
        eq, trades = port["equity"], port["trades"]

        m = metrics(eq, trades, years)
        mc = monte_carlo(eq, rules)

        # split-half expectancy (robustness)
        mid = trades["entry_time"].quantile(0.5) if len(trades) else None
        if mid is not None:
            e1 = trades[trades.entry_time < mid]["r"].mean()
            e2 = trades[trades.entry_time >= mid]["r"].mean()
        else:
            e1 = e2 = float("nan")

        rows.append({"config": label, **m,
                     "expH1": round(e1, 3), "expH2": round(e2, 3),
                     "P(pass)%": round(mc.get("p_pass", 0) * 100, 1)})

    print("=== Opening-range breakout — non-JPY FX portfolio (risk 0.25%/mkt) ===\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nLong-only / short-only check on best config would come next if any config shows promise.")
    print("Good = positive Sharpe, expectancy positive in BOTH halves, maxDD modest.")


if __name__ == "__main__":
    main()
