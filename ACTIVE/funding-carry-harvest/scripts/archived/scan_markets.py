"""
Diagnostic scan: run ONE strategy across every cached market and compare.

This is a PROBE, not a finished edge. Its job is to reveal which markets suit a
given strategy style (e.g. mean reversion) so the real hypothesis work is aimed
at the right place. Judged by FTMO-relevant columns, not raw return.

    python scripts/scan_markets.py
    python scripts/scan_markets.py --timeframe H1 --risk 0.005
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantresearch.backtest import BacktestConfig, ExitConfig, run_backtest
from quantresearch.evaluation.metrics import summarize
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo
from quantresearch.strategies import BollingerMeanReversion, BollingerReversionExitAtMean

CACHE = Path(__file__).resolve().parents[1] / "data_cache"


def build(variant: str):
    """Return (strategy, exit_cfg) for the chosen probe variant."""
    if variant == "exit_at_mean":
        # exit at the mean; empty TP ladder so the rule-based exit governs the trade
        return BollingerReversionExitAtMean(atr_mult=2.0), ExitConfig(tp_ladder=[])
    # 'basic': fixed R-based scale-out (the original momentum-style exit)
    return (BollingerMeanReversion(bb_period=20, bb_std=2.0, atr_period=14, atr_mult=1.5),
            ExitConfig(tp_ladder=[(1.0, 0.5), (2.0, 0.5)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="H1")
    p.add_argument("--risk", type=float, default=0.005)
    p.add_argument("--balance", type=float, default=100000.0)
    p.add_argument("--variant", default="exit_at_mean", choices=["basic", "exit_at_mean"])
    args = p.parse_args()

    files = sorted(CACHE.glob(f"*_{args.timeframe}.parquet"))
    if not files:
        print(f"No cached *_{args.timeframe}.parquet files in {CACHE}")
        return

    rules = FTMORules()
    rows = []

    for f in files:
        symbol = f.stem[: -(len(args.timeframe) + 1)]  # strip "_H1"
        df = pd.read_parquet(f)

        strat, exit_cfg = build(args.variant)
        sig = strat.generate_signals(df)
        cfg = BacktestConfig(risk_pct=args.risk, initial_equity=args.balance, compounding=True)
        res = run_backtest(df, sig, exit_cfg, cfg)

        s = summarize(res["equity"], res["trades"])
        mc = monte_carlo(res["equity"], rules)
        rows.append({
            "market": symbol,
            "trades": s.get("n_trades", 0),
            "win%": round(s.get("win_rate_pct", 0), 1),
            "expectancy_R": round(s.get("expectancy_r", 0), 3),
            "ret%": round(s["total_return_pct"], 1),
            "maxDD%": round(s["max_dd_pct"], 1),
            "sharpe": round(s["sharpe"], 2),
            "P(pass)%": round(mc.get("p_pass", 0) * 100, 1) if "error" not in mc else None,
        })

    table = pd.DataFrame(rows).sort_values("expectancy_R", ascending=False)
    pd.set_option("display.max_columns", None, "display.width", 160)
    print(f"\n=== Mean-reversion probe across markets ({args.timeframe}, risk {args.risk:.1%}) ===")
    print(table.to_string(index=False))
    print("\nReading it: positive expectancy_R = mean-reversion has SOME edge there.")
    print("This is a probe to aim the hunt — not a tradeable strategy yet.")


if __name__ == "__main__":
    main()
