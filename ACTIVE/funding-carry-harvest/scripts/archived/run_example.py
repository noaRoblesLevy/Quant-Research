"""
End-to-end example: load data -> generate signals -> backtest -> metrics + FTMO.

Run WITHOUT any data or MT5 (sanity check the harness):
    python scripts/run_example.py --synthetic

Run on real MT5 data (after `pip install MetaTrader5` and downloading):
    python scripts/run_example.py --symbol US100.cash --timeframe H1 --start 2018-01-01

Note: the demo strategy (Bollinger mean reversion) is a TEMPLATE, not an edge.
On synthetic random-walk data it should land near break-even minus costs — which
confirms the engine isn't manufacturing P&L.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantresearch.backtest import BacktestConfig, ExitConfig, run_backtest
from quantresearch.evaluation import print_summary, print_ftmo, FTMORules
from quantresearch.strategies import BollingerMeanReversion


def load_data(args):
    if args.synthetic:
        from quantresearch.data.synthetic import make_ohlc
        print("Using SYNTHETIC random-walk data (no edge by construction).")
        return make_ohlc(n_bars=args.bars)
    from quantresearch.data.mt5_loader import fetch_bars
    print(f"Loading {args.symbol} {args.timeframe} from MT5 (or cache)...")
    return fetch_bars(args.symbol, args.timeframe, args.start, args.end)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="use synthetic data, no MT5 needed")
    p.add_argument("--bars", type=int, default=30000, help="synthetic bar count")
    p.add_argument("--symbol", default="US100.cash")
    p.add_argument("--timeframe", default="H1")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--risk", type=float, default=0.005, help="risk per trade (fraction)")
    p.add_argument("--balance", type=float, default=100000.0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    df = load_data(args)
    print(f"  bars: {len(df)}  range: {df.index.min()} -> {df.index.max()}")

    # --- strategy: pure Bollinger mean reversion (demo) ---
    strat = BollingerMeanReversion(bb_period=20, bb_std=2.0, atr_period=14, atr_mult=1.5)
    signals = strat.generate_signals(df)

    # --- exits: scale out 50% at 1R, rest at 2R (simple demo ladder) ---
    exit_cfg = ExitConfig(tp_ladder=[(1.0, 0.5), (2.0, 0.5)])
    cfg = BacktestConfig(risk_pct=args.risk, initial_equity=args.balance, compounding=True)

    result = run_backtest(df, signals, exit_cfg, cfg)
    equity, trades = result["equity"], result["trades"]

    print_summary(equity, trades, title=f"{strat.name} on {'synthetic' if args.synthetic else args.symbol}")
    print_ftmo(equity, FTMORules(label="FTMO 2-step Phase 1"), initial_balance=args.balance)

    if args.plot:
        import matplotlib.pyplot as plt
        equity.plot(title="Equity curve", figsize=(11, 4))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
