"""
Download + cache MT5 history for a set of symbols/timeframes.

Usage:
    python scripts/download_data.py --symbols US100.cash EURUSD XAUUSD --timeframe H1 --start 2015-01-01

Find your broker's exact symbol names first:
    python scripts/download_data.py --list xau
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=[])
    p.add_argument("--timeframe", default="H1")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--list", default=None, help="list symbols matching this substring and exit")
    args = p.parse_args()

    from quantresearch.data.mt5_loader import fetch_bars, list_symbols

    if args.list is not None:
        syms = list_symbols(args.list)
        print(f"{len(syms)} symbols matching '{args.list}':")
        for s in syms:
            print(f"  {s}")
        return

    if not args.symbols:
        print("No --symbols given. Use --list <substring> to find names first.")
        return

    for sym in args.symbols:
        try:
            df = fetch_bars(sym, args.timeframe, args.start, args.end, refresh=True)
            print(f"  {sym} {args.timeframe}: {len(df)} bars  "
                  f"({df.index.min()} -> {df.index.max()})  cached.")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: FAILED — {e}")


if __name__ == "__main__":
    main()
