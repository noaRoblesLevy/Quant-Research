"""
Pull the REAL broker swap rates from MT5 for every symbol.

Two uses:
  1. Exact financing costs (replace the estimates in eval_swap_costs.py)
  2. A carry signal: ranking by swap_long is ranking by interest-rate differential

Swap handling depends on SYMBOL_SWAP_MODE:
  POINTS (1)            -> swap is in points/lot/day -> bps/day = swap*point/price*1e4
  INTEREST_CURRENT (5) -> swap_long is an annual % -> bps/day = swap/360*100

    python scripts/pull_swap_rates.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data_cache" / "swap_rates.parquet"
CCYS = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}


def _mt5():
    import MetaTrader5 as mt5  # noqa: N813
    return mt5


def main():
    mt5 = _mt5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    try:
        syms = [s.name for s in mt5.symbols_get()]
        fx = [s for s in syms if len(s) == 6 and s[:3] in CCYS and s[3:] in CCYS]
        extra = [s for s in ["BTCUSD", "XAUUSD", "USOIL.cash", "US100.cash"] if s in syms]
        rows = []
        for sym in fx + extra:
            mt5.symbol_select(sym, True)
            info = mt5.symbol_info(sym)
            if info is None:
                continue
            price = info.bid or info.ask or 1.0
            mode = info.swap_mode
            if mode == 1:   # POINTS
                bl = info.swap_long * info.point / price * 1e4
                bs = info.swap_short * info.point / price * 1e4
            elif mode in (5, 6):  # INTEREST (% annual)
                bl = info.swap_long / 360 * 100
                bs = info.swap_short / 360 * 100
            else:
                bl = bs = float("nan")
            rows.append({"symbol": sym, "swap_mode": mode,
                         "swap_long_raw": info.swap_long, "swap_short_raw": info.swap_short,
                         "carry_long_bps": round(bl, 3), "carry_short_bps": round(bs, 3)})
    finally:
        mt5.shutdown()

    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    pd.set_option("display.max_rows", None, "display.width", 140)
    print("Swap rates (bps/day to HOLD that direction; positive = you EARN):\n")
    print(df[df.symbol.str.len() == 6].sort_values("carry_long_bps", ascending=False).to_string(index=False))
    print("\nNon-FX:")
    print(df[df.symbol.str.len() != 6].to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
