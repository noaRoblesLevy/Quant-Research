"""
Test the overnight-vs-intraday return anomaly on index CFDs.

Documented effect: equity indices earn their returns OVERNIGHT (cash close ->
next cash open), while the intraday session (open -> close) is flat or negative.
Tied to a structural overnight risk premium -> more robust to post-publication
decay than most anomalies. Low turnover (one hold/day) -> costs bite less.

We use the cash session detected for Market Intraday Momentum:
  overnight return = today_open / prev_close - 1   (the strategy: long close->open)
  intraday  return = today_close / today_open - 1

    python scripts/test_overnight.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
SESSIONS = {
    "US100_cash": ("16:30", "22:30"), "US500_cash": ("16:30", "22:30"),
    "US30_cash": ("16:30", "22:30"), "US2000_cash": ("16:30", "22:30"),
    "GER40_cash": ("10:00", "18:00"), "EU50_cash": ("10:00", "18:00"),
    "UK100_cash": ("10:00", "18:00"), "JP225_cash": ("02:30", "08:00"),
}


def sessions_prices(df, first_slot, last_slot):
    slot = df.index.strftime("%H:%M")
    date = df.index.normalize()
    d = pd.DataFrame({"date": date, "slot": slot,
                      "open": df["open"].values, "close": df["close"].values})
    sess_open = d[d.slot == first_slot].set_index("date")["open"]
    sess_close = d[d.slot == last_slot].set_index("date")["close"]
    j = pd.DataFrame({"open": sess_open, "close": sess_close}).dropna()
    j["overnight"] = j["open"] / j["close"].shift(1) - 1.0    # prev close -> today open
    j["intraday"] = j["close"] / j["open"] - 1.0              # today open -> today close
    return j.dropna()


def ann_sharpe(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0


def main():
    rows = []
    on_pool, id_pool = [], []
    for sym, (fs, ls) in SESSIONS.items():
        f = CACHE / f"{sym}_M30.parquet"
        if not f.exists():
            continue
        j = sessions_prices(pd.read_parquet(f), fs, ls)
        if len(j) < 200:
            continue
        on, intr = j["overnight"], j["intraday"]
        mid = len(on) // 2
        rows.append({
            "index": sym, "days": len(on),
            "ON_bps": round(on.mean() * 1e4, 2), "ID_bps": round(intr.mean() * 1e4, 2),
            "ON_Sharpe": round(ann_sharpe(on), 2),
            "ON_ShH1": round(ann_sharpe(on.iloc[:mid]), 2),
            "ON_ShH2": round(ann_sharpe(on.iloc[mid:]), 2),
            "ON_win%": round((on > 0).mean() * 100, 1),
        })
        on_pool.append(on)
        id_pool.append(intr)

    print("\n=== Overnight vs Intraday return anomaly (index CFDs) ===")
    print("    ON = overnight (close->open), ID = intraday (open->close), in bps/day\n")
    print(pd.DataFrame(rows).to_string(index=False))

    if on_pool:
        on = pd.concat(on_pool)
        idr = pd.concat(id_pool)
        print(f"\nPOOLED: overnight {on.mean()*1e4:+.2f} bps/day (Sharpe {ann_sharpe(on):.2f}, "
              f"win {(on>0).mean()*100:.1f}%) vs intraday {idr.mean()*1e4:+.2f} bps/day "
              f"(Sharpe {ann_sharpe(idr):.2f})")
        # crude cost check: index CFD spread ~ 1-2 bps; overnight gap must beat it
        print(f"  Overnight mean {on.mean()*1e4:.2f} bps vs ~1-2 bps spread -> "
              f"{'clears costs' if on.mean()*1e4 > 2 else 'MARGINAL vs costs'}")


if __name__ == "__main__":
    main()
