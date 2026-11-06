"""
Test the Market Intraday Momentum claim (Gao, Han, Li & Zhou 2018) on our index CFDs.

Core claim: the FIRST half-hour return of the cash session (from the prior close)
predicts the LAST half-hour return. We test this directly before building anything:
  1. auto-detect each index's cash session from its 30-min volume profile
  2. compute first-bar return r_first and last-bar return r_last per day
  3. regress r_last on r_first (slope > 0, t > 2 = momentum) and test the trading
     signal sign(r_first) * r_last (gross, then note costs)

    python scripts/test_intraday_momentum.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
INDICES = ["US100_cash", "US500_cash", "US30_cash", "US2000_cash",
           "GER40_cash", "UK100_cash", "JP225_cash", "EU50_cash"]

# Known cash sessions (server EET): US indices = NY 9:30-16:00 ET -> 16:30-22:30 server.
SESSIONS = {
    "US100_cash": ("16:30", "22:30"), "US500_cash": ("16:30", "22:30"),
    "US30_cash": ("16:30", "22:30"), "US2000_cash": ("16:30", "22:30"),
}


def detect_session(df):
    """Auto-detect the cash session as the longest contiguous high-RANGE block,
    after removing the midnight rollover artifact bar (extreme range outlier)."""
    rng = (df["high"] - df["low"]) / df["close"]
    slots = df.index.strftime("%H:%M")
    prof = rng.groupby(slots).median()
    prof = prof[prof < prof.median() * 8]          # drop the 00:00 rollover spike
    names, vals = list(prof.index), prof.values
    thr = np.median(vals) * 1.5
    above = vals > thr
    best = (0, 0, -1)
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            if j - i > best[2]:
                best = (i, j, j - i)
            i = j + 1
        else:
            i += 1
    return names[best[0]], names[best[1]]


def mim_series(df, first_slot, last_slot):
    """Per-day r_first (incl. overnight gap) and r_last (last 30-min move)."""
    slot = df.index.strftime("%H:%M")
    date = df.index.normalize()
    d = pd.DataFrame({"date": date, "slot": slot,
                      "open": df["open"].values, "close": df["close"].values})
    first = d[d.slot == first_slot].set_index("date")
    last = d[d.slot == last_slot].set_index("date")
    j = pd.DataFrame({
        "first_close": first["close"], "first_open": first["open"],
        "last_open": last["open"], "last_close": last["close"],
    }).dropna()
    prev_close = j["last_close"].shift(1)               # previous session close
    r_first = j["first_close"] / prev_close - 1.0       # prior close -> end of first 30min
    r_last = j["last_close"] / j["last_open"] - 1.0      # the last 30-min return
    return pd.DataFrame({"r_first": r_first, "r_last": r_last}).dropna()


def main():
    rows = []
    pooled = []
    for sym in INDICES:
        f = CACHE / f"{sym}_M30.parquet"
        if not f.exists():
            print(f"  (missing {sym}_M30.parquet — skip)")
            continue
        df = pd.read_parquet(f)
        fs, ls = SESSIONS.get(sym, detect_session(df))
        mm = mim_series(df, fs, ls)
        if len(mm) < 200:
            print(f"  {sym}: too few days ({len(mm)}) — skip")
            continue

        # regression r_last = a + b * r_first
        x, y = mm["r_first"].values, mm["r_last"].values
        b = np.cov(x, y)[0, 1] / np.var(x)
        resid = y - (y.mean() + b * (x - x.mean()))
        se = np.sqrt((resid @ resid) / (len(x) - 2) / (np.var(x) * len(x)))
        t_b = b / se

        # trading signal: position = sign(r_first), pnl = position * r_last
        pnl = np.sign(mm["r_first"]) * mm["r_last"]
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() else 0
        mid = len(pnl) // 2
        sh1 = pnl.iloc[:mid].mean() / pnl.iloc[:mid].std() * np.sqrt(252)
        sh2 = pnl.iloc[mid:].mean() / pnl.iloc[mid:].std() * np.sqrt(252)

        rows.append({"index": sym, "session": f"{fs}-{ls}", "days": len(mm),
                     "beta": round(b, 3), "t_beta": round(t_b, 2),
                     "sig_Sharpe": round(sharpe, 2), "win%": round((pnl > 0).mean() * 100, 1),
                     "ShH1": round(sh1, 2), "ShH2": round(sh2, 2)})
        pooled.append(pnl)

    print("\n=== Market Intraday Momentum test (first 30min -> last 30min) ===")
    print("    beta>0 & t>2 = predictability; sig_Sharpe = sign(r_first)*r_last strategy (GROSS)\n")
    print(pd.DataFrame(rows).to_string(index=False))

    if pooled:
        allp = pd.concat(pooled)
        sh = allp.mean() / allp.std() * np.sqrt(252)
        t = allp.mean() / (allp.std() / np.sqrt(len(allp)))
        print(f"\nPOOLED across indices: {len(allp)} index-days | gross Sharpe {sh:.2f} | "
              f"mean/day {allp.mean()*1e4:.2f} bps | t-stat {t:.2f} | win {(allp>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
