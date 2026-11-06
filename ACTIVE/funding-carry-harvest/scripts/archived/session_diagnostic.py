"""
Session / time-of-day diagnostic (EDA, not a strategy).

Measures, by hour-of-day (broker server time), across the USD FX majors:
  * activity   : mean |return| and mean range per bar -> reveals the sessions
  * direction  : mean signed return + a t-stat -> is there a tradeable hourly drift?
  * persistence: does an hour's move continue into the next hours (momentum) or
                 reverse (mean reversion)? -> tells us which kind of session edge fits

This locates the London/NY sessions empirically (from the volatility peaks) so we
don't have to guess the broker's timezone, and tells us what to build next.

    python scripts/session_diagnostic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]


def per_pair_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar stats grouped by hour-of-day."""
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    ret = (c - o) / o                  # this hour's signed return
    rng = (h - l) / o                  # this hour's range (activity)
    g = pd.DataFrame({"hour": df.index.hour, "ret": ret, "rng": rng, "absret": ret.abs()})
    agg = g.groupby("hour").agg(
        mean_ret=("ret", "mean"),
        std_ret=("ret", "std"),
        n=("ret", "size"),
        mean_absret=("absret", "mean"),
        mean_rng=("rng", "mean"),
        pct_up=("ret", lambda x: (x > 0).mean()),
    )
    return agg


def main():
    BPS = 1e4  # report in basis points

    # average the hourly profile across majors
    profiles = []
    for sym in MAJORS:
        df = pd.read_parquet(CACHE / f"{sym}_H1.parquet")
        profiles.append(per_pair_hourly(df))

    avg = sum(p[["mean_ret", "mean_absret", "mean_rng", "pct_up"]] for p in profiles) / len(profiles)
    # t-stat for directional drift, pooled (mean / (std/sqrt(n))) averaged across pairs
    tstats = []
    for p in profiles:
        tstats.append(p["mean_ret"] / (p["std_ret"] / np.sqrt(p["n"])))
    avg["t_stat"] = sum(tstats) / len(tstats)

    out = pd.DataFrame({
        "hour": avg.index,
        "activity_bps": (avg["mean_rng"] * BPS).round(1),     # range = volatility proxy
        "mean_ret_bps": (avg["mean_ret"] * BPS).round(2),     # directional drift
        "t_stat": avg["t_stat"].round(2),
        "pct_up": (avg["pct_up"] * 100).round(1),
    })

    print("\n=== Hour-of-day profile (avg of 7 USD majors, broker server time) ===")
    print("    activity = mean bar range (bps); high = active session")
    print("    mean_ret = average per-bar drift (bps); |t_stat|>2 ~ statistically notable\n")
    print(out.to_string(index=False))

    # locate sessions from activity peaks
    peak = out.loc[out["activity_bps"].idxmax(), "hour"]
    quiet = out.loc[out["activity_bps"].idxmin(), "hour"]
    print(f"\n  Most active hour: {peak:02d}:00 server  | Quietest: {quiet:02d}:00 server")
    print("  (London-NY overlap is the global vol peak ~ maps server time to UTC.)")

    # persistence test: correlation of consecutive-hour returns per pair, averaged
    print("\n=== Persistence: does an hour's move continue or reverse next hour? ===")
    acf1 = []
    for sym in MAJORS:
        df = pd.read_parquet(CACHE / f"{sym}_H1.parquet")
        r = ((df["close"] - df["open"]) / df["open"])
        acf1.append(r.corr(r.shift(1)))
    m = float(np.mean(acf1))
    kind = "MOMENTUM (continues)" if m > 0.02 else ("REVERSION (reverses)" if m < -0.02 else "neither (random)")
    print(f"  Lag-1 autocorrelation of hourly returns: {m:+.3f}  -> {kind}")


if __name__ == "__main__":
    main()
