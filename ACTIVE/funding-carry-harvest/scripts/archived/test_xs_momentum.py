"""
Test cross-sectional FX momentum across the 8 major currencies.

Sweeps the momentum lookback and portfolio breadth, reports gross AND net (after a
rough spread cost) performance, and a split-half robustness check on each config.

    python scripts/test_xs_momentum.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantresearch.cross_sectional import build_currency_returns, xs_momentum, perf

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
CCYS = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}   # JPY INCLUDED for momentum


def is_fx(s):
    return len(s) == 6 and s[:3] in CCYS and s[3:] in CCYS


def split_sharpe(curr, lookback, n_sel):
    res = xs_momentum(curr, lookback_weeks=lookback, n_sel=n_sel)
    port = res["weekly"]
    mid = port.index[len(port) // 2]
    p1, p2 = port[port.index < mid], port[port.index >= mid]
    def sh(p):
        return round((p.mean() * 52) / (p.std() * (52 ** 0.5)), 2) if p.std() > 0 else 0.0
    return sh(p1), sh(p2)


def main():
    data = {f.stem[:-3]: pd.read_parquet(f)
            for f in sorted(CACHE.glob("*_H1.parquet")) if is_fx(f.stem[:-3])}
    print(f"Loaded {len(data)} FX pairs -> building 8-currency strength matrix...")
    curr = build_currency_returns(data)
    print(f"Currencies: {list(curr.columns)}")
    print(f"Daily currency-return matrix: {curr.shape[0]} days, {curr.index.min().date()} -> {curr.index.max().date()}\n")

    rows = []
    for lookback in [4, 8, 12, 26, 52]:
        for n_sel in [2, 3]:
            res = xs_momentum(curr, lookback_weeks=lookback, n_sel=n_sel)
            p = perf(res["weekly"], res["equity"], res["turnover"], spread_bps_per_leg=1.0, n_sel=n_sel)
            s1, s2 = split_sharpe(curr, lookback, n_sel)
            rows.append({
                "lookback_w": lookback, "n_sel": n_sel,
                "gross_Sh": p["sharpe"], "net_Sh": p["net_sharpe"],
                "ann_ret%": p["ann_ret_pct"], "net_ann%": p["net_ann_ret_pct"],
                "maxDD%": p["max_dd_pct"], "hit%": p["hit_rate_pct"],
                "ShH1": s1, "ShH2": s2,
                "both+": "YES" if (s1 > 0 and s2 > 0) else "",
            })

    print("=== Cross-sectional FX momentum sweep (weekly rebalance, long/short top/bottom n_sel) ===\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nGood = positive net Sharpe, robust across lookbacks, positive in BOTH halves (both+).")


if __name__ == "__main__":
    main()
