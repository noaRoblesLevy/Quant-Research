"""
Stack-building hunt: find edges UNCORRELATED with the FX-reversion + index-overnight
book, to push the combined Sharpe from 0.71 toward the ~1.0 needed for a >50%
FTMO pass in ~2 months.

Candidates (chosen to be uncorrelated by construction):
  * crypto trend (BTC) - crypto TRENDS where FX reverts
  * turn-of-month seasonality (index basket) - a calendar effect
  * commodity trend (gold + oil) - different asset class

For each: daily return series, Sharpe, correlation with the existing book.
Then stack the positive low-correlation ones and report the new Sharpe + the
max FTMO pass-probability at optimal leverage (40-day horizon).

    python scripts/hunt_more_edges.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from quantresearch.cross_sectional import build_currency_returns
from quantresearch.edges import (daily_reversion_pnl, is_fx, basket,
                                 daily_close_ret, crypto_trend, turn_of_month)

CACHE = ROOT / "data_cache"
rng = np.random.default_rng(0)


def commodity_trend():
    """Long-short trend on gold + oil (TS momentum, 50-day)."""
    out = []
    for s in ["XAUUSD", "USOIL_cash"]:
        df = pd.read_parquet(CACHE / f"{s}_H1.parquet")
        c = df["close"].resample("1D").last().dropna()
        pos = np.sign(c.pct_change().rolling(50).sum()).shift(1)
        out.append((c.pct_change() * pos))
    r = pd.concat(out, axis=1).mean(axis=1).dropna()
    r.index = r.index.normalize()
    return r


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() else 0


def p_pass(ann_sharpe, days=40, n=6000):
    best = 0
    for vol in np.arange(0.015, 0.028, 0.002):
        mu = ann_sharpe * vol / np.sqrt(252)
        P = 0
        for _ in range(n):
            eq = 0.0; ok = False
            for dd in range(days):
                x = rng.normal(mu, vol)
                if x <= -0.05: break
                eq += x
                if eq <= -0.10: break
                if eq >= 0.10 and dd >= 3: ok = True; break
            P += ok
        best = max(best, P / n)
    return best


def main():
    # existing book
    data_fx = {f.stem[:-3]: pd.read_parquet(f) for f in CACHE.glob("*_H1.parquet") if is_fx(f.stem[:-3])}
    curr = build_currency_returns(data_fx)
    rev = daily_reversion_pnl(curr, 12, 2, 1.5); rev.index = pd.to_datetime(rev.index).normalize()
    ovn = basket(use_filter=False); ovn.index = pd.to_datetime(ovn.index).normalize()

    book = {"FX_reversion": rev, "index_overnight": ovn}
    cands = {"crypto_trend": crypto_trend(), "turn_of_month": turn_of_month(),
             "commodity_trend": commodity_trend()}

    # combined existing book (equal risk)
    base = pd.DataFrame(book).dropna()
    comb0 = sum(base[c] / base[c].std() for c in base) / len(base.columns)

    print("=== Candidate edges vs the existing book ===\n")
    print(f"{'edge':<18} {'Sharpe':>7} {'corr_book':>10} {'keep?':>6}")
    keep = {}
    for name, r in cands.items():
        a, b = r.align(comb0, join="inner")
        corr = a.corr(b)
        sh = sharpe(r)
        good = sh > 0.15 and abs(corr) < 0.3
        print(f"{name:<18} {sh:>7.2f} {corr:>10.2f} {'YES' if good else '':>6}")
        if good:
            keep[name] = r

    # stack book + kept candidates at equal risk
    allser = {**book, **keep}
    D = pd.DataFrame(allser).dropna()
    stacked = sum(D[c] / D[c].std() for c in D) / len(D.columns)
    print(f"\nStacked book = {len(D.columns)} edges: {list(D.columns)}")
    print(f"  combined Sharpe: {sharpe(stacked):.2f}  (was 0.71 with 2 edges)")
    print(f"  max P(pass FTMO in 40 days, optimal leverage): {p_pass(sharpe(stacked))*100:.0f}%")


if __name__ == "__main__":
    main()
