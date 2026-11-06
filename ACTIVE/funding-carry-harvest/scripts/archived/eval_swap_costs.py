"""
Re-evaluate the 4-edge stack NET of swap/financing costs (the gate before going live).

Swap is charged per day on held notional. Base-case estimates (today's ~5% rates):
  index overnight : 1.8 bps/day (financing - dividends, across the rollover)
  FX reversion    : 1.0 bps/day negative carry (long low-yield / short high-yield)
  crypto trend    : 5.0 bps/day when held long (crypto CFD financing is steep)
  turn-of-month   : 1.8 bps/day on its hold days

Applied only on days each edge holds; shown with a 0.5x/1x/1.5x sensitivity band.

    python scripts/eval_swap_costs.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from quantresearch.cross_sectional import build_currency_returns
from quantresearch.edges import (daily_reversion_pnl, is_fx, basket,
                                 crypto_with_mask, tom_with_mask)

CACHE = ROOT / "data_cache"
SWAP = {"reversion": 1.0, "overnight": 1.8, "crypto": 5.0, "tom": 1.8}   # bps/day, base case


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() else 0


def main():
    data_fx = {f.stem[:-3]: pd.read_parquet(f) for f in CACHE.glob("*_H1.parquet") if is_fx(f.stem[:-3])}
    curr = build_currency_returns(data_fx)
    rev = daily_reversion_pnl(curr, 12, 2, 1.5); rev.index = pd.to_datetime(rev.index).normalize()
    ovn = basket(use_filter=False); ovn.index = pd.to_datetime(ovn.index).normalize()
    cry, cry_held = crypto_with_mask()
    tom, tom_held = tom_with_mask()

    for mult in [0.0, 0.5, 1.0, 1.5]:
        s = {k: v * mult / 1e4 for k, v in SWAP.items()}
        rev_n = rev - s["reversion"]                         # held daily
        ovn_n = ovn - s["overnight"]                         # held daily
        cry_n = cry - s["crypto"] * cry_held.astype(float)   # held when long
        tom_n = tom - s["tom"] * tom_held.astype(float)      # held on ToM days
        D = pd.DataFrame({"rev": rev_n, "ovn": ovn_n, "cry": cry_n, "tom": tom_n}).dropna()
        stacked = sum(D[c] / D[c].std() for c in D) / 4
        tag = "GROSS (no swap)" if mult == 0 else f"swap x{mult}"
        print(f"  {tag:<16} | rev {sharpe(rev_n):5.2f} | ovn {sharpe(ovn_n):5.2f} | "
              f"cry {sharpe(cry_n):5.2f} | tom {sharpe(tom_n):5.2f} || STACK {sharpe(stacked):.2f}")

    print("\nReading it: 'swap x1.0' is the base case. Watch which edges survive and the stacked Sharpe.")


if __name__ == "__main__":
    main()
