"""
Option A: can dynamic risk management lift the combined book's Calmar into
FTMO-viable range? Tests volatility targeting and a drawdown brake on the
FX-reversion + index-overnight blend.

    python scripts/eval_risk_managed.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from quantresearch.cross_sectional import build_currency_returns
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo
from quantresearch.edges import daily_reversion_pnl, is_fx, basket


def combined_returns():
    data_fx = {f.stem[:-3]: pd.read_parquet(f)
               for f in (ROOT / "data_cache").glob("*_H1.parquet") if is_fx(f.stem[:-3])}
    curr = build_currency_returns(data_fx)
    rev = daily_reversion_pnl(curr, 12, 2, 1.5)
    rev.index = pd.to_datetime(rev.index).normalize()
    ovn = basket(use_filter=False)
    ovn.index = pd.to_datetime(ovn.index).normalize()
    d = pd.DataFrame({"rev": rev, "ovn": ovn}).dropna()
    tv = 0.005
    return (0.5 * d["rev"] * (tv / d["rev"].std()) + 0.5 * d["ovn"] * (tv / d["ovn"].std()))


def vol_target(r, target_ann=0.10, window=30, max_lev=3.0):
    rv = r.rolling(window).std().shift(1) * np.sqrt(252)
    lev = (target_ann / rv).clip(upper=max_lev).fillna(0.0)
    return (r * lev).dropna()


def dd_brake(r, threshold=0.07, resume=0.02):
    """Cut exposure to zero once drawdown exceeds `threshold`; resume after it
    recovers to within `resume` of the peak. A hard cap suited to FTMO's limit."""
    eq, peak, active = 1.0, 1.0, True
    out = []
    for x in r:
        ret = x if active else 0.0
        eq *= (1 + ret)
        peak = max(peak, eq)
        dd = eq / peak - 1
        out.append(ret)
        if dd <= -threshold:
            active = False
        elif dd >= -resume:
            active = True
    return pd.Series(out, index=r.index)


def report(name, r):
    eq = (1 + r).cumprod()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    dd = -(eq / eq.cummax() - 1).min()
    ann = r.mean() * 252
    calmar = ann / dd if dd else 0
    print(f"  {name:<28} Sharpe {sh:.2f} | ann {ann*100:5.1f}% | maxDD {dd*100:5.1f}% | "
          f"Calmar {calmar:.2f} | worstDay {r.min()*100:6.2f}%")
    return r, dd


def ftmo(r, base_dd, name):
    rules = FTMORules()
    print(f"  FTMO sizing — {name}:")
    for tdd in [0.06, 0.08]:
        lev = tdd / base_dd
        lr = r * lev
        eq = 100000 * (1 + lr).cumprod()
        eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, rules, n_sims=4000)
        print(f"    {tdd*100:.0f}% DD -> ann {lr.mean()*252*100:5.1f}% | worstDay {lr.min()*100:6.2f}% | "
              f"P(pass) {mc['p_pass']*100:5.1f}% | P(breach) {mc['p_breach']*100:5.1f}%")


def main():
    base = combined_returns()
    print("=== Risk management on the combined book ===")
    _, dd0 = report("combined (baseline)", base)
    vt = vol_target(base, target_ann=0.10, window=30)
    _, dd1 = report("+ vol-target (30d, 10%)", vt)
    vtb = dd_brake(vt, threshold=0.07, resume=0.02)
    _, dd2 = report("+ vol-target + DD brake", vtb)

    print()
    ftmo(base, dd0, "baseline")
    print()
    ftmo(vtb, dd2, "vol-target + DD brake")


if __name__ == "__main__":
    main()
