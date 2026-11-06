"""
Combined two-edge portfolio: FX cross-sectional reversion + index overnight.

Both are real, positive, cost-clearing edges in DIFFERENT asset classes. If their
returns are near-uncorrelated, an equal-risk blend should lift the Sharpe toward
FTMO-viable territory and diversify the equity-crash tail.

    python scripts/eval_combined.py
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

CACHE = ROOT / "data_cache"


def stats(r):
    eq = (1 + r).cumprod()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    dd = -(eq / eq.cummax() - 1).min() * 100
    def win_dd(a, b):
        e = eq[(eq.index >= a) & (eq.index <= b)]
        return -(e / e.cummax() - 1).min() * 100 if len(e) else 0
    return sh, dd, r.min() * 100, win_dd("2020-02-01", "2020-05-01"), win_dd("2022-01-01", "2022-12-31")


def line(name, r):
    sh, dd, wd, d20, d22 = stats(r)
    print(f"  {name:<14} Sharpe {sh:.2f} | maxDD {dd:5.1f}% | worstDay {wd:6.2f}% | "
          f"DD2020 {d20:4.1f}% | DD2022 {d22:4.1f}%")
    return sh


def main():
    # --- edge 1: FX cross-sectional reversion (daily) ---
    data_fx = {f.stem[:-3]: pd.read_parquet(f)
               for f in CACHE.glob("*_H1.parquet") if is_fx(f.stem[:-3])}
    curr = build_currency_returns(data_fx)
    rev = daily_reversion_pnl(curr, lookback=12, n_sel=2, spread_bps=1.5)
    rev.index = pd.to_datetime(rev.index).normalize()

    # --- edge 2: index overnight basket (daily) ---
    ovn = basket(use_filter=False)
    ovn.index = pd.to_datetime(ovn.index).normalize()

    # --- align on common dates ---
    d = pd.DataFrame({"rev": rev, "ovn": ovn}).dropna()
    corr = d["rev"].corr(d["ovn"])
    print(f"Common period: {d.index.min().date()} -> {d.index.max().date()}  ({len(d)} days)")
    print(f"Correlation(reversion, overnight): {corr:+.3f}\n")

    # --- equal-RISK blend: scale each to the SAME realistic daily vol, then 50/50 ---
    TARGET_VOL = 0.005                      # 0.5% daily vol per leg (realistic scale)
    rev_s = d["rev"] * (TARGET_VOL / d["rev"].std())
    ovn_s = d["ovn"] * (TARGET_VOL / d["ovn"].std())
    comb = 0.5 * rev_s + 0.5 * ovn_s

    print("Standalone (rescaled to equal vol) vs combined:")
    line("reversion", rev_s)
    line("overnight", ovn_s)
    sh_c = line("COMBINED", comb)
    print(f"\n  Diversification: combined Sharpe {sh_c:.2f} "
          f"vs individual ~0.5-0.6 (uplift from low correlation)\n")

    # --- FTMO sizing of the combined book ---
    base_dd = -((1 + comb).cumprod() / (1 + comb).cumprod().cummax() - 1).min()
    rules = FTMORules()
    print("FTMO sizing of COMBINED book:")
    print(f"  {'targetDD':>8} {'ann%':>6} {'worstDay%':>10} {'P(pass)%':>9} {'P(breach)%':>11} {'avgDays':>8}")
    for tdd in [0.05, 0.06, 0.08]:
        lev = tdd / base_dd
        lr = comb * lev
        eq = 100000 * (1 + lr).cumprod()
        eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, rules, n_sims=4000)
        print(f"  {tdd*100:>7.0f}% {lr.mean()*252*100:>6.1f} {lr.min()*100:>10.2f} "
              f"{mc['p_pass']*100:>9.1f} {mc['p_breach']*100:>11.1f} {(mc['avg_days_to_pass'] or 0):>8.0f}")


if __name__ == "__main__":
    main()
