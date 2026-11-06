"""
Develop the overnight edge into a diversified, FTMO-evaluated basket.

  * equal-weight overnight-long basket across 8 index CFDs
  * optional trend filter: only hold a given index overnight if it's above its
    200-day MA (skips downtrends -> cuts the equity-crash tail)
  * report Sharpe / maxDD / worst day, the 2020 & 2022 drawdowns specifically,
    and the FTMO pass-rate at several drawdown sizings

    python scripts/eval_overnight_basket.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantresearch.evaluation.ftmo import FTMORules, evaluate_run, monte_carlo
from quantresearch.edges import INDEX_SESSIONS as SESSIONS, overnight_frame, basket  # moved to package

CACHE = Path(__file__).resolve().parents[1] / "data_cache"


def stats(r, label):
    eq = (1 + r).cumprod()
    sh = r.mean() / r.std() * np.sqrt(252)
    dd = -(eq / eq.cummax() - 1).min() * 100
    dd2020 = -(eq[(eq.index >= "2020-02-01") & (eq.index <= "2020-05-01")].pipe(
        lambda e: e / e.cummax() - 1).min()) * 100 if len(eq["2020-02-01":"2020-05-01"]) else 0
    dd2022 = -(eq[(eq.index >= "2022-01-01") & (eq.index <= "2022-12-31")].pipe(
        lambda e: e / e.cummax() - 1).min()) * 100 if len(eq["2022-01-01":"2022-12-31"]) else 0
    print(f"  {label:<22} Sharpe {sh:.2f} | ann {r.mean()*252*100:5.1f}% | maxDD {dd:5.1f}% | "
          f"worstDay {r.min()*100:6.2f}% | DD2020 {dd2020:4.1f}% | DD2022 {dd2022:4.1f}%")
    return r


def ftmo_table(r, label):
    eq_raw = (1 + r).cumprod()
    base_dd = -(eq_raw / eq_raw.cummax() - 1).min()
    rules = FTMORules()
    print(f"\n  FTMO sizing — {label}:")
    print(f"  {'targetDD':>8} {'lev':>5} {'ann%':>6} {'worstDay%':>10} {'P(pass)%':>9} {'P(breach)%':>11}")
    for tdd in [0.05, 0.06, 0.08]:
        lev = tdd / base_dd
        lr = r * lev
        eq = 100000 * (1 + lr).cumprod()
        eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, rules, n_sims=4000)
        print(f"  {tdd*100:>7.0f}% {lev:>5.1f} {lr.mean()*252*100:>6.1f} {lr.min()*100:>10.2f} "
              f"{mc['p_pass']*100:>9.1f} {mc['p_breach']*100:>11.1f}")


def main():
    print("=== Overnight-long index basket ===")
    nof = basket(use_filter=False)
    stats(nof, "no filter")
    wf = basket(use_filter=True)
    stats(wf, "200MA trend filter")

    # develop the better one for FTMO
    better = wf if (wf.mean() / wf.std()) >= (nof.mean() / nof.std()) else nof
    label = "200MA filter" if better is wf else "no filter"
    ftmo_table(better, label)


if __name__ == "__main__":
    main()
