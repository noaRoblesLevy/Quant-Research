"""
Is the cross-sectional reversion edge actually FTMO-usable?

Reconstructs DAILY P&L of the 12-week reversion strategy (weekly positions held
through each week, earning daily currency returns), then sizes it to several
drawdown budgets and runs the real FTMO pass-rate Monte Carlo at each.

    python scripts/eval_reversion_ftmo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantresearch.cross_sectional import build_currency_returns
from quantresearch.evaluation.ftmo import FTMORules, evaluate_run, monte_carlo
from quantresearch.edges import is_fx, daily_reversion_pnl  # moved to the package

CACHE = Path(__file__).resolve().parents[1] / "data_cache"


def stats(daily_ret):
    ann = daily_ret.mean() * 252
    vol = daily_ret.std() * np.sqrt(252)
    eq = (1 + daily_ret).cumprod()
    dd = -(eq / eq.cummax() - 1).min()
    return ann, vol, (ann / vol if vol else 0), dd, daily_ret.min()


def main():
    data = {f.stem[:-3]: pd.read_parquet(f)
            for f in sorted(CACHE.glob("*_H1.parquet")) if is_fx(f.stem[:-3])}
    curr = build_currency_returns(data)
    base = daily_reversion_pnl(curr, lookback=12, n_sel=2, spread_bps=1.5)

    ann, vol, sharpe, dd, worst = stats(base)
    print(f"RAW (leverage 1.0) 12wk reversion, daily, net of 1.5bp spread:")
    print(f"  Sharpe {sharpe:.2f} | ann {ann*100:.1f}% | vol {vol*100:.1f}% | maxDD {dd*100:.1f}% | worst day {worst*100:.2f}%\n")

    rules = FTMORules()
    print("Sized to drawdown budgets (leverage scales return AND drawdown together):\n")
    print(f"{'targetDD':>9} {'lev':>5} {'ann%':>7} {'maxDD%':>7} {'worstDay%':>10} {'P(pass)%':>9} {'P(breach)%':>11} {'avgDays':>8}")
    for target_dd in [0.04, 0.05, 0.06, 0.08]:
        lev = target_dd / dd
        lr = base * lev
        a, v, sh, d, wst = stats(lr)
        eq = (100000 * (1 + lr).cumprod())
        eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, rules, n_sims=4000)
        print(f"{target_dd*100:>8.0f}% {lev:>5.1f} {a*100:>7.1f} {d*100:>7.1f} {wst*100:>10.2f} "
              f"{mc['p_pass']*100:>9.1f} {mc['p_breach']*100:>11.1f} "
              f"{(mc['avg_days_to_pass'] or 0):>8.0f}")

    # also: would the actual history have passed, sized to 6% DD?
    lev6 = 0.06 / dd
    eq6 = 100000 * (1 + base * lev6).cumprod()
    eq6.index = pd.to_datetime(eq6.index)
    run = evaluate_run(eq6, rules, 100000)
    print(f"\nActual-history run @ 6% DD sizing: {'PASS' if run['passed'] else ('BREACH' if run['breached'] else 'INCOMPLETE')}"
          f" ({run['reason']}, {run['days_used']} days)")


if __name__ == "__main__":
    main()
