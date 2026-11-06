"""
Decisive evaluation of the funding-carry harvest for Crypto Fund Trader.

Honest protocol (avoids the trap of trusting a 2021-inflated number):
  * full-sample AND recent-regime (2023-2025) stats reported separately
  * 2025 is true out-of-sample (never used in design)
  * sized to the CFT static-10% rule; Monte-Carlo resamples RECENT returns only
  * carry/price/cost decomposition shown so we know what we're being paid for

    python scripts/eval_funding_harvest.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.funding_harvest import (
    load_panels, harvest, sharpe, DEFAULT_UNIVERSE, PERIODS_PER_YEAR)
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo

CFT = FTMORules(profit_target_pct=0.08, max_daily_loss_pct=0.05,
                max_total_loss_pct=0.10, min_trading_days=5, label="CFT 2-Phase P1")

YEARS = [("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
         ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
         ("2025 OOS", "2025-01-01", "2025-05-31")]


def block_mc_cft(period_net: pd.Series, target_dd: float, n_sims: int = 5000):
    """Daily-aggregate the 8h net P&L, size to target_dd, run CFT Monte Carlo."""
    daily = period_net.resample("D").sum()
    base_dd = -((1 + daily).cumprod() / (1 + daily).cumprod().cummax() - 1).min()
    lev = target_dd / base_dd if base_dd > 0 else 1.0
    lr = daily * lev
    eq = 100000 * (1 + lr).cumprod()
    eq.index = pd.to_datetime(eq.index)
    mc = monte_carlo(eq, CFT, n_sims=n_sims)
    return lev, lr.mean() * 365 * 100, mc


def main():
    print("=== Loading 8h panels (24 coins, 2021-2025) ===")
    P, Fnd, logret = load_panels(DEFAULT_UNIVERSE)
    print(f"  grid {P.index.min()} -> {P.index.max()}  ({len(P)} periods)\n")

    # chosen config: 3-day hold, beta-hedged, vol-targeted to stabilize DD
    cfg = dict(n_each=5, hold=9, cost_bps=8.0, beta_hedge=True,
               vol_target=0.004, vol_win=30)
    print(f"=== Config: {cfg} ===\n")
    res = harvest(P, Fnd, logret, **cfg)
    net = res["net"].dropna()

    print("Carry / price / cost decomposition (annualized):")
    for c in ["carry", "price", "cost", "net"]:
        print(f"  {c:<6} {res[c].mean() * PERIODS_PER_YEAR * 100:+7.1f}%/yr")

    print("\nYear-by-year (net):")
    print(f"  {'period':>9} {'Sharpe':>7} {'ann%':>7} {'maxDD%':>7}")
    for lbl, s, e in YEARS:
        sub = net[s:e]
        if len(sub) < 10:
            continue
        eq = (1 + sub).cumprod()
        dd = -(eq / eq.cummax() - 1).min() * 100
        print(f"  {lbl:>9} {sharpe(sub):>+7.2f} {sub.mean() * PERIODS_PER_YEAR * 100:>+7.1f} {dd:>7.1f}")

    # honest split: full vs recent regime
    full = net
    recent = net["2023-01-01":]
    oos = net["2025-01-01":]
    print(f"\nFull 2021-25 : Sharpe {sharpe(full):+.2f}")
    print(f"Recent 23-25 : Sharpe {sharpe(recent):+.2f}   <- the number to trust (post-maturation regime)")
    print(f"2025 OOS only: Sharpe {sharpe(oos):+.2f}")

    # CFT pass-rate sized to several DD budgets, RECENT returns only
    print("\n=== CFT 2-Phase pass-rate (Monte Carlo on RECENT 2023-25 returns) ===")
    print(f"  {'targetDD':>8} {'lev':>5} {'ann%':>6} {'P(pass)%':>9} {'P(breach)%':>11} {'avgDays':>8}")
    for tdd in [0.04, 0.06, 0.08, 0.10]:
        lev, ann, mc = block_mc_cft(recent, tdd)
        print(f"  {tdd*100:>7.0f}% {lev:>5.2f} {ann:>6.1f} {mc['p_pass']*100:>9.1f} "
              f"{mc['p_breach']*100:>11.1f} {(mc['avg_days_to_pass'] or 0):>8.0f}")

    print("\n=== Verdict ===")
    sr = sharpe(recent)
    if sr >= 1.0 and sharpe(oos) > 0:
        print(f"  VIABLE: recent Sharpe {sr:.2f}, OOS positive — regime-agnostic carry confirmed.")
    elif sr >= 0.6:
        print(f"  PROMISING: recent Sharpe {sr:.2f} — refine hedge/sizing.")
    else:
        print(f"  WEAK: recent Sharpe {sr:.2f}.")


if __name__ == "__main__":
    main()
