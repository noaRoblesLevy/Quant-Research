"""
CFT 2-Phase completion time estimate at actual Bybit fee levels.

Runs the funding-carry harvest at:
  - maker fees: 4bps round-trip  (limit orders, realistic for 7-day rebalance)
  - taker fees: 11bps round-trip (market orders)

For each fee scenario, runs MC for Phase 1 (8% target) and Phase 2 (5% target)
at the 8% DD sizing that gave ~69% combined pass-rate in the original study.
Reports P(pass), P(breach), avg days, and combined 2-phase expected total days.

    python scripts/eval_cft_twophase.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.data.bybit_loader import load_bybit_panels
from quantresearch.funding_harvest import harvest, sharpe, DEFAULT_UNIVERSE, PERIODS_PER_YEAR
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo

UNIV = [c for c in DEFAULT_UNIVERSE if c != "MATICUSDT"]   # 23 coins
BASE_CFG = dict(n_each=11, hold=21, beta_hedge=True, vol_target=0.004)

P1 = FTMORules(profit_target_pct=0.08, max_daily_loss_pct=0.05,
               max_total_loss_pct=0.10, min_trading_days=5, label="Phase 1 (8%)")
P2 = FTMORules(profit_target_pct=0.05, max_daily_loss_pct=0.05,
               max_total_loss_pct=0.10, min_trading_days=5, label="Phase 2 (5%)")

TARGET_DD = 0.08    # 8% DD sizing — the binding sizing decision


def run_scenario(net_daily, label):
    base_dd = -((1 + net_daily).cumprod() / (1 + net_daily).cumprod().cummax() - 1).min()
    lev = TARGET_DD / base_dd
    lr = net_daily * lev
    ann = lr.mean() * 365 * 100
    eq = 100000 * (1 + lr).cumprod()
    eq.index = pd.to_datetime(eq.index)

    mc1 = monte_carlo(eq, P1, n_sims=10000)
    mc2 = monte_carlo(eq, P2, n_sims=10000)

    p_both = mc1["p_pass"] * mc2["p_pass"]
    d1 = mc1["avg_days_to_pass"] or 0
    d2 = mc2["avg_days_to_pass"] or 0
    total_days = d1 + d2   # sequential phases, conditional on passing

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  leverage: {lev:.2f}x   ann return: {ann:+.1f}%/yr")
    print(f"  {'':22} {'P(pass)':>8} {'P(breach)':>10} {'avg days':>9}")
    print(f"  {'Phase 1 (8% target)':22} {mc1['p_pass']*100:>8.1f}% {mc1['p_breach']*100:>9.1f}% {d1:>9.0f}")
    print(f"  {'Phase 2 (5% target)':22} {mc2['p_pass']*100:>8.1f}% {mc2['p_breach']*100:>9.1f}% {d2:>9.0f}")
    print(f"  {'-'*53}")
    print(f"  {'Combined (both phases)':22} {p_both*100:>8.1f}%  (seq, P1×P2)")
    print(f"  {'Total expected days':22} {'~'+str(int(total_days)):>8}  (P1 + P2 conditional)")
    print(f"  {'Calendar months':22} {'~'+f'{total_days/30:.1f}':>8}")


def main():
    print("Loading Bybit panels (23 coins, 2022-07 -> 2025-05)...")
    Pb, Fb, Rb = load_bybit_panels(UNIV, "2022-07", "2025-05")
    print(f"  {Pb.index.min()} -> {Pb.index.max()} ({len(Pb)} periods)\n")

    for label, bps in [("MAKER (4 bps round-trip)", 4.0),
                        ("BACKTEST BASELINE (8 bps)", 8.0),
                        ("TAKER (11 bps round-trip)", 11.0)]:
        net = harvest(Pb, Fb, Rb, **BASE_CFG, cost_bps=bps)["net"].dropna()
        rec = net["2023-01-01":]
        daily = rec.resample("D").sum()
        run_scenario(daily, label)

    print("\n\nNOTES")
    print("  - 'avg days' is conditional on eventually passing that phase (sims that time out excluded)")
    print("  - Combined P(pass) = P(pass P1) × P(pass P2) — independent sequential phases")
    print("  - At 7-day rebalance, limit orders at mid-price get maker fills almost always")
    print("  - taker 11bps scenario is the conservative floor; real cost likely between 4-8bps")


if __name__ == "__main__":
    main()
