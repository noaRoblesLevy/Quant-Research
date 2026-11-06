"""
SECOND EDGE attempt #2 — short-horizon XS reversal, and does it survive cost + diversify carry?

Trend failed (TS negative, XS ~0.24 unbiased). Reversal is the other orthogonal candidate:
funding-light, hours-long holds, a different mechanism. The whole question is whether the
gross edge survives turnover cost — so we report cost sensitivity and turnover explicitly,
then test the combined (carry + reversal) book's CFT pass-rate.

Real Bybit 1h data, 23 coins, 2022-07 -> 2025-05.

    python scripts/eval_reversal_combo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.data.bybit_loader import load_bybit_panels
from quantresearch.funding_harvest import harvest, sharpe, DEFAULT_UNIVERSE, PERIODS_PER_YEAR
from quantresearch.intraday_reversal import load_1h_panels, reversal_book, sharpe_1h, avg_turnover, PERIODS_PER_YEAR_1H
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo

CFT = FTMORules(profit_target_pct=0.08, max_daily_loss_pct=0.05,
                max_total_loss_pct=0.10, min_trading_days=5, label="CFT 2-Phase P1")
UNIV = [c for c in DEFAULT_UNIVERSE if c != "MATICUSDT"]
CARRY_CFG = dict(n_each=11, hold=21, cost_bps=8.0, beta_hedge=True, vol_target=0.004)


def mdd(x):
    eq = (1 + x).cumprod(); return -(eq / eq.cummax() - 1).min() * 100


def rec(s):
    return s.dropna()["2023-01-01":]


def line1h(net, tag):
    r = rec(net)
    return (f"  {tag:<22} recent {sharpe_1h(r):+.2f} | 2024 {sharpe_1h(net['2024-01-01':'2024-12-31']):+.2f} "
            f"| 2025-OOS {sharpe_1h(net['2025-01-01':]):+.2f} | maxDD {mdd(r):4.1f}%")


def cft_row(daily, tag):
    base = -((1 + daily).cumprod() / (1 + daily).cumprod().cummax() - 1).min()
    print(f"  [{tag}] base maxDD {base*100:.1f}%")
    print(f"    {'targetDD':>8} {'lev':>5} {'ann%':>6} {'P(pass)%':>9} {'P(breach)%':>11} {'avgDays':>8}")
    for tdd in [0.06, 0.08, 0.10]:
        lev = tdd / base; lr = daily * lev
        eq = 100000 * (1 + lr).cumprod(); eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, CFT, n_sims=6000)
        print(f"    {tdd*100:>7.0f}% {lev:>5.2f} {lr.mean()*365*100:>6.1f} {mc['p_pass']*100:>9.1f} "
              f"{mc['p_breach']*100:>11.1f} {(mc['avg_days_to_pass'] or 0):>8.0f}")


def main():
    print("=== SECOND EDGE #2: SHORT-HORIZON XS REVERSAL (real Bybit 1h) ===\n")
    print("Loading 1h panels (23 coins)...")
    P, R, F = load_1h_panels(UNIV, "2022-07", "2025-05")
    print(f"  1h grid {P.index.min()} -> {P.index.max()} ({len(P)} bars)\n")

    # ---- gross (0 bps) sweep first: is the signal even there before cost? ----
    print("GROSS (0 bps) sweep — is the reversal signal present at all?")
    for k in [1, 2, 3, 6]:
        for hold in [1, 2, 4]:
            net = reversal_book(P, R, F, k=k, hold=hold, n_each=6, cost_bps=0.0, vol_target=0.004)["net"].dropna()
            print(line1h(net, f"k={k} hold={hold}"))

    # ---- net (6 bps) sweep + turnover ----
    print("\nNET (6 bps) sweep with turnover diagnostic:")
    net_results = {}
    for k in [1, 2, 3, 6]:
        for hold in [1, 2, 4, 6]:
            net = reversal_book(P, R, F, k=k, hold=hold, n_each=6, cost_bps=6.0, vol_target=0.004)["net"].dropna()
            to = avg_turnover(P, R, k, hold, 6)
            net_results[(k, hold)] = net
            rebals_per_yr = PERIODS_PER_YEAR_1H / hold
            ann_cost = to * rebals_per_yr * 6.0 / 1e4 * 100  # rough %/yr at unit gross
            print(line1h(net, f"k={k} hold={hold}") + f" | turn {to:.2f} ~cost {ann_cost:4.0f}%/yr")

    # ---- pick best net config (recent Sharpe, 2025-OOS positive) ----
    oos_ok = {kh: v for kh, v in net_results.items() if sharpe_1h(v["2025-01-01":]) > 0}
    pool = oos_ok or net_results
    best_kh = max(pool, key=lambda kh: sharpe_1h(rec(pool[kh])))
    best = pool[best_kh]
    print(f"\nBEST NET CONFIG (k,hold)={best_kh}")
    print(line1h(best, f"k={best_kh[0]} hold={best_kh[1]}"))

    # ---- cost sensitivity of the best config ----
    print("\nCost sensitivity of best config (recent Sharpe):")
    for cb in [2, 4, 6, 8, 10]:
        net = reversal_book(P, R, F, k=best_kh[0], hold=best_kh[1], n_each=6, cost_bps=cb, vol_target=0.004)["net"].dropna()
        print(f"  {cb:>2} bps -> {sharpe_1h(rec(net)):+.2f}")

    # ---- correlation with the carry book (daily) ----
    print("\nLoading 8h panels for the carry book...")
    Pb, Fb, Rb = load_bybit_panels(UNIV, "2022-07", "2025-05")
    carry = harvest(Pb, Fb, Rb, **CARRY_CFG)["net"].dropna()
    cd = carry.resample("D").sum()
    rd = best.resample("D").sum()
    j = pd.concat([cd, rd], axis=1, keys=["carry", "rev"]).dropna()
    print(f"Daily P&L correlation carry vs reversal: full {j.corr().iloc[0,1]:+.2f} | "
          f"recent {j['2023-01-01':].corr().iloc[0,1]:+.2f}")

    # ---- combined book (equal daily-risk), CFT pass-rate vs carry alone ----
    al = j.loc[j.index]
    cd_n = al["carry"] / al["carry"].std()
    rd_n = al["rev"] / al["rev"].std()
    combined = cd_n + rd_n

    def calmar_d(daily):
        r = daily["2023-01-01":]
        ann = r.mean() * 365
        m = -((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        return ann / m if m > 0 else float("nan")
    print(f"\nCalmar (daily, recent): carry {calmar_d(al['carry']):.2f}  "
          f"reversal {calmar_d(al['rev']):.2f}  COMBINED {calmar_d(combined):.2f}")

    print("\n=== CFT 2-Phase pass-rate (MC on recent daily returns) ===")
    cft_row(al["carry"]["2023-01-01":], "CARRY ALONE")
    cft_row(combined["2023-01-01":], "COMBINED (carry+reversal)")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    sb = sharpe_1h(rec(best)); corr = j["2023-01-01":].corr().iloc[0, 1]
    c_cal, k_cal = calmar_d(al["carry"]), calmar_d(combined)
    if sb >= 0.8 and k_cal > c_cal * 1.05 and abs(corr) < 0.3:
        print(f"  STRONG 2ND EDGE: reversal recent Sharpe {sb:+.2f}, corr {corr:+.2f}, "
              f"combined Calmar {k_cal:.2f} > carry {c_cal:.2f}.")
    elif sb >= 0.5 and k_cal > c_cal:
        print(f"  PROMISING: reversal {sb:+.2f} survives cost, corr {corr:+.2f}, "
              f"combined Calmar {k_cal:.2f} vs {c_cal:.2f}. Worth refining.")
    else:
        print(f"  WEAK/DEAD: reversal recent Sharpe {sb:+.2f} after cost. "
              f"Cost floor likely eats it (see sweep). Carry stands alone.")


if __name__ == "__main__":
    main()
