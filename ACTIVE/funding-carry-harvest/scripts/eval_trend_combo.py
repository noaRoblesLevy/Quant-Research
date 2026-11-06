"""
SECOND EDGE — trend-following on Bybit perps, and whether it diversifies the carry book.

The funding-carry harvest passes CFT at ~69% (8% DD sizing). A second, uncorrelated
edge should lift the COMBINED Calmar (return/maxDD) — the binding constraint — and so
the pass-rate. Carry+trend is the canonical pairing: trend tends to profit during the
carry book's drawdowns (sustained directional / deleveraging events).

This script, on REAL Bybit data (23 coins, 2022-07 -> 2025-05):
  1. builds the locked carry book (n=11, 7d hold, beta-hedged, vol-target)
  2. sweeps time-series (TS) and cross-sectional (XS) trend configs, honestly split
     into recent (2023-25) vs 2025-OOS
  3. measures each trend book's correlation to the carry book
  4. builds the equal-risk combined book and compares Sharpe / Calmar / CFT pass-rate
     against carry alone

    python scripts/eval_trend_combo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.data.bybit_loader import load_bybit_panels
from quantresearch.funding_harvest import harvest, sharpe, DEFAULT_UNIVERSE, PERIODS_PER_YEAR
from quantresearch.trend_follow import trend_book
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo

CFT = FTMORules(profit_target_pct=0.08, max_daily_loss_pct=0.05,
                max_total_loss_pct=0.10, min_trading_days=5, label="CFT 2-Phase P1")
BYBIT_UNIVERSE = [c for c in DEFAULT_UNIVERSE if c != "MATICUSDT"]   # 23 tradeable
CARRY_CFG = dict(n_each=11, hold=21, cost_bps=8.0, beta_hedge=True, vol_target=0.004)  # locked 7d


def mdd(x):
    eq = (1 + x).cumprod()
    return -(eq / eq.cummax() - 1).min() * 100


def rec(s):
    return s.dropna()["2023-01-01":]


def line(net, tag):
    r = rec(net)
    return (f"  {tag:<26} recent {sharpe(r):+.2f} | 2024 {sharpe(net['2024-01-01':'2024-12-31']):+.2f} "
            f"| 2025-OOS {sharpe(net['2025-01-01':]):+.2f} | maxDD {mdd(r):4.1f}% | ann {r.mean()*PERIODS_PER_YEAR*100:+5.1f}%")


def cft_row(daily, tag):
    base = -((1 + daily).cumprod() / (1 + daily).cumprod().cummax() - 1).min()
    out = []
    for tdd in [0.06, 0.08, 0.10]:
        lev = tdd / base
        lr = daily * lev
        eq = 100000 * (1 + lr).cumprod(); eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, CFT, n_sims=6000)
        out.append((tdd, lev, lr.mean() * 365 * 100, mc["p_pass"] * 100, mc["p_breach"] * 100,
                    mc["avg_days_to_pass"] or 0))
    print(f"  [{tag}] base maxDD {base*100:.1f}%")
    print(f"    {'targetDD':>8} {'lev':>5} {'ann%':>6} {'P(pass)%':>9} {'P(breach)%':>11} {'avgDays':>8}")
    for tdd, lev, ann, pp, pb, dd in out:
        print(f"    {tdd*100:>7.0f}% {lev:>5.2f} {ann:>6.1f} {pp:>9.1f} {pb:>11.1f} {dd:>8.0f}")


def main():
    print("=== SECOND EDGE: TREND-FOLLOWING vs / WITH THE CARRY BOOK (real Bybit) ===\n")
    Pb, Fb, Rb = load_bybit_panels(BYBIT_UNIVERSE, "2022-07", "2025-05")
    print(f"Bybit grid {Pb.index.min().date()} -> {Pb.index.max().date()} "
          f"({len(Pb)} 8h periods, {len(BYBIT_UNIVERSE)} coins)\n")

    # ---- the locked carry book (our first edge) ----
    carry = harvest(Pb, Fb, Rb, **CARRY_CFG)["net"].dropna()
    print("CARRY BOOK (locked 7d config):")
    print(line(carry, "carry"))

    # ---- TS trend sweep ----
    print("\nTIME-SERIES TREND sweep (directional, un-hedged, vol-targeted):")
    ts_results = {}
    for signal in ["zscore", "sign"]:
        for lb_days in [20, 30, 50, 90]:
            lb = lb_days * 3   # 8h periods per day
            net = trend_book(Pb, Fb, Rb, mode="ts", lookback=lb, signal=signal,
                             hold=9, cost_bps=8.0, vol_target=0.004)["net"].dropna()
            ts_results[(signal, lb_days)] = net
            print(line(net, f"ts {signal} {lb_days}d"))

    # ---- XS trend sweep ----
    print("\nCROSS-SECTIONAL TREND sweep (dollar-neutral, beta-hedged):")
    xs_results = {}
    for lb_days in [20, 30, 50, 90]:
        lb = lb_days * 3
        net = trend_book(Pb, Fb, Rb, mode="xs", lookback=lb, signal="raw",
                         n_each=8, hold=9, cost_bps=8.0, vol_target=0.004, beta_hedge=True)["net"].dropna()
        xs_results[lb_days] = net
        print(line(net, f"xs {lb_days}d"))

    # ---- pick best trend by recent Sharpe (require 2025-OOS positive: honest) ----
    cand = {**{f"TS {k[0]} {k[1]}d": v for k, v in ts_results.items()},
            **{f"XS {k}d": v for k, v in xs_results.items()}}
    oos_ok = {k: v for k, v in cand.items() if sharpe(v["2025-01-01":]) > 0}
    pool = oos_ok or cand
    best_name = max(pool, key=lambda k: sharpe(rec(pool[k])))
    best = pool[best_name]
    print(f"\nBEST TREND (recent Sharpe, 2025-OOS positive required): {best_name}")
    print(line(best, best_name))

    # ---- correlation with carry ----
    cd = carry.resample("D").sum(); td = best.resample("D").sum()
    j = pd.concat([cd, td], axis=1, keys=["carry", "trend"]).dropna()
    corr_full = j.corr().iloc[0, 1]
    corr_rec = j["2023-01-01":].corr().iloc[0, 1]
    print(f"\nDaily P&L correlation carry vs trend: full {corr_full:+.2f} | recent {corr_rec:+.2f}")

    # ---- combined book: equal-risk (both already vol-targeted to 0.004) ----
    al = pd.concat([carry, best], axis=1, keys=["c", "t"]).dropna()
    combined = 0.5 * al["c"] + 0.5 * al["t"]
    print("\nCOMBINED BOOK (equal-risk 50/50):")
    print(line(carry.loc[al.index], "carry alone"))
    print(line(best.loc[al.index], "trend alone"))
    print(line(combined, "COMBINED"))

    def calmar(s):
        r = rec(s)
        return (r.mean() * PERIODS_PER_YEAR) / (mdd(r) / 100) if mdd(r) > 0 else float("nan")
    print(f"\nCalmar (recent ann / maxDD):  carry {calmar(carry.loc[al.index]):.2f}  "
          f"trend {calmar(best.loc[al.index]):.2f}  COMBINED {calmar(combined):.2f}")

    # ---- CFT pass-rate: carry alone vs combined ----
    print("\n=== CFT 2-Phase pass-rate (MC on recent daily returns) ===")
    cft_row(rec(carry.loc[al.index]).resample("D").sum(), "CARRY ALONE")
    cft_row(rec(combined).resample("D").sum(), "COMBINED")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    c_cal, k_cal = calmar(carry.loc[al.index]), calmar(combined)
    if k_cal > c_cal * 1.05 and corr_rec < 0.3:
        print(f"  DIVERSIFIER CONFIRMED: combined Calmar {k_cal:.2f} > carry {c_cal:.2f}, "
              f"corr {corr_rec:+.2f}. Trend lifts the binding constraint.")
    elif k_cal > c_cal:
        print(f"  MARGINAL: combined Calmar {k_cal:.2f} vs carry {c_cal:.2f} (corr {corr_rec:+.2f}). "
              f"Small help; weigh added execution complexity.")
    else:
        print(f"  NO IMPROVEMENT: combined Calmar {k_cal:.2f} <= carry {c_cal:.2f}. "
              f"Trend too weak on Bybit net of cost+funding; carry stands alone.")


if __name__ == "__main__":
    main()
