"""
BYBIT TRANSFER VALIDATION — does the funding-carry edge survive on REAL Bybit data?

We backtested on Binance funding; Crypto Fund Trader settles Bybit-based accounts
against Bybit's funding. This re-runs the exact same harvest on Bybit funding +
price (2022-07 -> 2025-05, so the 2023-25 recent window has clean warmup) and
compares the headline numbers head-to-head with the Binance result.

    python scripts/eval_bybit_transfer.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.data.bybit_loader import load_bybit_panels
from quantresearch.funding_harvest import (
    load_panels as load_binance_panels, harvest, sharpe, DEFAULT_UNIVERSE, PERIODS_PER_YEAR)
from quantresearch.evaluation.ftmo import FTMORules, monte_carlo

CFT = FTMORules(profit_target_pct=0.08, max_daily_loss_pct=0.05,
                max_total_loss_pct=0.10, min_trading_days=5, label="CFT 2-Phase P1")
CFG = dict(n_each=11, hold=9, cost_bps=8.0, beta_hedge=True, vol_target=0.004)

# MATIC -> POL rebrand: Bybit no longer lists MATICUSDT klines, so CFT's Bybit
# account won't have it either. Real tradeable universe = 23 coins.
BYBIT_UNIVERSE = [c for c in DEFAULT_UNIVERSE if c != "MATICUSDT"]


def mdd(x):
    eq = (1 + x).cumprod()
    return -(eq / eq.cummax() - 1).min() * 100


def summarize(net, tag):
    rec = net["2023-01-01":]
    oos = net["2025-01-01":]
    print(f"\n[{tag}]  recent(2023-25) Sharpe {sharpe(rec):+.2f} | "
          f"2024 {sharpe(net['2024-01-01':'2024-12-31']):+.2f} | "
          f"2025-OOS {sharpe(oos):+.2f} | maxDD {mdd(net):.1f}% | ann {rec.mean()*PERIODS_PER_YEAR*100:+.1f}%")
    return rec


def cft_table(rec, tag):
    daily = rec.resample("D").sum()
    base = -((1 + daily).cumprod() / (1 + daily).cumprod().cummax() - 1).min()
    print(f"  CFT pass-rate ({tag}, MC on recent daily returns, base maxDD {base*100:.1f}%):")
    print(f"    {'targetDD':>8} {'lev':>5} {'ann%':>6} {'P(pass)%':>9} {'P(breach)%':>11}")
    for tdd in [0.06, 0.08, 0.10]:
        lev = tdd / base
        lr = daily * lev
        eq = 100000 * (1 + lr).cumprod()
        eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, CFT, n_sims=6000)
        print(f"    {tdd*100:>7.0f}% {lev:>5.2f} {lr.mean()*365*100:>6.1f} "
              f"{mc['p_pass']*100:>9.1f} {mc['p_breach']*100:>11.1f}")


def main():
    print("=== BYBIT TRANSFER VALIDATION ===")
    print(f"Loading BYBIT panels ({len(BYBIT_UNIVERSE)} coins, 2022-07 -> 2025-05)...")
    Pb, Fb, Rb = load_bybit_panels(BYBIT_UNIVERSE, "2022-07", "2025-05")
    print(f"  Bybit grid {Pb.index.min()} -> {Pb.index.max()} ({len(Pb)} periods)")

    res_b = harvest(Pb, Fb, Rb, **CFG)
    net_b = res_b["net"].dropna()

    # funding-spread + carry sanity on Bybit
    carry_b = harvest(Pb, Fb, Rb, n_each=12, hold=9, cost_bps=0.0, beta_hedge=False)["carry"]
    print(f"\n  Bybit carry-only Sharpe: {sharpe(carry_b):.1f} | carry ann (recent): "
          f"{carry_b['2023-01-01':].mean()*PERIODS_PER_YEAR*100:+.1f}%/yr")

    # Binance comparison over the SAME window + SAME 23 coins for apples-to-apples
    print("\nLoading BINANCE panels for head-to-head over same window/universe...")
    Pn, Fn, Rn = load_binance_panels(BYBIT_UNIVERSE)
    # restrict binance to same date span
    mask = (Pn.index >= Pb.index.min()) & (Pn.index <= Pb.index.max())
    res_n = harvest(Pn[mask], Fn[mask], Rn[mask], **CFG)
    net_n = res_n["net"].dropna()

    print("\n================ HEAD TO HEAD ================")
    rec_b = summarize(net_b, "BYBIT  (real CFT venue)")
    rec_n = summarize(net_n, "BINANCE (original)")

    print()
    cft_table(rec_b, "BYBIT")
    cft_table(rec_n, "BINANCE")

    # correlation of the two daily P&L streams
    db = net_b.resample("D").sum()["2023-01-01":]
    dn = net_n.resample("D").sum()["2023-01-01":]
    corr = pd.concat([db, dn], axis=1).dropna().corr().iloc[0, 1]
    print(f"\n  Daily P&L correlation Bybit vs Binance (2023-25): {corr:.2f}")

    print("\n=== VERDICT ===")
    sb = sharpe(rec_b)
    if sb >= 1.0 and sharpe(net_b['2025-01-01':]) > 0:
        print(f"  TRANSFER CONFIRMED: Bybit recent Sharpe {sb:.2f}, OOS positive. "
              f"Edge survives on the real CFT venue.")
    elif sb >= 0.6:
        print(f"  PARTIAL: Bybit recent Sharpe {sb:.2f} — weaker than Binance; investigate coin/rate diffs.")
    else:
        print(f"  TRANSFER FAILS: Bybit recent Sharpe {sb:.2f}. Edge does not hold on Bybit rates.")


if __name__ == "__main__":
    main()
