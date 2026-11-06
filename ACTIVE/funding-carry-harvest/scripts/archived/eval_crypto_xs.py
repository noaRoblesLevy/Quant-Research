"""
Cross-sectional crypto basket evaluation for Crypto Fund Trader 2-Phase challenge.

Rules wired in:
    profit target  : 8% Phase 1, 5% Phase 2
    daily loss cap : 5% of starting balance
    max overall    : 10% STATIC (of initial balance)
    no time limit

Steps
-----
1. Load 24-coin 1h Binance UM perp data + funding history
2. Build composite ranking signal (reversal + momentum + funding contrarian)
3. Long-top-5 / short-bottom-5, rebalance every 24h (1 day hold)
4. Sweep: hold period, n_each, spread_bps sensitivity
5. Size to the CFT drawdown budget and run Monte Carlo pass-rate
6. Print honest summary

    python scripts/eval_crypto_xs.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.data.binance_loader import load_klines, load_funding
from quantresearch.crypto_xs import build_features, xs_portfolio_returns
from quantresearch.evaluation.ftmo import FTMORules, evaluate_run, monte_carlo

UNIVERSE = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
    "SOLUSDT","DOGEUSDT","LINKUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","FILUSDT","NEARUSDT","AVAXUSDT","XLMUSDT",
    "AAVEUSDT","UNIUSDT","TRXUSDT","ATOMUSDT","ETCUSDT",
    "MATICUSDT","SANDUSDT","AXSUSDT","COMPUSDT",
]

# CFT 2-Phase Phase-1 rules (static drawdown, same structure as FTMO static model)
# Phase 1: 8% target, 5% daily, 10% static max loss, 5 min days, no time limit
CFT_RULES = FTMORules(
    profit_target_pct=0.08,
    max_daily_loss_pct=0.05,
    max_total_loss_pct=0.10,
    min_trading_days=5,
    label="CFT 2-Phase P1",
)


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def calmar(r):
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    ann = r.mean() * 252
    return ann / abs(dd) if dd < 0 else 0.0


def print_stats(r, label):
    eq = (1 + r).cumprod()
    dd = -(eq / eq.cummax() - 1).min() * 100
    sh = sharpe(r)
    ca = calmar(r)
    ann = r.mean() * 252 * 100
    wday = r.min() * 100
    print(f"  {label:<32}  Sharpe {sh:5.2f}  Calmar {ca:5.2f}  ann {ann:+6.1f}%  maxDD {dd:5.1f}%  worstDay {wday:+6.2f}%")
    return r


def main():
    print("=== Loading universe ===")
    klines, funding = {}, {}
    for sym in UNIVERSE:
        try:
            klines[sym]  = load_klines(sym, "1h", "2021-01", "2024-12")
            funding[sym] = load_funding(sym, "2021-01")
            print(f"  {sym:<15} bars={len(klines[sym]):>6}")
        except Exception as e:
            print(f"  {sym:<15} SKIP: {e}")

    print(f"\n=== Building features ({len(klines)} coins) ===")
    print("  (ranking signals across coins — may take 1-2 min)")
    feats = build_features(klines, funding)
    print(f"  features shape: {feats.shape}")

    print("\n=== Signal IC check ===")
    # Check if composite predicts next-bar log return cross-sectionally
    fwd = feats["log_ret"].groupby(level="time").apply(lambda x: x)
    comp = feats["composite"]
    # compute cross-sectional rank-IC per time step, report mean ± std
    ics = []
    for t in feats.index.get_level_values("time").unique()[:-1]:
        try:
            c = comp.loc[t].dropna()
            f = fwd.loc[t].dropna()   # this bar's return (we'll shift below)
            common = c.index.intersection(f.index)
            if len(common) >= 8:
                ic = c.loc[common].corr(f.loc[common], method="spearman")
                ics.append(ic)
        except Exception:
            pass
    if ics:
        print(f"  Cross-sectional IC (composite vs same-bar return): mean={np.mean(ics):+.4f}  std={np.std(ics):.4f}")
        print(f"  NOTE: we trade at t+1 so what matters is the t->t+1 IC (evaluated in backtest)")

    print("\n=== Backtest sweep: hold_bars × n_each × spread ===")
    best_sh, best_r, best_label = -99, None, ""
    for hold in [8, 24, 48]:
        for n in [4, 5, 6]:
            for spread in [8.0, 12.0]:
                r, _ = xs_portfolio_returns(feats, n_each=n, hold_bars=hold, spread_bps=spread)
                r = r.dropna()
                sh = sharpe(r)
                label = f"hold={hold:>2}h n={n} spread={spread:.0f}bps"
                print_stats(r, label)
                if sh > best_sh:
                    best_sh, best_r, best_label = sh, r, label

    print(f"\n=== Best configuration: {best_label} ===")
    r = best_r
    print_stats(r, "best (gross)")

    # split-half validation
    mid = r.index[len(r)//2]
    print_stats(r[r.index < mid], "  H1 (2021-22)")
    print_stats(r[r.index >= mid], "  H2 (2023-24)")

    # CFT pass-rate at several leverage levels
    print("\n=== CFT 2-Phase sizing (static 10% drawdown) ===")
    eq_raw = (1 + r).cumprod()
    base_dd = -(eq_raw / eq_raw.cummax() - 1).min()
    print(f"  Base-case maxDD: {base_dd*100:.1f}%")
    print(f"\n  {'targetDD':>8} {'lev':>5} {'ann%':>6} {'worstDay%':>10} {'P(pass)%':>9} {'P(breach)%':>11} {'avgDays':>8}")
    for tdd in [0.04, 0.05, 0.06, 0.08, 0.10]:
        lev = tdd / base_dd
        lr = r * lev
        eq = 100000 * (1 + lr).cumprod()
        eq.index = pd.to_datetime(eq.index)
        mc = monte_carlo(eq, CFT_RULES, n_sims=5000)
        worst = lr.min() * 100
        print(f"  {tdd*100:>7.0f}% {lev:>5.1f}  {lr.mean()*252*100:>6.1f}  {worst:>10.2f}  "
              f"{mc['p_pass']*100:>9.1f}  {mc['p_breach']*100:>11.1f}  "
              f"{(mc['avg_days_to_pass'] or 0):>8.0f}")

    print("\n=== Summary ===")
    sh = sharpe(r)
    print(f"  Net Sharpe: {sh:.2f}")
    if sh >= 1.0:
        print("  VIABLE: Sharpe >=1.0 — likely >50% pass probability at optimal sizing")
    elif sh >= 0.7:
        print("  MARGINAL: Sharpe 0.7-1.0 — ~30-50% pass probability; consider adding coins or signals")
    else:
        print("  BELOW THRESHOLD: Sharpe <0.7 — <30% pass; breadth or signal improvement needed")


if __name__ == "__main__":
    main()
