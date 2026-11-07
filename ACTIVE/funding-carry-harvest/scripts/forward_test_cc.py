"""
Live shadow ledger for the cash-and-carry book selected by the 2026-11-07 improvement
study (docs/preregistration_2026-11-07.md, rule R6; results in docs/improvement_results/).

Configuration "CC L=21 K=7":
    - short the N=11 perpetuals with the highest mean funding over the last 7 days
      (21 eight-hour periods), long the SAME coins on spot; a slot whose 7-day mean
      funding is not positive stays in cash
    - 7 tranches of 1/7 capital, each rebalanced weekly, one day apart
    - 8 bps cost on the turnover of each leg (unlike the baseline ledger, costs are
      charged here: for a low-volatility carry book they are first-order)

Like forward_test_bybit.py it is a shadow book (no orders) that replays every 8h funding
grid step since the last run from Bybit's public history, so missed runs are caught up
and do not change the result. Positions are fractions of equity, held constant between
rebalances. Returns are per unit of spot notional (spot used as collateral for the short).

    python scripts/forward_test_cc.py --init     # once; starts at the next grid time
    python scripts/forward_test_cc.py --step     # replay to now (scheduled every 8h)
    python scripts/forward_test_cc.py --report
    python scripts/forward_test_cc.py --dashboard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from forward_test_bybit import (UNIVERSE, fetch_closes_range, fetch_funding_range,
                                grid_after, _utc_naive, SETTLE)

N_SLOTS = 11
K = 7
L_PERIODS = 21                     # 7 days of 8h funding periods
COST_BPS = 8.0
HOLD = pd.Timedelta(days=7)
H1 = pd.Timedelta(hours=1)

STATE = ROOT / "data_cache" / "forward_state_cc.json"
LEDGER = ROOT / "data_cache" / "forward_ledger_cc.csv"
DASH_PNG = ROOT / "data_cache" / "forward_progress_cc.png"

# expectation from the study (Bybit Aug 2022 - May 2025); the 2026 holdout was -1.8%/yr
EXPECT = dict(sharpe_8h=6.39, sharpe_monthly=2.89, ann_ret=0.082, holdout_ann_ret=-0.018)


def set_state_dir(d: Path | None):
    global STATE, LEDGER, DASH_PNG
    if d:
        d = Path(d)
        STATE, LEDGER, DASH_PNG = (d / "forward_state_cc.json", d / "forward_ledger_cc.csv",
                                   d / "forward_progress_cc.png")


def iso(t: pd.Timestamp) -> str:
    return t.tz_localize("UTC").isoformat()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))


def append_ledger(row: dict):
    pd.DataFrame([row]).to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)


# ----------------------------------------------------------------- book logic
def signal_at(t: pd.Timestamp, funding: dict) -> pd.Series:
    """Mean funding per 8h period over the 7 days ending at t (events in (t-7d, t])."""
    lo = t - pd.Timedelta(hours=8 * L_PERIODS)
    return pd.Series({s: float(f[(f.index > lo) & (f.index <= t)].sum()) / L_PERIODS
                      for s, f in funding.items() if len(f)})


def select_at(t: pd.Timestamp, funding: dict, perp: pd.DataFrame, spot: pd.DataFrame) -> list:
    """Coins to hold in a tranche rebalanced at t: top N by 7-day mean funding among
    coins with both a perp and a spot close at t; non-positive signals stay in cash.
    Ties are broken by universe order (stable sort)."""
    sig = signal_at(t, funding)
    ok = [s for s in UNIVERSE if s in sig.index
          and np.isfinite(perp[s].get(t - H1, np.nan)) and np.isfinite(spot[s].get(t - H1, np.nan))]
    ranked = sorted(ok, key=lambda s: (-sig[s], UNIVERSE.index(s)))[:N_SLOTS]
    return [s for s in ranked if sig[s] > 0]


def aggregate(st) -> tuple[dict, dict]:
    perp = {s: 0.0 for s in UNIVERSE}; spot = {s: 0.0 for s in UNIVERSE}
    for tr in st["tranches"]:
        for s, w in tr["perp"].items():
            perp[s] += w
        for s, w in tr["spot"].items():
            spot[s] += w
    return perp, spot


# ----------------------------------------------------------------- commands
def do_init(start: pd.Timestamp | None = None):
    if STATE.exists():
        print(f"State already exists. Delete {STATE.name} to restart."); return
    now = _utc_naive(pd.Timestamp.utcnow())
    start = start if start is not None else grid_after(now, now + pd.Timedelta(hours=8))[0]
    st = dict(start=iso(start), last_run=iso(start - pd.Timedelta(hours=8)), equity=100000.0,
              cum_carry=0.0, cum_price=0.0, cum_cost=0.0,
              tranches=[dict(next_rebalance=iso(start + pd.Timedelta(days=k)), perp={}, spot={})
                        for k in range(K)],
              last_perp={}, last_spot={})
    save_state(st)
    print(f"Initialized cash-and-carry ledger: first step and first tranche at {iso(start)}; "
          f"tranche k rebalances first at start + k days, then weekly.")


def do_step(now: pd.Timestamp | None = None, quiet: bool = False):
    st = load_state()
    if st is None:
        print("No state. Run --init first."); return
    now = _utc_naive(now if now is not None else pd.Timestamp.utcnow())
    last = _utc_naive(st["last_run"])
    grid = grid_after(last, now - SETTLE)
    if not grid:
        if not quiet:
            print(f"[{now.isoformat()}] up to date (last step {last.isoformat()})")
        return

    lo = min(last, grid[0] - pd.Timedelta(days=8)) - 2 * H1
    hi = grid[-1] - H1
    perp = pd.DataFrame({s: fetch_closes_range(s, lo, hi, "linear") for s in UNIVERSE})
    spot = pd.DataFrame({s: fetch_closes_range(s, lo, hi, "spot") for s in UNIVERSE})
    funding = {s: fetch_funding_range(s, lo - pd.Timedelta(days=2), grid[-1]) for s in UNIVERSE}

    prev = last
    for t in grid:
        Wp, Ws = aggregate(st)
        price = carry = 0.0
        for s in UNIVERSE:
            p1, s1 = perp[s].get(t - H1, np.nan), spot[s].get(t - H1, np.nan)
            p0, s0 = st["last_perp"].get(s), st["last_spot"].get(s)
            if Wp[s] and p0 and np.isfinite(p1):
                price += Wp[s] * (p1 / p0 - 1.0)
            if Ws[s] and s0 and np.isfinite(s1):
                price += Ws[s] * (s1 / s0 - 1.0)
            if Wp[s]:
                f = funding[s]
                carry += -Wp[s] * float(f[(f.index > prev) & (f.index <= t)].sum())
            if np.isfinite(p1):
                st["last_perp"][s] = float(p1)
            if np.isfinite(s1):
                st["last_spot"][s] = float(s1)

        cost, rebalanced = 0.0, []
        for k, tr in enumerate(st["tranches"]):
            if t >= _utc_naive(tr["next_rebalance"]):
                pick = select_at(t, funding, perp, spot)
                w = 1.0 / (K * N_SLOTS)
                new_p = {s: -w for s in pick}; new_s = {s: w for s in pick}
                turn = sum(abs(new_p.get(s, 0) - tr["perp"].get(s, 0)) +
                           abs(new_s.get(s, 0) - tr["spot"].get(s, 0)) for s in UNIVERSE) / 2
                cost += turn * COST_BPS / 1e4
                tr["perp"], tr["spot"] = new_p, new_s
                tr["next_rebalance"] = iso(_utc_naive(tr["next_rebalance"]) + HOLD)
                rebalanced.append(str(k))

        st["equity"] *= (1 + price + carry - cost)
        st["cum_price"] += price; st["cum_carry"] += carry; st["cum_cost"] += cost
        st["last_run"] = iso(t)
        save_state(st)
        append_ledger(dict(time=iso(t), equity=round(st["equity"], 2), step_price=round(price, 7),
                           step_carry=round(carry, 7), step_cost=round(cost, 7),
                           cum_carry=round(st["cum_carry"], 7), cum_price=round(st["cum_price"], 7),
                           cum_cost=round(st["cum_cost"], 7),
                           rebalanced="|".join(rebalanced)))
        if not quiet:
            print(f"[{iso(t)}] equity ${st['equity']:,.2f}  (price {price*100:+.4f}%  "
                  f"carry {carry*100:+.4f}%  cost {cost*100:.4f}%)"
                  f"{'  [tranche ' + ','.join(rebalanced) + ']' if rebalanced else ''}")
        prev = t


def do_report():
    st = load_state()
    if st is None or not LEDGER.exists():
        print("No ledger yet."); return
    led = pd.read_csv(LEDGER)
    days = (pd.Timestamp(st["last_run"]) - pd.Timestamp(st["start"])).total_seconds() / 86400
    held = sum(1 for v in aggregate(st)[0].values() if v)
    print(f"=== Forward test [cash-and-carry CC L=21 K=7] — {days:.1f} days live ===")
    print(f"  equity      ${st['equity']:,.2f}   ({(st['equity']/1e5-1)*100:+.3f}%)")
    print(f"  cum carry   {st['cum_carry']*100:+.3f}%   cum price {st['cum_price']*100:+.3f}%   "
          f"cum cost {st['cum_cost']*100:.3f}%")
    print(f"  coins short (all tranches): {held}   steps: {len(led)}")
    if days > 7:
        print(f"  carry ann   {st['cum_carry']/days*365*100:+.1f}%/yr   "
              f"net ann {(st['equity']/1e5-1)/days*365*100:+.1f}%/yr")
    print(f"\n  Study expectation (Bybit 2022-25): {EXPECT['ann_ret']*100:.1f}%/yr, monthly Sharpe "
          f"{EXPECT['sharpe_monthly']}; 2026 holdout {EXPECT['holdout_ann_ret']*100:+.1f}%/yr.")
    print("  -> Watch whether carry exceeds costs (~3%/yr) and a cash yield.")


def do_dashboard():
    do_report()
    if not LEDGER.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    led = pd.read_csv(LEDGER, parse_dates=["time"])
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(led["time"], led["equity"], color="teal", lw=1.5)
    ax[0].axhline(100000, color="gray", ls=":", lw=1)
    ax[0].set_title("Cash-and-carry forward test (CC L=21 K=7) - live shadow ledger")
    ax[0].set_ylabel("equity ($)"); ax[0].grid(alpha=0.3)
    for col, c in (("cum_carry", "green"), ("cum_price", "orange"), ("cum_cost", "red")):
        ax[1].plot(led["time"], led[col] * 100, color=c, label=col.replace("cum_", "cumulative "))
    ax[1].axhline(0, color="gray", ls=":", lw=1); ax[1].set_ylabel("% of start equity")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(DASH_PNG, dpi=90); plt.close()
    print(f"\n  dashboard -> {DASH_PNG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("init", "step", "report", "dashboard"):
        ap.add_argument(f"--{a}", action="store_true")
    ap.add_argument("--state-dir", default=None, help="use this dir instead of data_cache/ (tests)")
    a = ap.parse_args()
    set_state_dir(a.state_dir)
    if a.init:        do_init()
    elif a.step:      do_step()
    elif a.report:    do_report()
    elif a.dashboard: do_dashboard()
    else:             ap.print_help()
