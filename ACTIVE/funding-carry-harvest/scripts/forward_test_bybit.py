"""
Live forward-test ("shadow ledger") for the locked funding-carry strategy on Bybit.

No API key, no real orders: it tracks a simulated book on LIVE Bybit data. Each run
it pulls fresh prices + the funding events that actually printed since the last run,
marks the held book to market, accrues the real funding income, rebalances every 7
days, and appends to a ledger. Because data after you start didn't exist at backtest
time, this is a genuine out-of-sample test — and it confirms live funding accrues as
the model assumes (the one thing a backtest can't verify).

Locked config (from notebook 07): 23 Bybit perps, long 11 lowest-funding /
short 11 highest-funding, BTC beta-hedge, 7-day rebalance, unlevered (size to the
8% drawdown budget separately).

Usage:
    python scripts/forward_test_bybit.py --init     # start a fresh ledger (run once)
    python scripts/forward_test_bybit.py --step      # advance to now (run daily / per 8h)
    python scripts/forward_test_bybit.py --report    # show live stats vs backtest expectation

Run --step on a schedule (Windows Task Scheduler, or just manually once a day).

Variants (--variant, default "baseline"):
    baseline : the locked config above (1/N weights, BTC beta hedge). Files:
               forward_state.json / forward_ledger.csv / forward_progress.png
    c        : inverse-vol weights within each leg + 2-factor (BTC+ETH) hedge.
               Runs as a separate parallel shadow ledger so the baseline stays a
               clean out-of-sample control. Files get a "_c" suffix.
               Backtest (scripts/eval_hedge_variants.py): lower price-leg vol and
               drawdown on both venues; Sharpe up on Binance (1.32 -> 1.48), down on
               Bybit (1.24 -> 0.90). See paper/ for the full analysis.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.data.bybit_loader import _get, BASE
from quantresearch.funding_harvest import DEFAULT_UNIVERSE, PERIODS_PER_YEAR

UNIVERSE = [c for c in DEFAULT_UNIVERSE if c != "MATICUSDT"]   # 23 tradeable on Bybit
N_EACH = 11
HOLD_DAYS = 7
BETA_WIN = 90          # 8h-periods (30d) for rolling beta
HEDGE = "BTCUSDT"
VOL_WIN = 30           # 8h-periods (10d) of trailing vol for inverse-vol weights (variant c)
FACTORS_C = ["BTCUSDT", "ETHUSDT"]

VARIANT = "baseline"   # set from --variant in __main__
_SUFFIX = {"baseline": "", "c": "_c"}

STATE = ROOT / "data_cache" / "forward_state.json"
LEDGER = ROOT / "data_cache" / "forward_ledger.csv"
DASH_PNG = ROOT / "data_cache" / "forward_progress.png"

# backtest expectation to compare against (Bybit, 7-day hold, recent regime)
EXPECT = dict(sharpe=1.34, carry_ann=0.092, net_ann=0.30)
# variant c: scripts/eval_hedge_variants.py --venue bybit (Aug 2022..May 2025, same venue
# as this live test). Same harness: baseline Sharpe 1.24, c 0.90. On Binance 2021-25 c
# beats baseline (1.48 vs 1.32), so c is a risk reduction, not a reliable return boost.
EXPECT_C = dict(sharpe=0.90, carry_ann=0.060, net_ann=0.18)


def set_variant(v: str):
    """Point state/ledger/dashboard files and expectations at the chosen variant."""
    global VARIANT, STATE, LEDGER, DASH_PNG, EXPECT
    VARIANT = v
    sfx = _SUFFIX[v]
    STATE = ROOT / "data_cache" / f"forward_state{sfx}.json"
    LEDGER = ROOT / "data_cache" / f"forward_ledger{sfx}.csv"
    DASH_PNG = ROOT / "data_cache" / f"forward_progress{sfx}.png"
    if v == "c":
        EXPECT = EXPECT_C


# ----------------------------------------------------------------- live data
def fetch_klines(symbol: str, hours: int = 1000) -> pd.Series:
    """Last `hours` of 1h closes (most-recent), as a Series."""
    url = f"{BASE}/kline?category=linear&symbol={symbol}&interval=60&limit={min(hours,1000)}"
    lst = _get(url)["result"]["list"]
    s = pd.Series({pd.to_datetime(int(x[0]), unit="ms"): float(x[4]) for x in lst})
    return s.sort_index()


def fetch_funding_since(symbol: str, since_ms: int) -> pd.Series:
    """Funding events with timestamp > since_ms (rate per event)."""
    url = f"{BASE}/funding/history?category=linear&symbol={symbol}&limit=200"
    lst = _get(url)["result"]["list"]
    rows = {int(x["fundingRateTimestamp"]): float(x["fundingRate"]) for x in lst
            if int(x["fundingRateTimestamp"]) > since_ms}
    return pd.Series(rows).sort_index()


# ----------------------------------------------------------------- book construction
def target_weights() -> dict:
    """Rank by latest funding; long 11 lowest, short 11 highest; + BTC beta hedge."""
    # ~40 days of hourly closes -> 8h returns -> rolling beta + latest funding
    closes = {s: fetch_klines(s, 1000) for s in UNIVERSE}
    px = pd.DataFrame(closes).resample("8h").last().dropna(how="all")
    ret = np.log(px).diff()

    # latest funding per coin (most recent printed rate)
    fund = {}
    for s in UNIVERSE:
        f = _get(f"{BASE}/funding/history?category=linear&symbol={s}&limit=1")["result"]["list"]
        fund[s] = float(f[0]["fundingRate"]) if f else np.nan
        time.sleep(0.05)
    fr = pd.Series(fund).dropna()

    if VARIANT == "c":
        w = _weights_variant_c(fr, ret)
        last_px = {s: float(px[s].iloc[-1]) for s in UNIVERSE}
        return w, last_px

    w = {s: 0.0 for s in UNIVERSE}
    for s in fr.nsmallest(N_EACH).index:
        w[s] = 1.0 / N_EACH
    for s in fr.nlargest(N_EACH).index:
        w[s] = -1.0 / N_EACH

    # BTC beta hedge: net_beta = sum(w_i * beta_i), hedge = -net_beta
    btc = ret[HEDGE]
    var = btc.tail(BETA_WIN).var()
    net_beta = 0.0
    for s in UNIVERSE:
        if w[s] != 0 and s in ret:
            cov = ret[s].tail(BETA_WIN).cov(btc.tail(BETA_WIN))
            net_beta += w[s] * (cov / var if var > 0 else 1.0)
    w[HEDGE] = w.get(HEDGE, 0.0) - net_beta     # add hedge to BTC leg

    last_px = {s: float(px[s].iloc[-1]) for s in UNIVERSE}
    return w, last_px


def _weights_variant_c(fr: pd.Series, ret: pd.DataFrame) -> dict:
    """Variant c: inverse-vol within each leg (each leg gross 1.0) + BTC/ETH factor hedge.

    Inverse-vol stops the hot, high-funding shorts from dominating the book's risk.
    The 2-factor hedge solves h = -Cov(F)^-1 * exposure, where exposure_k is the
    book's covariance with factor k, so the book is neutral to BTC and ETH jointly.
    """
    longs, shorts = fr.nsmallest(N_EACH).index, fr.nlargest(N_EACH).index
    vol = ret.tail(VOL_WIN).std()
    iv = 1.0 / vol.replace(0, np.nan)

    w = {s: 0.0 for s in UNIVERSE}
    for leg, sign in ((longs, 1.0), (shorts, -1.0)):
        x = iv.reindex(leg).fillna(0.0)
        if x.sum() <= 0:                       # no vol history: fall back to 1/N
            x = pd.Series(1.0, index=leg)
        x = x / x.sum()
        for s in leg:
            w[s] = sign * float(x[s])

    F = ret[FACTORS_C].tail(BETA_WIN).dropna()
    R = ret.tail(BETA_WIN)
    cov_ff = np.atleast_2d(np.cov(F.values.T))
    expo = np.array([sum(w[s] * R[s].cov(F[f]) for s in UNIVERSE if w[s] != 0 and s in R)
                     for f in FACTORS_C])
    try:
        h = -np.linalg.solve(cov_ff, expo)
    except np.linalg.LinAlgError:
        h = -expo / np.diag(cov_ff)
    for f, hk in zip(FACTORS_C, h):
        w[f] = w.get(f, 0.0) + float(hk)        # hedge stacks on any leg weight
    return w


# ----------------------------------------------------------------- ledger ops
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))


def append_ledger(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)


def do_init():
    if STATE.exists():
        print(f"State already exists. Delete {STATE.name} to restart."); return
    w, last_px = target_weights()
    now = pd.Timestamp.utcnow().isoformat()
    st = dict(start=now, last_run=now, last_rebalance=now,
              equity=100000.0, positions=w, last_price=last_px,
              cum_carry=0.0, cum_price=0.0)
    save_state(st)
    longs = [s for s, x in w.items() if x > 0]; shorts = [s for s, x in w.items() if x < 0]
    print(f"Initialized forward test [{VARIANT}] at {now}")
    print(f"  LONG  (lowest funding): {', '.join(longs)}")
    print(f"  SHORT (highest funding): {', '.join(shorts)}")
    print(f"  BTC weight (leg + hedge): {w[HEDGE]:+.3f}")
    if VARIANT == "c":
        print(f"  ETH weight (leg + hedge): {w['ETHUSDT']:+.3f}")
    print(f"  equity = $100,000 (unlevered book)")


def do_step():
    st = load_state()
    if st is None:
        print("No state. Run --init first."); return
    last_run_ms = int(pd.Timestamp(st["last_run"]).timestamp() * 1000)

    # current prices for held coins
    held = {s: x for s, x in st["positions"].items() if x != 0}
    cur_px = {}
    for s in held:
        cur_px[s] = float(fetch_klines(s, 5).iloc[-1]); time.sleep(0.05)

    # mark-to-market price P&L since last run
    price_ret = 0.0
    for s, w in held.items():
        p0 = st["last_price"].get(s); p1 = cur_px[s]
        if p0:
            price_ret += w * (p1 / p0 - 1.0)

    # funding income from events printed since last run:  -w * sum(rates)
    carry_ret = 0.0
    for s, w in held.items():
        fs = fetch_funding_since(s, last_run_ms); time.sleep(0.05)
        if len(fs):
            carry_ret += -w * fs.sum()

    eq0 = st["equity"]
    st["equity"] = eq0 * (1 + price_ret + carry_ret)
    st["cum_carry"] += carry_ret
    st["cum_price"] += price_ret

    # rebalance if 7 days elapsed
    now = pd.Timestamp.utcnow()
    rebalanced = False
    if (now - pd.Timestamp(st["last_rebalance"])).total_seconds() >= HOLD_DAYS * 86400:
        w_new, last_px = target_weights()
        st["positions"] = w_new
        st["last_price"] = last_px
        st["last_rebalance"] = now.isoformat()
        rebalanced = True
    else:
        for s in held:
            st["last_price"][s] = cur_px[s]    # mark forward

    st["last_run"] = now.isoformat()
    save_state(st)
    append_ledger(dict(time=now.isoformat(), equity=round(st["equity"], 2),
                       step_price=round(price_ret, 6), step_carry=round(carry_ret, 6),
                       cum_carry=round(st["cum_carry"], 6), cum_price=round(st["cum_price"], 6),
                       rebalanced=rebalanced))
    print(f"[{now.isoformat()}] equity ${st['equity']:,.2f}  "
          f"(this step: price {price_ret*100:+.3f}%  carry {carry_ret*100:+.3f}%)"
          f"{'  [REBALANCED]' if rebalanced else ''}")


def do_report():
    if not LEDGER.exists():
        print("No ledger yet. Run --init then --step."); return
    led = pd.read_csv(LEDGER, parse_dates=["time"])
    st = load_state()
    days = (pd.Timestamp(st["last_run"]) - pd.Timestamp(st["start"])).total_seconds() / 86400
    ret = st["equity"] / 100000 - 1
    tag = "" if VARIANT == "baseline" else f" [variant {VARIANT}]"
    print(f"=== Forward test{tag} — {days:.1f} days live ===")
    print(f"  equity        ${st['equity']:,.2f}   ({ret*100:+.2f}% cumulative, unlevered)")
    print(f"  cum carry     {st['cum_carry']*100:+.2f}%   cum price {st['cum_price']*100:+.2f}%")
    if days > 3 and len(led) > 3:
        # annualize from per-step returns
        step_tot = led["step_price"] + led["step_carry"]
        ann = step_tot.mean() * (len(led) / days) * 365
        sh = step_tot.mean() / step_tot.std() * np.sqrt(len(led) / days * 365) if step_tot.std() > 0 else 0
        print(f"  annualized    net {ann*100:+.1f}%/yr   Sharpe ~{sh:.2f}")
        print(f"  carry ann     {st['cum_carry']/days*365*100:+.1f}%/yr   "
              f"(backtest expected ~{EXPECT['carry_ann']*100:.0f}%/yr)")
    print(f"\n  Backtest expectation: Sharpe ~{EXPECT['sharpe']}, carry ~{EXPECT['carry_ann']*100:.0f}%/yr, "
          f"net ~{EXPECT['net_ann']*100:.0f}%/yr (unlevered)")
    print(f"  -> If live carry tracks ~{EXPECT['carry_ann']*100:.0f}%/yr and net is positive, "
          f"the edge is confirmed live.")


def do_dashboard():
    """Print the report and (re)generate a progress PNG from the ledger."""
    if not LEDGER.exists():
        print("No ledger yet. Run --init then --step."); return
    do_report()

    import matplotlib
    matplotlib.use("Agg")          # headless: safe under Task Scheduler
    import matplotlib.pyplot as plt

    led = pd.read_csv(LEDGER, parse_dates=["time"])
    rebal = led["rebalanced"].astype(str).str.lower().isin(["true", "1"])

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(led["time"], led["equity"], color="teal", lw=1.5)
    ax[0].axhline(100000, color="gray", ls=":", lw=1)
    ax[0].set_ylabel("equity ($, unlevered)")
    title = "Funding-carry forward test — live shadow ledger"
    if VARIANT != "baseline":
        title += f" (variant {VARIANT}: inverse-vol + BTC/ETH hedge)"
    ax[0].set_title(title)
    ax[0].grid(alpha=0.3)

    ax[1].plot(led["time"], led["cum_carry"] * 100, color="green", label="cum carry %")
    ax[1].plot(led["time"], led["cum_price"] * 100, color="orange", label="cum price %")
    ax[1].axhline(0, color="gray", ls=":", lw=1)
    for t in led.loc[rebal, "time"]:
        ax[1].axvline(t, color="red", ls="--", lw=0.6, alpha=0.5)
    ax[1].set_ylabel("cumulative %")
    ax[1].set_xlabel("rebalances marked in red")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    out = DASH_PNG
    plt.savefig(out, dpi=90); plt.close()
    print(f"\n  dashboard -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dashboard", action="store_true", help="report + refresh progress PNG")
    ap.add_argument("--variant", choices=sorted(_SUFFIX), default="baseline",
                    help="which parallel shadow ledger to drive (default: baseline)")
    a = ap.parse_args()
    set_variant(a.variant)
    if a.init:        do_init()
    elif a.step:      do_step()
    elif a.report:    do_report()
    elif a.dashboard: do_dashboard()
    else:             ap.print_help()
