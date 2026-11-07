"""
Study 4: short-horizon cross-sectional crypto signals with order flow, as pre-registered
in docs/preregistration_study4_2026-11-07.md.

    python study4.py --checks     # engine checks only
    python study4.py              # checks, then the study -> results/

Hourly bars, index = bar open time. Bar t closes at t+1h. A signal formed with data up to
the close of bar t trades at the close of bar t+1, so its weights earn the returns of
bars t+2, t+3, ... (one-bar execution lag).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FCH = ROOT.parent / "funding-carry-harvest"
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'NEARUSDT', 'AVAXUSDT',
            'XLMUSDT', 'AAVEUSDT', 'UNIUSDT', 'TRXUSDT', 'ATOMUSDT', 'ETCUSDT', 'SANDUSDT',
            'AXSUSDT', 'COMPUSDT']
CACHE = ROOT / "data_cache"
OUT = ROOT / "results"
K = 5
ANN = 365
COST_BPS = 7.0
PERIODS = {"selection": ("2021-02-01", "2023-12-31"),
           "confirmation": ("2024-01-01", "2025-05-31"),
           "holdout": ("2025-06-01", "2026-10-31")}
SIGNALS = {  # code: (score, window_h, rebalance_h)
    "REV24_8": ("rev", 24, 8), "REV24_24": ("rev", 24, 24),
    "OFIC24_8": ("ofi", 24, 8), "OFIC24_24": ("ofi", 24, 24),
    "OFIR24_8": ("-ofi", 24, 8), "OFIR24_24": ("-ofi", 24, 24),
    "OFIC1_1": ("ofi", 1, 1),
}


def load() -> dict:
    C, V, TB, F = {}, {}, {}, {}
    for s in UNIVERSE:
        a = pd.read_parquet(FCH / "data_cache" / f"{s}_1h_um.parquet")[["close", "volume", "taker_buy_base"]]
        b = pd.read_parquet(CACHE / f"{s}_1h.parquet")
        k = pd.concat([a, b]).sort_index(); k = k[~k.index.duplicated()]
        C[s], V[s], TB[s] = k["close"], k["volume"], k["taker_buy_base"]
        fa = pd.read_parquet(FCH / "data_cache" / f"{s}_funding.parquet")["funding"]
        fb = pd.read_parquet(CACHE / f"{s}_funding.parquet")["funding"]
        f = pd.concat([fa, fb]).sort_index(); f = f[~f.index.duplicated()]
        F[s] = f
    idx = pd.date_range("2021-01-01", "2026-10-31 23:00", freq="h")
    C = pd.DataFrame(C).reindex(idx); V = pd.DataFrame(V).reindex(idx); TB = pd.DataFrame(TB).reindex(idx)
    # funding paid at time T is booked on the bar that closes at T (open time T-1h)
    fund = pd.DataFrame(0.0, index=idx, columns=UNIVERSE)
    for s, f in F.items():
        t = f.index.floor("h") - pd.Timedelta(hours=1)
        g = f.groupby(t).sum()
        fund.loc[fund.index.intersection(g.index), s] = g.reindex(fund.index.intersection(g.index)).values
    R = C.pct_change()
    return dict(C=C, V=V, TB=TB, R=R, fund=fund, idx=idx)


def scores(D: dict, kind: str, win: int) -> pd.DataFrame:
    """Score at bar t using data up to the close of bar t."""
    if kind == "rev":
        return -(D["C"] / D["C"].shift(win) - 1.0)
    flow = (2 * D["TB"] - D["V"]).rolling(win).sum() / D["V"].rolling(win).sum()
    return flow if kind == "ofi" else -flow


def targets(score: pd.DataFrame, reb: int) -> pd.DataFrame:
    """Long top K / short bottom K (+-1/(2K)) at rebalance bars; NaN elsewhere.
    Rebalance bars: those closing at 00:00 UTC + multiples of `reb` hours."""
    close_hour = (score.index.hour + 1) % 24
    is_reb = (close_hour % reb == 0) if reb > 1 else np.ones(len(score), bool)
    S = score.values
    W = np.full(S.shape, np.nan)
    for i in np.where(is_reb)[0]:
        row = S[i]
        ok = np.where(np.isfinite(row))[0]
        if len(ok) < 2 * K:
            continue
        order = ok[np.argsort(row[ok], kind="stable")]
        w = np.zeros(S.shape[1])
        w[order[-K:]] = 0.5 / K
        w[order[:K]] = -0.5 / K
        W[i] = w
    return pd.DataFrame(W, index=score.index, columns=score.columns)


def backtest(D: dict, T: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Hourly P&L of target weights with a one-bar lag: targets at t are held from t+2."""
    W = T.shift(2).ffill().fillna(0.0)
    R = D["R"].fillna(0.0)
    price = (W * R).sum(axis=1)
    carry = -(W * D["fund"]).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(0.0)
    cost = turn * cost_bps / 1e4
    return pd.DataFrame({"price": price, "carry": carry, "cost": cost, "turn": turn,
                         "net": price + carry - cost})


def daily(h: pd.Series) -> pd.Series:
    return (1 + h).groupby(h.index.floor("D")).prod() - 1


def vol_target(d: pd.Series, target=0.10, cap=5.0) -> pd.Series:
    s = d.rolling(60).std().shift(1) * np.sqrt(ANN)
    lev = (target / s).clip(upper=cap).fillna(0.0)
    return d * lev


def metrics(d: pd.Series) -> dict:
    d = d.dropna()
    if len(d) < 30 or d.std() == 0:
        return dict(sharpe=np.nan, cagr_10v=np.nan, maxdd_10v=np.nan)
    sc = vol_target(d)
    sc = sc[sc.ne(0).cummax()]
    eq = (1 + sc).cumprod()
    yrs = len(sc) / ANN
    return dict(sharpe=float(d.mean() / d.std() * np.sqrt(ANN)),
                cagr_10v=float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
                maxdd_10v=float((eq / eq.cummax() - 1).min()))


def per(x, p):
    a, b = PERIODS[p]
    return x.loc[a:b]


# ----------------------------------------------------------------- checks
def checks(D) -> bool:
    ok = True
    # 1. no look-ahead in scores: truncating the future does not change the score at t
    t = D["idx"].get_loc(pd.Timestamp("2023-06-15 07:00"))
    cut = {k: (v.iloc[: t + 1] if isinstance(v, pd.DataFrame) else v) for k, v in D.items()}
    for kind, win in (("rev", 24), ("ofi", 24), ("ofi", 1)):
        a = scores(D, kind, win).iloc[t]; b = scores(cut, kind, win).iloc[t]
        good = np.allclose(a.values, b.values, equal_nan=True); ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] no look-ahead in score {kind}{win}")
    # 2. lag: a target set at bar t first earns the return of bar t+2
    T = pd.DataFrame(np.nan, index=D["idx"], columns=UNIVERSE)
    T.iloc[t] = 0.0; T.iloc[t, 0] = 1.0
    bt = backtest(D, T, 0.0)
    good = bt["price"].iloc[t + 1] == 0 and np.isclose(bt["price"].iloc[t + 2], D["R"].iloc[t + 2, 0])
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] one-bar execution lag (earns from t+2)")
    # 3. book shape: dollar-neutral, gross 1
    Tr = targets(scores(D, "rev", 24), 24)
    rows = Tr.dropna(how="all")
    good = np.allclose(rows.sum(axis=1), 0) and np.allclose(rows.abs().sum(axis=1), 1.0); ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] books are dollar-neutral with gross 1 ({len(rows)} rebalances)")
    # 4. placebo: randomly shuffled scores give a Sharpe near zero (gross, 24h rebalance)
    rng = np.random.default_rng(7)
    sc = scores(D, "rev", 24)
    sh = pd.DataFrame(rng.permuted(sc.values, axis=1), index=sc.index, columns=sc.columns)
    srs = []
    for _ in range(5):
        sh = pd.DataFrame(rng.permuted(sc.values, axis=1), index=sc.index, columns=sc.columns)
        d = daily(backtest(D, targets(sh, 24), 0.0)["net"]).loc["2021-02-01":"2025-05-31"]
        srs.append(d.mean() / d.std() * np.sqrt(ANN))
    good = abs(np.mean(srs)) < 0.5; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] placebo (shuffled scores) Sharpe {np.mean(srs):+.2f} "
          f"(5 draws: {', '.join(f'{x:+.2f}' for x in srs)})")
    print("CHECKS PASSED" if ok else "CHECKS FAILED")
    return ok


# ----------------------------------------------------------------- study
def run(D):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    rows, nets = [], {}
    btc = daily(D["R"]["BTCUSDT"].fillna(0.0))
    for code, (kind, win, reb) in SIGNALS.items():
        T = targets(scores(D, kind, win), reb)
        for bps in (0.0, 2.0, 7.0, 12.0):
            bt = backtest(D, T, bps)
            d = daily(bt["net"])
            if bps == COST_BPS:
                nets[code] = d
            for p in PERIODS:
                dp = per(d, p); hp = per(bt, p)
                yrs = len(dp) / ANN
                rows.append(dict(signal=code, cost_bps=bps, period=p, **metrics(dp),
                                 turnover_yr=float(hp["turn"].sum() / yrs),
                                 cost_drag_yr=float(hp["cost"].sum() / yrs),
                                 carry_yr=float(hp["carry"].sum() / yrs),
                                 corr_btc=float(dp.corr(btc.reindex(dp.index)))))
        print(f"{code:<10} done ({time.time()-t0:.0f}s)", flush=True)
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics.csv", index=False)
    m = met[met.cost_bps == COST_BPS].set_index(["signal", "period"])
    feas = [c for c in SIGNALS if all(m.loc[(c, p), "sharpe"] >= 1.0 and m.loc[(c, p), "maxdd_10v"] >= -0.20
                                      for p in ("selection", "confirmation"))]
    score = {c: float(min(m.loc[(c, "selection"), "sharpe"], m.loc[(c, "confirmation"), "sharpe"])) for c in feas}
    cand = max(score, key=score.get) if score else None
    summary = dict(feasible=feas, candidate=cand, maximin_sharpe=score, runtime_s=round(time.time() - t0))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(nets).to_csv(OUT / "daily_net_7bps.csv")
    figure(nets)
    print(json.dumps(summary, indent=2))


def figure(nets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, d in nets.items():
        eq = (1 + vol_target(d.loc["2021-02-01":])).cumprod()
        ax.plot(eq, lw=1.2, label=c)
    for p, (a, _) in PERIODS.items():
        ax.axvline(pd.Timestamp(a), color="grey", ls=":", lw=0.8)
    ax.set_yscale("log"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
    ax.set_title("Study 4: signals net of 7 bps per side, scaled to 10% volatility")
    fig.tight_layout(); fig.savefig(OUT / "equity_7bps.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--checks", action="store_true")
    a = ap.parse_args()
    D = load()
    if not checks(D):
        sys.exit("checks failed; study not run")
    if not a.checks:
        run(D)
