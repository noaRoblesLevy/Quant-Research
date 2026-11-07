"""
Run the multi-asset portfolio study exactly as pre-registered
(docs/preregistration_2026-11-07.md).

    python run_study.py --checks   # sanity checks only (no results are read)
    python run_study.py            # full study -> results/

Outputs in results/: calibration.csv, metrics.csv, yearly.csv, summary.json, figures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

import portfolio_lab as L

OUT = ROOT / "results"
SCALED = [s for s in L.STRATEGIES if s != "BENCH"]


def checks(px, rf) -> bool:
    """Engine sanity checks, run before any result is looked at."""
    ok = True
    # 1. BENCH must reproduce SPY buy-and-hold (minus the one-time entry cost)
    bench = L.simulate("BENCH", px, rf, None)
    spy = px["SPY"].pct_change()
    first_target = L.month_ends(px.index)[0]           # BENCH holds SPY from the first target
    live = bench.index[first_target + 3:]              # held from d+2; skip the entry-cost day
    diff = (bench.loc[live] - spy.loc[live]).abs().max()
    good = diff < 1e-12
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] BENCH equals SPY daily returns (max diff {diff:.1e})")
    # 2. fast path equals reference implementation
    for name in ("P1", "SMA"):
        a = L.simulate(name, px, rf, 0.10)
        b = L._simulate_reference(name, px, rf, 0.10)
        d = (a - b).abs().max()
        good = d < 1e-12
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: prepare/scaled_returns == reference (max diff {d:.1e})")
    # 3. no look-ahead: weights at a month end do not change when future data is removed
    rets = px.pct_change()
    d = L.month_ends(px.index)[150]
    for b in ("TF", "RP", "SMA"):
        full = L.block_weights(b, d, px, rets)
        cut = L.block_weights(b, d, px.iloc[: d + 1], rets.iloc[: d + 1])
        good = np.allclose(full.values, cut.values)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] no look-ahead in {b} at {px.index[d].date()}")
    # 4. realised vol of a scaled strategy is near its target (full sample)
    r = L.simulate("TF", px, rf, 0.10).loc["2008":]
    v = r.std() * np.sqrt(L.DAYS)
    good = 0.07 < v < 0.13
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] TF scaled to 10% target realises {v:.1%} volatility")
    print("CHECKS PASSED" if ok else "CHECKS FAILED")
    return ok


def run():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    px, rf = L.load_data()
    bench = L.simulate("BENCH", px, rf, None)
    spy = px["SPY"].pct_change()

    # calibration: highest target with selection max drawdown <= 15%
    calib, series = [], {"BENCH": bench}
    for name in SCALED:
        prep = L.prepare(name, px, rf)
        best, rows = None, []
        for tau in L.TARGETS:
            r = L.scaled_returns(prep, rf, tau)
            dd = L.metrics(L.period(r, "selection"), rf)["maxdd"]
            rows.append((tau, dd))
            if dd >= -L.SEL_DD_LIMIT:
                best = (tau, r)
        tau, r = best if best else (L.TARGETS[0], L.scaled_returns(prep, rf, L.TARGETS[0]))
        series[name] = r
        calib.append(dict(strategy=name, target_vol=tau, feasible=best is not None,
                          selection_dd_at_target=dict(rows)[tau]))
        print(f"{name:<6} target vol {tau:.0%} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(calib).to_csv(OUT / "calibration.csv", index=False)

    # metrics per period
    rows = []
    for name, r in series.items():
        for p in L.PERIODS:
            rows.append(dict(strategy=name, period=p, **L.metrics(L.period(r, p), rf, spy)))
        rows.append(dict(strategy=name, period="full 2007-07..2026-11",
                         **L.metrics(r.loc["2007-07-01":], rf, spy)))
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics.csv", index=False)

    yearly = pd.DataFrame({n: (1 + r.loc["2007-07-01":]).groupby(r.loc["2007-07-01":].index.year).prod() - 1
                           for n, r in series.items()})
    yearly.to_csv(OUT / "yearly.csv")

    # D1 feasibility, D2 selection (Selection and Confirmation only)
    m = met.set_index(["strategy", "period"])
    spy_sr = {p: m.loc[("BENCH", p), "sharpe"] for p in ("selection", "confirmation")}
    qual = []
    for name in SCALED:
        dd_ok = all(m.loc[(name, p), "maxdd"] >= -0.20 for p in ("selection", "confirmation"))
        sr_ok = all(m.loc[(name, p), "sharpe"] > spy_sr[p] for p in ("selection", "confirmation"))
        if dd_ok and sr_ok:
            qual.append(name)
    score = {n: min(m.loc[(n, "selection"), "cagr"], m.loc[(n, "confirmation"), "cagr"]) for n in qual}
    cand = max(score, key=score.get) if score else None

    summary = dict(qualifying=qual, maximin_cagr=score, candidate=cand,
                   calibration={c["strategy"]: c["target_vol"] for c in calib},
                   runtime_s=round(time.time() - t0))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    figures(series, met, cand)
    print(json.dumps(summary, indent=2))
    return summary


def figures(series, met, cand):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["BENCH", "P1", "P2", "P4", "TF", "RP"]
    if cand and cand not in show:
        show.append(cand)
    cols = {"BENCH": "#7E8CA8", "P1": "#4C86FF", "P2": "#E39B2D", "P4": "#2BA84A",
            "TF": "#9C6ADE", "RP": "#D6455D", "P3": "#1F4FE0", "P5": "#8C564B",
            "SMA": "#17BECF", "TFL": "#BCBD22"}
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    for n in show:
        r = series[n].loc["2007-07-01":]
        eq = (1 + r).cumprod()
        lbl = "S&P 500 (SPY)" if n == "BENCH" else n
        ax[0].plot(eq, lw=2.2 if n == cand else 1.2, color=cols.get(n), label=lbl)
        ax[1].plot(eq / eq.cummax() - 1, lw=1.0, color=cols.get(n))
    for p, (a, b) in L.PERIODS.items():
        ax[0].axvline(pd.Timestamp(a), color="grey", ls=":", lw=0.8)
        ax[0].text(pd.Timestamp(a), ax[0].get_ylim()[1] if False else 1.0, f" {p}", fontsize=8, color="grey")
    ax[0].set_yscale("log"); ax[0].set_ylabel("growth of 1 (log)"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.2); ax[1].set_ylabel("drawdown"); ax[1].axhline(-0.2, color="red", ls="--", lw=0.8)
    ax[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "equity_drawdown.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true")
    a = ap.parse_args()
    px, rf = L.load_data()
    if not checks(px, rf):
        sys.exit("checks failed; study not run")
    if not a.checks:
        run()
