"""
Funding-carry improvement study, run exactly as pre-registered in
docs/preregistration_2026-11-07.md.

    python scripts/eval_improvements.py --validate   # engine vs eval_hedge_variants.py
    python scripts/eval_improvements.py              # full study -> docs/improvement_results/

Outputs (docs/improvement_results/):
    grid.csv                 all 24 configurations x 2 venues
    schedule.csv             schedule sensitivity (7 start days) for every K=1 config
    holdout.csv              live-period (2026) evaluation of the selected configurations
    summary.json             selection, R3 check, deflated Sharpe
    *.png                    figures for the report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import pandas as pd

from quantresearch.carry_lab import (Config, Panel, simulate, stats, deflated_sharpe,
                                     HOLD, WARMUP, PERIODS_PER_YEAR)
from quantresearch.funding_harvest import load_panels

UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'NEARUSDT', 'AVAXUSDT',
            'XLMUSDT', 'AAVEUSDT', 'UNIUSDT', 'TRXUSDT', 'ATOMUSDT', 'ETCUSDT', 'SANDUSDT',
            'AXSUSDT', 'COMPUSDT']
CACHE = ROOT / "data_cache"
OUT = ROOT / "docs" / "improvement_results"

KINDS = ["XS0", "XSC", "PAIR", "CC"]
LS = [1, 9, 21]
KS = [1, 7]
GRID = [Config(k, L, K) for k, L, K in product(KINDS, LS, KS)]
BASE = Config("XS0", 1, 1)

LIVE_START = pd.Timestamp("2026-06-06 16:00")   # first 8h grid time after live start
LIVE_END = pd.Timestamp("2026-11-06 08:00")


# ----------------------------------------------------------------- panels
def _spot_logret(closes: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    S = pd.DataFrame({s: c.resample("8h").last() for s, c in closes.items()}).reindex(index)
    return np.log(S).diff()[UNIVERSE]


def load_venue(venue: str, with_spot: bool = True) -> Panel:
    if venue == "binance":
        P, F, R = load_panels(UNIVERSE, start="2021-01", end="2025-05")
        Rs = None
        if with_spot:
            closes = {s: pd.read_parquet(CACHE / f"{s}_1h_spot.parquet")["close"] for s in UNIVERSE}
            Rs = _spot_logret(closes, P.index)
    elif venue == "bybit":
        from quantresearch.data.bybit_loader import load_bybit_panels
        P, F, R = load_bybit_panels(UNIVERSE, start="2022-07", end="2025-05")
        Rs = None
        if with_spot:
            closes = {s: pd.read_parquet(CACHE / "improve" / f"{s}_bybit_spot_60.parquet")["close"]
                      for s in UNIVERSE}
            Rs = _spot_logret(closes, P.index)
    else:
        raise ValueError(venue)
    return Panel(R=R[UNIVERSE], F=F[UNIVERSE], Rs=Rs)


def load_live() -> tuple[Panel, int]:
    """Bybit perp/spot/funding for the live period, same construction as the Bybit panel
    (8h last close, funding summed per 8h). Returns the panel and the index of LIVE_START."""
    d = CACHE / "improve" / "live"
    perp = {s: pd.read_parquet(d / f"{s}_perp.parquet")["close"] for s in UNIVERSE}
    spot = {s: pd.read_parquet(d / f"{s}_spot.parquet")["close"] for s in UNIVERSE}
    P = pd.DataFrame({s: c.resample("8h").last() for s, c in perp.items()}).sort_index()
    f8 = {}
    for s in UNIVERSE:
        f = pd.read_parquet(d / f"{s}_funding.parquet")["funding"]
        f.index = f.index.floor("h")
        f8[s] = f.groupby(f.index.floor("8h")).sum()
    F = pd.DataFrame(f8).reindex(P.index)
    P = P[P.index <= LIVE_END]; F = F.loc[P.index]
    R = np.log(P).diff()[UNIVERSE]
    Rs = _spot_logret(spot, P.index)
    start = int(np.searchsorted(P.index, LIVE_START))
    return Panel(R=R, F=F[UNIVERSE], Rs=Rs), start


# ----------------------------------------------------------------- validation
def validate():
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_hedge_variants import run_all
    ok = True
    for venue in ("binance", "bybit"):
        _, _, ref = run_all(venue)
        panel = load_venue(venue, with_spot=False)
        for name, cfg in (("baseline", Config("XS0", 1, 1)), ("C", Config("XSC", 1, 1))):
            mine = stats(simulate(panel, cfg))["sharpe"]
            diff = mine - ref[name]["sharpe"]
            good = abs(diff) <= 0.05
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] {venue:<8} {name:<9} engine {mine:.3f}  "
                  f"reference {ref[name]['sharpe']:.3f}  diff {diff:+.3f}")
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    return ok


# ----------------------------------------------------------------- study
SINGLE_CHANGES = {                     # R1: one factor changed at a time vs BASE
    "signal 3d mean (L=9)": Config("XS0", 9, 1),
    "signal 7d mean (L=21)": Config("XS0", 21, 1),
    "7 tranches (K=7)": Config("XS0", 1, 7),
    "inverse vol + BTC/ETH (XSC)": Config("XSC", 1, 1),
    "correlation pairs (PAIR)": Config("PAIR", 1, 1),
    "cash-and-carry (CC)": Config("CC", 1, 1),
}


def schedule_spread(panel: Panel, cfg: Config, start: int, offsets) -> dict:
    """Final cumulative return of a K=1 book for each start offset (rebalance day)."""
    tot = [stats(simulate(panel, cfg, start=start, offset=o))["total"] for o in offsets]
    return dict(std=float(np.std(tot, ddof=1)), lo=float(min(tot)), hi=float(max(tot)),
                totals=[float(x) for x in tot])


def run_study():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panels = {v: load_venue(v) for v in ("binance", "bybit")}
    print(f"panels loaded ({time.time()-t0:.0f}s)", flush=True)

    rows, nets = [], {}
    for venue, pan in panels.items():
        for cfg in GRID:
            df = simulate(pan, cfg)
            nets[(venue, cfg)] = df
            rows.append(dict(venue=venue, kind=cfg.kind, L=cfg.L, K=cfg.K, config=cfg.name,
                             **stats(df)))
        print(f"grid {venue} done ({time.time()-t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid.csv", index=False)

    # schedule sensitivity for every K=1 configuration (7 start days, one day apart)
    sched = []
    offs = [k * (HOLD // 7) for k in range(7)]
    for venue, pan in panels.items():
        for cfg in [c for c in GRID if c.K == 1]:
            sp = schedule_spread(pan, cfg, WARMUP, offs)
            sched.append(dict(venue=venue, config=cfg.name, kind=cfg.kind, L=cfg.L,
                              std=sp["std"], lo=sp["lo"], hi=sp["hi"]))
        print(f"schedule {venue} done ({time.time()-t0:.0f}s)", flush=True)
    sched = pd.DataFrame(sched)
    sched.to_csv(OUT / "schedule.csv", index=False)

    piv = grid.pivot_table(index="config", columns="venue", values="sharpe")
    piv["maximin"] = piv[["binance", "bybit"]].min(axis=1)

    # R1 separate effects
    base_sr = piv.loc[BASE.name]
    r1 = {lbl: dict(binance=float(piv.loc[c.name, "binance"] - base_sr["binance"]),
                    bybit=float(piv.loc[c.name, "bybit"] - base_sr["bybit"]))
          for lbl, c in SINGLE_CHANGES.items()}

    # R2 selection (maximin; ties -> lower mean schedule std of the K=1 version)
    best_val = piv["maximin"].max()
    tied = piv.index[np.isclose(piv["maximin"], best_val)]
    cand_name = tied[0]
    cand = next(c for c in GRID if c.name == cand_name)

    # R3 robustness
    r3 = bool((piv.loc[cand_name, "binance"] > base_sr["binance"]) and
              (piv.loc[cand_name, "bybit"] > base_sr["bybit"]))

    # R5 deflated Sharpe on Bybit, 48 trials
    ann = np.sqrt(PERIODS_PER_YEAR)
    trial_srs = [float(nets[k]["net"].mean() / nets[k]["net"].std()) for k in nets]
    dsr = deflated_sharpe(nets[("bybit", cand)]["net"], trial_srs, n_trials=len(nets))
    dsr_base = deflated_sharpe(nets[("bybit", BASE)]["net"], trial_srs, n_trials=len(nets))

    # best per construction family (maximin within kind), for the holdout
    fam = {}
    for kind in KINDS:
        sub = piv.loc[[c.name for c in GRID if c.kind == kind]]
        fam[kind] = sub["maximin"].idxmax()

    # R4 holdout (once)
    live, s_idx = load_live()
    hold_cfgs = {"baseline": BASE, "candidate": cand}
    hold_cfgs.update({f"best {k}": next(c for c in GRID if c.name == n) for k, n in fam.items()})
    hold_rows, hold_nets = [], {}
    for lbl, cfg in hold_cfgs.items():
        df = simulate(live, cfg, start=s_idx)
        hold_nets[lbl] = df
        hold_rows.append(dict(label=lbl, config=cfg.name, **stats(df)))
    hold = pd.DataFrame(hold_rows).drop_duplicates("label")
    hold.to_csv(OUT / "holdout.csv", index=False)

    # all 21 rebalance schedules (7 days x 3 times) on the holdout, baseline and candidate
    sched21 = {lbl: schedule_spread(live, cfg, s_idx, range(HOLD))
               for lbl, cfg in (("baseline", BASE), ("candidate", cand))
               if cfg.K == 1}

    summary = dict(
        n_configs=len(GRID), n_trials=len(nets),
        baseline=dict(config=BASE.name, binance=float(base_sr["binance"]),
                      bybit=float(base_sr["bybit"])),
        candidate=dict(config=cand_name, binance=float(piv.loc[cand_name, "binance"]),
                       bybit=float(piv.loc[cand_name, "bybit"]), maximin=float(best_val),
                       ties=list(tied)),
        R1_separate_effects=r1, R3_beats_baseline_on_both=r3,
        R5_deflated_sharpe=dict(candidate=dsr, baseline=dsr_base),
        best_per_family=fam,
        holdout_schedules=sched21,
        runtime_s=round(time.time() - t0),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    piv.sort_values("maximin", ascending=False).to_csv(OUT / "ranking.csv")
    make_figures(grid, sched, piv, nets, cand, hold_nets, sched21)
    print(f"study done ({time.time()-t0:.0f}s)")
    return summary


def make_figures(grid, sched, piv, nets, cand, hold_nets, sched21):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, amber, grey, green = "#4C86FF", "#E39B2D", "#7E8CA8", "#2BA84A"

    # 1. Sharpe of every configuration on both venues
    order = piv.sort_values("maximin").index
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(order))
    ax.barh(y - 0.2, piv.loc[order, "binance"], 0.4, color=blue, label="Binance 2021-25")
    ax.barh(y + 0.2, piv.loc[order, "bybit"], 0.4, color=amber, label="Bybit 2022-25")
    ax.axvline(piv.loc[BASE.name, "bybit"], color=grey, ls=":", lw=1)
    ax.set_yticks(y, order, fontsize=8)
    ax.set_xlabel("Sharpe ratio (net of costs)")
    ax.set_title("All 24 configurations, sorted by the weaker venue")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "sharpe_grid.png", dpi=110); plt.close(fig)

    # 2. schedule sensitivity (K=1): spread of outcomes across 7 start days
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, venue in zip(axes, ("binance", "bybit")):
        sub = sched[sched["venue"] == venue].set_index("config")
        ax.barh(sub.index, sub["std"] * 100, color=blue)
        ax.set_title(venue.capitalize()); ax.set_xlabel("std of total return across start days (pp)")
        ax.grid(axis="x", alpha=0.2)
    axes[0].tick_params(axis="y", labelsize=8)
    fig.suptitle("Rebalance-day sensitivity (K = 1 books)")
    fig.tight_layout(); fig.savefig(OUT / "schedule_sensitivity.png", dpi=110); plt.close(fig)

    # 3. equity: baseline vs candidate on both venues
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, venue in zip(axes, ("binance", "bybit")):
        for cfg, col, lbl in ((BASE, grey, "baseline"), (cand, blue, f"candidate: {cand.name}")):
            ax.plot((1 + nets[(venue, cfg)]["net"]).cumprod(), color=col, lw=1.4, label=lbl)
        ax.set_yscale("log"); ax.set_title(venue.capitalize()); ax.grid(alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "equity_candidate.png", dpi=110)
    plt.close(fig)

    # 4. holdout
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = [grey, blue, "#9C6ADE", green, amber, "#D6455D"]
    for (lbl, df), col in zip(hold_nets.items(), cols):
        ax.plot((1 + df["net"]).cumprod(), lw=1.3, color=col, label=lbl)
    ax.axhline(1, color=grey, ls=":", lw=1); ax.grid(alpha=0.2)
    ax.set_title("Holdout: live period 2026 (Bybit), evaluated once")
    ax.legend(fontsize=8, loc="lower left")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "holdout.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        sys.exit(0 if validate() else 1)
    if not validate():
        sys.exit("engine validation failed; study not run")
    s = run_study()
    print(json.dumps({k: s[k] for k in ("baseline", "candidate", "R3_beats_baseline_on_both")},
                     indent=2))
