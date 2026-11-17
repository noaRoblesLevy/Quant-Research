"""
Regime filter for cash-and-carry, run exactly as pre-registered in
docs/preregistration_2026-11-17.md.

    python scripts/eval_regime_filter.py   -> docs/regime_filter_results/

CC L=21 K=7 with an entry hurdle tau on the annualised 7-day mean funding, tau in
{0, 5, 10, 15}% a year. Idle capital earns cash; the primary metric is the Sharpe ratio
of the excess return over cash, excess_t = net_t - gross_t * cash_t.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import pandas as pd

from quantresearch.carry_lab import Config, simulate, stats, WARMUP, PERIODS_PER_YEAR
from eval_improvements import load_venue, load_live

OUT = ROOT / "docs" / "regime_filter_results"
TAUS = [0.0, 0.05, 0.10, 0.15]
CASH = {2021: 0.0005, 2022: 0.020, 2023: 0.051, 2024: 0.050, 2025: 0.042, 2026: 0.037}
ANNF = np.sqrt(PERIODS_PER_YEAR)


def cash_per_period(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([CASH[y] for y in index.year]) / PERIODS_PER_YEAR


def excess_stats(df: pd.DataFrame) -> dict:
    ex = df["net"] - df["gross"] * cash_per_period(df.index)
    yrs = len(ex) / PERIODS_PER_YEAR
    m = ex.groupby(ex.index.to_period("M")).sum()
    eq = (1 + df["net"]).cumprod()
    return dict(
        excess_sharpe=float(ex.mean() / ex.std() * ANNF) if ex.std() > 0 else 0.0,
        excess_ann=float(ex.sum() / yrs),
        excess_sharpe_monthly=float(m.mean() / m.std() * np.sqrt(12)) if len(m) >= 12 else None,
        net_ann=float(df["net"].sum() / yrs),
        carry_ann=float(df["carry"].sum() / yrs),
        cost_ann=float(df["cost"].sum() / yrs),
        vol=float(ex.std() * ANNF),
        maxdd=float((eq / eq.cummax() - 1).min()),
        gross=float(df["gross"].mean()),
        net_sharpe=stats(df)["sharpe"],
    ), ex


def validate(panels) -> bool:
    grid = pd.read_csv(ROOT / "docs" / "improvement_results" / "grid.csv")
    ok = True
    for venue, pan in panels.items():
        for cfg in (Config("CC", 21, 7), Config("XS0", 1, 1)):
            ref = grid[(grid.venue == venue) & (grid.config == cfg.name)]["sharpe"].iloc[0]
            mine = stats(simulate(pan, cfg))["sharpe"]
            good = abs(mine - ref) <= 0.01
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] {venue:<8} {cfg.name:<12} {mine:.4f} vs {ref:.4f}")
    return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {"binance": load_venue("binance"), "bybit": load_venue("bybit")}
    if not validate(panels):
        sys.exit("R1 validation failed; study not run")

    rows, years, curves = [], {}, {}
    for venue, pan in panels.items():
        for tau in TAUS:
            df = simulate(pan, Config("CC", 21, 7, tau))
            st, ex = excess_stats(df)
            rows.append(dict(venue=venue, tau=tau, **st))
            g = ex.groupby(ex.index.year)
            years[(venue, tau)] = (g.sum() * PERIODS_PER_YEAR / g.size()).to_dict()
            curves[(venue, tau)] = ex
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid.csv", index=False)

    piv = grid.pivot(index="tau", columns="venue", values="excess_sharpe")
    piv["maximin"] = piv[["binance", "bybit"]].min(axis=1)
    best = piv["maximin"].max()
    cand = float(min(t for t in TAUS if np.isclose(piv.loc[t, "maximin"], best)))   # R2
    r3 = bool(cand != 0.0 and (piv.loc[cand, "binance"] > piv.loc[0.0, "binance"])
              and (piv.loc[cand, "bybit"] > piv.loc[0.0, "bybit"]))

    live, s_idx = load_live()                                                          # R4
    hold_rows, hold_curves = [], {}
    for tau in TAUS:
        df = simulate(live, Config("CC", 21, 7, tau), start=s_idx)
        st, ex = excess_stats(df)
        hold_rows.append(dict(tau=tau, **st))
        hold_curves[tau] = ex
    hold = pd.DataFrame(hold_rows)
    hold.to_csv(OUT / "holdout.csv", index=False)

    summary = dict(
        candidate_tau=cand, R3_beats_control_on_both=r3,
        excess_sharpe={f"{t:.2f}": {v: float(piv.loc[t, v]) for v in ("binance", "bybit")}
                       for t in TAUS},
        by_year={f"{v} {t:.2f}": {int(k): float(x) for k, x in d.items()}
                 for (v, t), d in years.items()},
        decision=("deploy as extra shadow ledger (R5)" if r3 else
                  "no robust improvement; live configuration unchanged"),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(grid.round(4).to_string())
    print(hold.round(4).to_string())
    print(json.dumps({k: summary[k] for k in ("candidate_tau", "R3_beats_control_on_both",
                                               "decision")}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {0.0: "#7E8CA8", 0.05: "#4C86FF", 0.10: "#2BA84A", 0.15: "#E39B2D"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (name, getter) in zip(axes, (("Binance 2021-25", lambda t: curves[("binance", t)]),
                                         ("Bybit 2022-25", lambda t: curves[("bybit", t)]),
                                         ("Holdout 2026 (Bybit)", lambda t: hold_curves[t]))):
        for t in TAUS:
            ax.plot(getter(t).cumsum() * 100, color=cols[t], lw=1.3, label=f"tau {t:.0%}")
        ax.axhline(0, color="#7E8CA8", lw=0.8, ls=":")
        ax.set_title(name); ax.grid(alpha=0.2)
    axes[0].set_ylabel("cumulative excess over cash (%)"); axes[0].legend(fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "excess_curves.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
