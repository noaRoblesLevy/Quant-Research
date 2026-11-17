"""
Post-hoc robustness checks for the cash-and-carry candidate (CC L=21 K=7) of the
2026-11-07 improvement study. NOT part of the pre-registered protocol: these checks do
not change the selection, they test how much the selected result can be trusted.

    python scripts/eval_cc_robustness.py   -> docs/improvement_results/cc_robustness.json
                                            -> docs/improvement_results/cc_by_year.png

Checks
    1. carry, costs and net return per calendar year (is there a funding regime?)
    2. Sharpe on monthly instead of 8h returns (funding is autocorrelated, so the
       sqrt(1095) annualisation of 8h returns overstates the Sharpe)
    3. excess return over a cash yield (approximate 3-month US T-bill annual averages;
       CC has 1-2% volatility, so a few percent of cash yield matters a lot)
    4. cost sensitivity: 0, 8, 16 and 24 bps per leg turnover
    5. capital: returns if spot cannot serve as collateral (about 2x capital)
    6. exposure: share of the 11 slots that were filled (positive funding)
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

import quantresearch.carry_lab as lab
from quantresearch.carry_lab import Config, simulate, stats, PERIODS_PER_YEAR
from eval_improvements import load_venue, load_live, OUT

CAND = Config("CC", 21, 7)
# Approximate annual averages of the 3-month US T-bill yield (rounded). Used only to
# show the order of magnitude of the cash opportunity cost; 2026 is not assumed.
TBILL = {2021: 0.0005, 2022: 0.020, 2023: 0.051, 2024: 0.050, 2025: 0.042}


def by_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.year)
    n = g.size()
    out = pd.DataFrame({
        "periods": n,
        "carry": g["carry"].sum() * PERIODS_PER_YEAR / n,
        "price": g["price"].sum() * PERIODS_PER_YEAR / n,
        "cost": g["cost"].sum() * PERIODS_PER_YEAR / n,
        "net": g["net"].sum() * PERIODS_PER_YEAR / n,
    })
    out["tbill"] = [TBILL.get(y, np.nan) for y in out.index]
    out["excess"] = out["net"] - out["tbill"]
    return out


def monthly_sharpe(df: pd.DataFrame) -> float:
    m = df["net"].groupby(df.index.to_period("M")).sum()
    return float(m.mean() / m.std() * np.sqrt(12))


def exposure(panel, cfg: Config, start: int) -> float:
    """Average share of the 11 short slots filled over the evaluation period."""
    sig = panel.F.rolling(cfg.L, min_periods=cfg.L).mean().values
    spot = ~np.isnan(panel.Rs.values)
    filled = []
    for i in range(start, len(sig), lab.HOLD // cfg.K):
        s = np.where(spot[i] & ~np.isnan(sig[i]), sig[i], -np.inf)
        top = np.sort(s)[::-1][:lab.N_EACH]
        filled.append(float((top > 0).mean()))
    return float(np.mean(filled))


def main():
    res, years = {}, {}
    live, s_idx = load_live()
    panels = {"binance": (load_venue("binance"), lab.WARMUP),
              "bybit": (load_venue("bybit"), lab.WARMUP),
              "holdout 2026": (live, s_idx)}
    for name, (pan, start) in panels.items():
        df = simulate(pan, CAND, start=start)
        st = stats(df)
        yr = by_year(df)
        years[name] = yr
        tb = (yr["tbill"] * yr["periods"]).sum() / yr["periods"].sum() if yr["tbill"].notna().all() \
            else np.nan
        costs = {}
        base_cost = lab.COST_BPS
        for bps in (0.0, 8.0, 16.0, 24.0):
            lab.COST_BPS = bps
            costs[f"{bps:.0f}bps"] = stats(simulate(pan, CAND, start=start))["ann_ret"]
        lab.COST_BPS = base_cost
        res[name] = dict(
            sharpe_8h=st["sharpe"], sharpe_monthly=monthly_sharpe(df) if len(df) > 400 else None,
            ann_ret=st["ann_ret"], vol=st["vol"], maxdd=st["maxdd"],
            carry_ann=st["carry_ann"], cost_ann=st["cost_ann"],
            avg_tbill=None if np.isnan(tb) else float(tb),
            excess_ann=None if np.isnan(tb) else float(st["ann_ret"] - tb),
            excess_sharpe_8h=None if np.isnan(tb) else float((st["ann_ret"] - tb) / st["vol"]),
            ann_ret_by_cost=costs,
            ann_ret_separate_accounts=st["ann_ret"] / 2,
            exposure=exposure(pan, CAND, start),
            by_year={int(k): {c: float(v) for c, v in row.items()} for k, row in yr.iterrows()},
        )
        print(f"\n=== {name} ===")
        print(f"  Sharpe 8h {st['sharpe']:.2f} | monthly "
              f"{res[name]['sharpe_monthly'] if res[name]['sharpe_monthly'] is None else round(res[name]['sharpe_monthly'],2)}"
              f" | return {st['ann_ret']*100:.1f}%/yr, vol {st['vol']*100:.1f}%, maxDD {st['maxdd']*100:.1f}%")
        if res[name]["excess_ann"] is not None:
            print(f"  avg T-bill {tb*100:.1f}% -> excess {res[name]['excess_ann']*100:.1f}%/yr, "
                  f"excess Sharpe {res[name]['excess_sharpe_8h']:.2f}")
        print("  return by cost: " + ", ".join(f"{k} {v*100:+.1f}%" for k, v in costs.items()))
        print(f"  slots filled: {res[name]['exposure']*100:.0f}%")
        print((yr[["carry", "cost", "net", "tbill", "excess"]] * 100).round(1).to_string())

    (OUT / "cc_robustness.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.27
    for k, (name, col) in enumerate((("binance", "#4C86FF"), ("bybit", "#E39B2D"),
                                     ("holdout 2026", "#D6455D"))):
        yr = years[name]
        ax.bar(yr.index + (k - 1) * width, yr["carry"] * 100, width, color=col, label=f"{name}: carry")
        ax.scatter(yr.index + (k - 1) * width, yr["net"] * 100, color="black", s=12, zorder=3)
    tb = pd.Series(TBILL)
    ax.plot(tb.index, tb * 100, color="#7E8CA8", ls="--", lw=1.2, label="approx. T-bill yield")
    ax.axhline(0, color="#7E8CA8", lw=0.8)
    ax.set_ylabel("% per year"); ax.set_title("Cash-and-carry (CC L=21 K=7): carry (bars) and net (dots) by year")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "cc_by_year.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
