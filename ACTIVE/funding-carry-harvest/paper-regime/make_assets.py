"""
Build every number and figure used in the paper "Carry Needs a Regime".

    python paper-regime/make_assets.py

Outputs (all in paper-regime/):
    numbers.tex      LaTeX macros, one per number quoted in the paper
    results.json     the same numbers, for the web version
    figures/*.pdf    figures for the LaTeX paper
    figures/*.svg    the same figures for the web (transparent, theme-neutral colours)

Inputs: the result files of the two pre-registered studies (docs/improvement_results/,
docs/regime_filter_results/), the cached Binance and Bybit panels, and the live shadow
ledgers in data_cache/, frozen at LIVE_END.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
DATA = ROOT / "data_cache"
IMP = ROOT / "docs" / "improvement_results"
REG = ROOT / "docs" / "regime_filter_results"
LIVE_END = "2026-11-17T16:00:00+00:00"          # last step of the frozen ledgers

AZURE, COBALT, AMBER, GREY, GREEN, RED = "#4C86FF", "#1F4FE0", "#E39B2D", "#7E8CA8", "#2BA84A", "#D6455D"
N: dict[str, str] = {}
YEARWORD = {2021: "TwentyOne", 2022: "TwentyTwo", 2023: "TwentyThree", 2024: "TwentyFour",
            2025: "TwentyFive", 2026: "TwentySix"}
MINUS, PLUS = r"\ensuremath{-}", r"\ensuremath{+}"


def num(x, d=2, sign=False):
    x = round(float(x), d) + 0.0          # + 0.0 turns -0.0 into 0.0
    s = f"{x:+.{d}f}" if sign and x != 0 else f"{x:.{d}f}"
    return s.replace("-", MINUS).replace("+", PLUS)


def pct(x, d=1, sign=False):
    return num(x * 100, d, sign)


def style(web: bool):
    ink = GREY if web else "#222222"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.edgecolor": ink, "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": ink, "ytick.color": ink, "axes.grid": True, "grid.color": GREY,
        "grid.alpha": 0.18, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none",
        "figure.facecolor": "none" if web else "white", "axes.facecolor": "none" if web else "white",
        "savefig.transparent": web,
    })


def save(fig_fn, name):
    for web in (False, True):
        style(web)
        fig = fig_fn()
        fig.savefig(FIG / f"{name}.{'svg' if web else 'pdf'}", bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ study 1 (2026-11-07)
grid = pd.read_csv(IMP / "grid.csv")
summ = json.loads((IMP / "summary.json").read_text())
hold = pd.read_csv(IMP / "holdout.csv").set_index("label")
rob = json.loads((IMP / "cc_robustness.json").read_text())
sched = pd.read_csv(IMP / "schedule.csv")
piv = grid.pivot_table(index="config", columns="venue", values="sharpe")
piv["maximin"] = piv.min(axis=1)

for lbl, key in (("XS0 L=1 K=1", "Base"), ("CC L=21 K=7", "Cand"), ("XSC L=9 K=7", "BestXs")):
    for venue, tag in (("binance", "Bn"), ("bybit", "By")):
        row = grid[(grid.venue == venue) & (grid.config == lbl)].iloc[0]
        N[f"{tag}{key}Sharpe"] = num(row.sharpe)
        N[f"{tag}{key}Ret"] = pct(row.ann_ret)
        N[f"{tag}{key}Vol"] = pct(row.vol)
        N[f"{tag}{key}DD"] = pct(row.maxdd)
        N[f"{tag}{key}Carry"] = pct(row.carry_ann)
        N[f"{tag}{key}Cost"] = pct(row.cost_ann)

names = {"signal 3d mean (L=9)": "LNine", "signal 7d mean (L=21)": "LTwentyOne",
         "7 tranches (K=7)": "KSeven", "inverse vol + BTC/ETH (XSC)": "Xsc",
         "correlation pairs (PAIR)": "Pair", "cash-and-carry (CC)": "Cc"}
for lbl, key in names.items():
    d = summ["R1_separate_effects"][lbl]
    N[f"ROne{key}Bn"] = num(d["binance"], sign=True)
    N[f"ROne{key}By"] = num(d["bybit"], sign=True)
N["Dsr"] = num(summ["R5_deflated_sharpe"]["candidate"]["dsr"], 3)
N["DsrNull"] = num(summ["R5_deflated_sharpe"]["candidate"]["sr0_ann"], 1)
N["NBestCross"] = str(int((piv.drop(index=[i for i in piv.index if i.startswith("CC")])
                           ["maximin"] > piv.loc["XS0 L=1 K=1", "maximin"]).sum()))

for lbl, key in (("baseline", "Base"), ("candidate", "Cand"), ("best XS0", "BestXsZero"),
                 ("best XSC", "BestXsc"), ("best PAIR", "BestPair")):
    r = hold.loc[lbl]
    N[f"Hold{key}Total"] = pct(r.total, 1, sign=True)
    N[f"Hold{key}Sharpe"] = num(r.sharpe)
    N[f"Hold{key}Carry"] = pct(r.carry_ann)
    N[f"Hold{key}Cost"] = pct(r.cost_ann)
hs = summ["holdout_schedules"]["baseline"]
N["HoldSchedLo"] = pct(hs["lo"], 1, sign=True)
N["HoldSchedHi"] = pct(hs["hi"], 1, sign=True)
N["HoldSchedMed"] = pct(float(np.median(hs["totals"])), 1, sign=True)
N["HoldSchedNeg"] = str(int(sum(t < 0 for t in hs["totals"])))

for venue, tag in (("binance", "Bn"), ("bybit", "By")):
    s = sched[sched.venue == venue].set_index("config")["std"]
    N[f"{tag}SchedBase"] = num(s["XS0 L=1 K=1"] * 100, 0)
    N[f"{tag}SchedXsTwentyOne"] = num(s["XS0 L=21 K=1"] * 100, 0)
    N[f"{tag}SchedCc"] = num(s["CC L=21 K=1"] * 100, 1)

for name, tag in (("binance", "Bn"), ("bybit", "By"), ("holdout 2026", "Ho")):
    r = rob[name]
    N[f"{tag}CcMonthly"] = num(r["sharpe_monthly"]) if r["sharpe_monthly"] is not None else "n/a"
    N[f"{tag}CcRet"] = pct(r["ann_ret"])
    N[f"{tag}CcSep"] = pct(r["ann_ret_separate_accounts"])
    for y, d in r["by_year"].items():
        yw = YEARWORD[int(y)]                      # LaTeX macro names cannot hold digits
        N[f"{tag}CcCarry{yw}"] = pct(d["carry"])
        N[f"{tag}CcNet{yw}"] = pct(d["net"], 1)
    for k, v in r["ann_ret_by_cost"].items():
        N[f"{tag}CcCost{k.replace('bps', '')}".replace("0", "Zero").replace("8", "Eight")
          .replace("16", "Sixteen").replace("24", "TwentyFour")] = pct(v)
    if r["excess_ann"] is not None:
        N[f"{tag}CcExcess"] = pct(r["excess_ann"])
        N[f"{tag}CcTbill"] = pct(r["avg_tbill"])

# ------------------------------------------------------------------ live ledgers
def ledger(path):
    d = pd.read_csv(DATA / path)
    d["time"] = pd.to_datetime(d["time"], format="ISO8601", utc=True)
    return d[d["time"] <= pd.Timestamp(LIVE_END)].reset_index(drop=True)


def live_stats(d, start, cost_col=None):
    days = (d["time"].iloc[-1] - pd.Timestamp(start)).total_seconds() / 86400
    step = d["step_price"] + d["step_carry"] - (d[cost_col] if cost_col else 0)
    sh = step.mean() / step.std() * np.sqrt(len(d) / days * 365)
    return dict(days=days, total=d["equity"].iloc[-1] / 1e5 - 1, carry=d["cum_carry"].iloc[-1],
                price=d["cum_price"].iloc[-1], carry_ann=d["cum_carry"].iloc[-1] / days * 365,
                sharpe=sh)


base = ledger("forward_ledger.csv"); varc = ledger("forward_ledger_c.csv"); cc = ledger("forward_ledger_cc.csv")
st_b = json.loads((DATA / "forward_state.json").read_text())
st_c = json.loads((DATA / "forward_state_c.json").read_text())
st_cc = json.loads((DATA / "forward_state_cc.json").read_text())
lb = live_stats(base, st_b["start"]); lc = live_stats(varc, st_c["start"])
lcc = live_stats(cc, st_cc["start"], "step_cost")
N["LiveEnd"] = "17 November 2026"
N["LiveDays"] = f"{lb['days']:.0f}"
N["LiveTotal"] = pct(lb["total"], 1, sign=True)
N["LiveCarry"] = pct(lb["carry"], 1, sign=True)
N["LivePrice"] = pct(lb["price"], 1, sign=True)
N["LiveCarryAnn"] = pct(lb["carry_ann"])
N["LiveSharpe"] = num(lb["sharpe"])
# the first paper's window ended at the 6 November rebalance (equity 84,700.33)
e6 = float(base.loc[base["time"] <= pd.Timestamp("2026-11-06T14:00:00+00:00"), "equity"].iloc[-1])
N["LiveSinceFirst"] = pct(base["equity"].iloc[-1] / e6 - 1, 1, sign=True)
N["LiveSinceFirstDays"] = f"{(base['time'].iloc[-1] - pd.Timestamp('2026-11-06T13:16:46+00:00')).days}"
reg = base[base["time"] >= pd.Timestamp("2026-11-07T16:00:00+00:00")]
N["LiveStepVol"] = pct(reg["step_price"].std(), 2)

T = lb["days"] / 365
sr_bt = 1.24                                          # Bybit backtest, as in the first paper
se = np.sqrt((1 + sr_bt ** 2 / 2) / T)
N["LiveSe"] = num(se)
N["LiveZ"] = num((lb["sharpe"] - sr_bt) / se)
N["LiveZNaive"] = num((lb["sharpe"] - sr_bt) * np.sqrt(T))
N["LiveNetCarry"] = pct(lb["carry_ann"] - hold.loc["baseline"].cost_ann, 1, sign=True)

N["CDays"] = f"{lc['days']:.0f}"
N["CTotal"] = pct(lc["total"], 1, sign=True)
m = base.merge(varc, on="time", suffixes=("_b", "_c"))
rb = m["step_price_b"] + m["step_carry_b"]; rc = m["step_price_c"] + m["step_carry_c"]
N["CCorr"] = num(rb.corr(rc))
N["CVol"] = pct(m["step_price_c"].std(), 2)
N["CBaseVol"] = pct(m["step_price_b"].std(), 2)
N["BaseSameWindow"] = pct(m["equity_b"].iloc[-1] / m["equity_b"].iloc[0] - 1, 1, sign=True)

N["CcDays"] = f"{lcc['days']:.0f}"
N["CcLiveTotal"] = pct(lcc["total"], 2, sign=True)
N["CcLiveCarry"] = pct(cc["cum_carry"].iloc[-1], 3, sign=True)
N["CcLiveCost"] = pct(cc["cum_cost"].iloc[-1], 3)
N["CcLiveCarryAnn"] = pct(lcc["carry_ann"])

# how many 164-day backtest windows lost at least as much as the live ledger?
from eval_hedge_variants import run_all
win = int(round(lb["days"] * 3))
for venue, tag in (("binance", "Bn"), ("bybit", "By")):
    _, bk, _ = run_all(venue)
    roll = np.expm1(np.log1p(bk["baseline"]["net"]).rolling(win).sum().dropna())
    N[f"{tag}WinNeg"] = pct((roll < 0).mean(), 0)
    N[f"{tag}WinWorse"] = pct((roll <= lb["total"]).mean(), 1)

# ------------------------------------------------------------------ study 2 (2026-11-17)
rg = pd.read_csv(REG / "grid.csv")
rh = pd.read_csv(REG / "holdout.csv").set_index("tau")
rs = json.loads((REG / "summary.json").read_text())
TAUS = [0.0, 0.05, 0.10, 0.15]
TN = {0.0: "Zero", 0.05: "Five", 0.10: "Ten", 0.15: "Fifteen"}
for _, r in rg.iterrows():
    k = f"Tau{TN[r.tau]}{'Bn' if r.venue == 'binance' else 'By'}"
    N[k + "Sharpe"] = num(r.excess_sharpe)
    N[k + "Monthly"] = num(r.excess_sharpe_monthly)
    N[k + "Excess"] = pct(r.excess_ann)
    N[k + "Net"] = pct(r.net_ann)
    N[k + "Cost"] = pct(r.cost_ann)
    N[k + "Gross"] = pct(r.gross, 0)
for t in TAUS:
    r = rh.loc[t]
    k = f"HoTau{TN[t]}"
    N[k + "Excess"] = pct(r.excess_ann, 1, sign=True)
    N[k + "Net"] = pct(r.net_ann, 1, sign=True)
    N[k + "Cost"] = pct(r.cost_ann)
    N[k + "Gross"] = pct(r.gross, 0)
    for venue in ("binance", "bybit"):
        for y, v in rs["by_year"][f"{venue} {t:.2f}"].items():
            N[f"Yr{TN[t]}{'Bn' if venue == 'binance' else 'By'}{['A','B','C','D','E'][int(y)-2021]}"] = \
                pct(v, 1)

(OUT / "numbers.tex").write_text(
    "% generated by make_assets.py, do not edit\n"
    + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(N.items())) + "\n")
(OUT / "results.json").write_text(json.dumps(
    {k: v.replace(MINUS, "−").replace(PLUS, "+") for k, v in N.items()}, indent=1))
print(f"{len(N)} numbers written")


# ------------------------------------------------------------------ figures
def fig_grid():
    order = piv.sort_values("maximin").index
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(order))
    ax.barh(y - 0.2, piv.loc[order, "binance"], 0.4, color=AZURE, label="Binance 2021-25")
    ax.barh(y + 0.2, piv.loc[order, "bybit"], 0.4, color=AMBER, label="Bybit 2022-25")
    ax.axvline(piv.loc["XS0 L=1 K=1", "bybit"], color=GREY, ls=":", lw=1)
    ax.set_yticks(y, order, fontsize=8)
    ax.set_xlabel("Sharpe ratio, 8h returns, net of costs")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0)
    return fig


def fig_schedules():
    tot = np.array(hs["totals"]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    rng = np.random.default_rng(1)
    ax.scatter(tot, rng.uniform(-0.25, 0.25, len(tot)), color=AZURE, s=26, zorder=3,
               label="21 possible schedules, baseline")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.axvline(float(np.median(tot)), color=AMBER, lw=1.2, ls="--", label="median")
    ax.set_yticks([]); ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("total return over the holdout, 6 June to 6 November 2026 (%)")
    ax.legend(loc="upper left", fontsize=8, ncol=2); ax.grid(axis="y", alpha=0)
    return fig


def fig_cc_years():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.27
    tb = {2021: 0.0005, 2022: 0.020, 2023: 0.051, 2024: 0.050, 2025: 0.042, 2026: 0.037}
    for k, (name, col, lbl) in enumerate((("binance", AZURE, "Binance"), ("bybit", AMBER, "Bybit"),
                                          ("holdout 2026", RED, "Bybit holdout"))):
        yrs = {int(y): d for y, d in rob[name]["by_year"].items()}
        xs = np.array(list(yrs)) + (k - 1) * width
        ax.bar(xs, [d["carry"] * 100 for d in yrs.values()], width, color=col, label=f"{lbl}: carry")
        ax.scatter(xs, [d["net"] * 100 for d in yrs.values()], color="#222222" if not
                   plt.rcParams["savefig.transparent"] else GREY, s=12, zorder=3)
    ax.plot(list(tb), [v * 100 for v in tb.values()], color=GREY, ls="--", lw=1.2,
            label="cash (approx. T-bill)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("% per year"); ax.legend(fontsize=8, ncol=2); ax.grid(axis="x", alpha=0)
    return fig


def fig_live():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a = axes[0]
    a.plot(base["time"], base["equity"] / 1e3, color=AZURE, lw=1.4)
    a.axhline(100, color=GREY, lw=0.8, ls=":")
    a.axvline(pd.Timestamp("2026-11-06T13:16:46+00:00"), color=GREY, lw=0.8, ls="--")
    a.text(pd.Timestamp("2026-11-04", tz="UTC"), 101, "first paper", ha="right", fontsize=8)
    a.set_ylabel("equity (thousands)"); a.set_title("Baseline, 6 June to 17 November", fontsize=10)
    b = axes[1]
    t0 = pd.Timestamp("2026-11-07T16:00:00+00:00")
    for d, col, lbl in ((base, AZURE, "baseline"), (varc, AMBER, "variant C"), (cc, GREEN, "cash-and-carry")):
        s = d[d["time"] >= t0]
        b.plot(s["time"], (s["equity"] / s["equity"].iloc[0] - 1) * 100, color=col, lw=1.4, label=lbl)
    b.axhline(0, color=GREY, lw=0.8, ls=":")
    b.set_ylabel("% since 7 November"); b.set_title("All three ledgers, last ten days", fontsize=10)
    b.legend(fontsize=8)
    for x in axes:
        x.tick_params(axis="x", labelsize=8)
        for lab in x.get_xticklabels():
            lab.set_rotation(30)
    fig.tight_layout()
    return fig


def fig_regime():
    from quantresearch.carry_lab import Config, simulate
    from eval_improvements import load_venue, load_live
    from eval_regime_filter import excess_stats
    cols = {0.0: GREY, 0.05: AZURE, 0.10: GREEN, 0.15: AMBER}
    pans = {"Binance 2021-25": (load_venue("binance"), 105), "Bybit 2022-25": (load_venue("bybit"), 105)}
    live, s_idx = load_live(); pans["Bybit holdout 2026"] = (live, s_idx)
    curves = {n: {t: excess_stats(simulate(p, Config("CC", 21, 7, t), start=s))[1]
                  for t in TAUS} for n, (p, s) in pans.items()}

    def make():
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, (name, cv) in zip(axes, curves.items()):
            for t in TAUS:
                ax.plot(cv[t].index, cv[t].cumsum() * 100, color=cols[t], lw=1.3, label=f"hurdle {t:.0%}")
            ax.axhline(0, color=GREY, lw=0.8, ls=":")
            ax.set_title(name, fontsize=10); ax.tick_params(axis="x", labelsize=8)
            for lab in ax.get_xticklabels():
                lab.set_rotation(30)
        axes[0].set_ylabel("cumulative excess over cash (%)"); axes[0].legend(fontsize=8)
        fig.tight_layout()
        return fig
    return make


save(fig_grid, "sharpe_grid")
save(fig_schedules, "holdout_schedules")
save(fig_cc_years, "cc_by_year")
save(fig_live, "live_ledgers")
save(fig_regime(), "regime_excess")
print("figures written")
