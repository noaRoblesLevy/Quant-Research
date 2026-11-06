"""
Build every number and figure used in the funding-carry paper.

    python paper/make_assets.py

Outputs (all in paper/):
    numbers.tex      LaTeX macros, one per number quoted in the paper
    results.json     the same numbers, for the web version
    figures/*.pdf    figures for the LaTeX paper
    figures/*.svg    the same figures for the web (transparent, theme-neutral colours)

Inputs: cached Binance and Bybit panels (data_cache/), the live shadow ledger
(data_cache/forward_ledger.csv + forward_state.json) and live-period Bybit hourly
closes for BTC/ETH (data_cache/*_bybit_60_live.parquet).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_hedge_variants import run_all, run_book, load_venue, LABELS, VARIANTS
from matplotlib.ticker import FuncFormatter, LogLocator

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
DATA = ROOT / "data_cache"

AZURE, COBALT, AMBER, GREY = "#4C86FF", "#1F4FE0", "#E39B2D", "#7E8CA8"
N = {}          # macro name -> formatted string


# \ensuremath works both in text and inside $...$, unlike a bare $-$
MINUS, PLUS = r"\ensuremath{-}", r"\ensuremath{+}"


def pct(x, d=1, sign=False):
    s = f"{x*100:+.{d}f}" if sign else f"{x*100:.{d}f}"
    return s.replace("-", MINUS).replace("+", PLUS)


def num(x, d=2):
    return f"{x:.{d}f}".replace("-", MINUS)


# ------------------------------------------------------------------ figure style
def style(web: bool):
    ink = GREY if web else "#222222"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.edgecolor": ink, "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": ink, "ytick.color": ink,
        "axes.grid": True, "grid.color": GREY, "grid.alpha": 0.18,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "figure.facecolor": "none" if web else "white",
        "axes.facecolor": "none" if web else "white",
        "savefig.transparent": web,
    })


def save(fig_fn, name):
    for web in (False, True):
        style(web)
        fig = fig_fn()
        fig.savefig(FIG / f"{name}.{'svg' if web else 'pdf'}", bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ backtests
results, books = {}, {}
for venue, tag in (("binance", "Bn"), ("bybit", "By")):
    P, bk, res = run_all(venue)
    books[venue], results[venue] = bk, res
    first = bk["baseline"].index[0]
    N[f"{tag}Start"] = first.strftime("%B %Y")
    N[f"{tag}End"] = bk["baseline"].index[-1].strftime("%B %Y")
    N[f"{tag}Periods"] = f"{len(bk['baseline']):,}".replace(",", "{,}")
    N[f"{tag}Years"] = num(res["baseline"]["years"], 1)
    for k, s in res.items():
        key = "Base" if k == "baseline" else k
        N[f"{tag}{key}Sharpe"] = num(s["sharpe"])
        N[f"{tag}{key}Cagr"] = pct(s["cagr"])
        N[f"{tag}{key}Carry"] = pct(s["carry_ann"])
        N[f"{tag}{key}Price"] = pct(s["price_ann"])
        N[f"{tag}{key}Pvol"] = pct(s["price_vol"], 0)
        N[f"{tag}{key}MaxDD"] = pct(s["maxdd"])
        N[f"{tag}{key}Corr"] = num(s["corr_price_btc"])
    b = res["baseline"]
    N[f"{tag}CarrySharpe"] = num(b["carry_sharpe"], 1)
    N[f"{tag}CarryVol"] = pct(b["carry_vol"], 2)
    N[f"{tag}CarryLong"] = pct(b["carry_long_ann"], 1, sign=True)
    N[f"{tag}CarryShort"] = pct(b["carry_short_ann"], 1, sign=True)
    N[f"{tag}CarryShareVar"] = pct(books[venue]["baseline"]["carry"].var()
                                    / books[venue]["baseline"]["net"].var(), 2)

    # concentration: best 90-day window of the price leg, and which coins drove it
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Pv, Fv, Lv = load_venue(venue)
        bk_w, W = run_book(Pv, Fv, Lv, **VARIANTS["baseline"], return_weights=True)
    win = 270                                           # 90 days of 8h periods
    roll = bk_w["price"].rolling(win).sum()
    end_i = int(np.nanargmax(roll.values))
    sl = slice(end_i - win + 1, end_i + 1)
    best = float(roll.iloc[end_i])
    tot = float(bk_w["price"].sum())
    contrib = (W.iloc[sl] * Lv.reindex(W.index).iloc[sl].fillna(0.0)).sum().sort_values()
    N[f"{tag}BestWinStart"] = bk_w.index[sl.start].strftime("%B %Y")
    N[f"{tag}BestWinEnd"] = bk_w.index[end_i].strftime("%B %Y")
    N[f"{tag}BestWinPrice"] = pct(best, 0)
    N[f"{tag}TotalPrice"] = pct(tot, 0)
    N[f"{tag}BestWinShare"] = pct(best / tot, 0)
    top = contrib.tail(2)[::-1]
    N[f"{tag}TopOneName"] = top.index[0].replace("USDT", "")
    N[f"{tag}TopOneContrib"] = pct(top.iloc[0], 0, sign=True)
    N[f"{tag}TopTwoName"] = top.index[1].replace("USDT", "")
    N[f"{tag}TopTwoContrib"] = pct(top.iloc[1], 0, sign=True)
    N[f"{tag}TopOneWeight"] = num(W.iloc[sl][top.index[0]].mean(), 3)
    # how ordinary is the live outcome? all 153-day windows of the backtest
    w153 = 459                                          # 153 days of 8h periods
    roll_net = np.log1p(bk_w["net"]).rolling(w153).sum().dropna()
    roll_px = bk_w["price"].rolling(w153).sum().dropna()
    N[f"{tag}WinNegNet"] = pct((roll_net < 0).mean(), 0)
    N[f"{tag}WinNegPrice"] = pct((roll_px < 0).mean(), 0)
    N[f"{tag}WinWorseLive"] = pct((np.expm1(roll_net) <= -0.153).mean(), 0)
    print(f"[{venue}] 153-day windows: net<0 {N[tag+'WinNegNet']}%, price<0 "
          f"{N[tag+'WinNegPrice']}%, net <= -15.3% {N[tag+'WinWorseLive']}%")
    print(f"[{venue}] best 90d window {N[tag+'BestWinStart']} - {N[tag+'BestWinEnd']}: "
          f"price {best*100:.0f}% of total {tot*100:.0f}%; top contributors:\n"
          f"{(contrib.tail(4)[::-1]*100).round(1).to_string()}\n"
          f"worst contributors:\n{(contrib.head(3)*100).round(1).to_string()}")

# ------------------------------------------------------------------ live ledger
led = pd.read_csv(DATA / "forward_ledger.csv", parse_dates=["time"])
st = json.loads((DATA / "forward_state.json").read_text())
start = pd.Timestamp(st["start"])
led["time"] = pd.to_datetime(led["time"], utc=True)
t0 = start if start.tzinfo else start.tz_localize("UTC")
days = (led["time"].iloc[-1] - t0).total_seconds() / 86400
yrs = days / 365.0
eq = led["equity"]
total = eq.iloc[-1] / 1e5 - 1

# time-normalised estimates (steps are irregular: minutes to 3 weeks apart)
lr = np.log(eq / eq.shift(1).fillna(1e5))
ann_ret = lr.sum() / yrs
ann_vol = np.sqrt((lr ** 2).sum() / yrs)
live_sr = ann_ret / ann_vol
price_vol_live = np.sqrt((led["step_price"] ** 2).sum() / yrs)
carry_ann_live = st["cum_carry"] / yrs

# Sharpe standard error (Lo 2002, iid): se = sqrt((1 + SR^2/2) / T)
exp_sr = results["bybit"]["baseline"]["sharpe"]
se = np.sqrt((1 + exp_sr ** 2 / 2) / yrs)
z = (live_sr - exp_sr) / se

# gaps
dt_h = led["time"].diff().dt.total_seconds() / 3600
gaps = int((dt_h > 24 * 7).sum())
longest_gap = dt_h.max() / 24

sp = led["step_price"]
N.update({
    "LiveStart": t0.strftime("%-d %B %Y") if sys.platform != "win32" else t0.strftime("%#d %B %Y"),
    "LiveEnd": led["time"].iloc[-1].strftime("%#d %B %Y" if sys.platform == "win32" else "%-d %B %Y"),
    "LiveDays": f"{days:.0f}",
    "LiveSteps": f"{len(led)}",
    "LiveTotal": pct(total, 1),
    "LiveLoss": pct(-total, 1),                 # positive, for "lost X%" phrasing
    "LiveEquity": f"{eq.iloc[-1]:,.0f}".replace(",", "{,}"),
    "LiveCarry": pct(st["cum_carry"], 2, sign=True),
    "LivePrice": pct(st["cum_price"], 2, sign=True),
    "LiveAnnRet": pct(ann_ret, 1),
    "LiveAnnVol": pct(ann_vol, 0),
    "LiveSharpe": num(live_sr),
    "LiveCarryAnn": pct(carry_ann_live, 1),
    "LivePriceVol": pct(price_vol_live, 0),
    "LiveStepPriceStd": pct(sp.std(), 2),
    "LiveStepCarryStd": pct(led["step_carry"].std(), 3),
    "LiveStepRatio": f"{sp.std() / led['step_carry'].std():.0f}",
    "LiveWorstStep": pct(sp.min(), 2),
    "LiveSecondWorst": pct(sp.nsmallest(2).iloc[-1], 2),
    "LiveBestStep": pct(sp.max(), 2),
    "LiveBigSteps": f"{int((sp.abs() > 0.04).sum())}",
    "LiveSE": num(se),
    "LiveZ": num(z),
    "LiveGaps": f"{gaps}",
    "LiveLongestGap": f"{longest_gap:.0f}",
})

# ------------------------------------------------------------------ live factor regression
def live_close(sym):
    s = pd.read_parquet(DATA / f"{sym}_bybit_60_live.parquet")["close"]
    s.index = s.index + pd.Timedelta(hours=1)          # open time -> close time
    s.index = s.index.tz_localize("UTC")
    return s


btc_px, eth_px = live_close("BTCUSDT"), live_close("ETHUSDT")
t_prev = pd.Series([t0] + list(led["time"].iloc[:-1]), index=led.index)


def interval_ret(px, a, b):
    pa, pb = px.asof(a), px.asof(b)
    return float(np.log(pb / pa)) if pa and pb else np.nan


led["r_btc"] = [interval_ret(btc_px, a, b) for a, b in zip(t_prev, led["time"])]
led["r_eth"] = [interval_ret(eth_px, a, b) for a, b in zip(t_prev, led["time"])]
reg = led.dropna(subset=["r_btc", "r_eth"])
reg = reg[(t_prev.loc[reg.index] - t0).dt.total_seconds() >= 0]


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    return beta, r2


b1, r2_btc = ols(reg["step_price"].values, [reg["r_btc"].values])
b2, r2_both = ols(reg["step_price"].values, [reg["r_btc"].values, reg["r_eth"].values])
worst = reg.nsmallest(2, "step_price")
N.update({
    "LiveBetaBtc": num(b1[1]),
    "LiveRsqBtc": pct(r2_btc, 0),
    "LiveRsqBoth": pct(r2_both, 0),
    "LiveWorstBtc": pct(worst["r_btc"].iloc[0], 1, sign=True),
    "LiveSecondBtc": pct(worst["r_btc"].iloc[1], 1, sign=True),
    "LiveWorstDate": worst["time"].iloc[0].strftime("%#d %B" if sys.platform == "win32" else "%-d %B"),
    "LiveSecondDate": worst["time"].iloc[1].strftime("%#d %B" if sys.platform == "win32" else "%-d %B"),
})

# ------------------------------------------------------------------ figures
def fig_live():
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 1]))
    ax[0].plot(led["time"], led["equity"] / 1e3, color=AZURE, lw=1.6)
    ax[0].axhline(100, color=GREY, ls=":", lw=1)
    ax[0].set_ylabel("equity ($k)")
    ax[1].plot(led["time"], led["cum_carry"] * 100, color=AZURE, lw=1.6, label="cumulative carry")
    ax[1].plot(led["time"], led["cum_price"] * 100, color=AMBER, lw=1.6, label="cumulative price")
    ax[1].axhline(0, color=GREY, ls=":", lw=1)
    ax[1].set_ylabel("% of start equity")
    ax[1].legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_legs():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for a, venue, title in ((ax[0], "binance", "Binance"), (ax[1], "bybit", "Bybit")):
        bk = books[venue]["baseline"]
        a.plot(bk.index, bk["carry"].cumsum() * 100, color=AZURE, lw=1.5, label="carry")
        a.plot(bk.index, bk["price"].cumsum() * 100, color=AMBER, lw=1.1, label="price")
        a.axhline(0, color=GREY, ls=":", lw=1)
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("cumulative P&L (% of equity)")
    ax[0].legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_scatter():
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(reg["r_btc"] * 100, reg["step_price"] * 100, s=14, color=AZURE, alpha=0.75,
               edgecolor="none")
    xs = np.linspace(reg["r_btc"].min(), reg["r_btc"].max(), 50)
    ax.plot(xs * 100, (b1[0] + b1[1] * xs) * 100, color=AMBER, lw=1.4,
            label=f"OLS fit, $R^2$ = {r2_btc*100:.0f}%")
    ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("BTC log return over the step (%)")
    ax.set_ylabel("book price P&L over the step (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_variants():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, venue, title in ((ax[0], "binance", "Binance"), (ax[1], "bybit", "Bybit")):
        for k, col, lw in (("baseline", GREY, 1.3), ("C", AZURE, 1.6)):
            e = (1 + books[venue][k]["net"]).cumprod()
            a.plot(e.index, e, color=col, lw=lw, label=LABELS[k].split(":")[0]
                   if k == "C" else "Baseline")
        a.set_yscale("log")
        a.yaxis.set_major_locator(LogLocator(base=10, subs=(0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0)))
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("growth of 1 (log scale)")
    ax[0].legend(loc="upper left", labels=["Baseline", "Variant C"])
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


save(fig_live, "live_ledger")
save(fig_legs, "backtest_legs")
save(fig_scatter, "live_btc_scatter")
save(fig_variants, "variants_equity")

# ------------------------------------------------------------------ write numbers
with open(OUT / "numbers.tex", "w", encoding="utf-8") as f:
    f.write("% generated by paper/make_assets.py, do not edit by hand\n")
    for k, v in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

raw = dict(results={v: {k: {m: float(x) for m, x in s.items()} for k, s in r.items()}
                    for v, r in results.items()},
           live=dict(days=days, steps=len(led), total=total, ann_ret=ann_ret, ann_vol=ann_vol,
                     sharpe=live_sr, carry_ann=carry_ann_live, cum_carry=st["cum_carry"],
                     cum_price=st["cum_price"], price_vol=price_vol_live, se=se, z=z,
                     beta_btc=float(b1[1]), r2_btc=r2_btc, r2_btc_eth=r2_both,
                     worst=[(str(t.date()), float(p), float(r)) for t, p, r in
                            zip(worst["time"], worst["step_price"], worst["r_btc"])],
                     gaps_over_week=gaps, longest_gap_days=longest_gap))
(OUT / "results.json").write_text(json.dumps(raw, indent=2))

print(f"{len(N)} macros -> {OUT/'numbers.tex'}")
for k in ("LiveSharpe", "LiveSE", "LiveZ", "LiveRsqBtc", "LiveRsqBoth", "LiveBetaBtc",
          "LiveCarryAnn", "LivePriceVol", "LiveWorstBtc", "LiveSecondBtc", "LiveGaps",
          "LiveLongestGap", "BnCarryShareVar", "ByCarryShareVar"):
    print(f"  {k:<18} {N[k]}")
