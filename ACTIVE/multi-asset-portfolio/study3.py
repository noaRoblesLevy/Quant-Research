"""
Study 3: a portfolio of uncorrelated strategies, run as pre-registered in
docs/preregistration_study3_2026-11-07.md.

    python study3.py --checks    # engine checks only (no results are read)
    python study3.py             # checks, then the study -> results/study3/

Every sleeve is a daily EXCESS return over the T-bill rate on the SPY calendar (NaN
before it starts). A strategy earns the T-bill rate plus the vol-scaled sum of sleeve
excess returns, minus leverage financing and re-allocation costs.
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

CACHE = ROOT / "data_cache"
OUT = ROOT / "results" / "study3"
DAYS = 252
COST = 5e-4
SHORT_FEE = 0.005
LOOKBACKS = (21, 63, 126, 252)
PERIODS = {"selection": ("2007-07-01", "2016-12-31"),
           "confirmation": ("2017-01-01", "2022-12-31"),
           "holdout": ("2023-01-01", "2026-10-31")}
SLEEVES = ["EQ", "ETF_TF", "FX_CARRY", "FX_TF", "CRYPTO_TF", "CRYPTO_CC"]
COMBOS = ["EQRISK", "WF_SHARPE", "WF_TOP2", "EX_EQ"]


# ----------------------------------------------------------------- data
def load():
    px, rf = L.load_data()
    cal = px.index
    fx = pd.read_parquet(CACHE / "fx.parquet").reindex(cal).ffill()
    rates = pd.read_parquet(CACHE / "policy_rates.parquet")          # month-end index, %
    # accrual: the rate of the current month applies to each day of that month
    r_day = rates.copy(); r_day.index = r_day.index.to_period("M")
    rd = pd.DataFrame({c: [r_day[c].get(t.to_period("M"), np.nan) for t in cal] for c in rates.columns},
                      index=cal).ffill()
    fxx = fx.pct_change() + (rd[fx.columns].values - rd[["USD"]].values) / 100.0 / DAYS
    cr = pd.read_parquet(CACHE / "crypto.parquet").ffill().reindex(cal).ffill()
    crx = cr.pct_change().sub(rf, axis=0)
    cc = pd.read_parquet(CACHE / "cc_daily.parquet")["cc_net"]
    # compound crypto days onto the next SPY trading day
    pos = np.searchsorted(cal.values, cc.index.values, side="left")
    ok = pos < len(cal)
    cc_cal = (1 + cc[ok]).groupby(cal[pos[ok]]).prod() - 1
    ccx = (cc_cal.reindex(cal) - rf).where(cc_cal.reindex(cal).notna())
    etfx = px[L.TF_UNIVERSE].pct_change().sub(rf, axis=0)
    return dict(px=px, rf=rf, cal=cal, etfx=etfx, fxx=fxx, fx=fx, rates=rates, crx=crx, ccx=ccx)


# ----------------------------------------------------------------- sleeves
def month_ends(cal):
    return L.month_ends(cal)


def _apply(targets: dict[int, pd.Series], X: pd.DataFrame, short_fee: float) -> pd.Series:
    """Excess return of holding month-end targets (from d+2), with costs; NaN before start."""
    W = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    for d, w in targets.items():
        if d + 2 < len(X):
            W.iloc[d + 2] = w.reindex(X.columns).fillna(0.0).values
    started = W.notna().any(axis=1).cummax()
    W = W.ffill().fillna(0.0)
    turn = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    r = (W * X.fillna(0.0)).sum(axis=1) - COST * turn - short_fee / DAYS * W.clip(upper=0).abs().sum(axis=1)
    return r.where(started)


def tf_targets(X: pd.DataFrame, d: int) -> pd.Series:
    """TF rule on excess-return series X at month end d (data up to d)."""
    hist = X.iloc[: d + 1]
    w = pd.Series(0.0, index=X.columns)
    ok = [c for c in X.columns if hist[c].notna().sum() > 252]
    if not ok:
        return w
    idx = (1 + hist[ok].fillna(0.0)).cumprod()
    sig = sum(np.sign(idx.iloc[-1] / idx.iloc[-1 - k] - 1.0) for k in LOOKBACKS) / 4.0
    vol = hist[ok].iloc[-60:].std() * np.sqrt(DAYS)
    raw = (sig / vol.replace(0, np.nan)).fillna(0.0)
    g = raw.abs().sum()
    if g > 0:
        w[ok] = raw / g
    return w


def carry_targets(D: dict, d: int) -> pd.Series:
    """Long 3 highest / short 3 lowest policy rates (previous month's value), inverse-vol."""
    X = D["fxx"]; t = D["cal"][d]
    w = pd.Series(0.0, index=X.columns)
    prev = D["rates"][D["rates"].index < t.to_period("M").to_timestamp()]   # months before d's month
    if prev.empty:
        return w
    r = prev.iloc[-1][X.columns].dropna()
    hist = X.iloc[: d + 1]
    ok = [c for c in r.index if hist[c].notna().sum() > 60]
    if len(ok) < 6:
        return w
    r = r[ok].sort_values()
    vol = hist[ok].iloc[-60:].std() * np.sqrt(DAYS)
    for leg, sign in ((r.index[-3:], 1.0), (r.index[:3], -1.0)):
        iv = 1.0 / vol[leg]
        w[leg] = sign * 0.5 * iv / iv.sum()
    return w


def build_sleeves(D: dict) -> pd.DataFrame:
    cal = D["cal"]; mes = month_ends(cal)
    S = {}
    S["EQ"] = D["px"]["SPY"].pct_change() - D["rf"]
    S["ETF_TF"] = _apply({d: tf_targets(D["etfx"], d) for d in mes}, D["etfx"], SHORT_FEE)
    S["FX_CARRY"] = _apply({d: carry_targets(D, d) for d in mes}, D["fxx"], 0.0)
    S["FX_TF"] = _apply({d: tf_targets(D["fxx"], d) for d in mes}, D["fxx"], 0.0)
    S["CRYPTO_TF"] = _apply({d: tf_targets(D["crx"], d) for d in mes}, D["crx"], 0.0)
    S["CRYPTO_CC"] = D["ccx"]
    df = pd.DataFrame(S)
    # a TF sleeve with no eligible instrument yet is "not started", not "flat"
    for c in ["ETF_TF", "FX_TF", "CRYPTO_TF", "FX_CARRY"]:
        live = df[c].fillna(0.0).ne(0)          # NaN != 0 is True in pandas, so fill first
        if live.any():
            df.loc[: live.idxmax() - pd.Timedelta(days=1), c] = np.nan
    return df


# ----------------------------------------------------------------- combinations
def combo_weights(name: str, S: pd.DataFrame, d: int, prev: pd.Series | None) -> pd.Series:
    hist = S.iloc[: d + 1]
    vol = hist.iloc[-60:].std() * np.sqrt(DAYS)
    has60 = hist.iloc[-60:].notna().sum() >= 60
    w = pd.Series(0.0, index=S.columns)
    eq_risk = lambda cols: pd.Series({c: 1.0 / vol[c] for c in cols if has60[c] and vol[c] > 0})

    if name in SLEEVES:
        if has60[name] and vol[name] > 0:
            w[name] = 1.0
        return w
    if name in ("EQRISK", "EX_EQ"):
        cols = [c for c in S.columns if not (name == "EX_EQ" and c == "EQ")]
        e = eq_risk(cols)
        w[e.index] = e
        return w
    quarter_end = S.index[d].month in (3, 6, 9, 12)
    if not quarter_end and prev is not None and prev.abs().sum() > 0:
        return prev
    h252 = hist.iloc[-252:]
    full = [c for c in S.columns if h252[c].notna().sum() >= 252 and vol[c] > 0]
    sr = pd.Series({c: h252[c].mean() / h252[c].std() * np.sqrt(DAYS) for c in full})
    if name == "WF_SHARPE":
        pos = sr[sr > 0]
        if pos.empty:
            e = eq_risk(S.columns); w[e.index] = e
        else:
            w[pos.index] = pos / vol[pos.index]
        return w
    if name == "WF_TOP2":
        if len(sr) == 0:
            e = eq_risk(S.columns); w[e.index] = e
        else:
            top = sr.sort_values().index[-2:]
            w[top] = 1.0 / vol[top]
        return w
    raise ValueError(name)


def prepare(name: str, S: pd.DataFrame) -> dict:
    mes = month_ends(S.index)
    targets, prev = {}, None
    for d in mes:
        prev = combo_weights(name, S, d, prev)
        targets[d] = prev
    W = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for d, w in targets.items():
        if d + 2 < len(S):
            W.iloc[d + 2] = w.values
    W = W.ffill().fillna(0.0)
    unscaled = (W * S.fillna(0.0)).sum(axis=1)
    return dict(targets=targets, unscaled=unscaled, S=S, mes=mes)


def scaled(prep: dict, rf: pd.Series, tau: float) -> pd.Series:
    S, u = prep["S"], prep["unscaled"]
    W = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for d, w in prep["targets"].items():
        s = u.iloc[max(0, d - 59): d + 1].std() * np.sqrt(DAYS)
        lev = min(tau / s, L.MAX_LEV / max(w.abs().sum(), 1e-12)) if s > 0 else 0.0
        if d + 2 < len(S):
            W.iloc[d + 2] = (w * lev).values
    W = W.ffill().fillna(0.0)
    gross = W.abs().sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(gross)
    r = rf.fillna(0.0) + (W * S.fillna(0.0)).sum(axis=1) \
        - (gross - 1).clip(lower=0) * L.LEV_SPREAD / DAYS - COST * turn
    return r


def extra_metrics(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna(); ex = r - rf.reindex(r.index).fillna(0.0)
    mo = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    eq = (1 + r).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(r)) - 1
    dd = (eq / eq.cummax() - 1).min()
    dn = ex[ex < 0].std() * np.sqrt(DAYS)
    return dict(pf_monthly=float(mo[mo > 0].sum() / -mo[mo < 0].sum()) if (mo < 0).any() else np.inf,
                sortino=float(ex.mean() * DAYS / dn) if dn > 0 else np.nan,
                calmar=float(cagr / -dd) if dd < 0 else np.inf)


# ----------------------------------------------------------------- checks
def checks(D, S) -> bool:
    ok = True
    spy = D["px"]["SPY"].pct_change()
    d = (S["EQ"] + D["rf"] - spy).abs().max()
    good = d < 1e-12; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] EQ sleeve + T-bill = SPY (max diff {d:.1e})")
    jpy = D["fx"]["JPY"].dropna()
    good = 0.004 < jpy.median() < 0.02; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] JPY converted to USD per yen (median {jpy.median():.4f})")
    # carry accrual sign: AUD (high rate) vs USD in 2008-01 should accrue positively
    acc = (D["fxx"]["AUD"] - D["fx"]["AUD"].pct_change()).loc["2008-01"].mean()
    good = acc > 0; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] AUD earned a positive rate differential in 2008-01 ({acc*DAYS:.2%}/yr)")
    mes = month_ends(D["cal"]); dd = mes[160]
    for nm, f in (("ETF_TF", lambda X, k: tf_targets(X, k)),):
        full = f(D["etfx"], dd); cut = f(D["etfx"].iloc[: dd + 1], dd)
        good = np.allclose(full.values, cut.values); ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] no look-ahead in {nm} at {D['cal'][dd].date()}")
    Dcut = dict(D); Dcut["fxx"] = D["fxx"].iloc[: dd + 1]
    good = np.allclose(carry_targets(D, dd).values, carry_targets(Dcut, dd).values); ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] no look-ahead in FX_CARRY at {D['cal'][dd].date()}")
    for nm in ("WF_SHARPE", "WF_TOP2"):
        a = combo_weights(nm, S, dd, None); b = combo_weights(nm, S.iloc[: dd + 1], dd, None)
        good = np.allclose(a.values, b.values); ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] no look-ahead in {nm}")
    starts = {c: S[c].first_valid_index() for c in S.columns}
    print("   sleeve start dates:", {k: (v.date() if v is not None else None) for k, v in starts.items()})
    # a sleeve cannot start before its data plus the 12-month (TF) or 3-month (carry) history
    earliest = {"ETF_TF": "2003-01-01", "FX_CARRY": "2003-03-01", "FX_TF": "2004-01-01",
                "CRYPTO_TF": "2015-09-01", "CRYPTO_CC": "2022-08-01"}
    for c, e in earliest.items():
        good = starts[c] is not None and starts[c] >= pd.Timestamp(e); ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {c} starts after its data allows (>= {e})")
    print("CHECKS PASSED" if ok else "CHECKS FAILED")
    return ok


# ----------------------------------------------------------------- study
def period(r, p):
    a, b = PERIODS[p]
    return r.loc[a:b]


def run(D, S):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rf, spy = D["rf"], D["px"]["SPY"].pct_change()
    bench = spy.copy()
    series, calib = {"BENCH": bench}, []
    for name in SLEEVES + COMBOS:
        prep = prepare(name, S)
        best = None; dd_at = {}
        for tau in L.TARGETS:
            r = scaled(prep, rf, tau)
            # days without positions earn only the T-bill rate, so they add no drawdown
            dd = L.metrics(period(r, "selection"), rf)["maxdd"]
            dd_at[tau] = dd
            if dd >= -L.SEL_DD_LIMIT:
                best = (tau, r)
        if best is None:
            best = (L.TARGETS[0], scaled(prep, rf, L.TARGETS[0]))
        series[name] = best[1]
        calib.append(dict(strategy=name, target_vol=best[0], selection_dd=dd_at[best[0]],
                          in_selection=bool(S[name].loc[PERIODS['selection'][0]:PERIODS['selection'][1]].notna().any())
                          if name in SLEEVES else True))
        print(f"{name:<10} target {best[0]:.0%} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(calib).to_csv(OUT / "calibration.csv", index=False)

    rows = []
    for name, r in series.items():
        for p in PERIODS:
            rp = period(r, p)
            if name in SLEEVES and S[name].loc[PERIODS[p][0]:PERIODS[p][1]].notna().sum() < 120:
                continue
            rows.append(dict(strategy=name, period=p, **L.metrics(rp, rf, spy), **extra_metrics(rp, rf)))
        rf_full = r.loc["2007-07-01":"2026-10-31"]
        rows.append(dict(strategy=name, period="full", **L.metrics(rf_full, rf, spy), **extra_metrics(rf_full, rf)))
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics.csv", index=False)
    S.loc["2007-07-01":].corr().round(2).to_csv(OUT / "sleeve_correlations.csv")
    yearly = pd.DataFrame({n: (1 + r.loc["2007-07-01":"2026-10-31"]).groupby(
        r.loc["2007-07-01":"2026-10-31"].index.year).prod() - 1 for n, r in series.items()})
    yearly.to_csv(OUT / "yearly.csv")

    m = met.set_index(["strategy", "period"])
    spy_sr = {p: m.loc[("BENCH", p), "sharpe"] for p in ("selection", "confirmation")}
    feas = []
    for n in SLEEVES + COMBOS:
        if not all((n, p) in m.index for p in ("selection", "confirmation")):
            continue
        if all(m.loc[(n, p), "maxdd"] >= -0.20 and m.loc[(n, p), "sharpe"] > spy_sr[p]
               for p in ("selection", "confirmation")):
            feas.append(n)
    score = {n: float(min(m.loc[(n, "selection"), "cagr"], m.loc[(n, "confirmation"), "cagr"])) for n in feas}
    cand = max(score, key=score.get) if score else None
    h1 = bool(all(m.loc[("WF_SHARPE", p), "sharpe"] > m.loc[("EQRISK", p), "sharpe"]
                  for p in ("confirmation", "holdout")))
    summary = dict(feasible=feas, maximin_cagr=score, candidate=cand, H1_recent_winner_supported=h1,
                   calibration={c["strategy"]: c["target_vol"] for c in calib},
                   runtime_s=round(time.time() - t0))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    figure(series, cand)
    print(json.dumps(summary, indent=2))


def figure(series, cand):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["BENCH", "EQRISK", "WF_SHARPE", "WF_TOP2", "EX_EQ"] + ([cand] if cand and cand not in
                                                                ["EQRISK", "WF_SHARPE", "WF_TOP2", "EX_EQ"] else [])
    col = {"BENCH": "#7E8CA8", "EQRISK": "#4C86FF", "WF_SHARPE": "#E39B2D", "WF_TOP2": "#D6455D",
           "EX_EQ": "#2BA84A"}
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for n in show:
        r = series[n].loc["2007-07-01":"2026-10-31"]; eq = (1 + r).cumprod()
        ax[0].plot(eq, lw=2.2 if n == cand else 1.2, color=col.get(n, "#9C6ADE"),
                   label="S&P 500 (SPY)" if n == "BENCH" else n)
        ax[1].plot(eq / eq.cummax() - 1, lw=1.0, color=col.get(n, "#9C6ADE"))
    for p, (a, _) in PERIODS.items():
        for x in ax:
            x.axvline(pd.Timestamp(a), color="grey", ls=":", lw=0.8)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2); ax[0].set_ylabel("growth of 1 (log)")
    ax[1].axhline(-0.2, color="red", ls="--", lw=0.8); ax[1].grid(alpha=0.2); ax[1].set_ylabel("drawdown")
    fig.tight_layout(); fig.savefig(OUT / "equity_drawdown.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--checks", action="store_true")
    a = ap.parse_args()
    D = load(); S = build_sleeves(D)
    if not checks(D, S):
        sys.exit("checks failed; study not run")
    if not a.checks:
        run(D, S)
