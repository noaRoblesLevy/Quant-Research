"""
Multi-asset portfolio lab: data, building blocks, risk scaling and metrics for the study
pre-registered in docs/preregistration_2026-11-07.md. Every rule here is a literal
implementation of that document; nothing is tuned.

Conventions
    - daily adjusted closes (total return), US trading days (SPY calendar)
    - signals at the last trading day of each month (index d); trades at the close of
      d+1; new weights earn returns from d+2 on (one-day execution lag)
    - weights are held constant between rebalances (daily rebalanced to target); costs
      are charged on the monthly change in target weights
    - cash earns the 13-week T-bill; negative cash (leverage) pays T-bill + 1%; short
      proceeds earn T-bill minus a 0.5% borrow fee
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"

EQUITY = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
BONDS = ["TLT", "IEF", "LQD", "TIP"]
REAL = ["GLD", "DBC", "VNQ"]
FX = ["UUP"]
RP_UNIVERSE = EQUITY + BONDS + REAL                 # 12 ETFs
TF_UNIVERSE = RP_UNIVERSE + FX                      # 13 ETFs
CRYPTO = ["BTC-USD"]
ALL = TF_UNIVERSE + CRYPTO

DAYS = 252
VOL_WIN = 60
MIN_HISTORY = 252
COST = 5e-4
LEV_SPREAD = 0.01
BORROW_FEE = 0.005
MAX_LEV = 3.0

PERIODS = {
    "selection": ("2007-07-01", "2016-12-31"),
    "confirmation": ("2017-01-01", "2022-12-31"),
    "holdout": ("2023-01-01", "2026-11-06"),
}
TARGETS = [t / 100 for t in range(4, 26)]
SEL_DD_LIMIT = 0.15


# ----------------------------------------------------------------- data
def load_data(refresh: bool = False):
    """Returns (prices, rf_daily): adjusted closes on the SPY calendar and the daily
    T-bill rate. Cached in data_cache/."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "prices.parquet"
    if refresh or not f.exists():
        import yfinance as yf
        raw = yf.download(ALL + ["^IRX"], start="2002-01-01", end="2026-11-07",
                          auto_adjust=True, progress=False, threads=True)["Close"]
        raw.to_parquet(f)
    raw = pd.read_parquet(f)
    cal = raw["SPY"].dropna().index
    px = raw[ALL].reindex(cal)
    px["BTC-USD"] = raw["BTC-USD"].ffill().reindex(cal)     # crypto trades daily: last close
    irx = raw["^IRX"].reindex(cal).ffill() / 100.0
    rf = irx / DAYS
    return px, rf


# ----------------------------------------------------------------- building blocks
def month_ends(index: pd.DatetimeIndex) -> list[int]:
    s = pd.Series(np.arange(len(index)), index=index)
    return list(s.groupby([index.year, index.month]).max().values)


def block_weights(block: str, d: int, px: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Unscaled target weights of a building block at month-end index d (data up to d)."""
    hist = px.iloc[: d + 1]
    avail = [c for c in px.columns if hist[c].notna().sum() > MIN_HISTORY and np.isfinite(hist[c].iloc[-1])]
    vol = rets.iloc[max(0, d - VOL_WIN + 1): d + 1].std() * np.sqrt(DAYS)
    w = pd.Series(0.0, index=px.columns)

    if block == "BENCH":
        w["SPY"] = 1.0
    elif block == "SMA":
        if "SPY" not in avail:
            return w
        me = hist["SPY"].groupby([hist.index.year, hist.index.month]).last().iloc[-10:]
        w["SPY"] = 1.0 if hist["SPY"].iloc[-1] > me.mean() else 0.0
    elif block == "RP":
        u = [c for c in RP_UNIVERSE if c in avail and vol[c] > 0]
        if not u:
            return w
        iv = 1.0 / vol[u]
        w[u] = iv / iv.sum()
    elif block in ("TF", "TFL", "BTC"):
        uni = CRYPTO if block == "BTC" else TF_UNIVERSE
        u = [c for c in uni if c in avail and vol[c] > 0]
        if not u:                       # early months: nothing has 12 months of history
            return w
        p = hist[u]
        sig = sum(np.sign(p.iloc[-1] / p.iloc[-1 - k] - 1.0) for k in (21, 63, 126, 252)) / 4.0
        if block == "TFL":
            sig = sig.clip(lower=0.0)
        raw = sig / vol[u]
        g = raw.abs().sum()
        if g > 0:
            w[u] = raw / g
    else:
        raise ValueError(block)
    return w


STRATEGIES = {
    "BENCH": {"BENCH": 1.0},
    "SMA": {"SMA": 1.0},
    "RP": {"RP": 1.0},
    "TF": {"TF": 1.0},
    "TFL": {"TFL": 1.0},
    "P1": {"RP": 0.5, "TF": 0.5},
    "P2": {"BENCH": 0.5, "TF": 0.5},
    "P3": {"RP": 0.45, "TF": 0.45, "BTC": 0.10},
    "P4": {"SMA": 0.5, "TF": 0.5},
    "P5": {"RP": 0.5, "TFL": 0.5},
}


# ----------------------------------------------------------------- simulation
def _run(W: pd.DataFrame, rets: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """Daily portfolio return for a frame of held weights (row t = weights earning day t)."""
    R = rets.fillna(0.0).values
    Wv = W.values
    gross = (Wv * R).sum(1)
    long = np.clip(Wv, 0, None).sum(1)
    short = np.clip(-Wv, 0, None).sum(1)
    cash = 1.0 - long
    rfv = rf.fillna(0.0).values
    fin = np.where(cash >= 0, cash * rfv, cash * (rfv + LEV_SPREAD / DAYS))
    shortc = short * (rfv - BORROW_FEE / DAYS)
    turn = np.abs(np.diff(Wv, axis=0, prepend=np.zeros((1, Wv.shape[1])))).sum(1)
    return pd.Series(gross + fin + shortc - COST * turn, index=rets.index)


def _hold(targets: dict[int, pd.Series], index, cols) -> pd.DataFrame:
    """Turn month-end targets (decided at d) into held weights from d+2 on."""
    W = pd.DataFrame(np.nan, index=index, columns=cols)
    for d, w in targets.items():
        if d + 2 < len(index):
            W.iloc[d + 2] = w.reindex(cols).fillna(0.0).values
    return W.ffill().fillna(0.0)


def prepare(name: str, px: pd.DataFrame, rf: pd.Series) -> dict:
    """Everything that does not depend on the volatility target: block weights, the
    equal-risk combination and the unscaled return series. Compute once, scale often."""
    rets = px.pct_change()
    mes = month_ends(px.index)
    spec = STRATEGIES[name]
    blocks = {b: {d: block_weights(b, d, px, rets) for d in mes} for b in spec}
    zero_rf = rf * 0.0
    braw = {b: _run(_hold(t, px.index, px.columns), rets, zero_rf) for b, t in blocks.items()}
    comb = {}
    for d in mes:
        w = pd.Series(0.0, index=px.columns)
        for b, budget in spec.items():
            if len(spec) == 1:
                k = 1.0
            else:
                s = braw[b].iloc[max(0, d - VOL_WIN + 1): d + 1].std() * np.sqrt(DAYS)
                k = budget / s if s > 0 else 0.0
            w = w + k * blocks[b][d]
        comb[d] = w
    unscaled = _run(_hold(comb, px.index, px.columns), rets, zero_rf)
    return dict(name=name, rets=rets, comb=comb, unscaled=unscaled, index=px.index, cols=px.columns)


def scaled_returns(prep: dict, rf: pd.Series, target_vol: float | None) -> pd.Series:
    """Daily returns with the portfolio scaled monthly to target_vol (None = unscaled)."""
    if target_vol is None:
        return _run(_hold(prep["comb"], prep["index"], prep["cols"]), prep["rets"], rf)
    scaled = {}
    for d, w in prep["comb"].items():
        s = prep["unscaled"].iloc[max(0, d - VOL_WIN + 1): d + 1].std() * np.sqrt(DAYS)
        lev = min(target_vol / s, MAX_LEV / max(w.abs().sum(), 1e-12)) if s > 0 else 0.0
        scaled[d] = w * lev
    return _run(_hold(scaled, prep["index"], prep["cols"]), prep["rets"], rf)


def simulate(name: str, px: pd.DataFrame, rf: pd.Series, target_vol: float | None) -> pd.Series:
    """Daily returns of a strategy. Blocks are first scaled to equal risk (60-day realised
    vol of their unscaled returns), then the portfolio is scaled to target_vol
    (None = unscaled, used for BENCH). Same result as prepare + scaled_returns."""
    return scaled_returns(prepare(name, px, rf), rf, target_vol)


def _simulate_reference(name: str, px: pd.DataFrame, rf: pd.Series, target_vol: float | None) -> pd.Series:
    """Original single-pass implementation, kept to test that prepare/scaled_returns
    gives identical results."""
    rets = px.pct_change()
    mes = month_ends(px.index)
    spec = STRATEGIES[name]
    blocks = {b: {d: block_weights(b, d, px, rets) for d in mes} for b in spec}
    zero_rf = rf * 0.0
    braw = {b: _run(_hold(t, px.index, px.columns), rets, zero_rf) for b, t in blocks.items()}

    comb = {}
    for d in mes:
        w = pd.Series(0.0, index=px.columns)
        for b, budget in spec.items():
            if len(spec) == 1:
                k = 1.0
            else:
                s = braw[b].iloc[max(0, d - VOL_WIN + 1): d + 1].std() * np.sqrt(DAYS)
                k = budget / s if s > 0 else 0.0
            w = w + k * blocks[b][d]
        comb[d] = w
    if target_vol is None:
        return _run(_hold(comb, px.index, px.columns), rets, rf)

    unscaled = _run(_hold(comb, px.index, px.columns), rets, zero_rf)
    scaled = {}
    for d, w in comb.items():
        s = unscaled.iloc[max(0, d - VOL_WIN + 1): d + 1].std() * np.sqrt(DAYS)
        lev = min(target_vol / s, MAX_LEV / max(w.abs().sum(), 1e-12)) if s > 0 else 0.0
        scaled[d] = w * lev
    return _run(_hold(scaled, px.index, px.columns), rets, rf)


# ----------------------------------------------------------------- metrics
def period(r: pd.Series, name: str) -> pd.Series:
    a, b = PERIODS[name]
    return r.loc[a:b]


def metrics(r: pd.Series, rf: pd.Series, spy: pd.Series | None = None) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / DAYS
    ex = r - rf.reindex(r.index).fillna(0.0)
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    out = dict(
        cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
        vol=float(r.std() * np.sqrt(DAYS)),
        sharpe=float(ex.mean() / ex.std() * np.sqrt(DAYS)) if ex.std() > 0 else 0.0,
        maxdd=float((eq / eq.cummax() - 1).min()),
        worst_year=float(yearly.min()),
        pos_years=float((yearly > 0).mean()),
    )
    if spy is not None:
        out["corr_spy"] = float(r.corr(spy.reindex(r.index)))
    return out
