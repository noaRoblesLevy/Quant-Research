"""
Reusable edge-construction functions (the strategy building blocks).

These were prototyped in scripts/ during research; they live here in the package
so they import cleanly everywhere (IDE-resolvable, no sys.path hacks) and so the
scripts and notebooks share a single source of truth.

Each function returns a daily return Series (or a (return, held-mask) pair) for one
edge, net of spread where noted. Swap/financing is applied separately (see
scripts/eval_swap_costs.py and pull_swap_rates.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
CCYS = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}

# index cash sessions, broker EET server time (first bar, last bar)
INDEX_SESSIONS = {
    "US100_cash": ("16:30", "22:30"), "US500_cash": ("16:30", "22:30"),
    "US30_cash": ("16:30", "22:30"), "US2000_cash": ("16:30", "22:30"),
    "GER40_cash": ("10:00", "18:00"), "EU50_cash": ("10:00", "18:00"),
    "UK100_cash": ("10:00", "18:00"), "JP225_cash": ("02:30", "08:00"),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def is_fx(symbol: str) -> bool:
    """True for a 6-letter major FX pair (both legs in the 8 majors)."""
    return len(symbol) == 6 and symbol[:3] in CCYS and symbol[3:] in CCYS


def daily_close_ret(symbol: str, tf: str = "H1") -> pd.Series:
    """Daily close-to-close % returns for one symbol (date-indexed)."""
    df = pd.read_parquet(CACHE / f"{symbol}_{tf}.parquet")
    r = df["close"].resample("1D").last().dropna().pct_change()
    r.index = r.index.normalize()
    return r.dropna()


# --------------------------------------------------------------------------- #
# edge 1: FX cross-sectional reversion (daily P&L, net of spread)
# --------------------------------------------------------------------------- #
def daily_reversion_pnl(curr: pd.DataFrame, lookback: int = 12, n_sel: int = 2,
                        spread_bps: float = 1.5) -> pd.Series:
    """Daily returns of long-weakest / short-strongest currency reversion.

    curr: daily currency-strength matrix (from cross_sectional.build_currency_returns).
    Weekly positions held through the week earn daily currency returns; spread is
    charged on the weekly rebalance (Fridays).
    """
    w = curr.resample("W-FRI").sum()
    n = w.shape[1]
    sig = w.rolling(lookback).sum().shift(1)
    ranks = sig.rank(axis=1)
    long = ranks <= n_sel                 # reversion: long the weakest
    short = ranks > (n - n_sel)           # short the strongest
    weight = long.astype(float) / n_sel - short.astype(float) / n_sel

    daily_w = weight.reindex(curr.index, method="bfill").dropna(how="all")
    daily_ret = (curr.reindex(daily_w.index) * daily_w).sum(axis=1)

    pos = long.astype(int) - short.astype(int)
    turnover = pos.diff().abs().sum(axis=1) / (2 * n_sel)
    weekly_cost = (2 * turnover * spread_bps / 1e4).reindex(daily_w.index, method="bfill").fillna(0)
    is_friday = pd.Series(daily_w.index.dayofweek == 4, index=daily_w.index)
    return (daily_ret - weekly_cost.where(is_friday, 0.0)).dropna()


# --------------------------------------------------------------------------- #
# edge 2: index overnight (cash close -> next cash open)
# --------------------------------------------------------------------------- #
def overnight_frame(sym: str, first_slot: str, last_slot: str, spread_bps: float = 1.5) -> pd.DataFrame:
    df = pd.read_parquet(CACHE / f"{sym}_M30.parquet")
    slot = df.index.strftime("%H:%M")
    date = df.index.normalize()
    d = pd.DataFrame({"date": date, "slot": slot, "open": df["open"].values, "close": df["close"].values})
    so = d[d.slot == first_slot].set_index("date")["open"]
    sc = d[d.slot == last_slot].set_index("date")["close"]
    j = pd.DataFrame({"open": so, "close": sc}).dropna()
    overnight = j["open"] / j["close"].shift(1) - 1.0 - spread_bps / 1e4
    uptrend = (j["close"] > j["close"].rolling(200).mean()).shift(1)
    return pd.DataFrame({"ret": overnight, "uptrend": uptrend}).dropna()


def basket(use_filter: bool = False) -> pd.Series:
    """Equal-weight overnight-long index basket (optionally 200MA trend-filtered)."""
    rets, ups = {}, {}
    for sym, (fs, ls) in INDEX_SESSIONS.items():
        fr = overnight_frame(sym, fs, ls)
        rets[sym] = fr["ret"]
        ups[sym] = fr["uptrend"].astype(bool)
    R = pd.DataFrame(rets)
    U = pd.DataFrame(ups).reindex(R.index, fill_value=False)
    if use_filter:
        R = R.where(U)
    return R.mean(axis=1).dropna()


# --------------------------------------------------------------------------- #
# edge 3: crypto trend (long BTC above its 50-day MA)
# --------------------------------------------------------------------------- #
def crypto_with_mask(ma: int = 50):
    """Return (daily returns, held-mask) for long-BTC-above-MA."""
    df = pd.read_parquet(CACHE / "BTCUSD_H1.parquet")
    c = df["close"].resample("1D").last().dropna()
    pos = (c > c.rolling(ma).mean()).shift(1)
    ret = (c.pct_change() * pos)
    ret.index = ret.index.normalize()
    pos.index = pos.index.normalize()
    ret = ret.dropna()
    return ret, pos.reindex(ret.index).fillna(False).astype(bool)


def crypto_trend(ma: int = 50) -> pd.Series:
    return crypto_with_mask(ma)[0]


# --------------------------------------------------------------------------- #
# edge 4: turn-of-month seasonality (index basket, last + first 3 trading days)
# --------------------------------------------------------------------------- #
def tom_with_mask(indices=("US100_cash", "US500_cash", "GER40_cash", "JP225_cash")):
    """Return (daily returns, held-mask) for the turn-of-month index basket."""
    rets = [pd.read_parquet(CACHE / f"{s}_M30.parquet")["close"].resample("1D").last().dropna().pct_change()
            for s in indices]
    bask = pd.concat(rets, axis=1).mean(axis=1).dropna()
    bask.index = bask.index.normalize()
    d = bask.to_frame("r")
    d["ym"] = d.index.to_period("M")
    hold = (d.groupby("ym").cumcount() <= 2) | (d.groupby("ym").cumcount(ascending=False) == 0)
    return (d["r"] * hold).dropna(), hold.reindex(d.index).fillna(False)


def turn_of_month(indices=("US100_cash", "US500_cash", "GER40_cash", "JP225_cash")) -> pd.Series:
    return tom_with_mask(indices)[0]
