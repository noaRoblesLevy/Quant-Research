"""
Funding-carry harvest — a market-neutral, beta-hedged perpetual-funding edge.

Economic rationale
------------------
On a perp, when funding is positive the SHORT receives it. A book that is
**long the lowest-funding coins / short the highest-funding coins** therefore
collects funding on both legs — its per-period income is the cross-sectional
funding *spread*. That income is a STRUCTURAL CASH FLOW paid every 8h whether the
market trends or chops, which makes it regime-agnostic (unlike price-momentum,
which died out-of-sample in 2024-25).

The tension: high funding marks high-beta bull momentum, so shorting it imports a
short-beta price tilt. We neutralize that with a rolling-beta BTC hedge leg, which
isolates the (ultra-steady, Sharpe ~30) carry from the volatile price overlay.

No look-ahead: ranking/beta at period t use only info up to t; funding and price
P&L are realized over t -> t+1. Costs are turnover-based.

    from quantresearch.funding_harvest import load_panels, harvest
    P, Fnd, logret = load_panels(UNIVERSE)
    res = harvest(P, Fnd, logret, n_each=5, hold=9, cost_bps=8.0, beta_hedge=True)
    res['net']  # 8h-period P&L series, decomposed into carry/price/cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantresearch.data.binance_loader import load_klines, load_funding

PERIODS_PER_YEAR = 3 * 365          # 8h funding periods
ANNF = np.sqrt(PERIODS_PER_YEAR)


def sharpe(x: pd.Series) -> float:
    return x.mean() / x.std() * ANNF if x.std() > 0 else 0.0


def load_panels(universe: list[str], start: str = "2021-01", end: str = "2025-05"):
    """Return (price_8h, funding_8h, logret_8h) DataFrames aligned on the 8h grid."""
    klines  = {s: load_klines(s, "1h", start, end) for s in universe}
    funding = {s: load_funding(s, start) for s in universe}

    P = pd.DataFrame({s: klines[s]["close"].resample("8h").last() for s in universe}).sort_index()

    f8 = {}
    for s in universe:
        f = funding[s]
        if isinstance(f, pd.DataFrame):
            f = f.iloc[:, 0]
        f.index = f.index.floor("h")
        f8[s] = f.groupby(f.index.floor("8h")).last()
    Fnd = pd.DataFrame(f8).reindex(P.index)

    common = P.dropna(how="all").index.intersection(Fnd.dropna(how="all").index)
    P, Fnd = P.loc[common], Fnd.loc[common]
    logret = np.log(P).diff()
    return P, Fnd, logret


def rolling_beta(logret: pd.DataFrame, ref: str = "BTCUSDT", win: int = 90) -> pd.DataFrame:
    """Rolling beta of each coin vs the reference, shifted 1 period (no look-ahead)."""
    r = logret[ref]
    var = r.rolling(win).var()
    beta = pd.DataFrame({c: logret[c].rolling(win).cov(r) / var for c in logret.columns})
    return beta.shift(1)


def harvest(P: pd.DataFrame, Fnd: pd.DataFrame, logret: pd.DataFrame,
            n_each: int = 5, hold: int = 9, cost_bps: float = 8.0,
            beta_hedge: bool = True, hedge_ref: str = "BTCUSDT",
            vol_target: float | None = None, vol_win: int = 30) -> pd.DataFrame:
    """
    Simulate the funding-harvest book.

    n_each     : coins per leg (long lowest-funding n, short highest-funding n)
    hold       : rebalance every `hold` periods (9 = 3 days)
    cost_bps   : round-trip cost on turnover
    beta_hedge : add a rolling-beta `hedge_ref` leg to neutralize net beta
    vol_target : if set, scale the whole book so trailing realized vol of net P&L
                 matches this per-period target (stabilizes DD across regimes)

    Returns a DataFrame with columns carry / price / cost / net (per-period P&L).
    """
    universe = list(P.columns)
    beta = rolling_beta(logret, hedge_ref) if beta_hedge else None
    hedge_fund = Fnd[hedge_ref]
    btc = logret[hedge_ref]
    times = Fnd.index

    cur_w = pd.Series(0.0, index=universe)
    cur_hedge = 0.0
    scale = 1.0
    C, Pr, Co = [], [], []
    net_hist: list[float] = []

    for i in range(len(times)):
        # ---- realize P&L from positions held since last rebalance ----
        if cur_w.abs().sum() > 0 and i > 0:
            pr = float((cur_w * logret.iloc[i].reindex(universe).fillna(0)).sum())
            fc = float((-cur_w * Fnd.iloc[i].reindex(universe).fillna(0)).sum())
            hp = cur_hedge * btc.iloc[i] - cur_hedge * hedge_fund.iloc[i]   # hedge price + funding
        else:
            pr = fc = hp = 0.0
        carry_i, price_i = scale * fc, scale * (pr + hp)
        cost_i = 0.0

        # ---- rebalance ----
        if i % hold == 0:
            fr = Fnd.iloc[i].dropna()
            if len(fr) >= 2 * n_each:
                new_w = pd.Series(0.0, index=universe)
                new_w[fr.nsmallest(n_each).index] =  1.0 / n_each
                new_w[fr.nlargest(n_each).index]  = -1.0 / n_each

                # vol-target scaling from trailing net P&L
                if vol_target is not None and len(net_hist) >= vol_win:
                    realized = np.std(net_hist[-vol_win:])
                    scale = vol_target / realized if realized > 0 else scale
                    scale = min(scale, 5.0)        # cap leverage

                turnover = (cur_w - new_w).abs().sum() / 2.0
                cost_i += scale * turnover * cost_bps / 1e4
                cur_w = new_w

                if beta_hedge:
                    b = beta.iloc[i].reindex(universe).fillna(1.0)
                    new_hedge = -float((cur_w * b).sum())
                    cost_i += scale * abs(new_hedge - cur_hedge) * cost_bps / 1e4
                    cur_hedge = new_hedge

        net_i = carry_i + price_i - cost_i
        C.append(carry_i); Pr.append(price_i); Co.append(cost_i); net_hist.append(net_i)

    df = pd.DataFrame({"carry": C, "price": Pr, "cost": Co}, index=times)
    df["net"] = df["carry"] + df["price"] - df["cost"]
    return df


DEFAULT_UNIVERSE = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","LINKUSDT",
    "DOTUSDT","LTCUSDT","BCHUSDT","FILUSDT","NEARUSDT","AVAXUSDT","XLMUSDT","AAVEUSDT",
    "UNIUSDT","TRXUSDT","ATOMUSDT","ETCUSDT","MATICUSDT","SANDUSDT","AXSUSDT","COMPUSDT",
]
