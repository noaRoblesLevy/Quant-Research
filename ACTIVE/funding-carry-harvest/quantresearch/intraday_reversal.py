"""
Short-horizon cross-sectional reversal on crypto perps — candidate 2nd edge.

Mechanism (different from carry, and funding-light)
---------------------------------------------------
Over short horizons (hours), a coin that jumps relative to its peers tends to
partially mean-revert — liquidity providers and short-term traders fade the move.
A dollar-neutral book that goes LONG the recent cross-sectional losers and SHORT
the recent winners harvests that reversion. Because it holds only hours and is
dollar-neutral, funding (an 8h flow) is incidental, not the edge — so this is a
genuinely orthogonal return driver to the funding-carry harvest.

Prior evidence: in the (otherwise dead) XS-momentum study, the *reversal* leg had a
gross 2025 Sharpe of +1.90 — the one bright spot. The open question is whether it
survives turnover cost, which is the entire challenge for a high-frequency book.

Runs on the 1h grid (Bybit 60m klines). No look-ahead: signal at bar t uses returns
through t-1; P&L realized t -> t+1; cost is turnover-based. Implemented in numpy
because the 1h panel is ~25k bars.

    from quantresearch.intraday_reversal import load_1h_panels, reversal_book
    P, R, F = load_1h_panels(UNIVERSE)
    res = reversal_book(P, R, F, k=3, hold=2, n_each=6, cost_bps=6.0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantresearch.data.bybit_loader import load_klines, load_funding

PERIODS_PER_YEAR_1H = 24 * 365
ANNF_1H = np.sqrt(PERIODS_PER_YEAR_1H)


def sharpe_1h(x: pd.Series) -> float:
    return x.mean() / x.std() * ANNF_1H if x.std() > 0 else 0.0


def load_1h_panels(universe: list[str], start: str = "2022-07", end: str = "2025-05"):
    """Return (price_1h, logret_1h, funding_1h) aligned on the 1h grid.

    funding_1h spreads each 8h Bybit rate across its 8 hourly bars (rate/8 per hour),
    so a position accrues the correct funding per hour held."""
    P = pd.DataFrame({s: load_klines(s, "60", start, end)["close"] for s in universe}).sort_index()
    logret = np.log(P).diff()

    f8 = {}
    for s in universe:
        f = load_funding(s, start, end)
        f.index = f.index.floor("h")
        f8[s] = f.groupby(f.index.floor("8h")).sum()
    F8 = pd.DataFrame(f8)
    # spread each 8h rate over the next 8 hourly bars
    F1h = F8.reindex(P.index, method="ffill") / 8.0

    common = P.dropna(how="all").index
    return P.loc[common], logret.loc[common], F1h.loc[common]


def reversal_book(P: pd.DataFrame, logret: pd.DataFrame, F1h: pd.DataFrame,
                  k: int = 3, hold: int = 2, n_each: int = 6, cost_bps: float = 6.0,
                  vol_target: float | None = None, vol_win: int = 72) -> pd.DataFrame:
    """Simulate the short-horizon XS reversal book (numpy core).

    k        : lookback bars for the reversal signal (fade the trailing k-bar return)
    hold     : rebalance every `hold` bars (1 = hourly; higher = less turnover/cost)
    n_each   : coins per leg (long n biggest losers / short n biggest winners)
    cost_bps : per-unit-turnover cost in bps (round-trip on traded notional)
    vol_target / vol_win : optional trailing-vol scaling of the book (per-bar target)

    Returns DataFrame with price / funding / cost / net per-bar P&L (1h).
    """
    cols = list(P.columns)
    R = logret.to_numpy()
    F = F1h.reindex(columns=cols).to_numpy()
    R = np.nan_to_num(R, nan=0.0)
    F = np.nan_to_num(F, nan=0.0)
    # signal = -trailing k-bar return, shifted 1 (high signal = recent loser = go long)
    sig = -pd.DataFrame(R, columns=cols).rolling(k).sum().shift(1).to_numpy()
    T, N = R.shape

    cur = np.zeros(N)
    scale = 1.0
    price = np.zeros(T); fund = np.zeros(T); cost = np.zeros(T)
    net_hist: list[float] = []

    for i in range(T):
        if i > 0 and np.any(cur):
            pr = float(cur @ R[i])
            fc = float(-(cur @ F[i]))
        else:
            pr = fc = 0.0
        price[i] = scale * pr
        fund[i] = scale * fc
        c_i = 0.0

        if i % hold == 0:
            s = sig[i]
            valid = np.where(~np.isnan(s))[0]
            if valid.size >= 2 * n_each:
                order = valid[np.argsort(s[valid])]      # ascending: winners first, losers last
                new = np.zeros(N)
                new[order[-n_each:]] = 1.0 / n_each       # biggest losers -> long
                new[order[:n_each]] = -1.0 / n_each       # biggest winners -> short

                if vol_target is not None and len(net_hist) >= vol_win:
                    realized = float(np.std(net_hist[-vol_win:]))
                    scale = min(vol_target / realized, 5.0) if realized > 0 else scale

                turnover = np.abs(cur - new).sum() / 2.0
                c_i = scale * turnover * cost_bps / 1e4
                cur = new

        cost[i] = c_i
        net_hist.append(price[i] + fund[i] - c_i)

    df = pd.DataFrame({"price": price, "funding": fund, "cost": cost}, index=P.index)
    df["net"] = df["price"] + df["funding"] - df["cost"]
    return df


def avg_turnover(P, logret, k, hold, n_each):
    """Mean one-way turnover per rebalance (diagnostic for cost exposure)."""
    cols = list(P.columns)
    R = np.nan_to_num(logret.to_numpy(), nan=0.0)
    sig = -pd.DataFrame(R, columns=cols).rolling(k).sum().shift(1).to_numpy()
    T, N = R.shape
    cur = np.zeros(N); tos = []
    for i in range(0, T, hold):
        s = sig[i]; valid = np.where(~np.isnan(s))[0]
        if valid.size >= 2 * n_each:
            order = valid[np.argsort(s[valid])]
            new = np.zeros(N)
            new[order[-n_each:]] = 1.0 / n_each
            new[order[:n_each]] = -1.0 / n_each
            tos.append(np.abs(cur - new).sum() / 2.0)
            cur = new
    return float(np.mean(tos)) if tos else 0.0
