"""
Cross-sectional FX momentum.

Ranks the 8 currencies by trailing relative strength and forms a market-neutral
long-short portfolio (long the strongest, short the weakest), rebalanced weekly.

Currency strength construction
------------------------------
For each currency c, its return on a given day is the average — over the 7 pairs
that contain c — of the signed pair return:
    * if c is the base  (cXXX): +pair_return
    * if c is the quote (XXXc): -pair_return
With all 28 major pairs present, every currency appears in exactly 7 pairs, so the
strengths are balanced and comparable. This is the standard academic construction
(Menkhoff et al. 2012; Okunev & White 2003).

No look-ahead: the ranking signal at week t uses returns only through week t-1
(`.shift(1)`); the portfolio then earns week t's realised returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_currency_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """data: {pair: H1 OHLC df}. Returns daily currency log-return matrix (cols=currencies)."""
    # daily close-to-close log returns per pair
    rets = {}
    for pair, df in data.items():
        daily_close = df["close"].resample("1D").last().dropna()
        rets[pair] = np.log(daily_close).diff()
    R = pd.DataFrame(rets)

    currencies = sorted(set(p[:3] for p in data) | set(p[3:] for p in data))
    acc = {c: pd.Series(0.0, index=R.index) for c in currencies}
    cnt = {c: 0 for c in currencies}

    for pair in data:
        base, quote = pair[:3], pair[3:]
        r = R[pair].fillna(0.0)
        acc[base] = acc[base] + r
        acc[quote] = acc[quote] - r
        cnt[base] += 1
        cnt[quote] += 1

    out = pd.DataFrame({c: acc[c] / cnt[c] for c in currencies})
    return out.dropna(how="all")


def xs_momentum(curr_daily: pd.DataFrame, lookback_weeks: int = 12,
                n_sel: int = 3, rebalance: str = "W-FRI") -> dict:
    """Long-short cross-sectional momentum on currency strengths.

    lookback_weeks : trailing window for the momentum signal
    n_sel          : number of currencies held long AND short (e.g. 3 = long top 3, short bottom 3)
    Returns weekly portfolio returns, equity curve, and turnover.
    """
    w = curr_daily.resample(rebalance).sum()        # weekly log returns per currency
    n = w.shape[1]

    signal = w.rolling(lookback_weeks).sum().shift(1)   # trailing return, known before the week
    ranks = signal.rank(axis=1)

    long = ranks > (n - n_sel)      # top n_sel
    short = ranks <= n_sel          # bottom n_sel

    # equal-weight long minus short, realised over the held week
    port = (w.where(long).mean(axis=1) - w.where(short).mean(axis=1)).dropna()

    # turnover: fraction of long/short slots that change week to week (for cost estimate)
    pos = long.astype(int) - short.astype(int)
    turnover = pos.diff().abs().sum(axis=1).reindex(port.index).fillna(0) / (2 * n_sel)

    equity = (1 + port).cumprod()
    return {"weekly": port, "equity": equity, "turnover": turnover,
            "long": long, "short": short}


def xs_reversion_composite(curr_daily: pd.DataFrame, lookbacks=(4, 8, 12, 26),
                           n_sel: int = 2, rebalance: str = "W-FRI") -> dict:
    """Multi-horizon cross-sectional REVERSION (long weakest / short strongest).

    For each lookback we z-score the trailing return cross-sectionally (so horizons
    are comparable), average the z-scores into one composite, then go long the
    LOWEST composite (weakest -> expected to bounce) and short the HIGHEST.
    Combining independent horizons that all carry the edge lifts Sharpe.
    """
    w = curr_daily.resample(rebalance).sum()
    n = w.shape[1]

    z_sum = None
    for lb in lookbacks:
        sig = w.rolling(lb).sum().shift(1)                 # trailing return, lagged
        z = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1), axis=0)  # cross-sec z-score
        z_sum = z if z_sum is None else z_sum.add(z, fill_value=np.nan)
    composite = z_sum / len(lookbacks)

    ranks = composite.rank(axis=1)
    long = ranks <= n_sel               # REVERSION: long the weakest (lowest composite)
    short = ranks > (n - n_sel)         # short the strongest

    port = (w.where(long).mean(axis=1) - w.where(short).mean(axis=1)).dropna()
    pos = long.astype(int) - short.astype(int)
    turnover = pos.diff().abs().sum(axis=1).reindex(port.index).fillna(0) / (2 * n_sel)
    equity = (1 + port).cumprod()
    return {"weekly": port, "equity": equity, "turnover": turnover}


def ts_momentum(curr_daily: pd.DataFrame, lookback_weeks: int = 52,
                rebalance: str = "W-FRI") -> dict:
    """Time-series (trend-following) momentum on currency strengths.

    Each currency is held LONG if its own trailing return is positive (uptrend),
    SHORT if negative — independent of the cross-section. This is the classic
    trend complement to reversion: it profits in persistent moves where reversion
    bleeds, so the two tend to be negatively correlated.
    """
    w = curr_daily.resample(rebalance).sum()
    sig = w.rolling(lookback_weeks).sum().shift(1)
    pos = np.sign(sig)
    port = (w * pos).mean(axis=1).dropna()          # avg trend-position return
    turnover = pos.diff().abs().sum(axis=1).reindex(port.index).fillna(0) / w.shape[1]
    equity = (1 + port).cumprod()
    return {"weekly": port, "equity": equity, "turnover": turnover}


def perf(port: pd.Series, equity: pd.Series, turnover: pd.Series | None = None,
         spread_bps_per_leg: float = 1.0, n_sel: int = 3) -> dict:
    """Annualised performance of a weekly return series, with optional cost drag."""
    ann_ret = port.mean() * 52
    ann_vol = port.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    out = {
        "ann_ret_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(-dd * 100, 2),
        "hit_rate_pct": round((port > 0).mean() * 100, 1),
        "weeks": len(port),
    }
    if turnover is not None:
        # cost per week ≈ turnover * (2*n_sel legs) * spread; rough, in return units
        weekly_cost = turnover * (2 * n_sel) * (spread_bps_per_leg / 1e4) / (2 * n_sel)
        net = port - weekly_cost
        net_eq = (1 + net).cumprod()
        out["net_ann_ret_pct"] = round(net.mean() * 52 * 100, 2)
        out["net_sharpe"] = round((net.mean() * 52) / (net.std() * np.sqrt(52)), 2) if net.std() > 0 else 0.0
        out["net_max_dd_pct"] = round(-(net_eq / net_eq.cummax() - 1).min() * 100, 2)
        out["avg_turnover"] = round(turnover.mean(), 2)
    return out
