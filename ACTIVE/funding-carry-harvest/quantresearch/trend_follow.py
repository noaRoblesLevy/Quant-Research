"""
Trend-following on crypto perps — the carry book's diversifier.

Why trend, next to the funding-carry harvest
---------------------------------------------
Carry and trend are the two canonical risk premia, and they are diversifying for a
structural reason, not a statistical accident:

  * The carry book draws down when funding spreads compress/invert — which happens
    during violent deleveraging cascades (everyone closes shorts, funding flips,
    correlations -> 1). That is a *sustained directional* event.
  * Trend-following is LONG exactly that event. A time-series trend book can be
    net-SHORT in a downtrend, so it tends to profit when the carry book bleeds.

Because the binding constraint is Calmar (return / maxDD), an edge that pays off in
the carry book's worst weeks attacks maxDD directly — the lever that matters.

This edge was previously validated (~0.79 Sharpe) but was killed on FTMO by the 1:1
crypto leverage cap and the -8.33 bps/day CFD swap. On a Bybit-based perp account
neither killer applies (leverage to 1:100, you pay funding not CFD swap), so it is
worth re-testing on real Bybit data.

Two modes
---------
  ts : time-series trend. Each coin's weight follows the sign / risk-adjusted size of
       its OWN trailing return. Directional (net-long in bull, net-short in bear) ->
       carries market beta, which is the crash hedge. NOT beta-hedged by default.
  xs : cross-sectional trend. Rank coins by trailing return, long top / short bottom.
       Dollar-neutral; gets the same rolling-beta BTC hedge as the carry book.

No look-ahead: the signal at period t uses only returns up to t-1 (rolling then
shift(1)); price and funding P&L are realized over t -> t+1. Costs are turnover-based.
Funding here is an INCIDENTAL credit/debit on held positions (-w_i f_i), not the edge.

    from quantresearch.trend_follow import trend_book
    res = trend_book(P, Fnd, logret, mode='ts', lookback=90, signal='zscore')
    res['net']   # 8h-period P&L, decomposed into trend / funding / cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantresearch.funding_harvest import rolling_beta, sharpe, PERIODS_PER_YEAR, ANNF  # noqa: F401


def trend_signal(logret: pd.DataFrame, lookback: int, signal: str = "zscore") -> pd.DataFrame:
    """Per-coin trend signal at each period, shifted 1 (no look-ahead).

    signal:
      'sign'     -> +/-1 from the sign of the trailing `lookback`-period return
      'zscore'   -> risk-adjusted trend = trailing return / (vol * sqrt(lookback)),
                    clipped to +/-3 (a continuous conviction, the usual TSMOM form)
      'raw'      -> the raw trailing log return
    """
    cum = logret.rolling(lookback).sum()
    if signal == "sign":
        S = np.sign(cum)
    elif signal == "zscore":
        vol = logret.rolling(lookback).std()
        S = (cum / (vol * np.sqrt(lookback))).clip(-3, 3)
    elif signal == "raw":
        S = cum
    else:
        raise ValueError(f"unknown signal {signal!r}")
    return S.shift(1)


def trend_book(P: pd.DataFrame, Fnd: pd.DataFrame, logret: pd.DataFrame,
               mode: str = "ts", lookback: int = 90, signal: str = "zscore",
               n_each: int = 8, hold: int = 9, cost_bps: float = 8.0,
               gross: float = 1.0, beta_hedge: bool | None = None,
               hedge_ref: str = "BTCUSDT",
               vol_target: float | None = 0.004, vol_win: int = 30) -> pd.DataFrame:
    """Simulate a trend-following book on the 8h perp panel.

    mode='ts' : weight_i proportional to coin i's own trend signal, normalized to
                gross exposure `gross`. Directional. beta_hedge defaults False.
    mode='xs' : long top-`n_each` / short bottom-`n_each` by trend. Dollar-neutral.
                beta_hedge defaults True (same as the carry book).

    Returns a DataFrame with columns trend / funding / cost / net (per-period P&L).
    """
    if beta_hedge is None:
        beta_hedge = (mode == "xs")

    universe = list(P.columns)
    S = trend_signal(logret, lookback, signal)
    beta = rolling_beta(logret, hedge_ref) if beta_hedge else None
    btc = logret[hedge_ref]
    hedge_fund = Fnd[hedge_ref]
    times = P.index

    cur_w = pd.Series(0.0, index=universe)
    cur_hedge = 0.0
    scale = 1.0
    Tr, Fc, Co = [], [], []
    net_hist: list[float] = []

    for i in range(len(times)):
        # ---- realize P&L from positions held since last rebalance ----
        if cur_w.abs().sum() > 0 and i > 0:
            pr = float((cur_w * logret.iloc[i].reindex(universe).fillna(0)).sum())
            fc = float((-cur_w * Fnd.iloc[i].reindex(universe).fillna(0)).sum())
            hp = cur_hedge * btc.iloc[i] - cur_hedge * hedge_fund.iloc[i]   # hedge price+funding
        else:
            pr = fc = hp = 0.0
        trend_i = scale * (pr + hp)
        fund_i = scale * fc
        cost_i = 0.0

        # ---- rebalance ----
        if i % hold == 0:
            s = S.iloc[i].dropna()
            enough = len(s) >= (2 * n_each if mode == "xs" else 4)
            if enough:
                new_w = pd.Series(0.0, index=universe)
                if mode == "ts":
                    denom = s.abs().sum()
                    if denom > 0:
                        new_w[s.index] = s / denom * gross      # conviction-weighted, gross-normalized
                elif mode == "xs":
                    new_w[s.nlargest(n_each).index] = 1.0 / n_each
                    new_w[s.nsmallest(n_each).index] = -1.0 / n_each
                else:
                    raise ValueError(f"unknown mode {mode!r}")

                if vol_target is not None and len(net_hist) >= vol_win:
                    realized = np.std(net_hist[-vol_win:])
                    scale = min(vol_target / realized, 5.0) if realized > 0 else scale

                turnover = (cur_w - new_w).abs().sum() / 2.0
                cost_i += scale * turnover * cost_bps / 1e4
                cur_w = new_w

                if beta_hedge:
                    b = beta.iloc[i].reindex(universe).fillna(1.0)
                    new_hedge = -float((cur_w * b).sum())
                    cost_i += scale * abs(new_hedge - cur_hedge) * cost_bps / 1e4
                    cur_hedge = new_hedge

        net_i = trend_i + fund_i - cost_i
        Tr.append(trend_i); Fc.append(fund_i); Co.append(cost_i); net_hist.append(net_i)

    df = pd.DataFrame({"trend": Tr, "funding": Fc, "cost": Co}, index=times)
    df["net"] = df["trend"] + df["funding"] - df["cost"]
    return df
