"""
Cross-sectional crypto ranking engine.

Signals (all computed PER COIN then ranked cross-sectionally):
  1. Short-term reversal  : -1 * 5-day log return (contrarian, mean-reverts)
  2. Funding contrarian   : -1 * z-scored 3-period smoothed funding (high funding = crowded longs)
  3. Intermediate momentum: 20-day log return (trend continuation, different horizon)

Composite = equal-weight z-score average.  Rank coins each bar; long top N / short bottom N.

Usage
-----
    from quantresearch.crypto_xs import build_features, xs_portfolio_returns
    feats = build_features(universe_map, funding_map)
    rets, stats = xs_portfolio_returns(feats, n_each=5, hold_bars=24)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zrank(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score of ranks (robust to outliers)."""
    r = s.rank()
    return (r - r.mean()) / r.std().clip(lower=1e-8)


def build_features(
    klines: dict[str, pd.DataFrame],
    funding: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    klines  : {symbol: DataFrame with 'close', 'ofi' columns, hourly DatetimeIndex}
    funding : {symbol: Series of funding rate every 8h}

    Returns a MultiIndex DataFrame (time × symbol) with columns:
        close, log_ret, reversal, momentum, fund_signal, composite, in_universe
    """
    syms = sorted(set(klines) & set(funding))

    # ---- 1. hourly close panel ------------------------------------------------
    close = pd.DataFrame({s: klines[s]["close"] for s in syms}).sort_index()

    # ---- 2. per-coin signals --------------------------------------------------
    log_ret = np.log(close).diff()

    # short-term reversal: negative of 5-bar (5h) return -> expect reversion
    reversal = -log_ret.rolling(5).sum()

    # intermediate momentum: 20-day (480 bars) return, skip last 1h to avoid overlap
    momentum = np.log(close.shift(1) / close.shift(481))

    # ---- 3. funding rate -> hourly panel (forward-fill the 8h grid) ----------
    fund_panel = pd.DataFrame(index=close.index, columns=syms, dtype=float)
    for s in syms:
        f = funding[s]
        if isinstance(f, pd.DataFrame):
            f = f.iloc[:, 0]
        f = f.reindex(close.index, method="ffill")
        fund_panel[s] = f

    # funding contrarian: negative of ewm-smoothed funding
    # ewm across time (row-wise), column by column — avoids deprecated axis kwarg
    fund_signal = -fund_panel.apply(lambda col: col.ewm(span=3).mean())

    # ---- 4. cross-sectional composite — fully vectorized ----------------------
    # For each signal: rank coins cross-sectionally at each bar, then z-score the ranks.
    def xs_zrank(df: pd.DataFrame) -> pd.DataFrame:
        """Rank across columns (coins) at each row (bar), then z-score the ranks."""
        ranks = df.rank(axis=1)
        mu = ranks.mean(axis=1)
        sd = ranks.std(axis=1).clip(lower=1e-8)
        return ranks.sub(mu, axis=0).div(sd, axis=0)

    zr_rev  = xs_zrank(reversal)
    zr_mom  = xs_zrank(momentum)
    zr_fund = xs_zrank(fund_signal)
    composite = (zr_rev + zr_mom + zr_fund) / 3.0

    # ---- 5. stack into long MultiIndex frame ---------------------------------
    frames = {
        "close":      close,
        "log_ret":    log_ret,
        "reversal":   reversal,
        "momentum":   momentum,
        "fund_signal": fund_signal,
        "composite":  composite,
    }
    out = pd.concat({k: v.stack() for k, v in frames.items()}, axis=1)
    out.index.names = ["time", "symbol"]
    out["in_universe"] = True
    return out.sort_index()


def xs_portfolio_returns(
    features: pd.DataFrame,
    n_each: int = 5,
    hold_bars: int = 24,
    spread_bps: float = 10.0,
    signal_col: str = "composite",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Long-top-N / short-bottom-N portfolio rebalanced every `hold_bars`.

    - Signal at bar t fills at bar t+1 (no lookahead).
    - Costs: spread_bps * |Δweight| charged on actual turnover at each rebalance.
      (Cost is proportional to CHANGED notional, not total positions — sticky
      momentum portfolios pay less than fully-rotating ones.)
    - Returns daily P&L Series + stats DataFrame.
    """
    sig_mat = features[signal_col].unstack("symbol")
    ret_mat = features["log_ret"].unstack("symbol")
    syms = sig_mat.columns
    times = sig_mat.index
    n = len(times)
    cost = spread_bps / 1e4   # round-trip cost as a fraction

    pnl_bar = pd.Series(0.0, index=times)
    current_w = pd.Series(0.0, index=syms)
    rebal_count = 0

    for i in range(n):
        # P&L earned on this bar using weights set at PREVIOUS rebalance
        if current_w.abs().sum() > 0:
            ret_row = ret_mat.iloc[i].reindex(syms).fillna(0.0)
            pnl_bar.iloc[i] = float((current_w * ret_row).sum())

        # rebalance at this bar — new weights apply from next bar onwards
        if i % hold_bars == 0:
            sig_row = sig_mat.iloc[i].dropna()
            if len(sig_row) >= 2 * n_each:
                new_w = pd.Series(0.0, index=syms)
                new_w[sig_row.nlargest(n_each).index]  =  1.0 / n_each
                new_w[sig_row.nsmallest(n_each).index] = -1.0 / n_each
                # cost = half-spread * changed notional (actual turnover)
                turnover = (current_w - new_w).abs().sum() / 2.0
                pnl_bar.iloc[i] -= turnover * cost
                current_w = new_w
                rebal_count += 1

    daily = pnl_bar.resample("D").sum()
    stats_df = pd.DataFrame({
        "rebalances": [rebal_count],
        "avg_positions": [2 * n_each],
        "spread_bps": [spread_bps],
        "hold_bars": [hold_bars],
    })
    return daily, stats_df
