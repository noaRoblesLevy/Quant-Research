"""
Portfolio runner — combine many single-market backtests into one account.

Model (deliberately simple and honest):
  * Each market trades with FIXED-fractional risk on the INITIAL account
    (compounding off), so per-market P&L streams are additive.
  * Portfolio equity(t) = account + sum over markets of (market_equity(t) - account).
  * Positions across markets run concurrently — so the WORST case is every market
    in a losing trade at once (= n_markets * per_market_risk). Diversification means
    that rarely happens, but the portfolio equity curve shows the REAL aggregate
    drawdown empirically, which is the whole point.

This is the right structure for FTMO: the firm's rules apply to total account
equity, and that's exactly what `equity` here represents (incl. floating P&L,
since each market's curve already includes it).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .backtest import BacktestConfig, ExitConfig, run_backtest


def run_portfolio(
    data: dict[str, pd.DataFrame],
    make_signals: Callable[[pd.DataFrame], pd.DataFrame],
    exit_cfg: ExitConfig,
    account: float = 100_000.0,
    per_market_risk: float = 0.0025,
) -> dict:
    """Run `make_signals` on every market and combine into one equity curve.

    data            : {symbol: ohlc_df}  (already at the desired timeframe)
    make_signals    : df -> signals df (so the same strategy is applied to each)
    per_market_risk : risk per trade as a fraction of the account (fixed).
    """
    pnl_frames = []
    all_trades = []

    for sym, d in data.items():
        sig = make_signals(d)
        cfg = BacktestConfig(risk_pct=per_market_risk, initial_equity=account, compounding=False)
        res = run_backtest(d, sig, exit_cfg, cfg)
        pnl = (res["equity"] - account).rename(sym)
        pnl_frames.append(pnl)
        t = res["trades"]
        if len(t):
            all_trades.append(t.assign(market=sym))

    # align all markets on a common index; forward-fill P&L between a market's bars
    pnl_df = pd.concat(pnl_frames, axis=1).sort_index().ffill().fillna(0.0)
    equity = (account + pnl_df.sum(axis=1)).rename("equity")

    trades = (pd.concat(all_trades, ignore_index=True).sort_values("entry_time")
              if all_trades else pd.DataFrame())

    return {"equity": equity, "trades": trades, "per_market_pnl": pnl_df, "account": account}
