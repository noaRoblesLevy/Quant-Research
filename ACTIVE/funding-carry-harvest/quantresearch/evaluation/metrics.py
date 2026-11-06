"""
Performance metrics computed from a bar-level equity curve and a trade list.

The equity curve is expected to be a pandas Series indexed by datetime, already
including floating P&L (as produced by backtest.engine.run_backtest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer how many bars make up a year, for annualisation."""
    if len(index) < 3:
        return 252.0
    deltas = np.diff(index.view("int64"))  # nanoseconds
    median_ns = np.median(deltas)
    if median_ns <= 0:
        return 252.0
    seconds = median_ns / 1e9
    bars_per_day = (24 * 3600) / seconds
    # forex trades ~5 days/week
    return bars_per_day * 5 * 52


def max_drawdown(equity: pd.Series) -> dict:
    """Peak-to-trough max drawdown on the equity curve (includes floating P&L)."""
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    trough_i = dd.idxmin()
    return {
        "max_dd_pct": float(-dd.min() * 100.0),
        "max_dd_abs": float((running_max - equity).max()),
        "trough_time": trough_i,
    }


def daily_returns(equity: pd.Series) -> pd.Series:
    """End-of-day equity percentage changes."""
    daily = equity.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = daily_returns(equity)
    if r.std() == 0 or len(r) < 2:
        return 0.0
    excess = r - rf / 252
    return float(np.sqrt(252) * excess.mean() / excess.std())


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    r = daily_returns(equity)
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    excess = r - rf / 252
    return float(np.sqrt(252) * excess.mean() / downside.std())


def worst_day(equity: pd.Series) -> float:
    """Largest single-day percentage loss (positive number = loss magnitude)."""
    r = daily_returns(equity)
    return float(-r.min() * 100.0) if len(r) else 0.0


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"n_trades": 0}
    r = trades["r"]
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    return {
        "n_trades": int(len(trades)),
        "win_rate_pct": float(len(wins) / len(trades) * 100.0),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_r": float(r.mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "avg_mae_r": float(trades["mae_r"].mean()),
        "avg_mfe_r": float(trades["mfe_r"].mean()),
        "worst_trade_r": float(r.min()),
        "best_trade_r": float(r.max()),
    }


def summarize(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """One-stop summary dict combining curve and trade metrics."""
    total_return = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)
    dd = max_drawdown(equity)
    out = {
        "total_return_pct": total_return,
        "max_dd_pct": dd["max_dd_pct"],
        "sharpe": sharpe(equity),
        "sortino": sortino(equity),
        "worst_day_pct": worst_day(equity),
        "return_dd_ratio": (total_return / dd["max_dd_pct"]) if dd["max_dd_pct"] > 0 else np.inf,
    }
    out.update(trade_stats(trades))
    return out


def print_summary(equity: pd.Series, trades: pd.DataFrame, title: str = "Backtest summary") -> dict:
    s = summarize(equity, trades)
    print(f"\n=== {title} ===")
    print(f"  Trades            : {s.get('n_trades', 0)}")
    print(f"  Total return      : {s['total_return_pct']:+.2f}%")
    print(f"  Max drawdown      : {s['max_dd_pct']:.2f}%")
    print(f"  Return/DD ratio   : {s['return_dd_ratio']:.2f}")
    print(f"  Sharpe            : {s['sharpe']:.2f}")
    print(f"  Sortino           : {s['sortino']:.2f}")
    print(f"  Worst day         : -{s['worst_day_pct']:.2f}%")
    if s.get("n_trades", 0):
        print(f"  Win rate          : {s['win_rate_pct']:.1f}%")
        print(f"  Expectancy        : {s['expectancy_r']:+.3f} R")
        print(f"  Profit factor     : {s['profit_factor']:.2f}")
        print(f"  Avg MAE / MFE     : {s['avg_mae_r']:.2f} R / {s['avg_mfe_r']:.2f} R")
    return s
