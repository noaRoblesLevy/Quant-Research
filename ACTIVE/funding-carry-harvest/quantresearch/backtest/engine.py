"""
Event-driven backtest engine (bar resolution).

Design goals
------------
* No look-ahead: a signal generated on bar `i` is filled at the OPEN of bar `i+1`.
* Realistic costs: spread + commission charged per round trip.
* Risk-based sizing: every trade risks a fixed fraction of equity (1R = risk_pct).
* Bar-level equity curve INCLUDING floating P&L of the open position — this is
  what the FTMO evaluator needs (daily-loss and drawdown rules look at equity,
  not just closed trades).
* Scale-out exits (a TP ladder in R-multiples), optional move-to-breakeven and
  trailing for the runner.

Accounting is driven by R-multiples, which keeps everything instrument-agnostic
and lines up with the percentage-based FTMO rules.

Conservative assumptions (documented, so you trust the numbers)
* If a bar's range touches BOTH a stop and a take-profit, the STOP is assumed to
  fill first. This understates results rather than flattering them.
* Gaps through the stop fill at the bar OPEN (worse than the stop price).
* trade.r is GROSS of entry costs; the equity curve / balance is NET of costs.
  (Use the equity curve for returns; use trade.r for the R-distribution.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExitConfig:
    """How a position is unwound.

    tp_ladder : list of (R_multiple, fraction) of the ORIGINAL position. Sum may
        be < 1.0; the remainder is the runner.
        Example: [(1.5, 0.4), (3.0, 0.2), (4.0, 0.2)] -> 0.2 runner.
    move_be_after : after the Nth take-profit fills (1-based), move the remaining
        stop to entry. None disables.
    trail_after   : after the Nth take-profit fills, start trailing the runner.
    trail_dist_r  : trailing distance in R-multiples.
    """
    tp_ladder: list = field(default_factory=lambda: [(1.5, 1.0)])
    move_be_after: Optional[int] = None
    trail_after: Optional[int] = None
    trail_dist_r: float = 1.0


@dataclass
class BacktestConfig:
    risk_pct: float = 0.005           # fraction of equity risked per trade (= 1R)
    initial_equity: float = 100_000.0
    commission_price: float = 0.0     # commission per round trip, in PRICE units
    fixed_spread_price: float = 0.0   # used if df has no 'spread_price' column
    compounding: bool = True          # risk_pct of current equity (True) or initial (False)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    r: float
    bars_held: int
    mae_r: float
    mfe_r: float
    reason: str


def _spread_series(df: pd.DataFrame, cfg: BacktestConfig) -> np.ndarray:
    if "spread_price" in df.columns:
        return df["spread_price"].to_numpy(dtype=float)
    return np.full(len(df), cfg.fixed_spread_price, dtype=float)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    exit_cfg: ExitConfig,
    cfg: BacktestConfig,
) -> dict:
    """Run a single-position backtest.

    df      : OHLC, DatetimeIndex, columns open/high/low/close (+ optional 'spread_price').
    signals : same index; columns 'signal' (+1/-1/0) and 'sl_distance' (price units).
    Returns : {'equity': Series, 'trades': DataFrame, 'config': {...}}.
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)

    sig = signals["signal"].to_numpy(float)
    sld = signals["sl_distance"].to_numpy(float)
    # optional rule-based exit: +1 = flatten an open long, -1 = flatten an open short
    exit_arr = (signals["exit"].to_numpy(float) if "exit" in signals.columns
                else np.zeros(n, dtype=float))
    spread = _spread_series(df, cfg)

    ladder = [list(t) for t in exit_cfg.tp_ladder]

    balance = cfg.initial_equity
    equity_curve = np.empty(n, dtype=float)
    trades: list[Trade] = []

    in_pos = False
    pos_dir = 0
    entry_price = 0.0
    r_price = 0.0
    units = 0.0
    sl_price = 0.0
    remaining = 0.0
    tp_hits = 0
    tps: list[list] = []
    realized_money = 0.0
    entry_i = 0
    mae_r = 0.0
    mfe_r = 0.0
    trail_active = False
    pending = 0
    pending_sl = 0.0
    pending_exit = False

    for i in range(n):
        floating = 0.0

        # ---------- 1. manage an open position on this bar ----------
        if in_pos:
            if pos_dir > 0:
                mae_r = min(mae_r, (l[i] - entry_price) / r_price)
                mfe_r = max(mfe_r, (h[i] - entry_price) / r_price)
            else:
                mae_r = min(mae_r, (entry_price - h[i]) / r_price)
                mfe_r = max(mfe_r, (entry_price - l[i]) / r_price)

            # rule-based exit scheduled on the previous bar -> fill at this bar's open
            if pending_exit:
                total = realized_money + remaining * units * (o[i] - entry_price) * pos_dir
                balance += total
                r_net = total / (units * r_price) if (units * r_price) else 0.0
                tag = "rule_exit" if tp_hits == 0 else "ladder+rule_exit"
                trades.append(Trade(idx[entry_i], idx[i], pos_dir, entry_price,
                                    o[i], r_net, i - entry_i, mae_r, mfe_r, tag))
                in_pos = False
                pending_exit = False

            if in_pos:
                closed = False
                exit_price = None
                reason = ""

                # stop first (conservative); gap-through fills at open
                if pos_dir > 0:
                    if o[i] <= sl_price:
                        exit_price, reason, closed = o[i], "gap_stop", True
                    elif l[i] <= sl_price:
                        exit_price, reason, closed = sl_price, "stop", True
                else:
                    if o[i] >= sl_price:
                        exit_price, reason, closed = o[i], "gap_stop", True
                    elif h[i] >= sl_price:
                        exit_price, reason, closed = sl_price, "stop", True

                if not closed:
                    # take-profit ladder
                    for tp in tps:
                        price, frac, hit = tp
                        if hit:
                            continue
                        reached = (h[i] >= price) if pos_dir > 0 else (l[i] <= price)
                        if reached:
                            realized_money += frac * units * (price - entry_price) * pos_dir
                            remaining -= frac
                            tp[2] = True
                            tp_hits += 1
                            if exit_cfg.move_be_after and tp_hits >= exit_cfg.move_be_after:
                                sl_price = entry_price
                            if exit_cfg.trail_after and tp_hits >= exit_cfg.trail_after:
                                trail_active = True

                    if remaining <= 1e-9:
                        balance += realized_money
                        r_net = realized_money / (units * r_price) if (units * r_price) else 0.0
                        trades.append(Trade(idx[entry_i], idx[i], pos_dir, entry_price,
                                            tps[-1][0], r_net, i - entry_i, mae_r, mfe_r, "ladder_full"))
                        in_pos = False
                        closed = True

                if closed and in_pos:
                    total = realized_money + remaining * units * (exit_price - entry_price) * pos_dir
                    balance += total
                    r_net = total / (units * r_price) if (units * r_price) else 0.0
                    tag = reason if tp_hits == 0 else f"ladder+{reason}"
                    trades.append(Trade(idx[entry_i], idx[i], pos_dir, entry_price,
                                        exit_price, r_net, i - entry_i, mae_r, mfe_r, tag))
                    in_pos = False

                if in_pos and trail_active:
                    td = exit_cfg.trail_dist_r * r_price
                    if pos_dir > 0:
                        sl_price = max(sl_price, c[i] - td)
                    else:
                        sl_price = min(sl_price, c[i] + td)

                # schedule a rule-based exit for next bar's open.
                #   exit == +1/-1 : flatten only if it matches our direction
                #   |exit| >= 2    : force-flatten ANY open position (e.g. session close)
                if in_pos and exit_arr[i] != 0 and (
                        int(np.sign(exit_arr[i])) == pos_dir or abs(exit_arr[i]) >= 2):
                    pending_exit = True

                if in_pos:
                    floating = remaining * units * (c[i] - entry_price) * pos_dir

        # ---------- 2. execute a pending entry (from previous bar) ----------
        if (not in_pos) and pending != 0:
            if pending_sl > 0:
                entry_price = o[i]
                pos_dir = pending
                r_price = pending_sl
                r_money = cfg.risk_pct * (balance if cfg.compounding else cfg.initial_equity)
                units = r_money / r_price
                balance -= (spread[i] + cfg.commission_price) * units  # round-trip cost
                sl_price = entry_price - r_price * pos_dir
                remaining = 1.0
                tp_hits = 0
                realized_money = 0.0
                trail_active = False
                pending_exit = False
                mae_r = 0.0
                mfe_r = 0.0
                entry_i = i
                tps = [[entry_price + R * r_price * pos_dir, frac, False] for R, frac in ladder]
                in_pos = True
                floating = remaining * units * (c[i] - entry_price) * pos_dir
            pending = 0

        # ---------- 3. register a new pending entry from this bar's signal ----------
        if (not in_pos) and pending == 0 and sig[i] != 0 and not np.isnan(sld[i]) and sld[i] > 0:
            pending = int(np.sign(sig[i]))
            pending_sl = float(sld[i])

        equity_curve[i] = balance + floating

    eq = pd.Series(equity_curve, index=idx, name="equity")
    tdf = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price",
                 "r", "bars_held", "mae_r", "mfe_r", "reason"])
    return {"equity": eq, "trades": tdf, "config": {"backtest": cfg, "exit": exit_cfg}}
