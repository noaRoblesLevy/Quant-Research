"""
Opening-range breakout (ORB) — an event-based session strategy.

Thesis: during a session open (London ~09:00 server / NY ~15:00 server on this
EET broker), real-money order flow drives a volatility expansion. Form a range
over the first `range_hours`, then trade the BREAK of that range — the direction
is set by the break, not predicted in advance. Exit at session close (no overnight
risk). A catastrophic ATR stop caps the downside on a failed break.

This is NOT refuted by the hourly mean-reversion finding: a range breakout is a
conditional, volatility-clustering signal, distinct from unconditional hourly drift.

No look-ahead: the range high/low come only from bars INSIDE the formation window,
and are used only on bars AFTER the window has closed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr
from .base import Strategy


class SessionBreakout(Strategy):
    name = "SessionBreakout"

    def __init__(self, open_hour: int = 9, range_hours: int = 2, close_hour: int = 18,
                 atr_period: int = 14, atr_mult: float = 1.5,
                 allow_long: bool = True, allow_short: bool = True):
        self.open_hour = open_hour          # session start, broker server time
        self.range_hours = range_hours      # bars used to build the opening range
        self.close_hour = close_hour        # flatten everything at/after this hour
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.allow_long = allow_long
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        hour = pd.Series(df.index.hour, index=df.index)
        day = pd.Series(df.index.normalize(), index=df.index)
        a = atr(df, self.atr_period)

        # --- opening range, formed strictly inside [open_hour, open_hour+range_hours) ---
        in_window = (hour >= self.open_hour) & (hour < self.open_hour + self.range_hours)
        hi_in = df["high"].where(in_window)
        lo_in = df["low"].where(in_window)
        range_high = hi_in.groupby(day).transform("max")   # full-window max (known once window closes)
        range_low = lo_in.groupby(day).transform("min")

        # --- only trade AFTER the window has closed and before session close ---
        tradable = (hour >= self.open_hour + self.range_hours) & (hour < self.close_hour)
        broke_up = tradable & (df["close"] > range_high)
        broke_dn = tradable & (df["close"] < range_low)

        signal = pd.Series(0.0, index=df.index)
        if self.allow_long:
            signal[broke_up] = 1.0
        if self.allow_short:
            signal[broke_dn] = -1.0

        # keep only the FIRST breakout each day (one shot per session)
        nz = signal != 0
        order = nz.groupby(day).cumsum()
        signal = signal.where(nz & (order == 1), 0.0)

        # --- exit: force-flatten any open position at/after session close ---
        exit_sig = pd.Series(0.0, index=df.index)
        exit_sig[hour >= self.close_hour] = 2.0   # code 2 = flatten regardless of direction

        out = pd.DataFrame(index=df.index)
        out["signal"] = signal
        out["sl_distance"] = a * self.atr_mult
        out["exit"] = exit_sig
        out.iloc[:self.atr_period, :] = 0.0
        return out
