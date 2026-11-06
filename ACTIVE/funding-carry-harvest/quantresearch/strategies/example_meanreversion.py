"""
DEMO strategy — Bollinger-band mean reversion.

This is NOT a recommended edge. It exists to demonstrate the harness end-to-end
and to give you a template for the Strategy interface. Mean reversion is included
because its high-win-rate / tight-stop profile is the kind that tends to survive
FTMO's daily-loss and drawdown rules — but you must validate any real candidate
with out-of-sample data and the FTMO evaluator before believing it.

Logic
-----
* Long  when close pierces BELOW the lower band (oversold), expecting reversion.
* Short when close pierces ABOVE the upper band (overbought).
* Optional trend filter: only take reversion in the direction of a slow EMA, or
  disable it for pure counter-trend reversion.
* Stop = ATR * atr_mult. Exits (TP ladder) are configured separately in ExitConfig.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import adx, atr, bollinger, ema
from .base import Strategy


class BollingerMeanReversion(Strategy):
    name = "BollingerMeanReversion"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_mult: float = 1.5,
        trend_filter_ema: int | None = None,   # None = pure reversion (no trend filter)
        allow_long: bool = True,
        allow_short: bool = True,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.trend_filter_ema = trend_filter_ema
        self.allow_long = allow_long
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        lower, mid, upper = bollinger(close, self.bb_period, self.bb_std)
        a = atr(df, self.atr_period)

        signal = pd.Series(0, index=df.index, dtype=float)

        long_cond = close < lower
        short_cond = close > upper

        if self.trend_filter_ema:
            trend = ema(close, self.trend_filter_ema)
            long_cond &= close > trend     # only buy dips in an uptrend
            short_cond &= close < trend    # only sell rips in a downtrend

        if self.allow_long:
            signal[long_cond] = 1
        if self.allow_short:
            signal[short_cond] = -1

        out = pd.DataFrame(index=df.index)
        out["signal"] = signal
        out["sl_distance"] = a * self.atr_mult
        # don't trade until indicators are warm
        warm = max(self.bb_period, self.atr_period, self.trend_filter_ema or 0)
        out.iloc[:warm, :] = 0.0
        return out


class BollingerReversionExitAtMean(Strategy):
    """Mean reversion whose EXIT matches its thesis: get out when price reverts
    to the mean (the middle band), not at an arbitrary R-multiple.

    Entry : close below lower band -> long ; close above upper band -> short.
    Exit  : close back at/through the middle band -> flatten (rule-based exit).
    Stop  : ATR * atr_mult — a catastrophic backstop only; most trades exit at mean.

    The natural R:R here is < 1 (small target, wider stop) with a HIGH win rate —
    exactly the profile FTMO's daily-loss / drawdown rules reward.
    """
    name = "BollingerReversionExitAtMean"

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 atr_period: int = 14, atr_mult: float = 2.0,
                 allow_long: bool = True, allow_short: bool = True,
                 adx_period: int = 14, adx_max: float | None = None):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.allow_long = allow_long
        self.allow_short = allow_short
        # regime filter: if set, only enter when ADX < adx_max (i.e. ranging, not trending)
        self.adx_period = adx_period
        self.adx_max = adx_max

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        lower, mid, upper = bollinger(close, self.bb_period, self.bb_std)
        a = atr(df, self.atr_period)

        signal = pd.Series(0, index=df.index, dtype=float)
        if self.allow_long:
            signal[close < lower] = 1
        if self.allow_short:
            signal[close > upper] = -1

        # REGIME FILTER: suppress entries while the market is trending (high ADX).
        # Mean reversion bleeds in trends, so we only fade when ADX says "ranging".
        if self.adx_max is not None:
            adx_val = adx(df, self.adx_period)
            signal[adx_val >= self.adx_max] = 0

        exit_sig = pd.Series(0, index=df.index, dtype=float)
        exit_sig[close >= mid] = 1    # reverted up -> exit any long
        exit_sig[close <= mid] = -1   # reverted down -> exit any short

        out = pd.DataFrame(index=df.index)
        out["signal"] = signal
        out["sl_distance"] = a * self.atr_mult
        out["exit"] = exit_sig
        warm = max(self.bb_period, self.atr_period, self.adx_period)
        out.iloc[:warm, :] = 0.0
        return out
