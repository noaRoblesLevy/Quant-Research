"""Strategy interface.

A strategy's only job is to turn OHLC data into:
    * 'signal'      : +1 long / -1 short / 0 flat  (entry intent, filled next bar open)
    * 'sl_distance' : the stop distance in PRICE units for that entry

The exit ladder (take-profits, trailing, breakeven) lives in ExitConfig and is
handled by the engine — keep that out of the strategy so you can A/B exits
independently of entries (the single most important lesson from the EA work:
exits can't fix a bad entry, so test them separately).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "unnamed"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed like df with columns 'signal' and 'sl_distance'.

        MUST NOT use future information: every value on bar i may only depend on
        data up to and including bar i. The engine fills on bar i+1's open.
        """
        ...

    def params(self) -> dict:
        """Return tunable parameters (for walk-forward logging)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
