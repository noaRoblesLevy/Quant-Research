"""
Synthetic OHLC generator — lets you exercise the whole harness with NO data and
NO MT5 installed. Used by the self-test and by `run_example.py --synthetic`.

It is a geometric random walk with intrabar high/low noise. There is no edge in
it by construction, so any strategy run on it should come out roughly break-even
minus costs — a good sanity check that the engine isn't inventing P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlc(
    n_bars: int = 20_000,
    start: str = "2018-01-01",
    freq: str = "1h",
    start_price: float = 15000.0,
    vol: float = 0.004,
    spread_price: float = 0.5,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # business-hour-ish continuous index (drop weekends to look like FX/CFD)
    idx = pd.date_range(start=start, periods=n_bars, freq=freq)
    idx = idx[idx.dayofweek < 5]
    n = len(idx)

    rets = rng.normal(0, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    noise = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "tick_volume": rng.integers(50, 500, n),
         "spread_price": np.full(n, spread_price)},
        index=idx,
    )
    return df
