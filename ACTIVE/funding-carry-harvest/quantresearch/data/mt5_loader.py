"""
MetaTrader 5 data loader.

Pulls historical bars straight from your installed MT5 terminal — which means the
data matches the feed you'll actually execute on (same symbols, same spreads).
Caches to parquet so you only download once.

Requires:  pip install MetaTrader5  (Windows only)
The MT5 terminal must be installed and logged in (a demo/FTMO account is fine).

Tip: in MT5, raise Tools > Options > Charts > "Max bars in chart" to a large
value (e.g. unlimited) before downloading long histories.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# string -> MT5 timeframe constant name (resolved lazily so import works without MT5)
_TF_NAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1",
}


def _mt5():
    try:
        import MetaTrader5 as mt5  # noqa: N813
    except ImportError as e:
        raise ImportError(
            "MetaTrader5 package not installed. Run:  pip install MetaTrader5"
        ) from e
    return mt5


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_").replace(".", "_")
    return CACHE_DIR / f"{safe}_{timeframe}.parquet"


def fetch_bars(
    symbol: str,
    timeframe: str = "H1",
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLC bars for one symbol/timeframe.

    Returns a DataFrame indexed by datetime with columns:
        open, high, low, close, tick_volume, spread, spread_price
    'spread_price' is the spread converted to price units (for realistic costs).
    """
    cache = _cache_path(symbol, timeframe)
    if use_cache and not refresh and cache.exists():
        df = pd.read_parquet(cache)
        return _slice(df, start, end)

    mt5 = _mt5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}. "
                           "Is the terminal installed and logged in?")
    try:
        if timeframe not in _TF_NAMES:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Options: {list(_TF_NAMES)}")
        tf = getattr(mt5, _TF_NAMES[timeframe])

        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol '{symbol}' not found in MT5. "
                               "Check Market Watch / the exact broker symbol name.")
        if not info.visible:
            mt5.symbol_select(symbol, True)
        point = info.point

        # MT5's copy_rates_range is finicky: it needs NAIVE datetimes (tz-aware ->
        # -2 Invalid params), it errors when the end is "now" with a live time, and
        # it rejects very large single requests. Fix: normalise to clean midnight
        # boundaries and fetch in ~180-day chunks, then stitch. Every chunk request
        # is small and reliable.
        def _naive(ts):
            ts = pd.Timestamp(ts)
            ts = ts.tz_localize(None) if ts.tzinfo else ts
            return ts.normalize()

        start_ts = _naive(start)
        end_ts = _naive(end) if end else pd.Timestamp.now().normalize()

        frames = []
        cur = start_ts
        step = pd.Timedelta(days=180)
        while cur < end_ts:
            nxt = min(cur + step, end_ts)
            chunk = mt5.copy_rates_range(symbol, tf, cur.to_pydatetime(), nxt.to_pydatetime())
            if chunk is not None and len(chunk):
                frames.append(pd.DataFrame(chunk))
            cur = nxt

        if not frames:
            raise RuntimeError(f"No data returned for {symbol} {timeframe}: {mt5.last_error()}")

        df = pd.concat(frames, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        # consecutive chunks share a boundary bar -> drop duplicates
        df = df.drop_duplicates(subset="time").set_index("time").sort_index()
        df["spread_price"] = df["spread"].astype(float) * point
        keep = ["open", "high", "low", "close", "tick_volume", "spread", "spread_price"]
        df = df[[c for c in keep if c in df.columns]]
    finally:
        mt5.shutdown()

    if use_cache:
        df.to_parquet(cache)
    return _slice(df, start, end)


def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def list_symbols(pattern: str = "") -> list[str]:
    """List available MT5 symbols (optionally filter by substring)."""
    mt5 = _mt5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    try:
        syms = [s.name for s in mt5.symbols_get()]
    finally:
        mt5.shutdown()
    return [s for s in syms if pattern.lower() in s.lower()] if pattern else syms
