"""
Free historical crypto data from Binance Vision (https://data.binance.vision).

No API key, no rate limit, no cost. The public dumps include the one field that
retail MT5 CFD feeds throw away: **taker-buy volume**, i.e. how much of each bar's
volume was aggressive buying (lifting the offer) vs aggressive selling (hitting the
bid). That lets us reconstruct *signed order flow* historically -- the raw material
for microstructure edges that simply don't exist in OHLCV.

Kline CSV layout (12 cols, no header in the zips):
    0 open_time  1 open  2 high  3 low  4 close  5 volume  6 close_time
    7 quote_volume  8 trades  9 taker_buy_base  10 taker_buy_quote  11 ignore

Markets:
    'um'   USD-margined perpetual futures (what a crypto prop firm trades) <- default
    'spot' spot

    from quantresearch.data.binance_loader import load_klines
    df = load_klines("BTCUSDT", "5m", "2022-01", "2024-12")   # -> parquet-cached frame
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
BASE = "https://data.binance.vision/data"
FAPI = "https://fapi.binance.com/fapi/v1"
_UA = {"User-Agent": "Mozilla/5.0"}

# Binance kline csv columns
_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _months(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    rng = pd.period_range(start=start, end=end, freq="M")
    return [p.strftime("%Y-%m") for p in rng]


def _to_datetime(open_time: pd.Series) -> pd.DatetimeIndex:
    """Binance switched ms -> us timestamps in 2025 (spot dumps from January 2025).
    Detect the unit per row, so multi-year frames that mix both are converted correctly;
    an all-ms frame (e.g. USD-M futures) gives exactly the same result as before."""
    v = open_time.astype("int64")
    ms = v.where(v <= 1e15, v // 1000)
    return pd.to_datetime(ms, unit="ms")


def _fetch_month(symbol: str, interval: str, ym: str, market: str) -> pd.DataFrame | None:
    seg = "futures/um" if market == "um" else "spot"
    fname = f"{symbol}-{interval}-{ym}.zip"
    url = f"{BASE}/{seg}/monthly/klines/{symbol}/{interval}/{fname}"
    try:
        with urlopen(Request(url, headers=_UA), timeout=60) as r:
            blob = r.read()
    except HTTPError as e:
        if e.code == 404:
            return None          # month not published (too early / current month)
        raise
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    # some recent files carry a header row; sniff the first byte
    header = 0 if raw[:9] == b"open_time" else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=_COLS)
    return df


def load_klines(symbol: str = "BTCUSDT", interval: str = "5m",
                start: str = "2021-01", end: str = "2024-12",
                market: str = "um", refresh: bool = False) -> pd.DataFrame:
    """Download (cached) monthly klines and return a clean OHLCV+flow frame.

    Adds derived order-flow columns:
        taker_buy_ratio : taker_buy_base / volume            in [0, 1]
        ofi             : (2*ratio - 1)  (signed flow imbalance, in [-1, 1])
    """
    tag = f"{symbol}_{interval}_{market}"
    out = CACHE / f"{tag}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    frames = []
    for ym in _months(start, end):
        m = _fetch_month(symbol, interval, ym, market)
        if m is not None and len(m):
            frames.append(m)
    if not frames:
        raise RuntimeError(f"no data for {symbol} {interval} {start}..{end} ({market})")

    df = pd.concat(frames, ignore_index=True)
    df.index = _to_datetime(df["open_time"])
    df.index.name = "time"
    df = df[~df.index.duplicated(keep="first")].sort_index()

    num = ["open", "high", "low", "close", "volume", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote"]
    df[num] = df[num].astype(float)

    vol = df["volume"].replace(0, np.nan)
    df["taker_buy_ratio"] = (df["taker_buy_base"] / vol).clip(0, 1)
    df["ofi"] = 2 * df["taker_buy_ratio"] - 1            # +1 all buys, -1 all sells

    # engine expects a 'tick_volume' column; map trade-count onto it
    df["tick_volume"] = df["trades"]
    keep = ["open", "high", "low", "close", "tick_volume", "volume",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_ratio", "ofi"]
    df = df[keep].dropna(subset=["open", "high", "low", "close"])

    df.to_parquet(out)
    return df


def load_funding(symbol: str = "BTCUSDT", start: str = "2021-01",
                 refresh: bool = False) -> pd.Series:
    """Perp funding-rate history (8-hourly) from the free keyless fapi endpoint.

    Funding is a real cash flow: when positive, longs pay shorts. Extreme funding
    flags crowded positioning -> a documented contrarian signal that is tradeable
    one-sided on a perp account (no spot leg needed).

    Returns a Series of funding rate (fraction per 8h) indexed by fundingTime.
    """
    out = CACHE / f"{symbol}_funding.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)["funding"]

    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    rows: list[tuple] = []
    while True:
        url = f"{FAPI}/fundingRate?symbol={symbol}&startTime={start_ms}&limit=1000"
        with urlopen(Request(url, headers=_UA), timeout=60) as r:
            batch = json.load(r)
        if not batch:
            break
        rows += [(b["fundingTime"], float(b["fundingRate"])) for b in batch]
        if len(batch) < 1000:
            break
        start_ms = batch[-1]["fundingTime"] + 1
        time.sleep(0.2)                       # be polite to the public endpoint

    s = pd.DataFrame(rows, columns=["t", "funding"])
    s.index = pd.to_datetime(s["t"], unit="ms")
    s.index.name = "time"
    s = s["funding"]
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s.to_frame().to_parquet(out)
    return s


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    itv = sys.argv[2] if len(sys.argv) > 2 else "5m"
    s = sys.argv[3] if len(sys.argv) > 3 else "2021-01"
    e = sys.argv[4] if len(sys.argv) > 4 else "2024-12"
    d = load_klines(sym, itv, s, e, refresh=True)
    print(f"{sym} {itv}: {len(d):,} bars  {d.index.min()} -> {d.index.max()}")
    print(d[["close", "taker_buy_ratio", "ofi"]].describe().round(4))
