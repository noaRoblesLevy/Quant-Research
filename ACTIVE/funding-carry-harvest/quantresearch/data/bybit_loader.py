"""
Free Bybit V5 public market data (no API key) — for validating the funding-carry
edge on the REAL venue Crypto Fund Trader settles against.

CFT confirmed (2026-06-06) that Bybit-based accounts mirror Bybit's live funding
rates. We backtested Binance funding; this loader re-pulls Bybit funding + price so
we can confirm the edge survives the transfer before risking a challenge fee.

Endpoints (public, keyless):
  funding : GET /v5/market/funding/history?category=linear&symbol=&limit=200
  kline   : GET /v5/market/kline?category=linear&symbol=&interval=60&limit=1000
Both return newest-first; we paginate backward with the time cursor.

    from quantresearch.data.bybit_loader import load_klines, load_funding, load_bybit_panels
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
BASE = "https://api.bybit.com/v5/market"
_UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str) -> dict:
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=_UA), timeout=40) as r:
                d = json.load(r)
            if d.get("retCode") == 0:
                return d
            raise RuntimeError(f"bybit retCode {d.get('retCode')}: {d.get('retMsg')}")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.0 + attempt)
    return {}


def load_klines(symbol: str, interval: str = "60", start: str = "2022-07",
                end: str = "2025-05", refresh: bool = False) -> pd.DataFrame:
    """1h (interval='60') OHLCV from Bybit linear perps, cached to parquet."""
    out = CACHE / f"{symbol}_bybit_{interval}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int((pd.Timestamp(end) + pd.offsets.MonthEnd(1)).timestamp() * 1000)
    rows: list = []
    cursor = end_ms
    while cursor > start_ms:
        url = (f"{BASE}/kline?category=linear&symbol={symbol}"
               f"&interval={interval}&end={cursor}&limit=1000")
        lst = _get(url)["result"]["list"]
        if not lst:
            break
        rows += lst
        oldest = int(lst[-1][0])
        if oldest <= start_ms or oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(0.12)

    if not rows:
        raise RuntimeError(f"no Bybit klines for {symbol}")
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "turnover"])
    df.index = pd.to_datetime(df["t"].astype("int64"), unit="ms")
    df.index.name = "time"
    df = df[~df.index.duplicated()].sort_index()
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = df[c].astype(float)
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end) + pd.offsets.MonthEnd(1))]
    df = df[["open", "high", "low", "close", "volume"]]
    df.to_parquet(out)
    return df


def load_funding(symbol: str, start: str = "2022-07", end: str = "2025-05",
                 refresh: bool = False) -> pd.Series:
    """8-hourly funding-rate history from Bybit, cached. Returns a Series."""
    out = CACHE / f"{symbol}_bybit_funding.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)["funding"]

    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int((pd.Timestamp(end) + pd.offsets.MonthEnd(1)).timestamp() * 1000)
    rows: list = []
    cursor = end_ms
    while cursor > start_ms:
        url = (f"{BASE}/funding/history?category=linear&symbol={symbol}"
               f"&endTime={cursor}&limit=200")
        lst = _get(url)["result"]["list"]
        if not lst:
            break
        rows += [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in lst]
        oldest = int(lst[-1]["fundingRateTimestamp"])
        if oldest <= start_ms or oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(0.12)

    if not rows:
        raise RuntimeError(f"no Bybit funding for {symbol}")
    s = pd.DataFrame(rows, columns=["t", "funding"]).drop_duplicates("t")
    s.index = pd.to_datetime(s["t"], unit="ms")
    s.index.name = "time"
    s = s["funding"].sort_index()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end) + pd.offsets.MonthEnd(1))]
    s.to_frame().to_parquet(out)
    return s


def load_bybit_panels(universe: list[str], start: str = "2022-07", end: str = "2025-05"):
    """Return (price_8h, funding_8h, logret_8h) on the 8h grid — same shape as the
    Binance panels in funding_harvest.load_panels, but from real Bybit data.

    Funding is SUMMED within each 8h bucket (robust to coins that fund more often
    than 8h), representing total funding paid per 8h window."""
    P = pd.DataFrame({s: load_klines(s, "60", start, end)["close"].resample("8h").last()
                      for s in universe}).sort_index()

    f8 = {}
    for s in universe:
        f = load_funding(s, start, end)
        f.index = f.index.floor("h")
        f8[s] = f.groupby(f.index.floor("8h")).sum()      # total funding per 8h
    Fnd = pd.DataFrame(f8).reindex(P.index)

    common = P.dropna(how="all").index.intersection(Fnd.dropna(how="all").index)
    P, Fnd = P.loc[common], Fnd.loc[common]
    logret = np.log(P).diff()
    return P, Fnd, logret
