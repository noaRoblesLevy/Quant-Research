"""
Download the holdout data for study 4: Binance USD-M 1h klines (with taker-buy volume)
and funding, 2025-06 to 2026-10, for the 23 coins. Writes data_cache/{SYM}_1h.parquet
and data_cache/{SYM}_funding.parquet. Training data (2021-01..2025-05) is read from the
funding-carry-harvest cache.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
FCH = ROOT.parent / "funding-carry-harvest"
sys.path.insert(0, str(FCH))

import pandas as pd

from quantresearch.data.binance_loader import _fetch_month, _to_datetime

UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'NEARUSDT', 'AVAXUSDT',
            'XLMUSDT', 'AAVEUSDT', 'UNIUSDT', 'TRXUSDT', 'ATOMUSDT', 'ETCUSDT', 'SANDUSDT',
            'AXSUSDT', 'COMPUSDT']
CACHE = ROOT / "data_cache"
MONTHS = [p.strftime("%Y-%m") for p in pd.period_range("2025-06", "2026-10", freq="M")]


def klines(sym: str) -> pd.DataFrame:
    frames = [m for ym in MONTHS if (m := _fetch_month(sym, "1h", ym, "um")) is not None]
    df = pd.concat(frames, ignore_index=True)
    df.index = _to_datetime(df["open_time"]); df.index.name = "time"
    df = df[~df.index.duplicated()].sort_index()
    for c in ["close", "volume", "taker_buy_base"]:
        df[c] = df[c].astype(float)
    return df[["close", "volume", "taker_buy_base"]]


def funding(sym: str) -> pd.Series:
    start = int(pd.Timestamp("2025-05-25").timestamp() * 1000)
    end = int(pd.Timestamp("2026-11-01").timestamp() * 1000)
    rows = []
    while start < end:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={start}&limit=1000"
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
            batch = json.load(r)
        if not batch:
            break
        rows += [(b["fundingTime"], float(b["fundingRate"])) for b in batch]
        if len(batch) < 1000:
            break
        start = batch[-1]["fundingTime"] + 1
        time.sleep(0.2)
    s = pd.Series({pd.to_datetime(t, unit="ms"): f for t, f in rows}).sort_index()
    return s[s.index < pd.Timestamp("2026-11-01")]


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    t0 = time.time()
    for i, s in enumerate(UNIVERSE, 1):
        kf, ff = CACHE / f"{s}_1h.parquet", CACHE / f"{s}_funding.parquet"
        if not kf.exists():
            klines(s).to_parquet(kf)
        if not ff.exists():
            funding(s).to_frame("funding").to_parquet(ff)
        k = pd.read_parquet(kf)
        print(f"[{i:>2}/23] {s:<10} {len(k)} bars {k.index.min().date()} -> {k.index.max().date()} "
              f"| {time.time()-t0:.0f}s", flush=True)
    print("done")
