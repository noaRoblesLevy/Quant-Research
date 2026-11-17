"""
Download the extra data needed for the 2026-11-07 improvement study (docs/).

    python scripts/fetch_improvement_data.py

Writes only NEW cache files, never touches the existing backtest caches:
    data_cache/{SYM}_1h_spot.parquet                 Binance spot 1h, 2021-01..2025-05
    data_cache/improve/{SYM}_bybit_spot_60.parquet   Bybit spot 1h, 2022-07..2025-05
    data_cache/improve/live/{SYM}_perp.parquet       Bybit perp 1h, live period 2026
    data_cache/improve/live/{SYM}_spot.parquet       Bybit spot 1h, live period 2026
    data_cache/improve/live/{SYM}_funding.parquet    Bybit funding, live period 2026
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from quantresearch.data.binance_loader import load_klines as binance_klines
from quantresearch.data.bybit_loader import _get, BASE

UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'NEARUSDT', 'AVAXUSDT',
            'XLMUSDT', 'AAVEUSDT', 'UNIUSDT', 'TRXUSDT', 'ATOMUSDT', 'ETCUSDT', 'SANDUSDT',
            'AXSUSDT', 'COMPUSDT']
CACHE = ROOT / "data_cache"
IMP = CACHE / "improve"
LIVE = IMP / "live"
LIVE_START, LIVE_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-11-07 08:00")


def bybit_klines(symbol: str, category: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Hourly closes (index = open time, UTC naive) from Bybit, paging backward."""
    rows, cursor, s_ms = {}, int(end.value // 10**6), int(start.value // 10**6)
    while cursor > s_ms:
        url = (f"{BASE}/kline?category={category}&symbol={symbol}&interval=60"
               f"&end={cursor}&limit=1000")
        lst = _get(url)["result"]["list"]
        if not lst:
            break
        for x in lst:
            t = int(x[0])
            if s_ms <= t <= int(end.value // 10**6):
                rows[t] = float(x[4])
        oldest = int(lst[-1][0])
        if oldest <= s_ms or oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(0.12)
    s = pd.Series(rows, dtype=float)
    s.index = pd.to_datetime(s.index, unit="ms")
    return s.sort_index()


def bybit_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    rows, cursor, s_ms = {}, int(end.value // 10**6), int(start.value // 10**6)
    while True:
        url = f"{BASE}/funding/history?category=linear&symbol={symbol}&endTime={cursor}&limit=200"
        lst = _get(url)["result"]["list"]
        if not lst:
            break
        for x in lst:
            t = int(x["fundingRateTimestamp"])
            if s_ms < t <= int(end.value // 10**6):
                rows[t] = float(x["fundingRate"])
        oldest = int(lst[-1]["fundingRateTimestamp"])
        if oldest <= s_ms or len(lst) < 200 or oldest >= cursor:
            break
        cursor = oldest - 1
        time.sleep(0.12)
    s = pd.Series(rows, dtype=float)
    s.index = pd.to_datetime(s.index, unit="ms")
    return s.sort_index()


def save(s: pd.Series, path: Path, col: str):
    s.to_frame(col).to_parquet(path)


if __name__ == "__main__":
    IMP.mkdir(exist_ok=True); LIVE.mkdir(exist_ok=True)
    t0 = time.time()
    for i, sym in enumerate(UNIVERSE, 1):
        # Binance spot (monthly dumps; cached by the loader itself)
        try:
            df = binance_klines(sym, "1h", "2021-01", "2025-05", market="spot")
            b = f"binance spot {len(df):>6}"
        except Exception as e:                       # noqa: BLE001
            b = f"binance spot FAILED ({e})"
        # Bybit spot, backtest range
        p = IMP / f"{sym}_bybit_spot_60.parquet"
        if not p.exists():
            save(bybit_klines(sym, "spot", pd.Timestamp("2022-07-01"),
                              pd.Timestamp("2025-06-01")), p, "close")
        # live period: perp, spot, funding
        for kind in ("perp", "spot"):
            q = LIVE / f"{sym}_{kind}.parquet"
            if not q.exists():
                save(bybit_klines(sym, "linear" if kind == "perp" else "spot",
                                  LIVE_START, LIVE_END), q, "close")
        q = LIVE / f"{sym}_funding.parquet"
        if not q.exists():
            save(bybit_funding(sym, LIVE_START - pd.Timedelta(days=1), LIVE_END), q, "funding")
        n_by = len(pd.read_parquet(p))
        n_lv = len(pd.read_parquet(LIVE / f"{sym}_spot.parquet"))
        print(f"[{i:>2}/23] {sym:<10} {b} | bybit spot {n_by:>6} | live spot {n_lv:>5} "
              f"| {time.time()-t0:5.0f}s", flush=True)
    print("done")
