"""
Data for study 3 (docs/preregistration_study3_2026-11-07.md).

    python study3_data.py

Writes data_cache/fx.parquet (USD value of 1 unit of each currency), data_cache/
policy_rates.parquet (BIS, monthly, %), data_cache/crypto.parquet (BTC, ETH) and
data_cache/cc_daily.parquet (Bybit cash-and-carry CC L=21 K=7, daily net return per unit
of spot notional, before subtracting the T-bill rate).
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
FCH = ROOT.parent / "funding-carry-harvest"
sys.path.insert(0, str(FCH)); sys.path.insert(0, str(FCH / "scripts"))
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

CACHE = ROOT / "data_cache"
CCY = {"EUR": ("EURUSD=X", False, "XM"), "JPY": ("JPY=X", True, "JP"),
       "GBP": ("GBPUSD=X", False, "GB"), "CHF": ("CHF=X", True, "CH"),
       "AUD": ("AUDUSD=X", False, "AU"), "NZD": ("NZDUSD=X", False, "NZ"),
       "CAD": ("CAD=X", True, "CA"), "SEK": ("SEK=X", True, "SE"),
       "NOK": ("NOK=X", True, "NO")}          # (ticker, quoted as FCY per USD?, BIS code)


def fx_and_rates():
    import yfinance as yf
    tick = [v[0] for v in CCY.values()]
    raw = yf.download(tick, start="2003-01-01", end="2026-11-07", auto_adjust=True,
                      progress=False)["Close"]
    fx = pd.DataFrame({c: (1.0 / raw[t]) if inv else raw[t] for c, (t, inv, _) in CCY.items()})
    fx.to_parquet(CACHE / "fx.parquet")
    rates = {}
    for c, (_, _, code) in list(CCY.items()) + [("USD", (None, None, "US"))]:
        txt = urlopen(Request(f"https://stats.bis.org/api/v1/data/WS_CBPOL/M.{code}?format=csv",
                              headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
        df = pd.read_csv(io.BytesIO(txt))
        s = df.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float)
        s.index = pd.PeriodIndex(s.index, freq="M").to_timestamp("M")
        rates[c] = s
    pd.DataFrame(rates).to_parquet(CACHE / "policy_rates.parquet")
    print(f"fx {fx.shape}, rates {pd.DataFrame(rates).shape}")


def crypto():
    import yfinance as yf
    raw = yf.download(["BTC-USD", "ETH-USD"], start="2014-01-01", end="2026-11-07",
                      auto_adjust=True, progress=False)["Close"]
    raw.to_parquet(CACHE / "crypto.parquet")
    print(f"crypto {raw.shape}")


def cc_series():
    """Bybit CC L=21 K=7 over Aug 2022 - Oct 2026, from the cached backtest data, a newly
    fetched gap (May 2025 - Apr 2026) and the cached live-period data."""
    from fetch_improvement_data import bybit_klines, bybit_funding, UNIVERSE
    from quantresearch.carry_lab import Config, Panel, simulate, WARMUP
    imp = FCH / "data_cache" / "improve"
    gap = imp / "gap"; gap.mkdir(exist_ok=True)
    g0, g1 = pd.Timestamp("2025-05-20"), pd.Timestamp("2026-04-05")
    perp, spot, fund = {}, {}, {}
    for s in UNIVERSE:
        for kind, cat in (("perp", "linear"), ("spot", "spot")):
            f = gap / f"{s}_{kind}.parquet"
            if not f.exists():
                bybit_klines(s, cat, g0, g1).to_frame("close").to_parquet(f)
        f = gap / f"{s}_funding.parquet"
        if not f.exists():
            bybit_funding(s, g0, g1).to_frame("funding").to_parquet(f)
        p_bt = pd.read_parquet(FCH / "data_cache" / f"{s}_bybit_60.parquet")["close"]
        s_bt = pd.read_parquet(imp / f"{s}_bybit_spot_60.parquet")["close"]
        f_bt = pd.read_parquet(FCH / "data_cache" / f"{s}_bybit_funding.parquet")["funding"]
        parts = lambda a, b, c: pd.concat([a, b, c]).sort_index().loc[lambda x: ~x.index.duplicated()]
        perp[s] = parts(p_bt, pd.read_parquet(gap / f"{s}_perp.parquet")["close"],
                        pd.read_parquet(imp / "live" / f"{s}_perp.parquet")["close"])
        spot[s] = parts(s_bt, pd.read_parquet(gap / f"{s}_spot.parquet")["close"],
                        pd.read_parquet(imp / "live" / f"{s}_spot.parquet")["close"])
        fund[s] = parts(f_bt, pd.read_parquet(gap / f"{s}_funding.parquet")["funding"],
                        pd.read_parquet(imp / "live" / f"{s}_funding.parquet")["funding"])
    P = pd.DataFrame({s: v.resample("8h").last() for s, v in perp.items()}).sort_index()
    P = P.loc["2022-07-01":"2026-10-31 16:00"]
    S = pd.DataFrame({s: v.resample("8h").last() for s, v in spot.items()}).reindex(P.index)
    f8 = {}
    for s, f in fund.items():
        f = f.copy(); f.index = f.index.floor("h")
        f8[s] = f.groupby(f.index.floor("8h")).sum()
    F = pd.DataFrame(f8).reindex(P.index)
    panel = Panel(R=np.log(P).diff()[UNIVERSE], F=F[UNIVERSE], Rs=np.log(S).diff()[UNIVERSE])
    df = simulate(panel, Config("CC", 21, 7), start=WARMUP)
    daily = (1 + df["net"]).groupby(df.index.floor("D")).prod() - 1
    daily.to_frame("cc_net").to_parquet(CACHE / "cc_daily.parquet")
    gaps = P.isna().all(axis=1).sum()
    print(f"cc daily {daily.index.min().date()} -> {daily.index.max().date()} ({len(daily)} days, "
          f"{gaps} empty 8h rows in panel)")


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    fx_and_rates()
    crypto()
    cc_series()
