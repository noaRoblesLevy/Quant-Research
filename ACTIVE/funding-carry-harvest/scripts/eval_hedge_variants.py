"""
Weighting and hedge variants for the funding-carry book.

Reproduces the locked baseline (equal 1/N legs, BTC beta hedge) and tests:
    A  inverse-vol weights within each leg
    B  2-factor hedge: neutral to BTC and ETH jointly
    C  A + B
    D  C, re-hedged every 8h period instead of only at rebalance

The hedge is written generally as h = -Cov(F)^-1 e, with e_k = sum_i w_i cov(r_i, F_k).
With F = (BTC,) this is exactly the baseline single-beta hedge.

No look-ahead: vol, betas and ranks at period t use data up to t-1; funding and price
P&L are realised over t -> t+1. Costs: COST_BPS on turnover of legs and hedge.

    python scripts/eval_hedge_variants.py                 # Binance 2021-01..2025-05
    python scripts/eval_hedge_variants.py --venue bybit   # Bybit   2022-07..2025-05
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from quantresearch.funding_harvest import load_panels, PERIODS_PER_YEAR

ANNF = np.sqrt(PERIODS_PER_YEAR)

UNIVERSE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'NEARUSDT', 'AVAXUSDT',
            'XLMUSDT', 'AAVEUSDT', 'UNIUSDT', 'TRXUSDT', 'ATOMUSDT', 'ETCUSDT', 'SANDUSDT',
            'AXSUSDT', 'COMPUSDT']
N_EACH = 11
HOLD = 21          # 7 days of 8h periods
COST_BPS = 8.0
VOL_WIN = 30       # 10 days, trailing per-coin vol for inverse-vol weights
BETA_WIN = 90      # 30 days, trailing window for factor covariances
WARMUP = 105       # first rebalance on the HOLD grid after BETA_WIN (5 * 21)

VARIANTS = {
    "baseline": dict(inverse_vol=False, factors=("BTCUSDT",), rehedge=False),
    "A":        dict(inverse_vol=True,  factors=("BTCUSDT",), rehedge=False),
    "B":        dict(inverse_vol=False, factors=("BTCUSDT", "ETHUSDT"), rehedge=False),
    "C":        dict(inverse_vol=True,  factors=("BTCUSDT", "ETHUSDT"), rehedge=False),
    "D":        dict(inverse_vol=True,  factors=("BTCUSDT", "ETHUSDT"), rehedge=True),
}
LABELS = {
    "baseline": "Baseline: 1/N, BTC hedge",
    "A": "A: inverse-vol, BTC hedge",
    "B": "B: 1/N, BTC+ETH hedge",
    "C": "C: inverse-vol, BTC+ETH hedge",
    "D": "D: C, re-hedged every period",
}


def load_venue(venue: str):
    if venue == "binance":
        P, Fnd, logret = load_panels(UNIVERSE, start="2021-01", end="2025-05")
    elif venue == "bybit":
        from quantresearch.data.bybit_loader import load_bybit_panels
        P, Fnd, logret = load_bybit_panels(UNIVERSE, start="2022-07", end="2025-05")
    else:
        raise ValueError(venue)
    return P[UNIVERSE], Fnd[UNIVERSE], logret[UNIVERSE]


def leg_weights(fr: pd.Series, universe: list[str], inverse_vol: bool, vol: pd.Series) -> pd.Series:
    """Long the N lowest-funding coins, short the N highest; each leg gross 1.0."""
    w = pd.Series(0.0, index=universe)
    longs, shorts = fr.nsmallest(N_EACH).index, fr.nlargest(N_EACH).index
    for leg, sign in ((longs, 1.0), (shorts, -1.0)):
        if inverse_vol:
            x = (1.0 / vol.replace(0, np.nan)).reindex(leg).fillna(0.0)
            x = x / x.sum() if x.sum() > 0 else pd.Series(1.0 / N_EACH, index=leg)
        else:
            x = pd.Series(1.0 / N_EACH, index=leg)
        w[leg] = sign * x.values
    return w


def factor_hedge(w: pd.Series, F: pd.DataFrame, logret: pd.DataFrame, i: int) -> pd.Series:
    """h = -Cov(F)^-1 e over the trailing BETA_WIN periods (excluding period i)."""
    factors = list(F.columns)
    Fw = F.iloc[max(0, i - BETA_WIN):i]
    if len(Fw) < 10:
        return pd.Series(0.0, index=factors)
    Rw = logret.iloc[max(0, i - BETA_WIN):i]
    cov_ff = np.atleast_2d(np.cov(Fw.values.T)) if len(factors) > 1 \
        else np.array([[Fw.iloc[:, 0].var()]])
    expo = np.array([float((w * Rw.apply(lambda c: c.cov(Fw[f])).fillna(0.0)).sum())
                     for f in factors])
    try:
        h = -np.linalg.solve(cov_ff, expo)
    except np.linalg.LinAlgError:
        h = -expo / np.diag(cov_ff)
    return pd.Series(h, index=factors)


def run_book(P, Fnd, logret, inverse_vol=False, factors=("BTCUSDT",), rehedge=False,
             return_weights=False):
    """Per-period P&L of the book, split into carry / price / cost / net.

    With return_weights=True also returns the weights HELD during each period
    (legs plus hedge stacked per coin), for per-coin price attribution."""
    universe = list(P.columns)
    F = logret[list(factors)]
    Ff = Fnd[list(factors)]
    vol = logret.rolling(VOL_WIN).std().shift(1)
    cur_w = pd.Series(0.0, index=universe)
    cur_h = pd.Series(0.0, index=list(factors))
    rows, held = [], []
    for i in range(len(Fnd.index)):
        if return_weights:
            held.append(cur_w.add(cur_h, fill_value=0.0).reindex(universe).fillna(0.0).values)
        if cur_w.abs().sum() > 0 and i > 0:
            r = logret.iloc[i].fillna(0.0)
            f = Fnd.iloc[i].fillna(0.0)
            price = float((cur_w * r).sum()) + float((cur_h * F.iloc[i].fillna(0.0)).sum())
            carry_long = float((-cur_w.clip(lower=0) * f).sum())
            carry_short = float((-cur_w.clip(upper=0) * f).sum())
            carry = carry_long + carry_short - float((cur_h * Ff.iloc[i].fillna(0.0)).sum())
        else:
            price = carry = carry_long = carry_short = 0.0
        cost = 0.0

        if rehedge and cur_w.abs().sum() > 0 and i % HOLD != 0:
            new_h = factor_hedge(cur_w, F, logret, i)
            cost += float((new_h - cur_h).abs().sum()) * COST_BPS / 1e4
            cur_h = new_h

        # warm-up: no variant trades before vol and beta windows are full, so every
        # variant is evaluated on exactly the same periods
        if i % HOLD == 0 and i >= WARMUP:
            fr = Fnd.iloc[i].dropna()
            if len(fr) >= 2 * N_EACH:
                new_w = leg_weights(fr, universe, inverse_vol, vol.iloc[i])
                cost += float((cur_w - new_w).abs().sum()) / 2.0 * COST_BPS / 1e4
                new_h = factor_hedge(new_w, F, logret, i)
                cost += float((new_h - cur_h).abs().sum()) * COST_BPS / 1e4
                cur_w, cur_h = new_w, new_h
        rows.append((carry, carry_long, carry_short, price, cost))

    df = pd.DataFrame(rows, columns=["carry", "carry_long", "carry_short", "price", "cost"],
                      index=Fnd.index)
    df = df.iloc[WARMUP:]
    df["net"] = df["carry"] + df["price"] - df["cost"]
    if return_weights:
        W = pd.DataFrame(held, index=Fnd.index, columns=universe).iloc[WARMUP:]
        return df, W
    return df


def stats(df: pd.DataFrame, btc: pd.Series | None = None) -> dict:
    net = df["net"]
    eq = (1 + net).cumprod()
    yrs = len(net) / PERIODS_PER_YEAR
    out = dict(
        sharpe=net.mean() / net.std() * ANNF,
        cagr=eq.iloc[-1] ** (1 / yrs) - 1,
        carry_ann=df["carry"].sum() / yrs,
        price_ann=df["price"].sum() / yrs,
        cost_ann=df["cost"].sum() / yrs,
        price_vol=df["price"].std() * ANNF,
        carry_sharpe=df["carry"].mean() / df["carry"].std() * ANNF,
        carry_vol=df["carry"].std() * ANNF,
        carry_long_ann=df["carry_long"].sum() / yrs,
        carry_short_ann=df["carry_short"].sum() / yrs,
        maxdd=(eq / eq.cummax() - 1).min(),
        final=eq.iloc[-1],
        years=yrs,
    )
    if btc is not None:
        out["corr_price_btc"] = df["price"].corr(btc)
    return out


def run_all(venue: str):
    P, Fnd, logret = load_venue(venue)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        books = {k: run_book(P, Fnd, logret, **v) for k, v in VARIANTS.items()}
    btc = logret["BTCUSDT"].reindex(books["baseline"].index)
    return P, books, {k: stats(b, btc) for k, b in books.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", choices=["binance", "bybit"], default="binance")
    venue = ap.parse_args().venue
    P, books, res = run_all(venue)
    print(f"{venue}: {len(P)} 8h periods, {P.index.min().date()} -> {P.index.max().date()}\n")
    print(f"{'variant':<32}{'Sharpe':>7}{'CAGR':>8}{'carry':>8}{'price':>8}"
          f"{'pvol':>7}{'maxDD':>8}{'corrBTC':>9}")
    for k, s in res.items():
        print(f"{LABELS[k]:<32}{s['sharpe']:>7.2f}{s['cagr']*100:>7.1f}%{s['carry_ann']*100:>7.1f}%"
              f"{s['price_ann']*100:>7.1f}%{s['price_vol']*100:>6.0f}%{s['maxdd']*100:>7.1f}%"
              f"{s['corr_price_btc']:>9.2f}")
    b = res['baseline']
    print(f"\ncarry leg alone (baseline): Sharpe {b['carry_sharpe']:.1f}, "
          f"vol {b['carry_vol']*100:.2f}%/yr  |  from long leg {b['carry_long_ann']*100:+.1f}%/yr, "
          f"from short leg {b['carry_short_ann']*100:+.1f}%/yr")
