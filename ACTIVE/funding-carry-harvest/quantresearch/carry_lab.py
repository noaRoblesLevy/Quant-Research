"""
Carry lab: one vectorised engine for every funding-carry construction in the
2026-11-07 improvement study (docs/preregistration_2026-11-07.md).

A configuration is three choices:

    L     signal: rank on the mean funding of the last L 8h periods (L=1: latest print)
    K     tranches: K sub-books of 1/K capital, rebalanced HOLD/K periods apart
    kind  construction:
          "XS0"  long N lowest / short N highest funding, 1/N legs, BTC beta hedge
          "XSC"  same legs, inverse-vol weights, BTC+ETH factor hedge
          "PAIR" each short matched to the most correlated remaining coin; long weight =
                 pair hedge ratio / N; BTC beta hedge on the residual book
          "CC"   cash-and-carry: short N highest-funding perps, long the same coins on
                 spot; slots whose signal is not positive stay in cash

Timing (identical to scripts/eval_hedge_variants.py, so results are comparable):
period i realises the price change in logret[i] and the funding print Fnd[i] on the
weights held since the previous rebalance; a rebalance at i then uses Fnd[i] (known at
t_i) and returns up to i-1, and its new weights first earn period i+1. That is a
one-period execution lag: conservative, no look-ahead.

Costs: cost_bps x (sum |change in leg weights| / 2) for perp and spot legs, plus
cost_bps x |change in hedge| for the factor hedge (same formula as the earlier engine).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 3 * 365
ANNF = np.sqrt(PERIODS_PER_YEAR)

N_EACH = 11
HOLD = 21
COST_BPS = 8.0
VOL_WIN = 30
BETA_WIN = 90
WARMUP = 105


@dataclass(frozen=True)
class Config:
    kind: str = "XS0"
    L: int = 1
    K: int = 1

    @property
    def name(self) -> str:
        return f"{self.kind} L={self.L} K={self.K}"


@dataclass
class Panel:
    """Aligned 8h panel. R: perp log returns, Rs: spot log returns (or None),
    F: funding per 8h period. All T x N DataFrames on the same index/columns."""
    R: pd.DataFrame
    F: pd.DataFrame
    Rs: pd.DataFrame | None = None


# ----------------------------------------------------------------- statistics helpers
def _pair_cov(X: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Pairwise-complete sample covariance of each column of X with vector f
    (matches pandas Series.cov, which drops rows where either side is NaN)."""
    m = ~np.isnan(X) & ~np.isnan(f)[:, None]
    n = m.sum(0)
    Xm = np.where(m, X, 0.0)
    fm = np.where(m, f[:, None], 0.0)
    mx = Xm.sum(0) / np.maximum(n, 1)
    mf = fm.sum(0) / np.maximum(n, 1)
    c = (np.where(m, (X - mx) * (f[:, None] - mf), 0.0)).sum(0) / np.maximum(n - 1, 1)
    return np.where(n > 1, c, np.nan)


def _factor_hedge(w: np.ndarray, Rw: np.ndarray, fidx: list[int]) -> np.ndarray:
    """h = -Cov(F)^-1 e over the window Rw (rows = periods), e_k = sum_i w_i cov(r_i, F_k)."""
    Fw = Rw[:, fidx]
    ok = ~np.isnan(Fw).any(1)
    if ok.sum() < 10:
        return np.zeros(len(fidx))
    if len(fidx) > 1:
        cov_ff = np.atleast_2d(np.cov(Fw[ok].T))
    else:
        cov_ff = np.array([[np.var(Fw[ok, 0], ddof=1)]])
    expo = np.array([np.nansum(w * np.nan_to_num(_pair_cov(Rw, Rw[:, k]), nan=0.0))
                     for k in fidx])
    try:
        return -np.linalg.solve(cov_ff, expo)
    except np.linalg.LinAlgError:
        return -expo / np.diag(cov_ff)


# ----------------------------------------------------------------- construction
def _weights(kind: str, i: int, sig: np.ndarray, R: np.ndarray, vol: np.ndarray,
             spot_ok: np.ndarray | None, cols: list[str]):
    """Return (perp weights, spot weights, hedge factor indices) at rebalance i,
    or None if the cross-section is too thin to trade."""
    N = len(cols)
    valid = ~np.isnan(sig)
    if kind == "CC" and spot_ok is not None:
        valid &= spot_ok
    if valid.sum() < (N_EACH if kind == "CC" else 2 * N_EACH):
        return None
    # Stable sorts so that ties (common on Bybit, where many coins sit exactly at the
    # 0.01% default rate) are broken by column order, exactly like pandas
    # nsmallest/nlargest(keep="first") in the earlier engine.
    nv = int(valid.sum())
    order = np.argsort(np.where(valid, sig, np.inf), kind="stable")[:nv]
    lows = order[:N_EACH]
    highs = np.argsort(np.where(valid, -sig, np.inf), kind="stable")[:N_EACH]  # highest first
    w = np.zeros(N)
    s = np.zeros(N)
    btc, eth = cols.index("BTCUSDT"), cols.index("ETHUSDT")
    win = R[max(0, i - BETA_WIN):i]

    if kind in ("XS0", "XSC"):
        if kind == "XS0":
            w[lows], w[highs] = 1.0 / N_EACH, -1.0 / N_EACH
            fidx = [btc]
        else:
            iv = 1.0 / np.where(vol > 0, vol, np.nan)
            for leg, sign in ((lows, 1.0), (highs, -1.0)):
                x = np.nan_to_num(iv[leg], nan=0.0)
                x = x / x.sum() if x.sum() > 0 else np.full(len(leg), 1.0 / N_EACH)
                w[leg] = sign * x
            fidx = [btc, eth]
        return w, s, fidx

    if kind == "PAIR":
        pool = [j for j in order[:nv] if j not in set(highs)]
        for j in highs:                                         # highest funding first
            if not pool:
                break
            cors = []
            for p in pool:
                a, b = win[:, j], win[:, p]
                ok = ~np.isnan(a) & ~np.isnan(b)
                cors.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 10 else -np.inf)
            p = pool.pop(int(np.nanargmax(cors)))
            a, b = win[:, j], win[:, p]
            ok = ~np.isnan(a) & ~np.isnan(b)
            hr = np.cov(a[ok], b[ok])[0, 1] / np.var(b[ok], ddof=1) if ok.sum() > 10 else 1.0
            w[j] += -1.0 / N_EACH
            w[p] += float(np.clip(hr, 0.0, 3.0)) / N_EACH
        return w, s, [btc]

    if kind == "CC":
        pos = [j for j in highs if sig[j] > 0]
        w[pos], s[pos] = -1.0 / N_EACH, 1.0 / N_EACH
        return w, s, []

    raise ValueError(kind)


def _simulate_one(panel: Panel, cfg: Config, offset: int, start: int) -> pd.DataFrame:
    """One book rebalanced at periods i >= start with (i - start - offset) % HOLD == 0."""
    cols = list(panel.R.columns)
    R = panel.R.values
    Fd = panel.F.values
    Rs = panel.Rs.values if panel.Rs is not None else None
    T, N = R.shape
    sig_all = panel.F.rolling(cfg.L, min_periods=cfg.L).mean().values
    vol_all = panel.R.rolling(VOL_WIN).std().shift(1).values
    spot_avail = (~np.isnan(panel.Rs.values)) if Rs is not None else None

    w = np.zeros(N); s = np.zeros(N); h = np.zeros(0); fidx: list[int] = []
    out = np.zeros((T, 5))                     # carry, carry_long, carry_short, price, cost
    R0, F0 = np.nan_to_num(R), np.nan_to_num(Fd)
    Rs0 = np.nan_to_num(Rs) if Rs is not None else None
    for i in range(T):
        if i > 0 and (np.abs(w).sum() > 0 or np.abs(s).sum() > 0):
            price = float(w @ R0[i])
            if Rs0 is not None:
                price += float(s @ Rs0[i])
            cl = float(-np.clip(w, 0, None) @ F0[i])
            cs = float(-np.clip(w, None, 0) @ F0[i])
            hp = float(h @ R0[i, fidx]) if len(fidx) else 0.0
            hf = float(h @ F0[i, fidx]) if len(fidx) else 0.0
            out[i, :4] = (cl + cs - hf, cl, cs, price + hp)
        if i >= start and (i - start - offset) % HOLD == 0:
            spot_ok = spot_avail[i] if spot_avail is not None else None   # known at t_i
            res = _weights(cfg.kind, i, sig_all[i], R, vol_all[i], spot_ok, cols)
            if res is not None:
                nw, ns, nf = res
                cost = (np.abs(nw - w).sum() + np.abs(ns - s).sum()) / 2.0 * COST_BPS / 1e4
                nh = _factor_hedge(nw, R[max(0, i - BETA_WIN):i], nf) if nf else np.zeros(0)
                oh = np.zeros(len(nf)) if len(nf) != len(fidx) else h
                cost += float(np.abs(nh - oh).sum()) * COST_BPS / 1e4 if len(nf) else \
                    float(np.abs(h).sum()) * COST_BPS / 1e4
                out[i, 4] = cost
                w, s, h, fidx = nw, ns, nh, nf
    df = pd.DataFrame(out, index=panel.R.index,
                      columns=["carry", "carry_long", "carry_short", "price", "cost"])
    df["net"] = df["carry"] + df["price"] - df["cost"]
    return df


def simulate(panel: Panel, cfg: Config, start: int = WARMUP, offset: int = 0) -> pd.DataFrame:
    """Per-period P&L from `start` on. With K > 1 the K sub-books (offsets 0, HOLD/K, ...)
    each hold 1/K of capital and their returns are averaged."""
    if cfg.K == 1:
        df = _simulate_one(panel, cfg, offset, start)
    else:
        step = HOLD // cfg.K
        parts = [_simulate_one(panel, cfg, k * step, start) for k in range(cfg.K)]
        df = sum(parts) / cfg.K
    return df.iloc[start:]


def stats(df: pd.DataFrame) -> dict:
    net = df["net"]
    yrs = len(net) / PERIODS_PER_YEAR
    eq = (1 + net).cumprod()
    return dict(
        sharpe=float(net.mean() / net.std() * ANNF) if net.std() > 0 else 0.0,
        ann_ret=float(eq.iloc[-1] ** (1 / yrs) - 1),
        vol=float(net.std() * ANNF),
        maxdd=float((eq / eq.cummax() - 1).min()),
        carry_ann=float(df["carry"].sum() / yrs),
        price_ann=float(df["price"].sum() / yrs),
        cost_ann=float(df["cost"].sum() / yrs),
        total=float(eq.iloc[-1] - 1),
        log_total=float(np.log1p(net).sum()),
    )


def deflated_sharpe(net: pd.Series, trial_srs: list[float], n_trials: int) -> dict:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) on per-period returns.
    trial_srs: per-period (non-annualised) Sharpe ratios of all trials."""
    from scipy.stats import norm, skew, kurtosis
    x = net.values
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    emc = 0.5772156649
    v = np.var(trial_srs, ddof=1)
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1 / n_trials)
                        + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return dict(sr_ann=float(sr * ANNF), sr0_ann=float(sr0 * ANNF), dsr=float(norm.cdf(z)),
                skew=float(g3), kurt=float(g4))
