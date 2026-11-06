"""
FTMO challenge evaluation.

Turns the firm's rules into a measurable objective: would this equity curve PASS
the challenge, and how likely is it to pass vs. breach?

Standard 2-step FTMO (2026):
    Phase 1: +10% target, -5% daily loss, -10% static max loss, min 4 days
    Phase 2: +5%  target, -5% daily loss, -10% static max loss, min 4 days

The "max loss" is a STATIC floor at (1 - max_total_loss) * initial_balance — it
does NOT trail the equity peak (that is the harsher 1-step product).

Two tools:
  1. evaluate_run() — walk the actual historical bar-level equity once, find the
     first terminal event (target / daily breach / total breach). Uses intraday
     equity, so the daily-loss check is accurate.
  2. monte_carlo() — block-bootstrap daily returns to estimate P(pass), P(breach
     by total loss), and average days-to-pass. NOTE: works on daily closes, so it
     approximates the daily-loss rule (close-to-close) and is slightly optimistic
     about intraday daily breaches. Use evaluate_run() for the precise daily check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FTMORules:
    profit_target_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    max_total_loss_pct: float = 0.10
    min_trading_days: int = 4
    label: str = "FTMO 2-step Phase 1"


def evaluate_run(equity: pd.Series, rules: FTMORules, initial_balance: float | None = None) -> dict:
    """Walk the bar-level equity curve once and report the first terminal event.

    The daily-loss rule is evaluated against each day's INTRADAY minimum equity
    relative to that day's starting balance (start = prior day's closing equity,
    first day = initial_balance).
    """
    if initial_balance is None:
        initial_balance = float(equity.iloc[0])

    target_level = initial_balance * (1 + rules.profit_target_pct)
    total_floor = initial_balance * (1 - rules.max_total_loss_pct)
    daily_loss_abs = initial_balance * rules.max_daily_loss_pct

    by_day = equity.groupby(equity.index.normalize())
    day_start = initial_balance
    trading_days = 0
    result = {
        "passed": False,
        "breached": False,
        "reason": "incomplete",
        "days_used": 0,
        "end_equity": float(equity.iloc[-1]),
        "label": rules.label,
    }

    for day, day_equity in by_day:
        trading_days += 1
        day_min = float(day_equity.min())
        day_max = float(day_equity.max())
        day_close = float(day_equity.iloc[-1])

        # 1. total static floor (checked on intraday minimum)
        if day_min <= total_floor:
            result.update(passed=False, breached=True, reason="max_total_loss",
                          days_used=trading_days)
            return result

        # 2. daily loss (intraday min vs this day's start)
        if (day_start - day_min) >= daily_loss_abs:
            result.update(passed=False, breached=True, reason="max_daily_loss",
                          days_used=trading_days)
            return result

        # 3. profit target (intraday max), only valid once min days met
        if day_max >= target_level and trading_days >= rules.min_trading_days:
            result.update(passed=True, breached=False, reason="target_reached",
                          days_used=trading_days)
            return result

        day_start = day_close

    return result


def monte_carlo(
    equity: pd.Series,
    rules: FTMORules,
    n_sims: int = 5000,
    block: int = 5,
    max_days: int = 250,
    seed: int | None = 42,
) -> dict:
    """Estimate pass / breach probabilities by block-bootstrapping daily returns.

    Daily returns are taken as fractions of the PREVIOUS day's close, then applied
    additively to a fixed initial balance (challenge-style: targets/limits are vs
    the initial balance and traders typically risk a fixed amount during a challenge).
    """
    rng = np.random.default_rng(seed)
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna().to_numpy()
    if len(rets) < block + 1:
        return {"error": "not enough daily data for Monte Carlo"}

    target = rules.profit_target_pct
    daily_cap = rules.max_daily_loss_pct
    total_cap = rules.max_total_loss_pct
    min_days = rules.min_trading_days

    n_blocks_needed = int(np.ceil(max_days / block))
    passes = 0
    breaches = 0
    daily_breaches = 0
    days_to_pass = []

    starts = np.arange(0, len(rets) - block)

    for _ in range(n_sims):
        # build a synthetic return path from random blocks
        chosen = rng.choice(starts, size=n_blocks_needed, replace=True)
        path = np.concatenate([rets[s:s + block] for s in chosen])[:max_days]

        equity_frac = 0.0     # cumulative profit as fraction of initial
        outcome = None
        for d, r in enumerate(path, start=1):
            day_pnl = r           # treat daily return as fraction-of-initial increment
            # daily loss check
            if day_pnl <= -daily_cap:
                outcome = "daily_breach"
                daily_breaches += 1
                break
            equity_frac += day_pnl
            if equity_frac <= -total_cap:
                outcome = "total_breach"
                break
            if equity_frac >= target and d >= min_days:
                outcome = "pass"
                days_to_pass.append(d)
                break
        if outcome == "pass":
            passes += 1
        elif outcome in ("daily_breach", "total_breach"):
            breaches += 1

    return {
        "p_pass": passes / n_sims,
        "p_breach": breaches / n_sims,
        "p_daily_breach": daily_breaches / n_sims,
        "p_timeout": 1 - (passes + breaches) / n_sims,
        "avg_days_to_pass": float(np.mean(days_to_pass)) if days_to_pass else None,
        "n_sims": n_sims,
        "label": rules.label,
    }


def print_ftmo(equity: pd.Series, rules: FTMORules | None = None,
               initial_balance: float | None = None, run_mc: bool = True) -> dict:
    rules = rules or FTMORules()
    run = evaluate_run(equity, rules, initial_balance)
    print(f"\n=== {rules.label} ===")
    print(f"  Single historical run : {'PASS' if run['passed'] else ('BREACH' if run['breached'] else 'INCOMPLETE')}"
          f"  ({run['reason']}, {run['days_used']} days)")
    out = {"run": run}
    if run_mc:
        mc = monte_carlo(equity, rules)
        if "error" not in mc:
            print(f"  Monte Carlo (n={mc['n_sims']}):")
            print(f"    P(pass)         : {mc['p_pass']*100:5.1f}%")
            print(f"    P(breach)       : {mc['p_breach']*100:5.1f}%")
            print(f"    P(daily breach) : {mc['p_daily_breach']*100:5.1f}%")
            if mc["avg_days_to_pass"]:
                print(f"    Avg days to pass: {mc['avg_days_to_pass']:.0f}")
        else:
            print(f"    Monte Carlo: {mc['error']}")
        out["monte_carlo"] = mc
    return out
