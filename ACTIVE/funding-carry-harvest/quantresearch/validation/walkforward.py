"""
Out-of-sample and walk-forward validation utilities.

These don't run a backtest themselves — they hand you (train, test) date windows.
You supply a function that optimises params on `train` and evaluates on `test`.
The whole point: never judge a strategy on data you fitted it to.
"""
from __future__ import annotations

from typing import Callable, Iterator

import pandas as pd


def train_test_split(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple chronological split. Everything before split_date is train."""
    split = pd.Timestamp(split_date)
    return df.loc[df.index < split], df.loc[df.index >= split]


def rolling_walkforward(
    df: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, test) windows rolling forward through the data.

    Anchored = False (rolling): each train window is `train_months` long.
    """
    start = df.index.min()
    end = df.index.max()
    cur = start
    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        train = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test = df.loc[(df.index >= train_end) & (df.index < test_end)]
        if len(train) and len(test):
            yield train, test
        cur = cur + pd.DateOffset(months=step_months)


def run_walkforward(
    df: pd.DataFrame,
    optimise_fn: Callable[[pd.DataFrame], dict],
    evaluate_fn: Callable[[pd.DataFrame, dict], dict],
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> pd.DataFrame:
    """Run a full walk-forward.

    optimise_fn(train_df) -> best_params (dict)
    evaluate_fn(test_df, best_params) -> metrics (dict)
    Returns a DataFrame, one row per fold, with the out-of-sample metrics.
    """
    rows = []
    for k, (train, test) in enumerate(rolling_walkforward(df, train_months, test_months, step_months)):
        params = optimise_fn(train)
        metrics = evaluate_fn(test, params)
        rows.append({
            "fold": k,
            "test_start": test.index.min(),
            "test_end": test.index.max(),
            **{f"param_{kk}": vv for kk, vv in params.items()},
            **metrics,
        })
    return pd.DataFrame(rows)
