"""
research.py - Model Research & Development Utilities
====================================================
Reusable boilerplate code for machine learning research and model development.
This module eliminates copy-paste by centralizing common functions for:

- Data preprocessing and time series aggregation
- Feature engineering for financial data
- Model architecture inspection and debugging
- Visualization and exploratory data analysis
- Training utilities

Author: MemLabs
Course: Build a Quant Trading System
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Data manipulation and analysis
import polars as pl                         # Fast dataframes for financial data
from typing import Dict, List, Tuple, Union                     # Type hints for function signatures

# Machine learning framework
import torch                                # PyTorch for neural networks
import torch.nn as nn                       # Neural network modules
import torch.optim as optim                 # Optimization algorithms

# Numerical computing and datetime
import numpy as np                          # Numerical operations
import numpy.typing as npt
from datetime import datetime, timedelta    # Date and time handling

# Visualization
import altair                               # Interactive plotting library
import matplotlib.pyplot as plt

import random
import re
import itertools
from pathlib import Path
from tqdm import tqdm
import os

SEED = 42

# ============================================================================
# TIME SERIES AGGREGATION
# ============================================================================
OHLC_AGGS = [
    pl.col("price").first().alias("open"),
    pl.col("price").max().alias("high"),
    pl.col("price").min().alias("low"),
    pl.col("price").last().alias("close"),
]


def get_trade_files(directory: str, sym: str) -> List[Path]:
    dir_path = Path(directory)
    pattern = f"{sym}-trades*"
    return sorted(dir_path.glob(pattern))


from pathlib import Path
from typing import List, Optional

def load_ohlc_timeseries(sym: str, time_interval: str):
    return load_timeseries(sym, time_interval, OHLC_AGGS)

def load_timeseries(
    sym: str,
    time_interval: str,
    aggs: List[pl.Expr],
    data_path: Optional[str] = None
) -> pl.DataFrame:
    if data_path is None:
        data_path = './cache'

    files = get_trade_files(data_path, sym)

    if not files:
        raise FileNotFoundError(f"No files found for {sym} in {data_path}")

    ts_list = []

    for file in tqdm(files, desc=f"Loading {sym}", unit="file"):
        trades = pl.read_parquet(file)

        if "datetime" not in trades.columns:
            raise ValueError(f"Column 'datetime' not found in {file.name}")

        trades = trades.with_columns(
            pl.col("datetime").cast(pl.Datetime)
        ).sort("datetime")

        ts = trades.group_by_dynamic(
            "datetime",
            every=time_interval,
            offset="0m"
        ).agg(aggs)

        ts_list.append(ts)

    result = pl.concat(ts_list)
    result = result.sort("datetime").unique(subset=["datetime"])

    return result


def load_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    agg_cols: Union[pl.Expr,List[pl.Expr]],
    data_path: Optional[str] = None
) -> pl.DataFrame:
    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    ts_list = []
    total_days = (end_date - start_date).days + 1

    for i in tqdm(range(total_days), desc=f"Loading {sym}", unit="day"):
        current_date = start_date + timedelta(days=i)
        file_name = f"{sym}-trades-{current_date.strftime('%Y-%m-%d')}.parquet"
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            tqdm.write(f"[WARNING] Missing file: {file_name}")
            continue

        try:
            trades = pl.read_parquet(file_path)

            if "datetime" not in trades.columns:
                raise ValueError(f"Column 'datetime' not found in {file_name}")

            trades = trades.with_columns(pl.col("datetime").cast(pl.Datetime))

            ts = trades.group_by_dynamic("datetime", every=time_interval, offset="0m").agg(agg_cols)
            ts_list.append(ts)

        except Exception as e:
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(f"No trade data found for {sym} in range {start_date} to {end_date}")

    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    return result

def load_ohlc_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    data_path: Optional[str] = None
) -> pl.DataFrame:
    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    ts_list = []
    total_days = (end_date - start_date).days + 1

    for i in tqdm(range(total_days), desc=f"Loading {sym}", unit="day"):
        current_date = start_date + timedelta(days=i)
        file_name = f"{sym}-trades-{current_date.strftime('%Y-%m-%d')}.parquet"
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            tqdm.write(f"[WARNING] Missing file: {file_name}")
            continue

        try:
            trades = pl.read_parquet(file_path)

            if "datetime" not in trades.columns:
                raise ValueError(f"Column 'datetime' not found in {file_name}")

            trades = trades.with_columns(pl.col("datetime").cast(pl.Datetime))

            ts = trades.group_by_dynamic("datetime", every=time_interval, offset="0m").agg(OHLC_AGGS)
            ts_list.append(ts)

        except Exception as e:
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(f"No trade data found for {sym} in range {start_date} to {end_date}")

    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    return result


def sharpe_annualization_factor(interval: str,
                                trading_days_per_year: int = 365,
                                trading_hours_per_day: float = 24) -> float:
    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("Interval must be like '1d', '2h', '15m', '30s'")

    value, unit = int(match.group(1)), match.group(2)

    if unit == 'd':
        periods = trading_days_per_year / value
    elif unit == 'h':
        periods = trading_days_per_year * (trading_hours_per_day / value)
    elif unit == 'm':
        periods = trading_days_per_year * (trading_hours_per_day * 60 / value)
    elif unit == 's':
        periods = trading_days_per_year * (trading_hours_per_day * 3600 / value)
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    return np.sqrt(periods)


def ohlc_timeseries(df: pl.DataFrame, time_interval: str) -> pl.DataFrame:
    return timeseries(df, time_interval, OHLC_AGGS)

def lag_col_names(col: str, n: int) -> List[str]:
    return [f'{col}_lag_{i}' for i in range(1, n+1)]

def auto_reg_corr_matrx(df, target, max_no_lags) -> pl.DataFrame:
    return df.drop_nulls().select([target]+lag_col_names(target, max_no_lags)).corr()

def log_returns_col(name: str, step_size = 1) -> pl.Expr:
    return (pl.col(name)/pl.col(name).shift(step_size)).log().alias(f'{name}_log_return')

def timeseries(
    df: pl.DataFrame,
    time_interval: str,
    aggs: Union[List[pl.Expr],pl.Expr]
) -> pl.DataFrame:
    return df.group_by_dynamic(
        "datetime",
        every=time_interval,
        offset="0m"
    ).agg(aggs)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot(df: pl.DataFrame, col: str, title: str = "") -> altair.Chart:
    return altair.Chart(df).mark_area(
        opacity=0.7,
        interpolate='basis'
    ).transform_density(
        col,
        as_=[col, 'density']
    ).encode(
        x=altair.X(f'{col}:Q', title=col),
        y=altair.Y('density:Q', title='Density')
    ).properties(
        width=600,
        height=400,
        title=title if title else f'Distribution of {col}'
    )

def plot_distribution(data: pl.DataFrame, col: str, label = None, no_bins = 100):
    return altair.Chart(data).mark_bar().encode(
        altair.X(f'{col}:Q', bin=altair.Bin(maxbins=no_bins)),
        y='count()'
    ).properties(
        width=600,
        height=400,
        title=f'Distribution of {label if label else col}'
    ).configure_scale(zero=False).add_params(
        altair.selection_interval(bind='scales')
    )

def plot_static_timeseries(ts: pl.DataFrame, sym: str, col: str, interval_size: str):
    plt.figure(figsize=(12, 6))
    plt.plot(ts['datetime'], ts[col], label=col)
    plt.title(f'{sym} {interval_size} Bars')
    plt.xlabel('time')
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_multiple_lines(
    df: pl.DataFrame,
    cols_to_plot: List[str],
    sym: str,
    width: int = 15,
    height: int = 6,
    xlabel_unit: str = "Time Step"
):
    import matplotlib.pyplot as plt
    x_index = np.arange(len(df))
    plt.figure(figsize=(width, height))

    for col in cols_to_plot:
        if col in df.columns:
            y_values = df[col].to_numpy()
            plt.plot(x_index, y_values, label=col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    title_cols = ', '.join(cols_to_plot)
    plt.title(f'{sym} Series: {title_cols}')
    plt.xlabel(xlabel_unit)
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_dyn_timeseries(ts: pl.DataFrame, sym: str, col: str, time_interval: str):
    return altair.Chart(ts).mark_line(tooltip=True).encode(
        x="datetime",
        y=col
    ).properties(
        width=800,
        height=400,
        title=f"{sym} {time_interval} {col}"
    ).configure_scale(zero=False).add_params(
        altair.selection_interval(bind='scales', encodings=['x']),
        altair.selection_interval(bind='scales', encodings=['y'])
    )

def to_tensor(x, dtype=None) -> torch.Tensor:
    return torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)

# ============================================================================
# MODEL ANALYSIS
# ============================================================================

def print_model_complexity_ratio(m1, m1_name, m2, m2_name):
    m1_params = total_model_params(m1)
    m2_params = total_model_params(m2)
    complexity_ratio = m2_params / m1_params
    print(f"Complexity Comparsion:")
    print(f"\t{m2_name} has {complexity_ratio:.1f}x more parameters than {m1_name}")
    print(f"\tParametric difference: {m2_params - m1_params:,} additional parameters")

def total_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def print_model_info(model: torch.nn.Module, model_name: str) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"\nArchitecture:")
    print(f"  {model}")
    print(f"\nParameter Count:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")

    if total_params != trainable_params:
        frozen_params = total_params - trainable_params
        print(f"  Frozen parameters:     {frozen_params:,}")

    print(f"{'='*60}\n")

def _prefix_cols(df, prefix):
    return df.rename({col: f"{prefix}_{col}" for col in df.columns})

def _prefix_close_ts(trades, time_interval, prefix):
    return _prefix_cols(ohlc_timeseries(trades, time_interval), prefix)

def compare_ts_corr(x_df, x_prefix, y_df, y_prefix, time_interval, col='close'):
    x_col, y_col = f'{x_prefix}_{col}', f'{y_prefix}_{col}'
    joined_ts = pl.concat([
        _prefix_close_ts(x_df, time_interval, x_prefix),
        _prefix_close_ts(y_df, time_interval, y_prefix)
    ], how="horizontal")
    return joined_ts.select(pl.corr(x_col, y_col)).item()

def log_return_col(col: str) -> str:
    return f"{col}_log_return"

def log_return(col: str, shift_size: int = 1) -> pl.Expr:
    return (pl.col(col)/pl.col(col).shift(shift_size)).log().alias(log_return_col(col))

def lag_cols(col: str, forecast_horizon: str, no_lags: int) -> List[pl.Expr]:
    return [pl.col(col).shift(forecast_horizon * i).alias(f'{col}_lag_{i}') for i in range(1, no_lags + 1)]

def add_lags(df: pl.DataFrame, col: str, max_no_lags: int, forecast_step: int) -> pl.DataFrame:
    return df.with_columns([pl.col(col).shift(i * forecast_step).alias(f'{col}_lag_{i}') for i in range(1, max_no_lags + 1)])

def batch_train_reg(
    model: nn.Module,
    X_train,
    X_test,
    y_train,
    y_test,
    no_epochs: int,
    criterion=None,
    optimizer=None,
    logging=True,
    lr=None
):
    if criterion is None:
        criterion = nn.L1Loss()

    if lr is None:
        lr = 0.0002

    if optimizer is None:
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=1,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-7,
            tolerance_change=1e-9
        )

    if logging:
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
        print("Model architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel()} params)")
        print("\nTraining model...")

    train_loss = None
    log_tick_size = max(no_epochs // 10, 1)

    if isinstance(optimizer, torch.optim.LBFGS):
        for epoch in range(no_epochs):
            def closure():
                optimizer.zero_grad()
                predictions = model(X_train)
                loss = criterion(predictions, y_train)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                train_loss = criterion(model(X_train), y_train).item()

            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch+1}/{no_epochs}], Loss: {train_loss:.6f}")

    else:
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch+1}/{no_epochs}], Loss: {loss.item():.6f}")

    if logging:
        print("\nLearned parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}:\n{param.data.numpy()}")

    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
        test_loss = criterion(y_hat, y_test)
        if logging:
            print(f'\nTest Loss: {test_loss.item():.6f}, Train Loss: {train_loss:.6f}')

    return y_hat


def timeseries_train_test_split(df: pl.DataFrame, features, target, test_size=0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = df.drop_nulls()
    X = to_tensor(df[features])
    y = to_tensor(df[target]).reshape(-1, 1)
    X_train, X_test = timeseries_split(X, test_size)
    y_train, y_test = timeseries_split(y, test_size)
    return X_train, X_test, y_train, y_test

def timeseries_split(t, test_size=0.25):
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1 (got {test_size})")
    split_idx = int(len(t) * (1 - test_size))
    return t[:split_idx], t[split_idx:]


def plot_column(df, col_name, figsize=(15, 6), title=None, xlabel='Index'):
    if title is None:
        title = col_name
    chart = df[col_name].plot.line()
    return chart.properties(width=800, height=400, title=title)


def plot_columns(df, col_name, figsize=(15, 6), title=None, xlabel='Index'):
    if title is None:
        title = col_name
    chart = df[col_name].plot.line()
    return chart.properties(width=800, height=400, title=title)


def model_trade_results(y_true, y_pred) -> pl.DataFrame:
    trade_results = pl.DataFrame({
        'y_pred': y_pred.squeeze(),
        'y_true': y_true.squeeze()
    }).with_columns([
        (pl.col('y_pred').sign() == pl.col('y_true').sign()).alias('is_won'),
        pl.col('y_pred').sign().alias('position')
    ]).with_columns([
        (pl.col('position') * pl.col('y_true')).alias('trade_log_return')
    ]).with_columns([
        pl.col('trade_log_return').cum_sum().alias('equity_curve')
    ]).with_columns(
        (pl.col('equity_curve') - pl.col('equity_curve').cum_max()).alias('drawdown_log_return'),
    )
    return trade_results


def add_tx_fee(trades: pl.DataFrame, tx_fee: float, name: str):
    tx_fee_col = (pl.col('exit_trade_value') * tx_fee + pl.col('entry_trade_value') * tx_fee).alias(f"tx_fee_{name}")
    return trades.with_columns(tx_fee_col)


def add_tx_fees(trades: pl.DataFrame, maker_fee: float, taker_fee: float):
    trades = add_tx_fee(trades, maker_fee, 'maker')
    trades = add_tx_fee(trades, taker_fee, 'taker')
    return trades

def add_tx_fees_log(trades: pl.DataFrame, maker_fee, taker_fee):
    return trades.with_columns(
        (pl.col('trade_log_return') + np.log(maker_fee)).alias('trade_log_return_net_maker'),
        (pl.col('trade_log_return') + np.log(taker_fee)).alias('trade_log_return_net_taker'),
    ).with_columns(
        pl.col('trade_log_return_net_maker').cum_sum().alias('equity_curve_net_maker'),
        pl.col('trade_log_return_net_taker').cum_sum().alias('equity_curve_net_taker'),
    )

def eval_model_performance(y_actual, y_pred, feature_names: List[str], target_name: str, annualized_rate: float) -> Dict[str, any]:
    trade_results = model_trade_results(y_actual, y_pred)
    accuracy = trade_results['is_won'].mean()
    avg_win = trade_results.filter(pl.col('is_won'))['trade_log_return'].mean()
    avg_loss = trade_results.filter(~pl.col('is_won'))['trade_log_return'].mean()
    expected_value = accuracy * avg_win + (1 - accuracy) * avg_loss
    drawdown = (trade_results['equity_curve'] - trade_results['equity_curve'].cum_max())
    max_drawdown = drawdown.min()
    sharpe = trade_results['trade_log_return'].mean() / trade_results['trade_log_return'].std() if trade_results['trade_log_return'].std() > 0 else 0
    annualized_sharpe = sharpe * annualized_rate
    equity_trough = trade_results['equity_curve'].min()
    equity_peak = trade_results['equity_curve'].max()
    total_log_return = trade_results['trade_log_return'].sum()
    std = trade_results['trade_log_return'].std()
    return {
        'features': ','.join(list(feature_names)),
        'target': target_name,
        'no_trades': len(trade_results),
        'win_rate': accuracy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'best_trade': trade_results['trade_log_return'].max(),
        'worst_trade': trade_results['trade_log_return'].min(),
        'ev': expected_value,
        'std': std,
        'total_log_return': total_log_return,
        'compound_return': np.exp(total_log_return),
        'max_drawdown': max_drawdown,
        'equity_trough': equity_trough,
        'equity_peak': equity_peak,
        'sharpe': annualized_sharpe,
    }

def train_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate, test_size=0.25, loss=None, optimizer=None, no_epochs=None, log=False, lr=None):
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, lr=lr, logging=log)


def benchmark_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate, test_size=0.25, loss=None, optimizer=None, no_epochs=None, log=False, lr=None):
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, lr=lr, logging=log)
    perf = eval_model_performance(y_test, y_hat, features, target, annualized_rate)
    weights, biases = get_linear_params(model)
    perf['weights'] = str(weights)
    perf['biases'] = str(biases)
    return perf


def learn_model_trades(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, test_size=0.25, loss=None, optimizer=None, no_epochs=None, log=False, lr=None):
    df = df.drop_nulls()
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, criterion=loss, optimizer=optimizer, lr=lr, logging=log)
    return model_trade_results(y_test, y_hat)


def learn_model_trade_pnl(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, test_size=0.25, loss=None, optimizer=None, no_epochs=None, log=False, lr=None):
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, lr=lr, logging=log)
    trade_results = model_trade_results(y_test, y_hat)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(42)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_linear_params(model: nn.Module) -> tuple:
    weight = model.linear.weight.detach().cpu().numpy().flatten()
    bias = model.linear.bias.detach().cpu().numpy().item()
    return weight, bias

def add_log_return_features(df: pl.DataFrame, col: str, forecast_horizon: int, max_no_lags=None):
    if max_no_lags is None:
        max_no_lags = 0
    df = df.with_columns(log_return(col, forecast_horizon))
    if max_no_lags > 0:
        df = add_lags(df, log_return_col('close'), max_no_lags, forecast_horizon)
    return df

def benchmark_linear_models(ts: pl.DataFrame, target: str, feature_pool: List[str], annualized_rate: int, max_no_features: int = 1, no_epochs=200, loss=None, test_size=0.25) -> pl.DataFrame:
    import models
    ts = ts.drop_nulls()
    benchmarks = []
    fs = []
    for i in range(1, max_no_features + 1):
        fs += list(itertools.combinations(feature_pool, i))
    for features in fs:
        m = models.LinearModel(len(features))
        m.apply(init_weights)
        benchmarks.append(benchmark_reg_model(ts, list(features), target, m, annualized_rate, no_epochs=no_epochs, loss=loss, test_size=test_size))
    benchmark = pl.DataFrame(benchmarks)
    return benchmark.sort('sharpe', descending=True)


def print_model_params(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:\n{param.data.numpy()}")


def add_model_predictions(test_trades: pl.DataFrame, model: nn.Module, features: Union[str, List[str]]) -> pl.DataFrame:
    if type(features) != list:
        features = [features]
    X_test = torch.tensor(test_trades[features].to_numpy(), dtype=torch.float32)
    y_hat = model(X_test)
    s = pl.Series('y_hat', model(X_test).detach().cpu().numpy().squeeze())
    return test_trades.with_columns(s)


def add_equity_curve(trades: pl.DataFrame, initial_capital: float, col_name: str, suffix: str) -> pl.DataFrame:
    return trades.with_columns(
        (initial_capital + pl.col(col_name).cum_sum()).alias(f'equity_curve_{suffix}')
    )

def add_compounding_trades(trades, capital, leverage, maker_fee, taker_fee):
    lev_capital = capital * leverage
    trades = trades.with_columns(
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).shift().fill_null(lev_capital).alias('entry_trade_value'),
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).alias('exit_trade_value'),
    ).with_columns(
        (pl.col('entry_trade_value') / pl.col('open') * pl.col('dir_signal')).alias('signed_trade_qty'),
        (pl.col('exit_trade_value') - pl.col('entry_trade_value')).alias('trade_gross_pnl'),
    )
    trades = add_tx_fees(trades, maker_fee, taker_fee)
    trades = trades.with_columns(
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_taker')).alias('trade_net_taker_pnl'),
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_maker')).alias('trade_net_maker_pnl'),
    )
    trades = add_equity_curve(trades, capital, 'trade_gross_pnl', 'gross')
    trades = add_equity_curve(trades, capital, 'trade_net_taker_pnl', 'taker')
    trades = add_equity_curve(trades, capital, 'trade_net_maker_pnl', 'maker')
    return trades
