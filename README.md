# Quant Research

Personal repository for quantitative trading research and model development.
Each project is self-contained in its own folder with its own notebook(s) and data pipeline.

**Stack:** Python · PyTorch · Polars · Altair · Binance Futures data

---

## Projects

### 1. BTCUSD Regression Model
**Folder:** `BTCUSD_regression-model/`

Predicting future log returns on BTCUSD using a linear regression model trained with PyTorch.
Data is sourced directly from Binance's public trade archive and cached locally as Parquet.

| | |
|---|---|
| Asset | BTCUSDT Perpetual Futures |
| Target | Future log return |
| Model | Linear / Non-linear (PyTorch) |
| Data | Binance tick trades → OHLC bars |

---

## Shared Utilities

| File | Description |
|---|---|
| `research.py` | Core library — time series aggregation, training loops, plotting, metrics |
| `binance.py` | Downloads and caches trade data from `data.binance.vision` |
| `models.py` | PyTorch model definitions (`LinearModel`, `NonLinearModel`) |

---

## Setup

```bash
pip install torch polars altair numpy matplotlib tqdm requests
```

Data is downloaded automatically on first run and cached in `cache/` (gitignored).
