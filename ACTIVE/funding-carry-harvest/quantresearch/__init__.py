"""
quantresearch — a small futures/forex strategy research harness built around the
FTMO challenge objective.

Modules
-------
data        : MT5 loader, synthetic generator (+ Dukascopy optional)
indicators  : EMA / SMA / ATR / RSI / Bollinger
backtest    : event-driven engine with scale-out exits and bar-level equity
evaluation  : performance metrics + FTMO challenge simulator (pass-probability)
validation  : out-of-sample and walk-forward splitters
strategies  : Strategy interface + a demo mean-reversion strategy
"""
__version__ = "0.1.0"
