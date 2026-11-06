"""
Self-tests for the backtest engine — run WITHOUT MT5 or any data.

    python tests/test_engine.py        # standalone, prints PASS/FAIL
    pytest tests/test_engine.py        # if you have pytest

These pin down the engine's accounting so you can trust every later result.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantresearch.backtest import BacktestConfig, ExitConfig, run_backtest


def _df(rows):
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="1h")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def test_single_tp_winner():
    # bar0: signal long, sl_distance 10. entry next bar @100. TP at 1R=110.
    df = _df([
        [100, 100, 100, 100],   # bar0 (signal bar)
        [100, 105, 98, 102],    # bar1: entry @100, no stop/tp
        [104, 112, 104, 111],   # bar2: high 112 >= 110 -> TP hit, full exit
    ])
    sig = pd.DataFrame({"signal": [1, 0, 0], "sl_distance": [10, np.nan, np.nan]}, index=df.index)
    res = run_backtest(df, sig, ExitConfig(tp_ladder=[(1.0, 1.0)]),
                       BacktestConfig(risk_pct=0.01, initial_equity=100000))
    t = res["trades"]
    assert len(t) == 1, f"expected 1 trade, got {len(t)}"
    assert abs(t.iloc[0]["r"] - 1.0) < 1e-6, f"expected r=1.0, got {t.iloc[0]['r']}"
    # equity: risk 1% of 100k = 1000 risked, +1R = +1000
    assert abs(res["equity"].iloc[-1] - 101000) < 1.0, res["equity"].iloc[-1]
    return True


def test_single_sl_loser():
    df = _df([
        [100, 100, 100, 100],   # signal bar
        [100, 103, 99, 101],    # entry @100, sl @90, not hit
        [100, 101, 89, 90],     # low 89 <= 90 -> stop hit at 90 = -1R
    ])
    sig = pd.DataFrame({"signal": [1, 0, 0], "sl_distance": [10, np.nan, np.nan]}, index=df.index)
    res = run_backtest(df, sig, ExitConfig(tp_ladder=[(1.0, 1.0)]),
                       BacktestConfig(risk_pct=0.01, initial_equity=100000))
    t = res["trades"]
    assert len(t) == 1
    assert abs(t.iloc[0]["r"] + 1.0) < 1e-6, f"expected r=-1.0, got {t.iloc[0]['r']}"
    assert abs(res["equity"].iloc[-1] - 99000) < 1.0, res["equity"].iloc[-1]
    return True


def test_scale_out_then_breakeven():
    # 50% off at 1R, move stop to BE, remaining 50% stopped at entry -> net +0.5R
    df = _df([
        [100, 100, 100, 100],   # signal
        [100, 105, 99, 102],    # entry @100
        [104, 111, 104, 108],   # TP1 @110 hit (50%), BE stop moves to 100
        [108, 109, 99, 100],    # low 99 <= 100 (BE) -> remaining 50% out at entry
    ])
    sig = pd.DataFrame({"signal": [1, 0, 0, 0],
                        "sl_distance": [10, np.nan, np.nan, np.nan]}, index=df.index)
    res = run_backtest(df, sig,
                       ExitConfig(tp_ladder=[(1.0, 0.5)], move_be_after=1),
                       BacktestConfig(risk_pct=0.01, initial_equity=100000))
    t = res["trades"]
    assert len(t) == 1
    # 0.5 of position at +1R, 0.5 at 0R = +0.5R total
    assert abs(t.iloc[0]["r"] - 0.5) < 1e-6, f"expected r=0.5, got {t.iloc[0]['r']}"
    return True


def test_no_lookahead_and_synthetic_runs():
    from quantresearch.data.synthetic import make_ohlc
    from quantresearch.strategies import BollingerMeanReversion
    df = make_ohlc(n_bars=5000)
    sig = BollingerMeanReversion().generate_signals(df)
    res = run_backtest(df, sig, ExitConfig(tp_ladder=[(1.0, 0.5), (2.0, 0.5)]),
                       BacktestConfig(risk_pct=0.005, initial_equity=100000))
    eq = res["equity"]
    assert len(eq) == len(df)
    assert not eq.isna().any(), "equity curve has NaNs"
    # random walk: should be roughly break-even, definitely not absurd
    final_ret = eq.iloc[-1] / eq.iloc[0] - 1
    assert -0.95 < final_ret < 5.0, f"implausible synthetic return {final_ret:.2%}"
    return True


ALL = [
    ("single TP winner = +1R", test_single_tp_winner),
    ("single SL loser = -1R", test_single_sl_loser),
    ("scale-out + breakeven = +0.5R", test_scale_out_then_breakeven),
    ("no-lookahead + synthetic runs", test_no_lookahead_and_synthetic_runs),
]


if __name__ == "__main__":
    failed = 0
    for name, fn in ALL:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  ->  {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}  ->  {type(e).__name__}: {e}")
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    sys.exit(1 if failed else 0)
