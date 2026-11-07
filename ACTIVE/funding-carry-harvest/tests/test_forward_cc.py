"""
Tests for the cash-and-carry shadow ledger (scripts/forward_test_cc.py). Uses live Bybit
history (network) and temporary state directories; never touches data_cache/.

    python tests/test_forward_cc.py

1. gap invariance: one replay over ~4 weeks == the same period in three runs with gaps
2. no look-ahead: the coins picked at a rebalance are identical with data truncated at t
3. cadence: tranche k first rebalances at start + k days, then exactly weekly
4. book shape: equal and opposite perp/spot weights; gross per side <= 1
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import forward_test_cc as CC

START = pd.Timestamp("2026-10-08 16:00")
END = pd.Timestamp("2026-11-07 08:00")
SETTLE = pd.Timedelta(minutes=15)


def replay(stops):
    d = Path(tempfile.mkdtemp())
    CC.set_state_dir(d)
    CC.do_init(start=START)
    for s in stops:
        CC.do_step(now=s + SETTLE, quiet=True)
    st = json.loads(CC.STATE.read_text())
    led = pd.read_csv(CC.LEDGER)
    shutil.rmtree(d)
    return st, led


fails = 0
one, led1 = replay([END])
three, led3 = replay([START + pd.Timedelta(days=3, hours=8), START + pd.Timedelta(days=16), END])

ok1 = abs(one["equity"] - three["equity"]) < 1e-6 and led1.equals(led3)
fails += not ok1
print(f"[{'PASS' if ok1 else 'FAIL'}] gap invariance: equity {one['equity']:.6f} vs "
      f"{three['equity']:.6f}, ledgers equal={led1.equals(led3)} ({len(led1)} steps)")

reb = led1[led1["rebalanced"].notna()]
times = {}
for _, r in reb.iterrows():
    for k in str(r["rebalanced"]).split("|"):
        times.setdefault(int(float(k)), []).append(pd.Timestamp(r["time"]).tz_localize(None))
want = {k: [START + pd.Timedelta(days=k + 7 * j) for j in range(5)
            if START + pd.Timedelta(days=k + 7 * j) <= END] for k in range(CC.K)}
ok3 = all(times.get(k, []) == want[k] for k in range(CC.K))
fails += not ok3
print(f"[{'PASS' if ok3 else 'FAIL'}] cadence: " +
      ", ".join(f"T{k}:{len(times.get(k, []))}x" for k in range(CC.K)))

Wp, Ws = CC.aggregate(one)
ok4 = all(abs(Wp[s] + Ws[s]) < 1e-12 for s in CC.UNIVERSE) and sum(Ws.values()) <= 1 + 1e-12
fails += not ok4
print(f"[{'PASS' if ok4 else 'FAIL'}] book shape: perp = -spot per coin, gross spot "
      f"{sum(Ws.values()):.3f}, coins held {sum(1 for v in Ws.values() if v)}")

# no look-ahead: selection at a rebalance with data up to t vs data up to END
t = START + pd.Timedelta(days=9)
H1 = pd.Timedelta(hours=1)
lo = t - pd.Timedelta(days=9)
perp = pd.DataFrame({s: CC.fetch_closes_range(s, lo, END - H1, "linear") for s in CC.UNIVERSE})
spot = pd.DataFrame({s: CC.fetch_closes_range(s, lo, END - H1, "spot") for s in CC.UNIVERSE})
fund = {s: CC.fetch_funding_range(s, lo, END) for s in CC.UNIVERSE}
full = CC.select_at(t, fund, perp, spot)
cut = CC.select_at(t, {s: f[f.index <= t] for s, f in fund.items()},
                   perp[perp.index <= t - H1], spot[spot.index <= t - H1])
ok2 = full == cut
fails += not ok2
print(f"[{'PASS' if ok2 else 'FAIL'}] no look-ahead at {t}: same coins={ok2} ({len(full)} coins)")

print(f"\n   summary over {START.date()} to {END.date()}: equity {one['equity']:,.2f}, "
      f"carry {one['cum_carry']*100:+.3f}%, price {one['cum_price']*100:+.3f}%, "
      f"cost {one['cum_cost']*100:.3f}%")
print("ALL TESTS PASSED" if fails == 0 else f"{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
