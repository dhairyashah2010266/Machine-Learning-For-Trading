"""
Robustness checks for the two strongest results from experiments.py:
SPY 5-day SVM and Nifty 50 1-day Random Forest.
Run after placing experiments.py in the same folder:  python robustness.py
"""
import os, numpy as np, pandas as pd
if not os.path.exists("experiments.py"):
    raise SystemExit("experiments.py must be in the same folder as this script.")
import experiments as E
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
rng = np.random.default_rng(0)
print("Downloading VIX...")
VIX = E.load("^VIX")["Close"]

def prep(ticker, h):
    df = E.load(ticker)
    lag = 1 if ticker.startswith("^NSE") else 0
    vix = VIX.reindex(df.index.union(VIX.index)).ffill().shift(lag).reindex(df.index)
    v = df["Volume"]
    return E.make_dataset(df, vix, h, (v.isna() | (v <= 0)).mean() < 0.05)

def block_p(y, s, auc, n=1000, gap=63):
    """Circular-shift test: keeps the time structure of the labels, fair for overlapping windows."""
    yv, sv, N = y.values, s.values, len(y)
    hits = sum(roc_auc_score(np.roll(yv, rng.integers(gap, N - gap)), sv) >= auc for _ in range(n))
    return (hits + 1) / (n + 1)

def bt(preds, fwd, h, k=0):
    sig, ret = preds.iloc[k::h], fwd.iloc[k::h]
    strat = sig * ret - sig.diff().abs().fillna(sig.iloc[0]) * E.COST_BPS / 1e4
    return strat, ret

def sharpe(r, h):
    return E.perf(r, h)[0]["Sharpe"]

def evaluate(X, y, fwd, h, model, start, end=None):
    E.TEST_START = pd.Timestamp(start)
    preds, scores = E.walk_forward(model, X, y, h)
    if end:
        keep = preds.index < pd.Timestamp(end)
        preds, scores = preds[keep], scores[keep]
    yt, ft = y.loc[preds.index], fwd.loc[preds.index]
    auc = roc_auc_score(yt, scores)
    return preds, scores, yt, ft, auc

CANDIDATES = [("SPY", 5, "SVM"), ("^NSEI", 1, "Random Forest")]
verdicts = []

for ticker, h, mname in CANDIDATES:
    title = f"{ticker} {h}d / {mname}"
    print("\n" + "=" * 90 + f"\nROBUSTNESS CHECK: {title}\n" + "=" * 90)
    X, y, fwd = prep(ticker, h)
    base, grid = E.model_specs()[mname]
    passed = 0

    # TEST 1: original period, stricter p-value
    tr = X.index < pd.Timestamp("2021-01-01")
    model, params, _ = E.tune(base, grid, X[tr], y[tr], h)
    preds, scores, yt, ft, auc = evaluate(X, y, fwd, h, model, "2021-01-01")
    p = block_p(yt, scores, auc)
    ok1 = p < 0.05; passed += ok1
    print(f"\n[1] Stricter significance test (2021 onward, params {params})")
    print(f"    AUC {auc:.3f}   block-permutation p-value {p:.3f}   -> {'PASS' if ok1 else 'FAIL'}")

    # TEST 2: different rebalancing start day
    rows = []
    for k in range(h):
        s, r = bt(preds, ft, h, k)
        rows.append({"start_offset": k, "Sharpe": sharpe(s, h), "BH_Sharpe": sharpe(r, h)})
    t2 = pd.DataFrame(rows)
    wins = (t2["Sharpe"] > t2["BH_Sharpe"]).mean()
    ok2 = wins >= 0.8; passed += ok2
    print(f"\n[2] Different rebalancing start days (beats Buy & Hold in {wins:.0%} of them) -> {'PASS' if ok2 else 'FAIL'}")
    print(t2.round(3).to_string(index=False))

    # TEST 3: year by year
    s, r = bt(preds, ft, h, 0)
    yrs = []
    for yr in sorted(set(preds.index.year)):
        m = preds.index.year == yr
        ms, mr = s.index.year == yr, r.index.year == yr
        a = roc_auc_score(yt[m], scores[m]) if yt[m].nunique() > 1 else np.nan
        yrs.append({"Year": yr, "AUC": a, "Strategy_ret": (1 + s[ms]).prod() - 1, "BH_ret": (1 + r[mr]).prod() - 1})
    t3 = pd.DataFrame(yrs)
    good_years = (t3["AUC"] > 0.5).sum()
    ok3 = good_years >= len(t3) - 1; passed += ok3
    print(f"\n[3] Year by year (AUC above 0.5 in {good_years} of {len(t3)} years) -> {'PASS' if ok3 else 'FAIL'}")
    print(t3.round(3).to_string(index=False))

    # TEST 4: brand-new period never looked at before (2016-2020), tuned only on data before 2016
    tr16 = X.index < pd.Timestamp("2016-01-01")
    model16, params16, _ = E.tune(base, grid, X[tr16], y[tr16], h)
    p16, s16, yt16, ft16, auc16 = evaluate(X, y, fwd, h, model16, "2016-01-01", "2021-01-01")
    pv16 = block_p(yt16, s16, auc16)
    st16, r16 = bt(p16, ft16, h, 0)
    ok4 = pv16 < 0.05; passed += ok4
    print(f"\n[4] Fresh period 2016-2020 (params {params16})")
    print(f"    AUC {auc16:.3f}   p-value {pv16:.3f}   Sharpe {sharpe(st16, h):.3f} vs Buy & Hold {sharpe(r16, h):.3f}"
          f"   -> {'PASS' if ok4 else 'FAIL'}")

    # TEST 5: does it depend on lucky settings?
    rows = []
    for gp in ParameterGrid(grid):
        m = clone(base).set_params(**gp)
        _, sc, ytt, _, a = evaluate(X, y, fwd, h, m, "2021-01-01")
        rows.append({**gp, "AUC": a})
    t5 = pd.DataFrame(rows)
    share = (t5["AUC"] > 0.52).mean()
    ok5 = share >= 0.5; passed += ok5
    print(f"\n[5] All settings tried (AUC above 0.52 for {share:.0%} of them) -> {'PASS' if ok5 else 'FAIL'}")
    print(t5.round(3).to_string(index=False))

    verdicts.append((title, passed))
    E.TEST_START = pd.Timestamp("2021-01-01")

print("\n" + "=" * 90 + "\nFINAL ROBUSTNESS VERDICT\n" + "=" * 90)
for title, passed in verdicts:
    label = "LIKELY REAL" if passed >= 4 else ("MIXED" if passed >= 2 else "LIKELY LUCK")
    print(f"{title:<28} passed {passed}/5 tests  ->  {label}")
print("\nDONE")
