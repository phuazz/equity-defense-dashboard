"""WS4 H4 — composite attribution: leave-one-out and solo runs per sub-signal.

Pre-registered spec: C:/dev/KICKOFF_ws4-breadth-stresstest.md. Reads the
committed daily rows from data/signals.json and re-runs the S5 composite
engine with sub-signal subsets. The engine below replicates
compute_signals.py verbatim (score latching lines 245-261, _bt_loop lines
588-610, comp_fn lines 694-714) rather than importing it, because that module
executes its pipeline at import time. Replication is validated two ways
before any variant runs: recomputed scores must equal the stored
compositeScore on every row, and the recomputed full-composite equity curves
must match the stored bt.comp / bt.compIef curves exactly.

Costs: the repo's own model — 10 bps per unit of allocation change
(TX_COST_BPS), changes below 0.01 free, cash fallback 2% annualised. The
spec's 5 bps placeholder is superseded by the repo model and logged as such.

Output: reviews/2026-07-03_ws4_attribution.json plus a printed summary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIGNALS = ROOT / "data" / "signals.json"
OUT = ROOT / "reviews" / "2026-07-03_ws4_attribution.json"

THRESHOLD = 115
TX_COST_BPS = 10
CASH_RATE = 0.02 / 252
ALLOC_MAP = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
COMP_COOLDOWN = 42

SUBSIGNALS = ["blowup", "ma200", "mom12", "sma10m", "vix"]


def compute_scores(rolling, include):
    """Replicates compute_signals.py:245-261 for a sub-signal subset."""
    scores = []
    monthly_mom = [0]
    monthly_sma = [0]
    for i in range(len(rolling)):
        prev = rolling[i - 1] if i > 0 else rolling[i]
        prev_month = rolling[i - 1]["date"][:7] if i > 0 else ""
        curr_month = rolling[i]["date"][:7]
        if prev_month != curr_month:
            monthly_mom[0] = 1 if prev.get("spy12mNeg", False) else 0
            monthly_sma[0] = 1 if prev.get("belowSMA10m", False) else 0
        score = 0
        if "blowup" in include and prev["rolling8d"] >= THRESHOLD:
            score += 1
        if "ma200" in include and prev.get("sma200") is not None and prev["spx"] < prev["sma200"]:
            score += 1
        if "mom12" in include:
            score += monthly_mom[0]
        if "sma10m" in include:
            score += monthly_sma[0]
        if "vix" in include and prev.get("vixInverted", False):
            score += 1
        scores.append(score)
    return scores


def bt_comp(rolling, scores, def_ret_key):
    """Replicates _bt_loop + make_comp_fn, driven by a supplied score list.
    Compounds off the ROUNDED equity, as the source engine does."""
    eq = [{"date": rolling[0]["date"], "equity": 100000, "dd": 0, "mode": "invested"}]
    peak = 100000.0
    prev_def = 0.0
    total_tx = 0.0
    switches = 0
    days_def = 0
    cd = 0
    for i in range(1, len(rolling)):
        r = (rolling[i]["spx"] - rolling[i - 1]["spx"]) / rolling[i - 1]["spx"]
        # exact source semantics: `or` coerces None AND 0.0 to CASH_RATE
        def_r = rolling[i].get(def_ret_key, CASH_RATE) or CASH_RATE
        sc = min(scores[i], 5)
        raw_d = ALLOC_MAP[sc]
        if raw_d > 0 and cd <= 0:
            cd = COMP_COOLDOWN
        if cd > 0:
            cd -= 1
            d = max(raw_d, 0.5)
        else:
            d = raw_d
        mode = "invested" if d == 0 else ("partial" if d < 1.0 else "defensive")
        alloc_change = abs(d - prev_def)
        tx = alloc_change * TX_COST_BPS / 10000 if alloc_change > 0.01 else 0
        if alloc_change > 0.01:
            switches += 1
        total_tx += tx
        prev_def = d
        if d > 0:
            days_def += 1
        ret = d * def_r + (1 - d) * r - tx
        e = round(eq[-1]["equity"] * (1 + ret), 2)
        if e > peak:
            peak = e
        eq.append({"date": rolling[i]["date"], "equity": e,
                   "dd": round((e - peak) / peak, 6), "mode": mode})
    return eq, {"switches": switches, "days_defensive": days_def,
                "total_tx_frac": round(total_tx, 6)}


def metrics(eq):
    """WS4 metrics — one definition across all variants: daily net returns,
    Sharpe = mean/std * sqrt(252) (no cash subtraction), CAGR, MaxDD."""
    rets = []
    for i in range(1, len(eq)):
        rets.append(eq[i]["equity"] / eq[i - 1]["equity"] - 1)
    n = len(rets)
    yrs = n / 252
    cagr = (eq[-1]["equity"] / eq[0]["equity"]) ** (1 / yrs) - 1
    mu = sum(rets) / n
    var = sum((x - mu) ** 2 for x in rets) / (n - 1)
    sharpe = mu / math.sqrt(var) * math.sqrt(252) if var > 0 else float("nan")
    maxdd = min(row["dd"] for row in eq)
    return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 3),
            "maxdd": round(maxdd, 4), "years": round(yrs, 1),
            "final_equity": eq[-1]["equity"]}


def run_variant(rolling, include, def_ret_key):
    scores = compute_scores(rolling, include)
    eq, ops = bt_comp(rolling, scores, def_ret_key)
    return {**metrics(eq), **ops}


def main():
    d = json.loads(SIGNALS.read_text(encoding="utf-8"))
    rolling = d["rolling"]
    print(f"rolling rows: {len(rolling)} {rolling[0]['date']} -> {rolling[-1]['date']}")

    # ── Validation 1: recomputed full score == stored compositeScore
    full_scores = compute_scores(rolling, set(SUBSIGNALS))
    mism = [i for i in range(len(rolling))
            if full_scores[i] != rolling[i].get("compositeScore")]
    print(f"score mismatches vs stored compositeScore: {len(mism)}")
    if mism:
        for i in mism[:5]:
            print("  ", rolling[i]["date"], "recomputed", full_scores[i],
                  "stored", rolling[i].get("compositeScore"))
        raise SystemExit("FAIL-LOUD: score replication mismatch — do not proceed")

    # ── Validation 2: recomputed curves == stored bt curves
    checks = {}
    for key, leg in (("comp", "shyRet"), ("compIef", "iefRet")):
        eq, _ = bt_comp(rolling, full_scores, leg)
        stored = d["bt"][key]
        n_diff = sum(1 for a, b in zip(eq, stored)
                     if abs(a["equity"] - b["equity"]) > 0.01)
        checks[key] = {"len_mine": len(eq), "len_stored": len(stored),
                       "rows_differing": n_diff,
                       "final_mine": eq[-1]["equity"],
                       "final_stored": stored[-1]["equity"]}
        print(f"curve check {key}: {checks[key]}")
    if any(c["rows_differing"] > 0 or c["len_mine"] != c["len_stored"]
           for c in checks.values()):
        raise SystemExit("FAIL-LOUD: equity-curve replication mismatch — do not proceed")

    # ── Variants: full, leave-one-out, solo × {IEF, SHY} × {full window, 2014→}
    windows = {"full": rolling,
               "2014on": [r for r in rolling if r["date"] >= "2014-01-01"]}
    # Leave-one-out only. Solo-through-ALLOC_MAP is degenerate (a single
    # sub-signal maxes at score 1 and ALLOC_MAP[1] is 0.0, i.e. buy-and-hold),
    # so the solo view is served by the repo's own standalone strategies
    # (S1-S4), quoted below as standalone_reference.
    variants = {"full": set(SUBSIGNALS)}
    for s in SUBSIGNALS:
        variants[f"minus_{s}"] = set(SUBSIGNALS) - {s}

    results = {}
    for wname, rows in windows.items():
        results[wname] = {}
        for leg_name, leg in (("ief", "iefRet"), ("shy", "shyRet")):
            results[wname][leg_name] = {}
            for vname, inc in variants.items():
                results[wname][leg_name][vname] = run_variant(rows, inc, leg)

    # ── Pre-registered decision rule (spec H4): Blowup, full window, IEF leg
    base = results["full"]["ief"]["full"]
    minus = results["full"]["ief"]["minus_blowup"]
    d_sharpe = round(minus["sharpe"] - base["sharpe"], 3)
    d_maxdd_ppt = round((minus["maxdd"] - base["maxdd"]) * 100, 2)
    if abs(d_sharpe) <= 0.05 and abs(d_maxdd_ppt) <= 2.0:
        verdict = "ON_NOTICE"
    elif d_sharpe > 0.05 or d_maxdd_ppt > 2.0:
        verdict = "RECOMMEND_DEMOTION"
    else:
        verdict = "KEEP"
    decision = {"rule": "spec H4 bands: |dSharpe|<=0.05 and |dMaxDD|<=2ppt -> on notice",
                "delta_sharpe_removing_blowup": d_sharpe,
                "delta_maxdd_ppt_removing_blowup": d_maxdd_ppt,
                "verdict": verdict}

    out = {"spec": "KICKOFF_ws4-breadth-stresstest.md", "generated": "2026-07-03",
           "engine": {"tx_cost_bps": TX_COST_BPS, "alloc_map": ALLOC_MAP,
                      "cooldown_days": COMP_COOLDOWN, "threshold": THRESHOLD,
                      "note": "repo cost model used; spec 5bps placeholder superseded"},
           "replication_checks": checks, "variants": results,
           "standalone_reference": {
               "note": ("Repo-computed standalone strategy metrics (S1 enhanced "
                        "blowup, 200d MA, Faber 10m, dual momentum, VIX term "
                        "structure, buy-and-hold). Definitions differ from bare "
                        "sub-signals: S1 carries a 63-day defence window and "
                        "200dMA exit. Quoted, not recomputed."),
               "metrics": {k: d["metrics"][k] for k in
                           ("bh", "enh", "ma200", "fab", "dm", "vix")
                           if k in d.get("metrics", {})}},
           "decision_blowup": decision}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n== full window, IEF leg ==")
    hdr = f"{'variant':<16}{'sharpe':>8}{'cagr':>8}{'maxdd':>8}{'switches':>10}{'daysDef':>9}"
    print(hdr)
    for vname in variants:
        m = results["full"]["ief"][vname]
        print(f"{vname:<16}{m['sharpe']:>8.3f}{m['cagr']:>8.2%}{m['maxdd']:>8.1%}"
              f"{m['switches']:>10}{m['days_defensive']:>9}")
    print("\ndecision:", decision)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
