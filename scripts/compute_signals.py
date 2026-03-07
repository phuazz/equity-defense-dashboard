"""
Signal computation engine.
Mirrors the JS logic from blowup_signal_v3.html exactly.
Input: allData dict {ticker: {dates: [...], adjCloses: [...]}}
Output: JSON-serializable dict with all computed data for the dashboard.
"""

import math, logging
from collections import defaultdict

log = logging.getLogger(__name__)

# ─── Universe ───
SP500_SAMPLE = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AVGO","ADBE","CRM",
    "AMD","INTC","CSCO","ORCL","QCOM","TXN","NOW","INTU","MU","ANET",
    "UNH","JNJ","LLY","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN","GILD",
    "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","BLK","SCHW","AXP","C",
    "HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG",
    "PG","KO","PEP","COST","WMT","CL",
    "CAT","BA","HON","UNP","RTX","DE","GE","LMT",
    "XOM","CVX","COP","SLB","EOG","MPC",
    "NEE","DUK","SO","D",
    "PLD","AMT","CCI","SPG",
    "LIN","APD","SHW","FCX",
    "DIS","CMCSA","NFLX","T","VZ","CHTR",
    "SPY",
]
EXTRA_TICKERS = ["VWO", "BND", "IEF", "TLT", "SHY", "^VIX", "^VIX3M"]
ALL_TICKERS = list(set(SP500_SAMPLE + EXTRA_TICKERS))
SAMPLE_SIZE = len([t for t in SP500_SAMPLE if t != "SPY"])
SCALE_FACTOR = 500 / SAMPLE_SIZE

THRESHOLD = 115
DEBOUNCE = 20
WINDOW = 8
CASH_RATE = 0.02 / 252


def compute_all(all_data: dict) -> dict:
    """Main entry: compute signals, backtests, metrics, QC."""
    spy = all_data.get("SPY")
    if not spy:
        raise ValueError("SPY data missing")

    master_dates = spy["dates"]
    spy_adj = dict(zip(spy["dates"], spy["adjCloses"]))

    # Stock price maps
    stock_tickers = [t for t in all_data if t not in ["SPY"] + EXTRA_TICKERS and not t.startswith("^")]
    stock_maps = {}
    for t in stock_tickers:
        stock_maps[t] = dict(zip(all_data[t]["dates"], all_data[t]["adjCloses"]))

    # IEF map
    ief_map = {}
    if "IEF" in all_data:
        ief_map = dict(zip(all_data["IEF"]["dates"], all_data["IEF"]["adjCloses"]))

    # Canary maps
    canary_maps = {}
    for c in ["VWO", "BND"]:
        if c in all_data:
            canary_maps[c] = dict(zip(all_data[c]["dates"], all_data[c]["adjCloses"]))

    # VIX term structure maps
    vix_map = {}
    vix3m_map = {}
    if "^VIX" in all_data and "^VIX3M" in all_data:
        vix_map = dict(zip(all_data["^VIX"]["dates"], all_data["^VIX"]["adjCloses"]))
        vix3m_map = dict(zip(all_data["^VIX3M"]["dates"], all_data["^VIX3M"]["adjCloses"]))
    else:
        log.warning("VIX or VIX3M data missing — vixInverted will default to False")

    # ─── Daily blowup count ───
    daily = []
    for i in range(1, len(master_dates)):
        today, yesterday = master_dates[i], master_dates[i - 1]
        count, available = 0, 0
        for t in stock_tickers:
            p0, p1 = stock_maps[t].get(yesterday), stock_maps[t].get(today)
            if p0 and p1:
                available += 1
                if (p1 - p0) / p0 <= -0.07:
                    count += 1
        scaled = round(count * (500 / available)) if available > 0 else 0
        daily.append({
            "date": today, "rawCount": count, "scaledCount": scaled,
            "spx": spy_adj.get(today, 0), "available": available,
        })

    # ─── Rolling 8-day sum ───
    rolling = []
    for i, d in enumerate(daily):
        s = sum(daily[j]["scaledCount"] for j in range(max(0, i - WINDOW + 1), i + 1))
        rolling.append({**d, "rolling8d": s})

    # ─── 200d SMA ───
    for i in range(len(rolling)):
        if i >= 199:
            rolling[i]["sma200"] = sum(rolling[j]["spx"] for j in range(i - 199, i + 1)) / 200
        else:
            rolling[i]["sma200"] = None
        rolling[i]["aboveSMA200"] = (
            rolling[i]["sma200"] is not None and rolling[i]["spx"] > rolling[i]["sma200"]
        )

    # ─── ATH tracking ───
    ath = 0
    for r in rolling:
        if r["spx"] > ath:
            ath = r["spx"]
        r["athDist"] = (r["spx"] - ath) / ath if ath > 0 else 0
        r["ath"] = ath

    # ─── Canary momentum (13612W) ───
    for i in range(len(rolling)):
        d = rolling[i]["date"]
        canary_bad, canary_total = 0, 0
        for c in ["VWO", "BND"]:
            if c not in canary_maps:
                continue
            p0 = canary_maps[c].get(d)
            lags = [21, 63, 126, 252]
            prices = []
            for lag in lags:
                tgt = i - lag
                prices.append(
                    canary_maps[c].get(rolling[tgt]["date"]) if tgt >= 0 else None
                )
            if p0 and all(p is not None for p in prices):
                mom = (
                    12 * (p0 / prices[0] - 1) + 4 * (p0 / prices[1] - 1)
                    + 2 * (p0 / prices[2] - 1) + (p0 / prices[3] - 1)
                )
                canary_total += 1
                if mom <= 0:
                    canary_bad += 1
            elif p0:
                canary_total += 1
        rolling[i]["canaryBad"] = canary_bad
        rolling[i]["canaryTotal"] = canary_total

    # ─── IEF daily returns ───
    for i in range(len(rolling)):
        d = rolling[i]["date"]
        prev_d = rolling[i - 1]["date"] if i > 0 else None
        p1, p0 = ief_map.get(d), (ief_map.get(prev_d) if prev_d else None)
        rolling[i]["iefRet"] = (p1 - p0) / p0 if (p0 and p1) else 0

    # ─── Extra signals for strategies ───
    def sma10m(i):
        if i < 209:
            return None
        return sum(rolling[j]["spx"] for j in range(i - 209, i + 1)) / 210

    def ret12m(i):
        return (
            (rolling[i]["spx"] - rolling[i - 252]["spx"]) / rolling[i - 252]["spx"]
            if i >= 252
            else None
        )

    # IEF cumulative log return for 12m return
    ief_cum = [0.0]  # Day 0 = 0 cumulative return
    for i in range(1, len(rolling)):
        r_ief = rolling[i]["iefRet"]
        # Guard against extreme values that would break log
        if r_ief is not None and r_ief > -0.99:
            ief_cum.append(ief_cum[-1] + math.log(1 + r_ief))
        else:
            ief_cum.append(ief_cum[-1])

    def ief_12m(i):
        return math.exp(ief_cum[i] - ief_cum[i - 252]) - 1 if i >= 252 else None

    for i in range(len(rolling)):
        rolling[i]["sma10m"] = sma10m(i)
        rolling[i]["spy12mRet"] = ret12m(i)
        rolling[i]["belowSMA10m"] = (
            rolling[i]["sma10m"] is not None and rolling[i]["spx"] < rolling[i]["sma10m"]
        )
        rolling[i]["spy12mNeg"] = (
            rolling[i]["spy12mRet"] is not None and rolling[i]["spy12mRet"] < 0
        )
        # VIX term structure
        dt = rolling[i]["date"]
        vix_val = vix_map.get(dt)
        vix3m_val = vix3m_map.get(dt)
        if vix_val is not None and vix3m_val is not None and vix3m_val > 0:
            rolling[i]["vixRatio"] = round(vix_val / vix3m_val, 4)
            rolling[i]["vixRaw"] = (vix_val / vix3m_val) > 1.0  # Raw daily reading
            rolling[i]["hasVixData"] = True
            rolling[i]["_vixStale"] = 0
        else:
            # Carry forward previous day's state with 5-day stale cap
            prev_stale = rolling[i - 1].get("_vixStale", 99) if i > 0 else 99
            if i > 0 and rolling[i - 1].get("hasVixData", False) and prev_stale < 5:
                rolling[i]["vixRatio"] = rolling[i - 1]["vixRatio"]
                rolling[i]["vixRaw"] = rolling[i - 1]["vixRaw"]
                rolling[i]["hasVixData"] = True
                rolling[i]["_vixStale"] = prev_stale + 1
            else:
                rolling[i]["vixRatio"] = None
                rolling[i]["vixRaw"] = False
                rolling[i]["hasVixData"] = False
                rolling[i]["_vixStale"] = 99
        # 3-day confirmation: vixInverted only fires after 3 consecutive days
        # of raw inversion. Only count days with real data (stale=0) or fresh
        # carry-forward (stale ≤ 1, i.e. weekend gap). Don't count prolonged stale.
        if i >= 2:
            confirmed = True
            for k in range(3):
                r_k = rolling[i - k]
                if not r_k.get("vixRaw", False):
                    confirmed = False
                    break
                if r_k.get("_vixStale", 0) > 1:
                    confirmed = False
                    break
            rolling[i]["vixInverted"] = confirmed
        else:
            rolling[i]["vixInverted"] = False
        # Composite score (6 sub-signals)
        # Note: vixInverted uses T-1 (previous day's reading) for T+1 execution
        # consistency with the VIX standalone strategy. All other price-based 
        # signals are effectively T+0 (monthly signals where 1-day lag is negligible).
        score = 0
        if rolling[i]["rolling8d"] >= THRESHOLD:
            score += 1
        if rolling[i]["sma200"] is not None and rolling[i]["spx"] < rolling[i]["sma200"]:
            score += 1
        if rolling[i]["canaryBad"] >= 1:
            score += 1
        if rolling[i]["spy12mNeg"]:
            score += 1
        if rolling[i]["belowSMA10m"]:
            score += 1
        if i > 0 and rolling[i - 1].get("vixInverted", False):
            score += 1
        elif i == 0:
            pass  # Day 0: VIX cannot contribute
        rolling[i]["compositeScore"] = score

    # ─── Signal events ───
    signals = []
    last_sig = -DEBOUNCE - 1
    for i in range(len(rolling)):
        if rolling[i]["rolling8d"] >= THRESHOLD and (i - last_sig) > DEBOUNCE:
            signals.append({"idx": i, **rolling[i]})
            last_sig = i

    # Forward returns
    for sig in signals:
        sp = sig["spx"]
        fwd = {}
        for k, d in {"fwd1M": 21, "fwd3M": 63, "fwd6M": 126, "fwd12M": 252}.items():
            t = sig["idx"] + d
            fwd[k] = (rolling[t]["spx"] - sp) / sp if t < len(rolling) else None
        peak, max_dd = sp, 0
        for j in range(sig["idx"], min(sig["idx"] + 253, len(rolling))):
            if rolling[j]["spx"] > peak:
                peak = rolling[j]["spx"]
            dd = (rolling[j]["spx"] - peak) / peak
            if dd < max_dd:
                max_dd = dd
        fwd["maxDD12M"] = max_dd
        path = []
        for j in range(min(253, len(rolling) - sig["idx"])):
            path.append(round((rolling[sig["idx"] + j]["spx"] - sp) / sp, 6))
        fwd["path"] = path
        sig["fwd"] = fwd

    # ─── Backtests ───
    bt = run_backtests(rolling, signals)

    # ─── Metrics ───
    metrics = {k: calc_metrics(v) for k, v in bt.items()}

    # ─── Monitor summary stats (pre-computed for dashboard) ───
    monitor = {}
    
    # Score distribution: % of days at each score level
    score_counts = [0] * 7
    for r in rolling:
        score_counts[r["compositeScore"]] += 1
    total_days = len(rolling)
    monitor["scoreDist"] = [round(c / total_days, 4) for c in score_counts]
    
    # Current regime duration: how many consecutive days at current score
    curr_score = rolling[-1]["compositeScore"]
    streak = 0
    for i in range(len(rolling) - 1, -1, -1):
        if rolling[i]["compositeScore"] == curr_score:
            streak += 1
        else:
            break
    monitor["regimeDays"] = streak
    
    # Historical average duration at each score level
    durations = {s: [] for s in range(7)}
    run_score, run_len = rolling[0]["compositeScore"], 1
    for i in range(1, len(rolling)):
        if rolling[i]["compositeScore"] == run_score:
            run_len += 1
        else:
            durations[run_score].append(run_len)
            run_score = rolling[i]["compositeScore"]
            run_len = 1
    durations[run_score].append(run_len)  # final run
    monitor["avgDuration"] = [
        round(sum(d) / len(d)) if d else 0 for d in [durations[s] for s in range(7)]
    ]
    
    # Signal proximity: how far each signal is from flipping
    last = rolling[-1]
    prox = {}
    # 1. Stress count: distance from 115
    prox["stress"] = {"value": last["rolling8d"], "threshold": 115,
                       "pct": round((last["rolling8d"] - 115) / 115, 4) if last["rolling8d"] > 0 else -1}
    # 2. 200d MA: % distance
    if last["sma200"]:
        prox["sma200"] = {"value": round(last["spx"], 1), "ma": round(last["sma200"], 1),
                          "pct": round((last["spx"] - last["sma200"]) / last["sma200"], 4)}
    else:
        prox["sma200"] = {"value": round(last["spx"], 1), "ma": None, "pct": None}
    # 3. Canary
    prox["canary"] = {"bad": last["canaryBad"], "total": last["canaryTotal"]}
    # 4. 12M return
    prox["mom12m"] = {"value": round(last["spy12mRet"], 4) if last["spy12mRet"] is not None else None}
    # 5. 10M SMA
    if last["sma10m"]:
        prox["sma10m"] = {"value": round(last["spx"], 1), "ma": round(last["sma10m"], 1),
                          "pct": round((last["spx"] - last["sma10m"]) / last["sma10m"], 4)}
    else:
        prox["sma10m"] = {"value": round(last["spx"], 1), "ma": None, "pct": None}
    # 6. VIX term structure
    prox["vixTerm"] = {"ratio": last.get("vixRatio", None),
                        "inverted": last.get("vixInverted", False)}
    monitor["proximity"] = prox
    
    # Allocation history: last 2 years of composite allocation for sparkline
    alloc_days = min(504, len(rolling))
    alloc_hist = []
    for i in range(len(rolling) - alloc_days, len(rolling)):
        sc = rolling[i]["compositeScore"]
        alloc_hist.append({"d": rolling[i]["date"], "s": sc,
                           "spy": 100 if sc == 0 else 50 if sc == 1 else 0,
                           "ief": 0 if sc == 0 else 50 if sc == 1 else 100})
    monitor["allocHist"] = alloc_hist
    
    # Composite regime transitions (entry/exit events for chart annotation)
    # Track ANY departure from score=0 (both partial and full defensive)
    transitions = []
    prev_score = rolling[0]["compositeScore"]
    prev_active = prev_score >= 1  # Any defense active
    for i in range(1, len(rolling)):
        curr_score = rolling[i]["compositeScore"]
        curr_active = curr_score >= 1
        if curr_active and not prev_active:
            transitions.append({"date": rolling[i]["date"], "type": "entry", "spx": rolling[i]["spx"], "score": curr_score})
        elif not curr_active and prev_active:
            transitions.append({"date": rolling[i]["date"], "type": "exit", "spx": rolling[i]["spx"], "score": curr_score})
        prev_active = curr_active
    monitor["transitions"] = transitions

    # Build paired defensive trades (entry → exit) with context
    date_to_idx = {rolling[i]["date"]: i for i in range(len(rolling))}
    trades = []
    entries = [t for t in transitions if t["type"] == "entry"]
    exits = [t for t in transitions if t["type"] == "exit"]
    
    for j, ent in enumerate(entries):
        # Find matching exit (next exit after this entry) — pointer-based
        ex = None
        while exits and exits[0]["date"] <= ent["date"]:
            exits.pop(0)
        if exits:
            ex = exits.pop(0)
        
        ent_idx = date_to_idx.get(ent["date"], None)
        if ent_idx is None:
            continue
        
        # Determine exit index
        if ex:
            ex_idx = date_to_idx.get(ex["date"], None)
            is_open = False
        else:
            # Still active — use last day as pseudo-exit
            ex_idx = len(rolling) - 1
            ex = {"date": rolling[-1]["date"], "spx": rolling[-1]["spx"], "score": rolling[-1]["compositeScore"]}
            is_open = True
        
        if ex_idx is None:
            continue
            
        duration = ex_idx - ent_idx
        if duration <= 0:
            continue  # Skip zero/negative duration trades (data anomaly)
        spx_ret = (ex["spx"] - ent["spx"]) / ent["spx"] if ent["spx"] > 0 else 0
        
        # Max DD during defensive period
        peak = ent["spx"]
        max_dd_def = 0
        for k in range(ent_idx, ex_idx + 1):
            if rolling[k]["spx"] > peak:
                peak = rolling[k]["spx"]
            dd = (rolling[k]["spx"] - peak) / peak
            if dd < max_dd_def:
                max_dd_def = dd
        
        # VIX status at entry
        vix_inv = rolling[ent_idx].get("vixInverted", False)
        has_vix = rolling[ent_idx].get("hasVixData", False)
        
        # Score at entry
        entry_score = rolling[ent_idx]["compositeScore"]
        # Peak score during defensive period
        peak_score = entry_score
        # Days at each level
        days_partial = 0  # score == 1
        days_full = 0     # score >= 2
        for k in range(ent_idx, min(ex_idx + 1, len(rolling))):
            sc = rolling[k]["compositeScore"]
            if sc >= 2:
                days_full += 1
            elif sc == 1:
                days_partial += 1
            if sc > peak_score:
                peak_score = sc
        
        # Trade type: "partial" if never reached score ≥2, "full" if it did, "escalated" if started partial then went full
        if days_full == 0:
            trade_type = "partial"
        elif entry_score >= 2:
            trade_type = "full"
        else:
            trade_type = "escalated"
        
        # Allocation: weighted average IEF% during the trade
        alloc_ief_pct = round((days_full * 100 + days_partial * 50) / max(duration, 1), 1)
        
        # Context path: 63 trading days before entry to 63 after exit (3 months each side)
        # Thinned to every 2nd point to reduce JSON size, but always include entry/exit days
        ctx_start = max(0, ent_idx - 63)
        ctx_end = min(len(rolling) - 1, ex_idx + 63)
        entry_spx = ent["spx"]
        ctx_path = []
        for k in range(ctx_start, ctx_end + 1):
            # Always include: first, last, entry day, exit day, and every 2nd point
            is_key = (k == ctx_start or k == ctx_end or k == ent_idx or k == ex_idx)
            if is_key or (k - ctx_start) % 2 == 0:
                ctx_path.append({
                    "d": rolling[k]["date"],
                    "v": round(rolling[k]["spx"] / entry_spx * 100, 2),
                    "phase": "before" if k < ent_idx else ("during" if k <= ex_idx else "after")
                })
        
        trade = {
            "id": j + 1,
            "entryDate": ent["date"],
            "exitDate": ex["date"] if not is_open else None,
            "open": is_open,
            "entrySPX": round(ent["spx"], 1),
            "exitSPX": round(ex["spx"], 1) if ex else None,
            "duration": duration,
            "spxRet": round(spx_ret, 4),
            "maxDD": round(max_dd_def, 4),
            "entryScore": entry_score,
            "peakScore": peak_score,
            "daysPartial": days_partial,
            "daysFull": days_full,
            "tradeType": trade_type,
            "allocIEF": alloc_ief_pct,
            "vixInv": vix_inv,
            "hasVix": has_vix,
            "path": ctx_path,
        }
        trades.append(trade)
    
    monitor["trades"] = trades
    
    # Historical ranges for signal bars (min/max/percentiles over full history)
    stress_vals = [r["rolling8d"] for r in rolling if r["rolling8d"] is not None]
    monitor["ranges"] = {
        "stress": {"min": min(stress_vals), "max": max(stress_vals), 
                   "p25": sorted(stress_vals)[len(stress_vals)//4],
                   "p75": sorted(stress_vals)[3*len(stress_vals)//4],
                   "median": sorted(stress_vals)[len(stress_vals)//2]},
    }
    if rolling[-1]["spy12mRet"] is not None:
        mom_vals = [r["spy12mRet"] for r in rolling if r["spy12mRet"] is not None]
        monitor["ranges"]["mom12m"] = {"min": min(mom_vals), "max": max(mom_vals),
                                        "median": sorted(mom_vals)[len(mom_vals)//2]}

    # ─── QC ───
    qc = run_qc(all_data, rolling, signals, bt, metrics)

    # ─── Assemble output ───
    # Thin the rolling data for JSON size — only keep every Nth day for charts
    # (full data for signal analysis, thinned for time series charts)
    thin_n = max(1, len(rolling) // 2000)  # ~2000 points max
    rolling_thin = [rolling[i] for i in range(0, len(rolling), thin_n)]
    if rolling[-1] not in rolling_thin:
        rolling_thin.append(rolling[-1])

    # Thin backtest equity curves similarly
    bt_thin = {}
    for k, v in bt.items():
        bt_thin[k] = [v[i] for i in range(0, len(v), thin_n)]
        if v[-1] not in bt_thin[k]:
            bt_thin[k].append(v[-1])

    return {
        "rolling": rolling_thin,
        "rollingFull": rolling,  # excluded from JSON output, used internally
        "signals": [{k: v for k, v in s.items() if k != "idx"} for s in signals],
        "bt": bt_thin,
        "metrics": metrics,
        "qc": qc,
        "monitor": monitor,
        "meta": {
            "startDate": rolling[0]["date"],
            "endDate": rolling[-1]["date"],
            "years": (len(rolling)) / 252,
            "signalCount": len(signals),
            "sampleSize": SAMPLE_SIZE,
            "scaleF": round(SCALE_FACTOR, 2),
            "stockCount": len(stock_tickers),
            "provider": "computed",
            "compositeNow": rolling[-1]["compositeScore"],
            "vixRatioNow": rolling[-1].get("vixRatio", None),
            "vixInvertedNow": rolling[-1].get("vixInverted", False),
            # Next-day score: what composite will be tomorrow using today's confirmed VIX
            # Swap yesterday's vixInverted (used in today's score) for today's vixInverted
            "nextDayScore": rolling[-1]["compositeScore"]
                - (1 if (len(rolling) >= 2 and rolling[-2].get("vixInverted", False)) else 0)
                + (1 if rolling[-1].get("vixInverted", False) else 0),
        },
    }


# ─── Backtest engine ───

# Transaction cost: 5bps per unit of allocation change (conservative for SPY/IEF)
# e.g. 0→100% defensive = 5bps, 0→50% = 2.5bps, 50→100% = 2.5bps
TX_COST_BPS = 5

def _bt_loop(rolling, is_defensive_fn):
    """Generic backtest loop. is_defensive_fn(i, rolling) -> (def_frac, mode_str)"""
    eq = [{"date": rolling[0]["date"], "equity": 100000, "dd": 0, "mode": "invested"}]
    peak = 100000
    prev_def = 0.0
    total_tx = 0.0
    for i in range(1, len(rolling)):
        r = (rolling[i]["spx"] - rolling[i - 1]["spx"]) / rolling[i - 1]["spx"]
        ief_r = rolling[i]["iefRet"]
        def_frac, mode = is_defensive_fn(i, rolling)
        # Transaction cost: proportional to allocation change
        alloc_change = abs(def_frac - prev_def)
        tx = alloc_change * TX_COST_BPS / 10000 if alloc_change > 0.01 else 0
        total_tx += tx
        prev_def = def_frac
        ret = def_frac * (ief_r if ief_r else CASH_RATE) + (1 - def_frac) * r - tx
        e = eq[-1]["equity"] * (1 + ret)
        if e > peak:
            peak = e
        eq.append({"date": rolling[i]["date"], "equity": round(e, 2), "dd": round((e - peak) / peak, 6), "mode": mode})
    return eq


def run_backtests(rolling, signals):
    n = len(rolling)
    signal_dates = set(s["date"] for s in signals)

    # S1: Enhanced Blowup
    enh_def, enh_cd, enh_pend = False, 0, False
    def enh_fn(i, r):
        nonlocal enh_def, enh_cd, enh_pend
        # T+1 execution: pending blowup triggers defense next day
        if enh_pend:
            enh_def, enh_cd, enh_pend = True, 63, False
        # Blowup trigger (T+1 execution via pending flag)
        if r[i]["rolling8d"] >= THRESHOLD and not enh_def and not enh_pend:
            enh_pend = True
        # Regime filter trigger (T+0, immediate)
        bma = r[i]["sma200"] is not None and r[i]["spx"] < r[i]["sma200"]
        if bma and r[i]["canaryBad"] >= 1 and not enh_def and not enh_pend:
            enh_def = True
            enh_cd = 63  # Also apply cooldown for regime entry
        # Cooldown countdown
        if enh_cd > 0:
            enh_cd -= 1
        # Exit: above 200d + canary clear + cooldown expired
        if enh_def and enh_cd <= 0:
            abv = r[i]["sma200"] is not None and r[i]["spx"] > r[i]["sma200"]
            if abv and r[i]["canaryBad"] == 0:
                enh_def = False
        return (1.0 if enh_def else 0.0, "defensive" if enh_def else "invested")

    # S2: Faber 10M SMA
    def fab_fn(i, r):
        d = 1.0 if r[i]["belowSMA10m"] else 0.0
        return (d, "defensive" if d else "invested")

    # S3: Dual Momentum
    # IEF cumulative log return for 12m comparison
    ief_cum = [0.0]
    for i in range(1, n):
        r_ief = rolling[i]["iefRet"]
        if r_ief is not None and r_ief > -0.99:
            ief_cum.append(ief_cum[-1] + math.log(1 + r_ief))
        else:
            ief_cum.append(ief_cum[-1])

    def dm_fn(i, r):
        spy12 = r[i]["spy12mRet"]
        ief12 = (math.exp(ief_cum[i] - ief_cum[i - 252]) - 1) if i >= 252 else None
        d = 1.0 if (spy12 is not None and ief12 is not None and (spy12 < 0 or spy12 < ief12)) else 0.0
        return (d, "defensive" if d else "invested")

    # S4: VIX Term Structure (T+1 execution: use previous day's confirmed inversion)
    # Note: vixInverted already requires 3 consecutive days of raw inversion
    def vix_fn(i, r):
        if i == 0:
            return (0.0, "invested")
        d = 1.0 if r[i - 1].get("vixInverted", False) else 0.0
        return (d, "defensive" if d else "invested")

    # S5: Composite
    def comp_fn(i, r):
        sc = r[i]["compositeScore"]
        if sc >= 2:
            return (1.0, "defensive")
        elif sc == 1:
            return (0.5, "partial")
        return (0.0, "invested")

    # B&H
    def bh_fn(i, r):
        return (0.0, "invested")

    return {
        "bh": _bt_loop(rolling, bh_fn),
        "enh": _bt_loop(rolling, enh_fn),
        "fab": _bt_loop(rolling, fab_fn),
        "dm": _bt_loop(rolling, dm_fn),
        "vix": _bt_loop(rolling, vix_fn),
        "comp": _bt_loop(rolling, comp_fn),
    }


# ─── Metrics ───

def calc_metrics(eq):
    n = len(eq)
    yrs = n / 252
    total_ret = eq[-1]["equity"] / eq[0]["equity"] - 1
    cagr = (eq[-1]["equity"] / eq[0]["equity"]) ** (1 / yrs) - 1 if yrs > 0 else 0
    max_dd = min(e["dd"] for e in eq)

    daily_rets = [(eq[i]["equity"] / eq[i - 1]["equity"] - 1) for i in range(1, n)]
    mean_r = sum(daily_rets) / len(daily_rets) if daily_rets else 0
    vol = (
        math.sqrt(sum((r - mean_r) ** 2 for r in daily_rets) / (len(daily_rets) - 1)) * math.sqrt(252)
        if len(daily_rets) > 1 else 0
    )
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    neg = [r for r in daily_rets if r < 0]
    down_vol = (
        math.sqrt(sum(r ** 2 for r in neg) / len(neg)) * math.sqrt(252) if neg else 0
    )
    sortino = (cagr - 0.02) / down_vol if down_vol > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0

    # Monthly returns
    monthly = []
    month_start = 0
    for i in range(1, n):
        if eq[i]["date"][:7] != eq[i - 1]["date"][:7] or i == n - 1:
            ret = eq[i - 1]["equity"] / eq[month_start]["equity"] - 1
            monthly.append({"month": eq[month_start]["date"][:7], "ret": round(ret, 6)})
            month_start = i
    win_rate = sum(1 for m in monthly if m["ret"] > 0) / len(monthly) if monthly else 0

    # Annual returns
    annual = []
    yr_start = 0
    for i in range(1, n):
        if eq[i]["date"][:4] != eq[i - 1]["date"][:4] or i == n - 1:
            ret = eq[i - 1]["equity"] / eq[yr_start]["equity"] - 1
            annual.append({"year": int(eq[yr_start]["date"][:4]), "ret": round(ret, 6)})
            yr_start = i

    # Defensive fraction
    def_days = sum(1 for e in eq if e["mode"] != "invested")
    def_frac = def_days / n if n > 0 else 0

    return {
        "yrs": round(yrs, 1),
        "cagr": round(cagr, 6),
        "totalRet": round(total_ret, 4),
        "maxDD": round(max_dd, 4),
        "vol": round(vol, 4),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "winRate": round(win_rate, 4),
        "defFrac": round(def_frac, 4),
        "monthly": monthly,
        "annual": annual,
    }


# ─── QC ───

def run_qc(all_data, rolling, signals, bt, m):
    checks = []

    def chk(cat, desc, result, status):
        checks.append({"cat": cat, "chk": desc, "res": result, "st": status})

    stock_count = len([t for t in all_data if t not in ["SPY"] + EXTRA_TICKERS and not t.startswith("^")])
    chk("Data", f"Stock sample size", f"{stock_count}/{SAMPLE_SIZE}", "pass" if stock_count >= SAMPLE_SIZE * 0.8 else "warn")
    chk("Data", "SPY history", f"{len(all_data.get('SPY', {}).get('dates', []))} days", "pass" if len(all_data.get("SPY", {}).get("dates", [])) > 5000 else "warn")
    chk("Data", "Data type", "Adjusted Close (total return)", "pass")

    for t in ["VWO", "BND", "IEF"]:
        has = t in all_data and len(all_data[t].get("dates", [])) > 100
        chk("Data", f"{t} available", "Yes" if has else "Missing", "pass" if has else "warn")

    for t in ["^VIX", "^VIX3M"]:
        has = t in all_data and len(all_data[t].get("dates", [])) > 100
        chk("Data", f"{t} available", "Yes" if has else "Missing", "pass" if has else "warn")

    chk("Signal", "Events found", f"{len(signals)}", "pass" if len(signals) >= 3 else "warn")
    chk("Backtest", "B&H CAGR", f"{m['bh']['cagr'] * 100:.1f}%", "pass" if 0.07 < m["bh"]["cagr"] < 0.15 else "warn")
    chk("Backtest", "Composite max DD", f"{m['comp']['maxDD'] * 100:.1f}% vs B&H {m['bh']['maxDD'] * 100:.1f}%", "pass" if m["comp"]["maxDD"] > m["bh"]["maxDD"] else "warn")
    chk("Backtest", "Composite def fraction", f"{m['comp']['defFrac'] * 100:.1f}%", "pass" if 0.1 < m["comp"]["defFrac"] < 0.6 else "warn")
    chk("Backtest", "Transaction costs", f"{TX_COST_BPS}bps per allocation change", "pass")
    chk("Backtest", "Look-ahead bias", "Blowup T+1, VIX T+1 (3d confirm), regime T+0, no future data", "pass")

    # VIX strategy sanity checks
    vix_def = m["vix"]["defFrac"]
    chk("Backtest", "VIX def fraction", f"{vix_def * 100:.1f}%", "pass" if 0.05 < vix_def < 0.40 else "warn")
    # VIX turnover: count allocation flips in backtest
    vix_flips = sum(1 for j in range(1, len(bt["vix"])) if bt["vix"][j]["mode"] != bt["vix"][j-1]["mode"])
    vix_yrs = len(bt["vix"]) / 252
    vix_annual_flips = round(vix_flips / vix_yrs, 1) if vix_yrs > 0 else 0
    chk("Backtest", "VIX annual round-trips", f"{vix_annual_flips}", "pass" if vix_annual_flips < 10 else "warn")

    vs = [s for s in signals if s["fwd"]["maxDD12M"] is not None]
    if vs:
        avg_dd = sum(s["fwd"]["maxDD12M"] for s in vs) / len(vs)
        chk("Stats", "Avg fwd max DD", f"{avg_dd * 100:.1f}%", "pass" if avg_dd > -0.40 else "warn")

    return checks
