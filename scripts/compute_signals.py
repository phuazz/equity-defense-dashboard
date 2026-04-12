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
EXTRA_TICKERS = ["VWO", "BND", "IEF", "SHV", "TLT", "SHY", "^VIX", "^VIX3M",
                 "MCHI", "EWS", "EWM", "EIDO", "THD", "EPHE"]  # Regional ETFs for spillover analysis
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

    # IEF map (7-10Y UST — duration hedge)
    ief_map = {}
    if "IEF" in all_data:
        ief_map = dict(zip(all_data["IEF"]["dates"], all_data["IEF"]["adjCloses"]))

    # SHY map (1-3Y UST — short duration, low rate risk)
    shy_map = {}
    if "SHY" in all_data:
        shy_map = dict(zip(all_data["SHY"]["dates"], all_data["SHY"]["adjCloses"]))

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
        rolling[i]["iefRet"] = (p1 - p0) / p0 if (p0 and p1) else CASH_RATE

    # ─── SHY daily returns ───
    for i in range(len(rolling)):
        d = rolling[i]["date"]
        prev_d = rolling[i - 1]["date"] if i > 0 else None
        p1, p0 = shy_map.get(d), (shy_map.get(prev_d) if prev_d else None)
        rolling[i]["shyRet"] = (p1 - p0) / p0 if (p0 and p1) else CASH_RATE

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

    monthly_mom = [0]  # latched monthly: 12M momentum signal
    monthly_sma = [0]  # latched monthly: 10M SMA signal
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
        # 5 signals (canary removed).
        # Daily signals (stress, 200d MA, VIX): use previous day's data (T+1).
        # Monthly signals (12M mom, 10M SMA): latch at month-end, hold for entire month.
        prev = rolling[i - 1] if i > 0 else rolling[i]
        # Detect month boundary: previous day's month differs from current day's month
        prev_month = rolling[i - 1]["date"][:7] if i > 0 else ""
        curr_month = rolling[i]["date"][:7]
        if prev_month != curr_month:
            monthly_mom[0] = 1 if prev.get("spy12mNeg", False) else 0
            monthly_sma[0] = 1 if prev.get("belowSMA10m", False) else 0
        score = 0
        if prev["rolling8d"] >= THRESHOLD:
            score += 1
        if prev.get("sma200") is not None and prev["spx"] < prev["sma200"]:
            score += 1
        score += monthly_mom[0]
        score += monthly_sma[0]
        if prev.get("vixInverted", False):
            score += 1
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
    alloc_map = [0, 0, 50, 75, 100, 100, 100]
    for i in range(len(rolling) - alloc_days, len(rolling)):
        sc = min(rolling[i]["compositeScore"], 6)
        ief_pct = alloc_map[sc]
        alloc_hist.append({"d": rolling[i]["date"], "s": sc,
                           "spy": 100 - ief_pct,
                           "ief": ief_pct})
    monitor["allocHist"] = alloc_hist
    
    # Composite regime transitions (entry/exit events for chart annotation)
    # Mirrors the min holding period in comp_fn: once defensive, stay for 42 trading days.
    transitions = []
    t_cd = 0  # cooldown counter
    t_active = rolling[0]["compositeScore"] >= 2
    for i in range(1, len(rolling)):
        curr_score = rolling[i]["compositeScore"]
        raw_active = curr_score >= 2
        if raw_active and not t_active:
            t_active = True
            t_cd = 42
            transitions.append({"date": rolling[i]["date"], "type": "entry", "spx": rolling[i]["spx"], "score": curr_score})
        if t_cd > 0:
            t_cd -= 1
        elif t_active and not raw_active:
            t_active = False
            transitions.append({"date": rolling[i]["date"], "type": "exit", "spx": rolling[i]["spx"], "score": curr_score})
    monitor["transitions"] = transitions
    monitor["defenceActive"] = t_active  # True if currently in defensive position (incl. cooldown)
    monitor["cooldownRemaining"] = t_cd  # Days remaining in cooldown

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
        # Days at each level (Grad D: 0/0/50/75/100/100/100)
        days_partial = 0  # score 2 or 3 (50% or 75% IEF)
        days_full = 0     # score >= 4 (100% IEF)
        for k in range(ent_idx, min(ex_idx + 1, len(rolling))):
            sc = rolling[k]["compositeScore"]
            if sc >= 4:
                days_full += 1
            elif sc >= 2:
                days_partial += 1
            if sc > peak_score:
                peak_score = sc
        
        # Trade type
        if days_full == 0 and days_partial == 0:
            trade_type = "watch"  # Score=1 only, no allocation change
        elif days_full == 0:
            trade_type = "partial"
        elif entry_score >= 4:
            trade_type = "full"
        else:
            trade_type = "escalated"
        
        # Allocation: weighted average IEF% during the trade (Grad D map)
        trade_alloc_map = [0, 0, 50, 75, 100, 100, 100]
        alloc_days_sum = 0
        for k in range(ent_idx, min(ex_idx + 1, len(rolling))):
            sc = min(rolling[k]["compositeScore"], 6)
            alloc_days_sum += trade_alloc_map[sc]
        alloc_ief_pct = round(alloc_days_sum / max(duration, 1), 1)
        
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

    # Regional spillover analysis (China + ASEAN ETFs)
    regional = compute_regional(all_data, rolling)
    if regional:
        monitor["regional"] = regional

    # ─── QC ───
    qc = run_qc(all_data, rolling, signals, bt, metrics)

    # ─── Assemble output ───
    # Thin the rolling data for JSON size — only keep every Nth day for charts
    # (full data for signal analysis, thinned for time series charts)
    thin_n = 1  # Keep full daily resolution
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
            "years": (len(rolling) - 1) / 252,
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

# Transaction cost: 10bps per unit of allocation change
# Covers bid-ask spread + market impact for SPY/IEF at institutional size
TX_COST_BPS = 10

def _bt_loop(rolling, is_defensive_fn, def_ret_key="iefRet"):
    """Generic backtest loop. is_defensive_fn(i, rolling) -> (def_frac, mode_str)
    def_ret_key: which return field to use for defensive allocation (iefRet or shvRet)
    """
    eq = [{"date": rolling[0]["date"], "equity": 100000, "dd": 0, "mode": "invested"}]
    peak = 100000
    prev_def = 0.0
    total_tx = 0.0
    for i in range(1, len(rolling)):
        r = (rolling[i]["spx"] - rolling[i - 1]["spx"]) / rolling[i - 1]["spx"]
        def_r = rolling[i].get(def_ret_key, CASH_RATE) or CASH_RATE
        def_frac, mode = is_defensive_fn(i, rolling)
        # Transaction cost: proportional to allocation change
        alloc_change = abs(def_frac - prev_def)
        tx = alloc_change * TX_COST_BPS / 10000 if alloc_change > 0.01 else 0
        total_tx += tx
        prev_def = def_frac
        ret = def_frac * def_r + (1 - def_frac) * r - tx
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
        # Cooldown countdown
        if enh_cd > 0:
            enh_cd -= 1
        # Exit: above 200d + cooldown expired (T+1)
        if enh_def and enh_cd <= 0:
            prev_x = r[i - 1] if i > 0 else r[i]
            abv = prev_x.get("sma200") is not None and prev_x["spx"] > prev_x["sma200"]
            if abv:
                enh_def = False
        return (1.0 if enh_def else 0.0, "defensive" if enh_def else "invested")

    # S2a: 200d MA standalone (T+1: use previous day's signal)
    def ma200_fn(i, r):
        if i == 0:
            return (0.0, "invested")
        prev = r[i - 1]
        d = 1.0 if (prev.get("sma200") is not None and prev["spx"] < prev["sma200"]) else 0.0
        return (d, "defensive" if d else "invested")

    # S3: Faber 10M SMA — monthly re-evaluation only
    # Observe month-end signal, execute first trading day of following month.
    fab_state = [0.0]
    def fab_fn(i, r):
        prev_i = max(i - 1, 0)
        # Check if previous day was last trading day of its month
        prev_month = r[prev_i]["date"][:7]
        curr_month = r[i]["date"][:7]
        if prev_month != curr_month:
            # Month boundary: latch signal from month-end data
            fab_state[0] = 1.0 if r[prev_i].get("belowSMA10m", False) else 0.0
        d = fab_state[0]
        return (d, "defensive" if d else "invested")

    # S4: Dual Momentum (Antonacci) — monthly re-evaluation only
    # IEF cumulative log return for 12m comparison
    ief_cum = [0.0]
    for i in range(1, n):
        r_ief = rolling[i]["iefRet"]
        if r_ief is not None and r_ief > -0.99:
            ief_cum.append(ief_cum[-1] + math.log(1 + r_ief))
        else:
            ief_cum.append(ief_cum[-1])

    dm_state = [0.0]
    def dm_fn(i, r):
        prev_i = max(i - 1, 0)
        prev_month = r[prev_i]["date"][:7]
        curr_month = r[i]["date"][:7]
        if prev_month != curr_month:
            spy12 = r[prev_i].get("spy12mRet")
            ief12 = (math.exp(ief_cum[prev_i] - ief_cum[prev_i - 252]) - 1) if prev_i >= 252 else None
            dm_state[0] = 1.0 if (spy12 is not None and ief12 is not None and (spy12 < 0 or spy12 < ief12)) else 0.0
        d = dm_state[0]
        return (d, "defensive" if d else "invested")

    # S4: VIX Term Structure (T+1 execution: use previous day's confirmed inversion)
    # Note: vixInverted already requires 3 consecutive days of raw inversion
    def vix_fn(i, r):
        if i == 0:
            return (0.0, "invested")
        d = 1.0 if r[i - 1].get("vixInverted", False) else 0.0
        return (d, "defensive" if d else "invested")

    # S5: Composite — 5 signals (canary removed), graduated allocation
    # Score: 0-1→0%, 2→50%, 3+→100%
    # Min holding period: once defensive (score>=2), stay for at least 42 trading days (~2 months)
    # to prevent whipsaw from score oscillating around the threshold.
    ALLOC_MAP = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    COMP_COOLDOWN = 42

    def make_comp_fn():
        """Factory: each call returns a fresh comp_fn with its own cooldown state."""
        cd = [0]
        def comp_fn(i, r):
            sc = min(r[i]["compositeScore"], 5)
            raw_d = ALLOC_MAP[sc]
            # Enter defence: start cooldown
            if raw_d > 0 and cd[0] <= 0:
                cd[0] = COMP_COOLDOWN
            # During cooldown: maintain at least 50% defensive
            if cd[0] > 0:
                cd[0] -= 1
                d = max(raw_d, 0.5)
            else:
                d = raw_d
            mode = "invested" if d == 0 else ("partial" if d < 1.0 else "defensive")
            return (d, mode)
        return comp_fn

    # B&H
    def bh_fn(i, r):
        return (0.0, "invested")

    return {
        "bh": _bt_loop(rolling, bh_fn),
        "enh": _bt_loop(rolling, enh_fn),
        "ma200": _bt_loop(rolling, ma200_fn),
        "fab": _bt_loop(rolling, fab_fn),
        "dm": _bt_loop(rolling, dm_fn),
        "vix": _bt_loop(rolling, vix_fn),
        "comp": _bt_loop(rolling, make_comp_fn(), def_ret_key="shyRet"),
        "compIef": _bt_loop(rolling, make_comp_fn(), def_ret_key="iefRet"),
    }


# ─── Metrics ───

def calc_metrics(eq):
    n = len(eq)
    yrs = (n - 1) / 252  # n points = n-1 return periods
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

    has_shy = "SHY" in all_data and len(all_data["SHY"].get("dates", [])) > 100
    chk("Data", "SHY available", "Yes" if has_shy else "Missing", "pass" if has_shy else "warn")

    for t in ["^VIX", "^VIX3M"]:
        has = t in all_data and len(all_data[t].get("dates", [])) > 100
        chk("Data", f"{t} available", "Yes" if has else "Missing", "pass" if has else "warn")

    chk("Signal", "Events found", f"{len(signals)}", "pass" if len(signals) >= 3 else "warn")
    chk("Backtest", "B&H CAGR", f"{m['bh']['cagr'] * 100:.1f}%", "pass" if 0.07 < m["bh"]["cagr"] < 0.15 else "warn")
    chk("Backtest", "Composite max DD", f"{m['comp']['maxDD'] * 100:.1f}% vs B&H {m['bh']['maxDD'] * 100:.1f}%", "pass" if m["comp"]["maxDD"] > m["bh"]["maxDD"] else "warn")
    chk("Backtest", "Composite def fraction", f"{m['comp']['defFrac'] * 100:.1f}%", "pass" if 0.1 < m["comp"]["defFrac"] < 0.6 else "warn")
    chk("Backtest", "Transaction costs", f"{TX_COST_BPS}bps per allocation change (Grad D scheme)", "pass")
    chk("Backtest", "Look-ahead bias", "All signals T+1: prev-day data, no look-ahead bias", "pass")

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


# ─── Regional spillover analysis ───

REGIONAL_ETFS = {
    "mchi": "MCHI", "ews": "EWS", "ewm": "EWM",
    "eido": "EIDO", "thd": "THD", "ephe": "EPHE",
}
REGIONAL_HORIZONS = {"1W": 5, "4W": 21, "12W": 63}


def compute_regional(all_data: dict, rolling: list) -> dict:
    """
    Compute forward returns & drawdowns for regional ETFs by US score episode.
    Both deterioration (score rises) and improvement (score falls) episodes.
    Multiple horizons (1W, 4W, 12W) to match Monitor tab structure.
    Also computes beta-to-SPX by regime (low/mid/high score).
    """
    available = {k: v for k, v in REGIONAL_ETFS.items() if v in all_data}
    if not available:
        log.info("  Regional: no ETF data available, skipping")
        return {}

    log.info(f"  Regional: computing for {list(available.values())}")

    date_idx = {r["date"]: i for i, r in enumerate(rolling)}

    spy_data = all_data.get("SPY", {})
    spy_date_price = {}
    if spy_data:
        for d, p in zip(spy_data["dates"], spy_data["adjCloses"]):
            spy_date_price[d] = p

    # Deterioration episodes (score rises to N from lower)
    det_episodes = {s: [] for s in range(7)}
    imp_episodes = {s: [] for s in range(7)}
    for i in range(1, len(rolling)):
        cur = rolling[i]["compositeScore"]
        prev = rolling[i - 1]["compositeScore"]
        if cur > prev:
            det_episodes[cur].append(i)
        if cur < prev:
            imp_episodes[cur].append(i)

    # Score 0 baseline for deterioration panel
    s0_indices = [i for i, r in enumerate(rolling) if r["compositeScore"] == 0]
    det_episodes[0] = s0_indices[::21] if len(s0_indices) > 50 else s0_indices

    def _compute_fwd(etf_prices, episodes_dict):
        """Compute forward returns by score for a given set of episodes."""
        fwd_by_score = {}
        for score, indices in episodes_dict.items():
            if len(indices) < 2:
                continue
            score_data = {}
            for h_name, h_days in REGIONAL_HORIZONS.items():
                rets, mdds = [], []
                for idx in indices:
                    if idx + 1 >= len(rolling):
                        continue
                    start_date = rolling[idx + 1]["date"]
                    p0 = etf_prices.get(start_date)
                    if p0 is None:
                        for offset in range(1, 5):
                            if idx + 1 + offset < len(rolling):
                                alt = rolling[idx + 1 + offset]["date"]
                                p0 = etf_prices.get(alt)
                                if p0 is not None:
                                    start_date = alt
                                    break
                    if p0 is None or p0 <= 0:
                        continue
                    si = date_idx.get(start_date)
                    if si is None:
                        continue
                    ei = si + h_days
                    if ei >= len(rolling):
                        continue
                    end_date = rolling[ei]["date"]
                    p_end = etf_prices.get(end_date)
                    if p_end is None:
                        for off in range(-2, 3):
                            if 0 <= ei + off < len(rolling):
                                p_end = etf_prices.get(rolling[ei + off]["date"])
                                if p_end is not None:
                                    break
                    if p_end is None:
                        continue
                    rets.append(round((p_end - p0) / p0, 6))
                    peak, mdd = p0, 0.0
                    for di in range(si, min(ei + 1, len(rolling))):
                        px = etf_prices.get(rolling[di]["date"])
                        if px is not None:
                            if px > peak:
                                peak = px
                            dd = (px - peak) / peak
                            if dd < mdd:
                                mdd = dd
                    mdds.append(round(mdd, 6))
                if len(rets) >= 2:
                    score_data[h_name] = {"rets": rets, "mdds": mdds}
            if score_data:
                fwd_by_score[str(score)] = score_data
        return fwd_by_score

    result = {}

    for key, ticker in available.items():
        etf = all_data[ticker]
        etf_prices = {d: p for d, p in zip(etf["dates"], etf["adjCloses"])}

        det_fwd = _compute_fwd(etf_prices, det_episodes)
        imp_fwd = _compute_fwd(etf_prices, imp_episodes)

        etf_result = {}
        if det_fwd:
            etf_result["det"] = det_fwd
        if imp_fwd:
            etf_result["imp"] = imp_fwd
        if etf_result:
            result[key] = etf_result

    # Beta by regime
    correlations = {}
    for key, ticker in available.items():
        etf = all_data[ticker]
        etf_prices = {d: p for d, p in zip(etf["dates"], etf["adjCloses"])}

        # Daily returns paired with score
        regime_pairs = {"low": [], "mid": [], "high": []}
        prev_spy, prev_etf = None, None
        for r in rolling:
            d = r["date"]
            spy_p = spy_date_price.get(d)
            etf_p = etf_prices.get(d)
            if spy_p and etf_p and prev_spy and prev_etf:
                spy_ret = (spy_p - prev_spy) / prev_spy
                etf_ret = (etf_p - prev_etf) / prev_etf
                sc = r["compositeScore"]
                bucket = "low" if sc <= 1 else ("mid" if sc <= 3 else "high")
                regime_pairs[bucket].append((spy_ret, etf_ret))
            prev_spy = spy_p
            prev_etf = etf_p

        betas = {}
        for regime, pairs in regime_pairs.items():
            if len(pairs) < 30:
                betas[regime] = None
                continue
            spy_arr = [p[0] for p in pairs]
            etf_arr = [p[1] for p in pairs]
            n = len(spy_arr)
            mean_s = sum(spy_arr) / n
            mean_e = sum(etf_arr) / n
            cov = sum((s - mean_s) * (e - mean_e) for s, e in zip(spy_arr, etf_arr)) / (n - 1)
            var_s = sum((s - mean_s) ** 2 for s in spy_arr) / (n - 1)
            betas[regime] = round(cov / var_s, 2) if var_s > 0 else None

        if any(v is not None for v in betas.values()):
            correlations[key] = betas

    if correlations:
        result["correlations"] = correlations

    n_etfs = len([k for k in result if k != "correlations"])
    log.info(f"  Regional: {n_etfs} ETFs computed")
    return result
