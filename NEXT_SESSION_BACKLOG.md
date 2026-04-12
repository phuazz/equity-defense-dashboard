# Next Session Backlog — equity-defense-dashboard

Last updated: 2026-04-12 (post major redesign)

## Current state
- 4 tabs: Monitor, Indicators, Performance, Methodology
- 5 signals (canary removed): Stress 8D, 200d MA, 12M Momentum, 10M SMA, VIX Term
- Primary defensive asset: SHY (1-3Y UST)
- Allocation: score 0-1 = 100% SPY, score 2 = 50% SHY, score 3+ = 100% SHY
- 42-day minimum holding period
- All signals T+1, monthly signals latch at month-end
- Full daily data (no thinning), ~8MB dashboard output
- Comp(SHY): CAGR 9.4%, MaxDD -20.1%, Sharpe 0.60

## Follow-up items

### Data / Performance
1. Dashboard output is ~8MB due to full daily data (7111 points x 8 strategies). Consider selective thinning for individual strategy curves while keeping composite and B&H at full resolution.
2. Validate data frequency assumption: `calc_metrics` uses sqrt(252) annualisation — confirm SPY cache dates are true daily (gaps of 1-3 days including weekends look correct, but worth a formal check).
3. The 200d MA standalone strategy shows high CAGR (check if it is overstated by the T+1 implementation — confirm the `ma200_fn` logic is genuinely using previous day's close).

### Monitor tab
4. Gauge "HOLDING (Xd remaining)" — consider showing the entry date and expected exit date instead of just days remaining.
5. CIO narrative text says "Active signals: ." (empty) when score = 1 from latched monthly signal — the active signal name is not being captured correctly for latched signals.
6. Next-day score calculation may need updating for the 5-signal model (currently swaps VIX only, does not account for monthly latch boundaries).

### Indicators tab
7. Composite Score History chart y-axis shows 6 but max is now 5 — update range.
8. Consider adding the composite allocation overlay (shaded regions showing 50%/100% defensive) to the score chart.

### Performance tab
9. Worst Drawdowns bar chart — label with event names (e.g. "GFC", "COVID") for the well-known episodes rather than just YYYY-MM dates.
10. Consider adding a rolling 12-month return comparison chart (Defence vs B&H).

### Methodology tab
11. Add a "Changes from Previous Version" section documenting: canary removal rationale, SHY vs IEF decision, monthly signal latching, minimum holding period.
12. Consider adding a "Limitations" section: single-asset universe (US large cap only), backtest ≠ live performance, no hedging costs, SHY availability post-2002 only.

### Code quality
13. Faber/DM monthly signals: the `belowSMA10m` and `spy12mRet` fields are still computed daily on every rolling row. The backtest correctly latches at month-end, but the rolling data could be simplified to only update these fields at month boundaries.
14. Enhanced strategy (`enh_fn`): the blowup trigger reads `r[i]` (today) with a pending flag, while all other signals read `r[i-1]`. Functionally equivalent but inconsistent style — consider aligning.
