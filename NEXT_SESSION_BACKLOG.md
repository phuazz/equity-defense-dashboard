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

### Chart rendering
13. **Legend truncation bug — diagnostic findings**

**Symptom:** Legend text clipped on multiple charts across Monitor, Overview, Performance, and Indicators tabs. Examples:
- "Composite" → "Composit" (Performance tab drawdown charts)
- "Defence (SHY)" → "Defence (SH" (Growth of $100 chart)
- "Exiting Defence" → "Exiting Defenc" (Monitor S&P 500 LAST 12 MONTHS chart)

**What was ruled out (width is NOT the cause):**

At desktop 1400px viewport, each legend item in ov-dd-bar has ~280-300px of allocated space. The actual text at 12px Source Sans 3 needs roughly 160px total for "B&H + Composite". The Growth of $100 chart has ~1300px available for a label needing perhaps 100px. The math shows legend content fits with significant room to spare.

All fixes targeting horizontal space were tried and failed:
- itemwidth:15 in global PL() legend
- entrywidth:0 with and without entrywidthmode:'fraction' (entrywidth:0 with fraction mode caused legends to disappear entirely)
- Removing overflow:hidden from .ch container
- Per-chart itemwidth overrides
- Label abbreviation (e.g. "Composite" → "Comp.") — cosmetic only, bug persists on other labels

The truncation mechanism is upstream of the width allocation.

**Top 3 hypotheses for the real cause:**

**Hypothesis 1 — Font loading race condition (most likely).** Plotly measures text width at render time using the browser's current font. If Source Sans 3 has not finished loading when Plotly.newPlot executes, the browser substitutes a fallback font for measurement. The fallback glyphs are wider, so Plotly allocates based on truncated text. Then the real font loads, renders narrower glyphs, and the truncation is visible as clipped text with ample whitespace to the right.

DevTools test: Open Performance tab, hard-refresh with cache disabled (Ctrl+Shift+R). In Network panel, check when Source Sans 3 finishes loading versus when Plotly.newPlot calls complete. Alternatively, in Console run:

    document.fonts.ready.then(() => Plotly.relayout('ov-dd-bar', {}))

If the legend renders correctly after this forced relayout, font timing is confirmed as the cause.

**Hypothesis 2 — CSS overflow:hidden on .ch clipping the SVG visual layer.** The SVG element's own overflow attribute is hidden by default. Combined with parent .ch overflow:hidden, there could be a double-clip scenario where the legend group's bounding box extends past a clipping boundary at the visual layer even though the coordinate math says it fits. Browser-specific.

DevTools test: Right-click the truncated legend text in Elements panel. Check the SVG `<text>` element — does it contain the full string "Composite" or the truncated "Composit"?
- If full string in DOM but visually clipped → CSS/SVG overflow issue
- If DOM already contains truncated string → Plotly truncated during render (points back to hypothesis 1)

**Hypothesis 3 — Stale container width during Plotly measurement.** If the chart div has width:0 or incorrect width when Plotly measures (e.g. because the Performance tab panel is display:none when first rendered), Plotly computes a tiny legend budget and truncates aggressively. The panel only becomes display:block when the tab is clicked, but Plotly.newPlot may run before that.

DevTools test: Click a different tab, then click back to Performance. If the legends are correct after the tab switch (because a resize event fires on tab click and triggers Plotly relayout), this is the cause.

**Do NOT retry these approaches (all failed):**
- itemwidth, entrywidth, entrywidthmode tweaks
- overflow:hidden changes on .ch
- Container width changes via margin
- Label abbreviation

**Starting point for next session:** Open Chrome DevTools on the Performance tab's Worst Drawdowns chart, right-click the truncated legend, Inspect, and check whether the SVG `<text>` element contains the full word or the truncated string. That single observation distinguishes hypothesis 1/3 from hypothesis 2 and tells you which direction to investigate first.

---

### Session 3 diagnostic findings (2026-04-13)

**Ruled out this session:**
- Typo in source strings (grepped template.html, all labels correct)
- Chrome-specific rendering (reproduces identically in Microsoft Edge)
- SVG overflow:hidden at container boundary (DOM measurement shows text rect inside SVG rect with 100+ pixels clearance on every visible legend; CSS overflow:visible override via DevTools did not change rendering)
- Plotly truncating DOM text content (every visible legend text has textContent === data-unformatted across all 34 elements)

**What we still do not know:**
- Why getBoundingClientRect says the text fits but the visible render shows truncation
- Whether the bug is consistent across charts or affects only some
- Whether template.html served directly renders differently from docs/index.html (user observation was ambiguous)

**Recommended next approach (fresh session, do NOT retry DevTools Console queries):**

1. First: compare template.html and docs/index.html side by side in two browser tabs. View template.html at http://localhost:3000/template.html and docs/index.html at http://localhost:3000/ . Navigate to the same chart on each (start with Growth of $100 on Performance tab). Take screenshots of each. If any chart renders differently between the two, the bug is in scripts/pipeline.py — grep both files for the literal legend trace names to find what the pipeline is corrupting.

2. If template and docs render identically: try the Firefox Fonts panel to check font substitution. Firefox DevTools has a dedicated Fonts tab that Chrome does not — it shows exactly which font face is being used for each element.

3. If Firefox confirms font mismatch: fix by either (a) preloading the font with font-display:block in the HTML head, (b) wrapping Plotly.newPlot calls in document.fonts.ready.then(...), or (c) switching to a system font stack for legends only.

4. As a last resort: screenshot the legend area and pixel-count the visible characters versus expected characters. This can reveal if the truncation is exactly 1 character (typo/off-by-one) or a fractional character (font metric issue).

**Do NOT retry (all tested and failed):**
- itemwidth, entrywidth, entrywidthmode tweaks
- overflow:hidden changes on .ch
- Container width adjustments via margin
- Label abbreviation
- Global CSS overflow:visible on Plotly SVG  
- Tab switching (no relayout triggered)
- Chrome-specific fixes

---

### Session 4 findings (2026-04-13)

**Approach this session:** Empirical fix-testing rather than further diagnosis. Applied candidate fixes one at a time, rebuilt, visually verified, reverted if unsuccessful.

**Candidate fixes tested and ruled out:**

1. **document.fonts.ready wrapper around render()** — Wrapped all chart rendering inside `document.fonts.ready.then(...)` to ensure Source Sans 3 was loaded before Plotly measured text. Result: caused rendering issues on multiple charts that were previously unaffected. Reverted immediately.

2. **Font preconnect + display:block** (SHIPPED as `30bfb9c`) — Added `<link rel="preconnect">` hints for Google Fonts CDN and changed `display=swap` to `display=block` on the font stylesheet. Result: did not fix legend truncation, but is a harmless font-loading optimisation. Kept in place.

3. **Plotly version upgrade (2.27.0 → latest/3.x)** — Swapped CDN URL to `plotly-latest.min.js`. Dashboard rendered correctly but truncation persisted identically. Confirms the bug is not version-specific. Reverted to pinned 2.27.0.

4. **Legend anchor: centered → left-aligned** — Changed PL() legend from `x:.5, xanchor:'center'` to `x:0, xanchor:'left'`. Result: legend shifted left as expected, but truncation was identical — the clipping moved with the text. This confirms the truncation is **internal to the legend item**, not caused by the legend group overrunning the plot area. Reverted.

5. **Trailing space in trace name** — Changed `name:'Defence (SHY)'` to `name:'Defence (SHY) '` to force Plotly to allocate a wider clip boundary. Result: no change. Reverted.

**Key diagnostic insight from this session:**

The left-anchor test (candidate 4) is the most informative result. The truncation is not at a container boundary — it is per-item clipping inside the Plotly legend SVG group. Plotly calculates each legend item's clip region based on its own text measurement, and that measurement is consistently too narrow by ~1-2 characters. This is independent of:
- Font loading timing (candidates 1, 2)
- Plotly version (candidate 3)
- Legend group positioning (candidate 4)
- Trailing whitespace in trace names (candidate 5)

**Additional affected chart discovered (2026-04-13 afternoon):** 10M SMA (Faber) chart on Indicators tab also shows legend truncation ("SPY 10M SMA" → "SPY 10M SM"). Verified via side-by-side comparison at pre-font-change commit `4b96c24` vs post-font-change commit `30bfb9c` that this truncation was present BEFORE today's font changes, not caused by them. This is additional evidence that the bug is ambient across multiple charts and not triggered by font-display mode.

**Do NOT retry (all tested and failed across sessions 2-4):**
- itemwidth, entrywidth, entrywidthmode tweaks
- overflow:hidden changes on .ch
- Container width adjustments via margin
- Label abbreviation
- Global CSS overflow:visible on Plotly SVG
- Tab switching (no relayout triggered)
- Chrome-specific fixes
- document.fonts.ready wrapper
- Plotly version upgrade
- Legend anchor position change (centered vs left)
- Trailing space in trace names

**Recommended next approach:** Custom HTML legend replacing Plotly's built-in SVG legend entirely. Disable `showlegend` on the Plotly chart config and render legend items as styled HTML elements below the chart div. This bypasses all Plotly text measurement and SVG clipping.

---

### Code quality
15. Faber/DM monthly signals: the `belowSMA10m` and `spy12mRet` fields are still computed daily on every rolling row. The backtest correctly latches at month-end, but the rolling data could be simplified to only update these fields at month boundaries.
16. Enhanced strategy (`enh_fn`): the blowup trigger reads `r[i]` (today) with a pending flag, while all other signals read `r[i-1]`. Functionally equivalent but inconsistent style — consider aligning.
