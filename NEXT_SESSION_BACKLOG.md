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

**Confirmed:**
- Source strings in template.html are correct — no typo. Grepped all occurrences of "Defence (SHY)", "Composite", "Composite (SHY)". Every string is present in full at lines 183, 269, 971, 974, 991, 1003, 1009, 1020. Rules out typo hypothesis.
- Bug reproduces identically in Microsoft Edge as in Chrome. Rules out Chrome-specific rendering or font substitution. The bug is in the HTML/CSS/SVG or in Plotly's render output, not the browser.
- Direct DOM inspection: every visible legend text element has `textContent === data-unformatted`. Plotly is NOT truncating strings at the text content level.
- `getBoundingClientRect()` on every visible legend text shows `clipped: false` against the containing SVG boundary. Text right edges are 100+ pixels inside SVG right edges for every affected chart.
- CSS `overflow: visible !important` injected into Plotly's SVG via DevTools did not change the visible rendering.

**What this means:**
The bug is a mismatch between what the DOM model says is rendered ("full text, within bounds, no clipping") and what is actually visible on screen ("text cut off at the end"). The typical causes for this kind of mismatch are:

1. **Font metrics mismatch** — the `<text>` element reports a bounding rect based on one font, but the browser renders with different glyph widths (e.g. Plotly measures with Open Sans but ends up rendering with Source Sans 3, or vice versa). The bounding rect is correct for the measurement font but wrong for the render font.

2. **SVG `textLength` / `lengthAdjust` attributes** being applied somewhere, causing the rendered glyphs to be squeezed or expanded relative to their natural widths.

3. **An ancestor element with `clip-path` or `mask`** that is clipping at an intermediate level, below the `<text>` element but above the container we measured.

**Recommended next approach (for a fresh session with different tools):**

Stop trying to diagnose via DevTools Console queries. The DOM model is lying about what is visually rendered, so DOM queries cannot find the cause. Instead:

1. Open the dashboard in Firefox (not Chrome or Edge). Firefox has different DevTools with a "Fonts" panel that shows exactly which font family is used for each element. Check whether the legend text elements have a font mismatch between what is declared and what is actually used.

2. Screenshot the legend area at 4x zoom via Windows Magnifier. Pixel-count the visible characters versus the expected characters. This tells you whether the truncation is exactly 1 character, 2 characters, or a fractional character width — which narrows down the mechanism.

3. Add a Plotly config option `config: {typesetMath: false}` to the `PC` constant — Plotly sometimes uses MathJax for text layout and it can cause metric drift. If this changes the rendering, that is the cause.

4. As a nuclear option: replace the global font import with a locally hosted font file (no external CDN fetch), and ensure the font is in `font-display: block` mode so Plotly cannot render before the font is loaded.

**Do NOT retry these (all tested and failed):**
- itemwidth, entrywidth, entrywidthmode tweaks
- overflow:hidden changes on .ch
- Container width adjustments
- Label abbreviation
- Global CSS overflow:visible on Plotly SVG
- Font-ready forced relayout (document.fonts.ready.then(...))
- Vertical legend orientation
- Tab switching (does not trigger relayout)
- Assumption that it is Chrome-specific

---

### Code quality
15. Faber/DM monthly signals: the `belowSMA10m` and `spy12mRet` fields are still computed daily on every rolling row. The backtest correctly latches at month-end, but the rolling data could be simplified to only update these fields at month boundaries.
16. Enhanced strategy (`enh_fn`): the blowup trigger reads `r[i]` (today) with a pending flag, while all other signals read `r[i-1]`. Functionally equivalent but inconsistent style — consider aligning.
