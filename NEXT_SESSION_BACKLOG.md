# Next session backlog for equity-defense-dashboard

## Monitor tab
1. "S&P 500 — LAST 12 MONTHS" chart: add range selector, add hovermode:'closest', fix "Def Exit" legend truncation, expand "Def Entry/Exit" labels to full words
2. ~~"Composite Score History" default range~~ — VERIFIED: default is 2Y via rangeStart(24) on line 707
3. "Composite Score History" legend: investigate why "≥2 Def" renders as "≥2 Del" on Monitor and Indicators composite score legends — diagnosis is uncertain, may be font rendering at small sizes, may be Plotly legend truncation. Possible fixes: full word "Defensive", trailing period "Def.", or Plotly legend font/width tweak

## Overview tab
4. "S&P 500 — LAST 12 MONTHS" chart: same bugs as Monitor version (tooltip clipping, legend truncation, label expansion)
5. Add range selector buttons to main Overview charts (match Monitor/Indicators pattern)

## Regional tab
6. NOW marker direction fix — same pattern as Monitor ch-score (computeNowConfig for deterioration vs improvement)
7. Add range selector buttons to Regional charts

## Indicators tab
8. Y-axis rescale-on-zoom behaviour: design decision on locked vs auto-scaling when clicking range buttons (mixed approach likely best)
9. SPX 200DMA chart: malformed y-axis tick values (mixing log-like and linear scales)

## Data investigation
10. "Avg duration at score 0 = 15 days" looks low — investigate calculation, compute median/mean/distribution to verify

## Cross-cutting
11. Audit all charts for hovermode:'x unified' clipping bug and apply hovermode:'closest' where tooltips extend past card boundaries
