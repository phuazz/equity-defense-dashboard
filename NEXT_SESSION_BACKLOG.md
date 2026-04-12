# Next session backlog for equity-defense-dashboard

## Monitor tab
1. "S&P 500 — LAST 12 MONTHS" chart: add range selector, add hovermode:'closest', fix "Def Exit" legend truncation, expand "Def Entry/Exit" labels to full words
2. "Composite Score History" chart: fix legend text truncation ("≥2 Del" should be "≥2 Def"), confirm default range is 2Y as proposed (currently appears to be 1Y)

## Overview tab
3. "S&P 500 — LAST 12 MONTHS" chart: same bugs as Monitor version (tooltip clipping, legend truncation, label expansion)
4. Add range selector buttons to main Overview charts (match Monitor/Indicators pattern)

## Regional tab
5. NOW marker direction fix — same pattern as Monitor ch-score (computeNowConfig for deterioration vs improvement)
6. Add range selector buttons to Regional charts

## Indicators tab
7. Y-axis rescale-on-zoom behaviour: design decision on locked vs auto-scaling when clicking range buttons (mixed approach likely best)
8. SPX 200DMA chart: malformed y-axis tick values (mixing log-like and linear scales)

## Data investigation
9. "Avg duration at score 0 = 15 days" looks low — investigate calculation, compute median/mean/distribution to verify

## Cross-cutting
10. Audit all charts for hovermode:'x unified' clipping bug and apply hovermode:'closest' where tooltips extend past card boundaries
