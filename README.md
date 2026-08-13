# Equity Defense Dashboard — S&P 500 Risk Overlay

A daily-updated defensive regime monitor for the S&P 500: a five-component
composite score (0–5) drives a defensive allocation ladder, with the full
indicator set, episode statistics and strategy comparison behind it.

**Live:** https://phuazz.github.io/equity-defense-dashboard/

> **Personal research artefact.** Not investment advice, not an offer, and not
> affiliated with any regulated fund. All returns shown are simulated; there is
> no live track record. The page footer carries the full disclosure.

## The composite score

One point per active signal, evaluated daily:

| # | Signal | Trigger |
|---|--------|---------|
| 1 | Stress count | Rolling 8-day count of S&P 500 members down ≥7%, scaled to index size (Batnik's "blowup" count) |
| 2 | 200-day MA | SPX below its 200-day simple moving average |
| 3 | 12-month momentum | SPY trailing 12-month total return negative (Antonacci absolute momentum) |
| 4 | 10-month SMA | SPX below its 10-month SMA at the monthly read (Faber 2007) |
| 5 | VIX term structure | VIX above VIX3M (inverted term structure) |

Allocation ladder by score: 0–1 → 100% SPY · 2 → 50% SPY / 50% defensive ·
3 → 25% SPY / 75% defensive · 4–5 → fully defensive. Defensive asset: SHY
(1–3y US Treasuries) primary, IEF (7–10y) shown for comparison. Episode
statistics are measured from the first day a score level is reached, with
forward returns from T+1.

Tabs: **Monitor** (regime dial, allocation, what-to-watch) · **Indicators**
(each signal against its trigger) · **Performance** (forward-return and
drawdown distributions by score level, episode analysis) · **Methodology**.

## Data

- Adjusted close throughout (dividends and splits reinvested — total return).
- Universe: 90 S&P 500 representative members, counts scaled by
  `500 / available` (currently ×5.56); plus SPY, SHY, IEF and VIX/VIX3M.
- Window: 1998-01-05 onward; the masthead shows the live end date.
- Providers: Yahoo Finance (default) or Alpha Vantage behind a common
  `DataProvider` interface, with an incremental committed cache
  (`data/cache.json`) so each daily run fetches only the delta.

## Architecture

```
.github/workflows/update.yml     daily cron 01:00 UTC
    scripts/data_providers.py    provider abstraction + cache
    scripts/compute_signals.py   pure-Python signal engine
    scripts/pipeline.py          fetch → compute → render
    data/signals.json            computed signals + metadata
    template.html                source page (edit this)
    docs/index.html              built page (never edit — GitHub Pages serves /docs)
```

Local build: `pip install -r requirements.txt` then `python scripts/pipeline.py`
(first run fetches full history; later runs use the cache). `--no-cache`
forces a refetch; `--provider alphavantage` switches provider (needs
`ALPHA_VANTAGE_KEY`).

## Status

- **Live and refreshing daily** — the Actions cron commits "📊 Update signals"
  each session day; data end date renders in the masthead.
- Cross-dashboard audit 2026-08-13 (`C:\dev\studies\2026-08-13_dashboard-audit.md`):
  disclaimer footer added, sub-11px type raised to the mobile floor, README
  rewritten (this file — the previous one described the pre-rename
  "blowup-signal" project with a placeholder URL).
- Related project: [breadth-thrust-signal](https://github.com/phuazz/breadth-thrust-signal)
  — the bullish mirror (risk-on conviction meter).

## References

- Batnik, M. — "The Blowup Signal" (The Compound)
- Faber, M. (2007) — "A Quantitative Approach to Tactical Asset Allocation"
- Antonacci, G. (2014) — "Dual Momentum Investing"
- Keller, W. & Keuning, J. (2018) — "Defensive Asset Allocation"
- Zweig, M. (1986) — Breadth Thrust indicator
- Siegel, J. (2006) — "Stocks for the Long Run"

MIT licence.

*Last updated: 2026-08-13*
