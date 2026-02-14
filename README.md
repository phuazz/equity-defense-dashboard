# Blowup Signal v3 — Multi-Strategy Defensive Dashboard

A daily-updated dashboard comparing 5 defensive equity strategies, built around Batnik's "Blowup Signal" (rolling count of S&P 500 stocks down ≥7%) and enhanced with literature-backed improvements.

**Live Dashboard:** [https://YOUR_USERNAME.github.io/blowup-signal/](https://YOUR_USERNAME.github.io/blowup-signal/)

## Strategies

| # | Strategy | Trigger | Source |
|---|----------|---------|--------|
| S1 | **Enhanced Blowup** | Blowup ≥115 OR (SPX < 200d MA + canary bad) → IEF | Batnik, Keller 2018, Faber 2007 |
| S2 | **Faber 10-Month SMA** | SPX < 10M SMA → IEF | Faber (2007), validated to 1886 |
| S3 | **Dual Momentum** | SPY 12M < 0 OR SPY 12M < IEF 12M → IEF | Antonacci (2014), Dow Award |
| S4 | **Composite (≥2/5)** | Score 5 sub-signals; ≥2 → full defensive | Ensemble approach |
| S5 | **Buy & Hold** | 100% SPY | Benchmark |

### Signal Components (Composite Score 0–5)
1. **Blowup count** ≥ 115 (rolling 8-day, scaled to S&P 500)
2. **SPX < 200-day SMA** (Faber regime filter)
3. **Canary warning** — VWO or BND negative 13612W momentum (Keller & Keuning 2018)
4. **SPY 12-month return** < 0 (Antonacci absolute momentum)
5. **SPX < 10-month SMA** (Faber timing model)

## Architecture

```
GitHub Actions (cron: daily 22:00 UTC / 5 PM ET)
    │
    ├── scripts/data_providers.py   ← Abstract data layer (Yahoo / Alpha Vantage)
    ├── scripts/compute_signals.py  ← Pure Python signal engine
    ├── scripts/pipeline.py         ← Orchestrator: fetch → compute → render
    │
    ├── data/cache.json             ← Incremental data cache (committed)
    ├── data/signals.json           ← Computed signals + metrics
    │
    ├── template.html               ← Dashboard template (Plotly.js)
    └── docs/index.html             ← Generated static page (GitHub Pages)
```

### Key Design Decisions

**Incremental caching:** `cache.json` stores all historical data. Each daily run only fetches the delta (new dates since last cache). This cuts Yahoo Finance API calls from ~95 full-history requests to ~95 small delta requests, avoiding rate limits.

**Compute/render separation:** All signal computation runs in Python (CI), not in the browser. The HTML dashboard is a pure renderer that reads a pre-injected JSON blob. Pages load instantly — no 2-minute fetch wait.

**Abstract data layer:** Add new providers by subclassing `DataProvider` in `data_providers.py`. Currently supports Yahoo Finance and Alpha Vantage. The `CachedProvider` wrapper adds file-based caching to any provider.

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/blowup-signal.git
cd blowup-signal
pip install -r requirements.txt
```

### 2. Run locally

```bash
# First run (fetches full history, takes ~2-3 minutes)
python scripts/pipeline.py

# Subsequent runs (uses cache, only fetches delta)
python scripts/pipeline.py

# Force full refetch
python scripts/pipeline.py --no-cache

# Use Alpha Vantage instead of Yahoo
export ALPHA_VANTAGE_KEY=your_key_here
python scripts/pipeline.py --provider alphavantage
```

Open `docs/index.html` in your browser.

### 3. Deploy to GitHub Pages

1. Push to GitHub
2. Go to **Settings → Pages → Source** → select `Deploy from a branch`
3. Set branch to `main`, folder to `/docs`
4. The Actions workflow runs daily and commits updated `docs/index.html`

### 4. (Optional) Alpha Vantage

If Yahoo Finance becomes unreliable:
1. Get a free API key from [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. Add it as a GitHub secret: **Settings → Secrets → ALPHA_VANTAGE_KEY**
3. Update `.github/workflows/update.yml`: change `--provider yahoo` to `--provider alphavantage`

Note: Alpha Vantage free tier allows 25 calls/day. The pipeline fetches ~95 tickers, so you'd need to run across multiple days to build the initial cache, or use a paid tier.

## Adding a New Data Provider

```python
# In scripts/data_providers.py:
class MyProvider(DataProvider):
    def name(self) -> str:
        return "myprovider"

    def fetch(self, tickers, start, end) -> dict:
        # Return {ticker: {"dates": [...], "adjCloses": [...]}}
        ...

# Register it:
PROVIDERS["myprovider"] = MyProvider
```

Then run: `python scripts/pipeline.py --provider myprovider`

## Data

All data uses **adjusted close** (dividends + splits reinvested = total return). Per Price Action Lab, ~42% of SPY's total return since inception comes from dividend reinvestment — using price-only data would bias all strategy comparisons.

**Universe:** 90 S&P 500 representative stocks, scaled to 500 using `500/available_stocks_that_day`. Plus canary assets (VWO, BND) and defensive asset (IEF).

## References

- Batnik, M. — "The Blowup Signal" (The Compound)
- Keller, W. & Keuning, J. (2018) — "Defensive Asset Allocation" (canary universe, 13612W momentum)
- Faber, M. (2007) — "A Quantitative Approach to Tactical Asset Allocation" (10M SMA)
- Antonacci, G. (2014) — "Dual Momentum Investing" (GEM, absolute + relative momentum)
- Siegel, J. (2006) — "Stocks for the Long Run" (200d MA validation back to 1886)
- Zweig, M. (1986) — Breadth Thrust indicator (used in Enhanced Blowup re-entry)

## License

MIT
