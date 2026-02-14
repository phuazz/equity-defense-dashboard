#!/usr/bin/env python3
"""
Blowup Signal v3 — Daily Pipeline
Fetches data, computes signals, generates static HTML dashboard.

Usage:
    python scripts/pipeline.py                          # Default: yahoo + cache
    python scripts/pipeline.py --provider alphavantage  # Use Alpha Vantage
    python scripts/pipeline.py --no-cache               # Force full refetch
    python scripts/pipeline.py --output docs/index.html # Custom output path
"""

import argparse, json, logging, os, sys
from datetime import datetime, timedelta
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from data_providers import get_provider
from compute_signals import compute_all, ALL_TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / "data" / "cache.json"
TEMPLATE_PATH = ROOT / "template.html"
OUTPUT_PATH = ROOT / "docs" / "index.html"


def main():
    parser = argparse.ArgumentParser(description="Blowup Signal pipeline")
    parser.add_argument("--provider", default="yahoo", choices=["yahoo", "alphavantage"])
    parser.add_argument("--no-cache", action="store_true", help="Skip cache, full refetch")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--start", default="1998-01-01", help="History start date")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Blowup Signal v3 — Pipeline Run")
    log.info(f"Provider: {args.provider} | Cache: {not args.no_cache}")
    log.info("=" * 60)

    # 1. Fetch data
    log.info("Step 1: Fetching market data...")
    cache = None if args.no_cache else str(CACHE_PATH)
    provider = get_provider(args.provider, cache_path=cache)

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    all_data = provider.fetch(ALL_TICKERS, args.start, end_date)
    log.info(f"  Got data for {len(all_data)} tickers")

    if "SPY" not in all_data:
        log.error("FATAL: SPY data missing!")
        sys.exit(1)

    # 2. Compute signals
    log.info("Step 2: Computing signals & backtests...")
    result = compute_all(all_data)

    # Remove rollingFull from JSON output (too large)
    result.pop("rollingFull", None)

    log.info(f"  Signals: {result['meta']['signalCount']}")
    log.info(f"  Composite now: {result['meta']['compositeNow']}/5")
    log.info(f"  Period: {result['meta']['startDate']} → {result['meta']['endDate']}")

    # 3. Write data JSON (for debugging / external consumers)
    data_json_path = ROOT / "data" / "signals.json"
    with open(data_json_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    size_mb = os.path.getsize(data_json_path) / 1024 / 1024
    log.info(f"  Data JSON: {data_json_path} ({size_mb:.1f} MB)")

    # 4. Inject into HTML template
    log.info("Step 3: Generating HTML dashboard...")
    if not TEMPLATE_PATH.exists():
        log.error(f"Template not found: {TEMPLATE_PATH}")
        sys.exit(1)

    template = TEMPLATE_PATH.read_text(encoding='utf-8')

    # Inject the pre-computed data as a JS variable
    data_js = f"const PRECOMPUTED_DATA = {json.dumps(result, separators=(',', ':'))};"
    html = template.replace("/* __INJECT_DATA__ */", data_js)

    # Write output
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    size_kb = os.path.getsize(output) / 1024
    log.info(f"  Dashboard: {output} ({size_kb:.0f} KB)")

    # 5. Summary
    log.info("=" * 60)
    m = result["metrics"]
    log.info(f"B&H:       CAGR {m['bh']['cagr']*100:.1f}% | MaxDD {m['bh']['maxDD']*100:.1f}% | Sharpe {m['bh']['sharpe']}")
    log.info(f"Composite: CAGR {m['comp']['cagr']*100:.1f}% | MaxDD {m['comp']['maxDD']*100:.1f}% | Sharpe {m['comp']['sharpe']}")
    log.info(f"Enh Blowup:CAGR {m['enh']['cagr']*100:.1f}% | MaxDD {m['enh']['maxDD']*100:.1f}% | Sharpe {m['enh']['sharpe']}")
    log.info(f"Faber 10M: CAGR {m['fab']['cagr']*100:.1f}% | MaxDD {m['fab']['maxDD']*100:.1f}% | Sharpe {m['fab']['sharpe']}")
    log.info(f"Dual Mom:  CAGR {m['dm']['cagr']*100:.1f}% | MaxDD {m['dm']['maxDD']*100:.1f}% | Sharpe {m['dm']['sharpe']}")

    cs = result["meta"]["compositeNow"]
    label = "HIGH RISK" if cs >= 3 else "ELEVATED" if cs >= 2 else "CAUTION" if cs == 1 else "ALL CLEAR"
    log.info(f"\n🚦 CURRENT SIGNAL: {cs}/5 → {label}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
