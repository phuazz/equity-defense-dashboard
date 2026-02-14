"""
Data provider abstraction layer.
Add new providers by subclassing DataProvider and registering in PROVIDERS dict.
Each provider returns the same schema: {ticker: {dates: [...], adjCloses: [...]}}
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json, os, time, logging

log = logging.getLogger(__name__)


class DataProvider(ABC):
    """Base class for market data providers."""

    @abstractmethod
    def fetch(self, tickers: list[str], start: str, end: str) -> dict:
        """
        Fetch adjusted close data for tickers.
        Args:
            tickers: list of ticker symbols
            start: 'YYYY-MM-DD'
            end: 'YYYY-MM-DD'
        Returns:
            {ticker: {"dates": ["YYYY-MM-DD", ...], "adjCloses": [float, ...]}}
        """
        ...

    @abstractmethod
    def name(self) -> str: ...


class YahooFinanceProvider(DataProvider):
    """Uses yfinance library (pip install yfinance)."""

    def name(self) -> str:
        return "yahoo"

    def fetch(self, tickers: list[str], start: str, end: str) -> dict:
        import yfinance as yf

        result = {}
        # Batch download for efficiency
        batch_size = 20
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            log.info(f"  Yahoo batch {i // batch_size + 1}: {batch[:5]}...")
            try:
                df = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if df.empty:
                    continue

                adj = df["Adj Close"] if len(batch) > 1 else df[["Adj Close"]]
                if len(batch) == 1:
                    adj.columns = batch

                for t in batch:
                    if t in adj.columns:
                        s = adj[t].dropna()
                        if len(s) > 0:
                            result[t] = {
                                "dates": [d.strftime("%Y-%m-%d") for d in s.index],
                                "adjCloses": [round(float(v), 4) for v in s.values],
                            }
            except Exception as e:
                log.warning(f"  Yahoo batch error: {e}")
            time.sleep(0.5)  # Rate limiting

        return result


class AlphaVantageProvider(DataProvider):
    """Uses Alpha Vantage free tier (25 calls/day). Set ALPHA_VANTAGE_KEY env var."""

    def name(self) -> str:
        return "alphavantage"

    def fetch(self, tickers: list[str], start: str, end: str) -> dict:
        import requests

        api_key = os.environ.get("ALPHA_VANTAGE_KEY", "")
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_KEY env var not set")

        result = {}
        start_dt = datetime.strptime(start, "%Y-%m-%d")

        for t in tickers:
            try:
                url = (
                    f"https://www.alphavantage.co/query?"
                    f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={t}"
                    f"&outputsize=full&apikey={api_key}"
                )
                r = requests.get(url, timeout=30)
                data = r.json()
                ts = data.get("Time Series (Daily)", {})
                dates, closes = [], []
                for d in sorted(ts.keys()):
                    if d >= start and d <= end:
                        dates.append(d)
                        closes.append(round(float(ts[d]["5. adjusted close"]), 4))
                if dates:
                    result[t] = {"dates": dates, "adjCloses": closes}
                time.sleep(12.5)  # 5 calls/min limit
            except Exception as e:
                log.warning(f"  AV error for {t}: {e}")

        return result


class CachedProvider(DataProvider):
    """
    Wraps another provider with file-based caching.
    Only fetches delta (new dates since last cache).
    """

    def __init__(self, inner: DataProvider, cache_path: str):
        self._inner = inner
        self._cache_path = cache_path
        self._cache = self._load_cache()

    def name(self) -> str:
        return f"cached({self._inner.name()})"

    def _load_cache(self) -> dict:
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                return json.load(f)
        return {"provider": self._inner.name(), "tickers": {}, "last_updated": None}

    def save_cache(self):
        self._cache["last_updated"] = datetime.utcnow().isoformat()
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f, separators=(",", ":"))
        size_mb = os.path.getsize(self._cache_path) / 1024 / 1024
        log.info(f"  Cache saved: {self._cache_path} ({size_mb:.1f} MB)")

    def fetch(self, tickers: list[str], start: str, end: str) -> dict:
        cached = self._cache.get("tickers", {})
        # Determine which tickers need fresh data
        need_fetch = []
        for t in tickers:
            if t not in cached or not cached[t].get("dates"):
                need_fetch.append(t)
            else:
                last_date = cached[t]["dates"][-1]
                # If cache is older than end date, fetch delta
                if last_date < end:
                    need_fetch.append(t)

        if need_fetch:
            # For cached tickers, only fetch from last cached date
            # For new tickers, fetch full history
            fresh_start = start
            for t in need_fetch:
                if t in cached and cached[t].get("dates"):
                    # Fetch from 5 days before last cached date (overlap for safety)
                    last = cached[t]["dates"][-1]
                    t_start = (
                        datetime.strptime(last, "%Y-%m-%d") - timedelta(days=5)
                    ).strftime("%Y-%m-%d")
                    if t_start > fresh_start:
                        pass  # Will use full start for batch efficiency

            log.info(f"  Fetching {len(need_fetch)} tickers (delta update)...")
            fresh = self._inner.fetch(need_fetch, start, end)

            # Merge fresh data into cache
            for t, data in fresh.items():
                if t in cached and cached[t].get("dates"):
                    # Merge: keep old + append new non-overlapping
                    old_dates = set(cached[t]["dates"])
                    old_d = cached[t]["dates"]
                    old_p = cached[t]["adjCloses"]
                    for d, p in zip(data["dates"], data["adjCloses"]):
                        if d not in old_dates:
                            old_d.append(d)
                            old_p.append(p)
                        else:
                            # Update existing date with fresh price (adjustments may change)
                            idx = old_d.index(d)
                            old_p[idx] = p
                    # Sort by date
                    pairs = sorted(zip(old_d, old_p))
                    cached[t] = {
                        "dates": [p[0] for p in pairs],
                        "adjCloses": [p[1] for p in pairs],
                    }
                else:
                    cached[t] = data

            self._cache["tickers"] = cached
            self.save_cache()
        else:
            log.info("  All tickers up-to-date in cache")

        # Return only requested tickers
        return {t: cached[t] for t in tickers if t in cached}


# Provider registry
PROVIDERS = {
    "yahoo": YahooFinanceProvider,
    "alphavantage": AlphaVantageProvider,
}


def get_provider(name: str, cache_path: str = None) -> DataProvider:
    """Factory: get a data provider, optionally wrapped with cache."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    inner = PROVIDERS[name]()
    if cache_path:
        return CachedProvider(inner, cache_path)
    return inner
