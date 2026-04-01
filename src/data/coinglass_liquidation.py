"""
CoinGlass Open API v4 client (liquidation + optional heatmap).

Auth: header ``CG-API-KEY`` (see https://docs.coinglass.com/reference/authentication).

Hobbyist plan (typical limits):
- ~30 requests / minute (exact cap in ``API-KEY-MAX-LIMIT`` response header).
- Aggregated liquidation history: interval must be **>= 4h**
  (https://docs.coinglass.com/reference/aggregated-liquidation-history).
- **Liquidation heatmap Model1** is **not** on Hobbyist (Professional+ only);
  :func:`fetch_liquidation_heatmap_model1` will fail with a clear error.

Pro heatmap UI (for reference): https://www.coinglass.com/pro/futures/LiquidationHeatMap
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import settings

log = structlog.get_logger()

BASE_URL_DEFAULT = "https://open-api-v4.coinglass.com"

# Hobbyist: docs say >= 4h for aggregated liquidation history
HOBBYIST_MIN_INTERVALS = frozenset({"4h", "6h", "8h", "12h", "1d", "1w"})


class RateLimiter:
    """Sliding window: at most *max_calls* per *period_sec* (thread-safe)."""

    def __init__(self, max_calls: int, period_sec: float = 60.0) -> None:
        self.max_calls = max(1, max_calls)
        self.period = period_sec
        self._lock = threading.Lock()
        self._times: list[float] = []

    def acquire(self) -> None:
        while True:
            sleep_for: float = 0.0
            with self._lock:
                now = time.monotonic()
                self._times = [t for t in self._times if t > now - self.period]
                if len(self._times) < self.max_calls:
                    self._times.append(now)
                    return
                oldest = min(self._times)
                sleep_for = self.period - (now - oldest) + 0.05
            time.sleep(max(sleep_for, 0.05))


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def _get_limiter() -> RateLimiter:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            # Stay slightly under advertised cap to avoid edge 429s
            rpm = max(1, settings.coinglass_max_requests_per_minute)
            _limiter = RateLimiter(max_calls=rpm, period_sec=60.0)
        return _limiter


def bybit_symbol_to_coinglass_coin(symbol: str) -> str:
    """BTCUSDT -> BTC, 1000PEPEUSDT -> strip quote heuristics."""
    s = symbol.upper().strip()
    for quote in ("USDT", "USDC", "USD", "BUSD"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)]
    return s


@dataclass
class CoinGlassClient:
    """Thin synchronous HTTP client with rate limiting and 429 retries."""

    api_key: str = field(default_factory=lambda: settings.coinglass_api_key)
    base_url: str = field(default_factory=lambda: settings.coinglass_base_url.strip().rstrip("/"))
    timeout_sec: float = 25.0
    max_retries: int = 4

    def _request(
        self,
        path: str,
        params: dict[str, str | int | None],
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("COINGLASS_API_KEY is not set")

        q = {k: str(v) for k, v in params.items() if v is not None and v != ""}
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(
            url,
            headers={
                "accept": "application/json",
                "CG-API-KEY": self.api_key,
            },
            method="GET",
        )

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            _get_limiter().acquire()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                    max_lim = resp.headers.get("API-KEY-MAX-LIMIT")
                    use_lim = resp.headers.get("API-KEY-USE-LIMIT")
                    if max_lim or use_lim:
                        log.debug(
                            "coinglass_quota_headers",
                            api_key_max_limit=max_lim,
                            api_key_use_limit=use_lim,
                        )
                data = json.loads(raw)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code == 429:
                    wait = 60.0 * (2**attempt) / 4
                    log.warning(
                        "coinglass_rate_limited",
                        attempt=attempt + 1,
                        sleep_s=wait,
                        body=body[:500],
                    )
                    time.sleep(wait)
                    last_err = e
                    continue
                log.error("coinglass_http_error", code=e.code, body=body[:800])
                raise RuntimeError(f"CoinGlass HTTP {e.code}: {body[:500]}") from e
            except urllib.error.URLError as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
                continue

            code = data.get("code")
            if str(code) != "0":
                msg = data.get("msg", str(data))
                raise RuntimeError(f"CoinGlass API error code={code} msg={msg}")
            return data

        raise RuntimeError(f"CoinGlass request failed after retries: {last_err}")

    def aggregated_liquidation_history(
        self,
        symbol: str,
        *,
        interval: str | None = None,
        exchange_list: str | None = None,
        limit: int = 100,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        GET /api/futures/liquidation/aggregated-history

        Hobbyist: use interval >= 4h (enforced if settings enforce flag is on).
        """
        coin = bybit_symbol_to_coinglass_coin(symbol)
        iv = interval or settings.coinglass_min_liquidation_interval
        if settings.coinglass_enforce_hobbyist_limits and iv not in HOBBYIST_MIN_INTERVALS:
            raise ValueError(
                f"interval={iv!r} may be rejected on Hobbyist; "
                f"use one of {sorted(HOBBYIST_MIN_INTERVALS)}"
            )
        ex = exchange_list or settings.coinglass_exchange_list
        payload = self._request(
            "/api/futures/liquidation/aggregated-history",
            {
                "exchange_list": ex,
                "symbol": coin,
                "interval": iv,
                "limit": limit,
                "start_time": start_time_ms,
                "end_time": end_time_ms,
            },
        )
        raw_data = payload.get("data")
        if not isinstance(raw_data, list):
            return []
        return raw_data

    def liquidation_heatmap_model1(
        self,
        symbol: str,
        *,
        range_param: str = "3d",
    ) -> dict[str, Any]:
        """
        GET /api/futures/liquidation/aggregated-heatmap/model1

        **Professional+ only** — Hobbyist receives an error from the API.
        """
        coin = bybit_symbol_to_coinglass_coin(symbol)
        payload = self._request(
            "/api/futures/liquidation/aggregated-heatmap/model1",
            {"symbol": coin, "range": range_param},
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            return {}
        return data


def fetch_liquidation_snapshot(
    symbol: str,
    *,
    interval: str | None = None,
) -> dict[str, Any] | None:
    """
    Fetch recent aggregated long/short liquidation USD (Hobbyist-safe defaults).

    Returns a small dict for strategies, e.g.::

        {
            "symbol": "BTCUSDT",
            "interval": "4h",
            "latest": {...},
            "rows": [...],
        }
    """
    if not settings.coinglass_api_key:
        log.debug("coinglass_skip_no_key")
        return None
    client = CoinGlassClient()
    iv = interval or settings.coinglass_min_liquidation_interval
    rows = client.aggregated_liquidation_history(symbol, interval=iv, limit=48)
    if not rows:
        return {"symbol": symbol, "interval": iv, "latest": None, "rows": []}
    latest = rows[-1]
    return {
        "symbol": symbol,
        "interval": iv,
        "latest": latest,
        "rows": rows,
    }


def sync_liquidation_bars_to_db(engine, symbol: str | None = None) -> int:
    """
    Pull aggregated liquidation history from CoinGlass and upsert into
    ``coinglass_liquidation_bars``. One API call per symbol (rate-limited).

    Returns total rows upserted.
    """
    from datetime import datetime, timezone

    from sqlalchemy import func
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from src.db.models import CoinglassLiquidationBar

    if not settings.coinglass_api_key:
        log.warning("coinglass_sync_skipped_no_key")
        return 0

    symbols = [symbol] if symbol else list(settings.symbols)
    client = CoinGlassClient()
    iv = settings.coinglass_min_liquidation_interval
    lim = min(1000, max(1, settings.coinglass_sync_history_limit))
    table = CoinglassLiquidationBar.__table__
    total = 0

    for sym in symbols:
        rows = client.aggregated_liquidation_history(
            sym, interval=iv, limit=lim,
        )
        batch: list[dict[str, Any]] = []
        for r in rows:
            t = r.get("time")
            if t is None:
                continue
            ts = datetime.fromtimestamp(int(t) / 1000.0, tz=timezone.utc)
            batch.append(
                {
                    "symbol": sym,
                    "bucket_time": ts,
                    "interval": iv,
                    "long_liquidation_usd": float(
                        r.get("aggregated_long_liquidation_usd") or 0
                    ),
                    "short_liquidation_usd": float(
                        r.get("aggregated_short_liquidation_usd") or 0
                    ),
                }
            )
        if not batch:
            log.info("coinglass_sync_no_rows", symbol=sym)
            continue
        stmt = pg_insert(table).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "bucket_time", "interval"],
            set_={
                "long_liquidation_usd": stmt.excluded.long_liquidation_usd,
                "short_liquidation_usd": stmt.excluded.short_liquidation_usd,
                "fetched_at": func.now(),
            },
        )
        with engine.begin() as conn:
            conn.execute(stmt)
        total += len(batch)
        log.info("coinglass_sync_upserted", symbol=sym, rows=len(batch))

    return total


def fetch_liquidation_heatmap_model1(
    symbol: str,
    *,
    range_param: str = "3d",
) -> dict[str, Any] | None:
    """Heatmap payload or None if not configured; raises on API/plan errors."""
    if not settings.coinglass_api_key:
        return None
    client = CoinGlassClient()
    return client.liquidation_heatmap_model1(symbol, range_param=range_param)


if __name__ == "__main__":
    # From repo root:  cd bybit_futures_bot && python -m src.data.coinglass_liquidation BTCUSDT
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    snap = fetch_liquidation_snapshot(sym)
    print(json.dumps(snap, indent=2, default=str)[:8000])
