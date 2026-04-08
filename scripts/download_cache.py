"""Download 5m candles to data_cache for symbols that are missing."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

FROM = "2023-01-01"
TO = "2026-04-03"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def fetch_5m_candles(symbol: str, since: str, until: str) -> pd.DataFrame:
    import ccxt
    exchange = ccxt.bybit({"enableRateLimit": True})
    exchange.load_markets()
    from src.live.order_manager import _to_ccxt_symbol
    bybit_symbol = _to_ccxt_symbol(symbol)
    if bybit_symbol not in exchange.markets:
        bybit_symbol = symbol.replace("USDT", "/USDT:USDT")
    since_ts = int(datetime.fromisoformat(since).replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    until_ts = int(datetime.fromisoformat(until).replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    all_candles: list[list] = []
    current = since_ts
    while current < until_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(
                bybit_symbol, "5m", since=current, limit=1000)
        except Exception as e:
            print(f"  [WARN] {symbol} fetch error: {e}", flush=True)
            time.sleep(2)
            continue
        if not ohlcv:
            break
        for candle in ohlcv:
            if candle[0] >= until_ts:
                break
            all_candles.append(candle)
        last_ts = ohlcv[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
        if len(all_candles) % 10000 == 0:
            print(f"    {symbol}: {len(all_candles):,} candles so far ...", flush=True)
    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low",
                               "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp").reset_index(drop=True))
    df["symbol"] = symbol
    df["buy_volume"] = df["volume"] * 0.5
    df["sell_volume"] = df["volume"] * 0.5
    df["volume_delta"] = 0.0
    df["quote_volume"] = df["volume"] * df["close"]
    df["trade_count"] = 0
    df["mark_price"] = df["close"]
    df["funding_rate"] = 0.0
    return df


def main():
    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(exist_ok=True)

    for sym in SYMBOLS:
        cache_path = cache_dir / f"{sym}_{FROM}_{TO}_5m.parquet"
        if cache_path.exists():
            print(f"{sym}: already cached ({cache_path.name})", flush=True)
            continue
        print(f"{sym}: downloading {FROM} -> {TO} ...", flush=True)
        t0 = time.time()
        df = fetch_5m_candles(sym, FROM, TO)
        if df.empty:
            print(f"  {sym}: NO DATA returned", flush=True)
            continue
        df.to_parquet(cache_path)
        print(f"  {sym}: {len(df):,} candles saved ({time.time()-t0:.0f}s)", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
