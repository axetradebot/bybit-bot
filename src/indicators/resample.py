"""Resample 5m candles to 15m candles for the 15m indicator pipeline."""

from __future__ import annotations

import pandas as pd


def resample_5m_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Build 15m OHLCV from 5m candles (3 bars -> 1 bar).

    Aggregation rules
    -----------------
    - OHLC: standard (first / max / min / last)
    - volume, buy_volume, sell_volume, volume_delta, quote_volume, trade_count: sum
    - mark_price, funding_rate: last value in window
    """
    df = df_5m.copy()
    df = df.set_index("timestamp")

    agg_rules = {
        "symbol": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "buy_volume": "sum",
        "sell_volume": "sum",
        "volume_delta": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
        "mark_price": "last",
        "funding_rate": "last",
    }

    existing = {k: v for k, v in agg_rules.items() if k in df.columns}

    df_15m = df.resample("15min", label="right", closed="right").agg(existing)
    df_15m = df_15m.dropna(subset=["close"])
    df_15m = df_15m.reset_index()

    return df_15m
