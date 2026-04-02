"""Resample 5m candles to higher timeframes for multi-TF analysis."""

from __future__ import annotations

import pandas as pd


TF_FREQ: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
}

CONTEXT_TF: dict[str, str] = {
    "5m": "1h",
    "15m": "1h",
    "1h": "4h",
    "4h": "4h",
}

_AGG_RULES = {
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


def resample_5m_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Build 15m OHLCV from 5m candles (backward compat)."""
    return resample_candles(df_5m, "15m")


def resample_candles(df_5m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 5m candles to any higher timeframe.

    Parameters
    ----------
    df_5m : DataFrame with columns: timestamp, symbol, open, high, low, close,
            volume, and optionally buy_volume, sell_volume, etc.
    target_tf : One of '5m', '15m', '30m', '1h', '2h', '4h'.
    """
    if target_tf == "5m":
        return df_5m.copy()
    freq = TF_FREQ.get(target_tf)
    if freq is None:
        raise ValueError(f"Unsupported timeframe: {target_tf}")

    df = df_5m.copy().set_index("timestamp")
    existing = {k: v for k, v in _AGG_RULES.items() if k in df.columns}
    resampled = df.resample(freq, label="right", closed="right").agg(existing)
    resampled = resampled.dropna(subset=["close"]).reset_index()
    return resampled


def build_bars_for_tf(df_5m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample 5m candles and compute the full indicator suite.

    Returns a DataFrame with DB-named columns (ema_9, rsi_14, etc.)
    ready to be consumed by the backtester/strategy.

    Raises ValueError for target_tf='5m' — use pre-computed DB indicators.
    """
    if target_tf == "5m":
        raise ValueError("For 5m use DB indicators; this function is for "
                         "resampled timeframes only")

    from src.indicators.compute_all import (
        TA_COL_MAP,
        compute_derived,
        compute_ta_indicators,
    )

    resampled = resample_candles(df_5m, target_tf)
    with_ta = compute_ta_indicators(resampled)
    with_derived = compute_derived(with_ta)

    rename_map = {
        ta: db for ta, db in TA_COL_MAP.items() if ta in with_derived.columns
    }
    result = with_derived.rename(columns=rename_map)

    for col in ("funding_8h", "funding_24h_cum", "liq_volume_1h"):
        if col not in result.columns:
            result[col] = 0.0
    if "extras" not in result.columns:
        result["extras"] = [{} for _ in range(len(result))]

    return result
