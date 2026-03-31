"""Custom indicators not available in pandas_ta — RSI divergence detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_rsi_divergence(
    df: pd.DataFrame,
    lookback: int = 20,
    min_bars_between_pivots: int = 3,
) -> pd.DataFrame:
    """
    Detect regular and hidden RSI divergence.

    Regular bullish:  price lower low  + RSI higher low  -> potential long
    Regular bearish:  price higher high + RSI lower high  -> potential short
    Hidden bullish:   price higher low  + RSI lower low   -> trend continuation long
    Hidden bearish:   price lower high  + RSI higher high -> trend continuation short

    Adds columns: div_regular_bull, div_regular_bear, div_hidden_bull, div_hidden_bear.
    Uses pivot high/low detection over *lookback* window.
    First *lookback* rows are always False.
    """
    n = len(df)
    reg_bull = np.zeros(n, dtype=bool)
    reg_bear = np.zeros(n, dtype=bool)
    hid_bull = np.zeros(n, dtype=bool)
    hid_bear = np.zeros(n, dtype=bool)

    if n < lookback:
        df["div_regular_bull"] = reg_bull
        df["div_regular_bear"] = reg_bear
        df["div_hidden_bull"] = hid_bull
        df["div_hidden_bear"] = hid_bear
        return df

    rsi_col = "RSI_14" if "RSI_14" in df.columns else "rsi_14"
    low = df["low"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    rsi = df[rsi_col].values.astype(np.float64)

    half = lookback // 2

    pivot_low_idx: list[int] = []
    pivot_high_idx: list[int] = []

    for i in range(half, n - half):
        lo_win = low[i - half : i + half + 1]
        hi_win = high[i - half : i + half + 1]

        if low[i] <= np.nanmin(lo_win):
            pivot_low_idx.append(i)
        if high[i] >= np.nanmax(hi_win):
            pivot_high_idx.append(i)

    # --- bullish divergence (compare consecutive pivot lows) ---
    for j in range(1, len(pivot_low_idx)):
        prev_i = pivot_low_idx[j - 1]
        curr_i = pivot_low_idx[j]
        if curr_i - prev_i < min_bars_between_pivots:
            continue
        if np.isnan(rsi[prev_i]) or np.isnan(rsi[curr_i]):
            continue

        if low[curr_i] < low[prev_i] and rsi[curr_i] > rsi[prev_i]:
            reg_bull[curr_i] = True
        if low[curr_i] > low[prev_i] and rsi[curr_i] < rsi[prev_i]:
            hid_bull[curr_i] = True

    # --- bearish divergence (compare consecutive pivot highs) ---
    for j in range(1, len(pivot_high_idx)):
        prev_i = pivot_high_idx[j - 1]
        curr_i = pivot_high_idx[j]
        if curr_i - prev_i < min_bars_between_pivots:
            continue
        if np.isnan(rsi[prev_i]) or np.isnan(rsi[curr_i]):
            continue

        if high[curr_i] > high[prev_i] and rsi[curr_i] < rsi[prev_i]:
            reg_bear[curr_i] = True
        if high[curr_i] < high[prev_i] and rsi[curr_i] > rsi[prev_i]:
            hid_bear[curr_i] = True

    df["div_regular_bull"] = reg_bull
    df["div_regular_bear"] = reg_bear
    df["div_hidden_bull"] = hid_bull
    df["div_hidden_bear"] = hid_bear
    return df
