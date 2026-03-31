"""
Main indicator pipeline: reads candles, computes all indicators, writes to
indicators_5m and indicators_15m.

Usage
-----
    python src/indicators/compute_all.py --symbol BTCUSDT [--from 2021-01-01]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401 — registers .ta accessor
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402
from src.db.models import Indicators5m, Indicators15m  # noqa: E402
from src.indicators.custom_indicators import detect_rsi_divergence  # noqa: E402
from src.indicators.resample import resample_5m_to_15m  # noqa: E402

log = structlog.get_logger()

BATCH_SIZE = 2000

# pandas_ta v0.4.x column name -> DB column name
TA_COL_MAP: dict[str, str] = {
    "EMA_9": "ema_9",
    "EMA_21": "ema_21",
    "EMA_50": "ema_50",
    "EMA_200": "ema_200",
    "RSI_14": "rsi_14",
    "STOCHRSIk_14_14_3_3": "stochrsi_k",
    "STOCHRSId_14_14_3_3": "stochrsi_d",
    "MACD_12_26_9": "macd_line",
    "MACDs_12_26_9": "macd_signal",
    "MACDh_12_26_9": "macd_hist",
    "MACD_5_13_1": "macd_fast_line",
    "MACDs_5_13_1": "macd_fast_signal",
    "MACDh_5_13_1": "macd_fast_hist",
    "BBU_20_2.0_2.0": "bb_upper",
    "BBM_20_2.0_2.0": "bb_mid",
    "BBL_20_2.0_2.0": "bb_lower",
    "BBB_20_2.0_2.0": "bb_width",
    "KCUe_20_1.5": "kc_upper",
    "KCLe_20_1.5": "kc_lower",
    "SUPERT_10_3.0": "supertrend",
    "SUPERTd_10_3.0": "supertrend_dir",
    "ATRr_14": "atr_14",
    "VWAP_D": "vwap",
    "OBV": "obv",
}

DERIVED_FLOAT_COLS = [
    "atr_pct_rank",
    "vwap_dev_upper1", "vwap_dev_lower1",
    "vwap_dev_upper2", "vwap_dev_lower2",
    "order_flow_imb",
    "ha_open", "ha_high", "ha_low", "ha_close",
    "funding_8h", "funding_24h_cum", "liq_volume_1h",
    "volume_delta",
]

DIV_COLS = [
    "div_regular_bull", "div_regular_bear",
    "div_hidden_bull", "div_hidden_bear",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _engine():
    return create_engine(settings.sync_db_url)


def load_candles(engine, symbol: str, from_date: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM candles_5m WHERE symbol = %(symbol)s"
    params: dict = {"symbol": symbol}
    if from_date:
        query += " AND timestamp >= %(from_date)s"
        params["from_date"] = from_date
    query += " ORDER BY timestamp"
    df = pd.read_sql(query, engine, params=params, parse_dates=["timestamp"])
    log.info("candles_loaded", symbol=symbol, rows=len(df))
    return df


def load_funding(engine, symbol: str) -> pd.DataFrame:
    query = (
        "SELECT timestamp, funding_rate FROM funding_history "
        "WHERE symbol = %(symbol)s ORDER BY timestamp"
    )
    return pd.read_sql(query, engine, params={"symbol": symbol},
                       parse_dates=["timestamp"])


def load_liquidations(engine, symbol: str) -> pd.DataFrame:
    query = (
        "SELECT timestamp, value_usd FROM liquidations "
        "WHERE symbol = %(symbol)s ORDER BY timestamp"
    )
    return pd.read_sql(query, engine, params={"symbol": symbol},
                       parse_dates=["timestamp"])


# ---------------------------------------------------------------------------
# pandas_ta indicator computation
# ---------------------------------------------------------------------------

def compute_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)

    df.ta.rsi(length=14, append=True)
    df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)

    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.macd(fast=5, slow=13, signal=1, append=True)

    df.ta.bbands(length=20, std=2.0, append=True)

    df.ta.kc(length=20, scalar=1.5, append=True)

    df.ta.supertrend(length=10, multiplier=3.0, append=True)

    df.ta.atr(length=14, append=True)

    df.ta.vwap(append=True)

    df.ta.obv(append=True)

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Derived / custom columns
# ---------------------------------------------------------------------------

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    # BB Squeeze: BB width (absolute) < KC width (absolute)
    bb_upper = df.get("BBU_20_2.0_2.0", pd.Series(dtype=float))
    bb_lower = df.get("BBL_20_2.0_2.0", pd.Series(dtype=float))
    kc_upper = df.get("KCUe_20_1.5", pd.Series(dtype=float))
    kc_lower = df.get("KCLe_20_1.5", pd.Series(dtype=float))
    df["bb_squeeze"] = (bb_upper - bb_lower) < (kc_upper - kc_lower)

    # ATR percentile rank (rolling 288 bars = 1 day of 5m candles)
    atr = df.get("ATRr_14", pd.Series(dtype=float))
    df["atr_pct_rank"] = atr.rolling(288, min_periods=1).rank(pct=True)

    # VWAP deviation bands (+-1 ATR, +-2 ATR)
    vwap = df.get("VWAP_D", pd.Series(dtype=float))
    df["vwap_dev_upper1"] = vwap + atr
    df["vwap_dev_lower1"] = vwap - atr
    df["vwap_dev_upper2"] = vwap + (atr * 2)
    df["vwap_dev_lower2"] = vwap - (atr * 2)

    # Order flow imbalance (3-bar rolling avg of buy_vol / total_vol)
    df["order_flow_imb"] = (
        df["buy_volume"]
        .div(df["volume"].replace(0, pd.NA))
        .rolling(3, min_periods=1)
        .mean()
    )

    # Heikin Ashi
    df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["ha_open"] = (df["open"].shift(1) + df["close"].shift(1)) / 2
    df.loc[df.index[0], "ha_open"] = (
        (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    )
    df["ha_high"] = df[["high", "ha_open", "ha_close"]].max(axis=1)
    df["ha_low"] = df[["low", "ha_open", "ha_close"]].min(axis=1)

    return df


def join_funding(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    if funding_df.empty:
        df["funding_8h"] = np.nan
        df["funding_24h_cum"] = np.nan
        return df

    funding_sorted = funding_df.sort_values("timestamp")
    df_sorted = df.sort_values("timestamp")

    merged = pd.merge_asof(
        df_sorted[["timestamp"]],
        funding_sorted[["timestamp", "funding_rate"]],
        on="timestamp",
        direction="backward",
    )
    df["funding_8h"] = merged["funding_rate"].values

    funding_sorted = funding_sorted.copy()
    funding_sorted["funding_24h_cum"] = (
        funding_sorted["funding_rate"].rolling(3, min_periods=1).sum()
    )
    merged_cum = pd.merge_asof(
        df_sorted[["timestamp"]],
        funding_sorted[["timestamp", "funding_24h_cum"]],
        on="timestamp",
        direction="backward",
    )
    df["funding_24h_cum"] = merged_cum["funding_24h_cum"].values
    return df


def join_liquidations(df: pd.DataFrame, liq_df: pd.DataFrame) -> pd.DataFrame:
    if liq_df.empty:
        df["liq_volume_1h"] = 0.0
        return df

    liq = liq_df.set_index("timestamp")
    liq_5m = liq.resample("5min")["value_usd"].sum().fillna(0)
    liq_rolling = liq_5m.rolling("60min", min_periods=1).sum()

    df = df.merge(
        liq_rolling.rename("liq_volume_1h"),
        left_on="timestamp",
        right_index=True,
        how="left",
    )
    df["liq_volume_1h"] = df["liq_volume_1h"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# DataFrame -> DB records
# ---------------------------------------------------------------------------

def _safe_float(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return float(val)


def _safe_int(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return int(val)


def _safe_bool(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return bool(val)


def build_records(df: pd.DataFrame, symbol: str) -> list[dict]:
    records: list[dict] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        rec: dict = {"symbol": symbol, "timestamp": row["timestamp"]}

        for ta_col, db_col in TA_COL_MAP.items():
            val = row.get(ta_col)
            if db_col == "supertrend_dir":
                rec[db_col] = _safe_int(val)
            else:
                rec[db_col] = _safe_float(val)

        for col in DERIVED_FLOAT_COLS:
            rec[col] = _safe_float(row.get(col))

        rec["bb_squeeze"] = _safe_bool(row.get("bb_squeeze"))

        extras = {}
        for c in DIV_COLS:
            v = row.get(c, False)
            extras[c] = bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False
        rec["extras"] = extras

        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def upsert_batch(engine, table, records: list[dict]) -> None:
    if not records:
        return
    pk_cols = [col.name for col in table.primary_key.columns]
    update_cols = [
        col.name for col in table.columns
        if col.name not in pk_cols
    ]
    stmt = pg_insert(table).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=pk_cols,
        set_={col: stmt.excluded[col] for col in update_cols},
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def write_indicators(engine, model, records: list[dict], label: str) -> None:
    table = model.__table__
    total = len(records)
    for i in range(0, total, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        upsert_batch(engine, table, batch)
        log.info(
            "batch_upserted",
            table=label,
            rows=len(batch),
            progress=f"{min(i + BATCH_SIZE, total)}/{total}",
        )
    log.info("write_complete", table=label, total_rows=total)


# ---------------------------------------------------------------------------
# Full pipeline for one symbol
# ---------------------------------------------------------------------------

def run_pipeline(
    engine,
    symbol: str,
    from_date: str | None = None,
) -> None:
    # ---- 5m indicators ----
    df = load_candles(engine, symbol, from_date)
    if df.empty:
        log.warning("no_candles", symbol=symbol)
        return

    funding_df = load_funding(engine, symbol)
    liq_df = load_liquidations(engine, symbol)

    log.info("computing_5m_indicators", symbol=symbol, candles=len(df))
    df = compute_ta_indicators(df)
    df = compute_derived(df)
    df = detect_rsi_divergence(df)
    df = join_funding(df, funding_df)
    df = join_liquidations(df, liq_df)

    records_5m = build_records(df, symbol)
    write_indicators(engine, Indicators5m, records_5m, "indicators_5m")

    # ---- 15m indicators ----
    log.info("resampling_to_15m", symbol=symbol)
    df_candles = load_candles(engine, symbol, from_date)
    df_15m = resample_5m_to_15m(df_candles)
    if df_15m.empty:
        log.warning("no_15m_candles", symbol=symbol)
        return

    log.info("computing_15m_indicators", symbol=symbol, candles=len(df_15m))
    df_15m = compute_ta_indicators(df_15m)
    df_15m = compute_derived(df_15m)
    df_15m = detect_rsi_divergence(df_15m)
    df_15m = join_funding(df_15m, funding_df)
    df_15m = join_liquidations(df_15m, liq_df)

    records_15m = build_records(df_15m, symbol)
    write_indicators(engine, Indicators15m, records_15m, "indicators_15m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute indicators for a symbol and write to DB",
    )
    parser.add_argument(
        "--symbol", required=True, help="Symbol (e.g. BTCUSDT)",
    )
    parser.add_argument(
        "--from", dest="from_date", default=None,
        help="Start date inclusive (e.g. 2021-01-01)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = _engine()
    log.info("pipeline_start", symbol=args.symbol, from_date=args.from_date)
    run_pipeline(engine, args.symbol, args.from_date)
    log.info("pipeline_complete", symbol=args.symbol)


if __name__ == "__main__":
    main()
