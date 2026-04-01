"""
Bybit Futures Bot — Analytics Dashboard (Streamlit)

Run:  streamlit run src/analytics/streamlit_dashboard.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@st.cache_resource
def _engine():
    return create_engine(settings.sync_db_url, pool_pre_ping=True)


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with _engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bybit Futures Bot",
    page_icon="📈",
    layout="wide",
)

tab_explorer, tab_regime, tab_live, tab_quality = st.tabs([
    "Strategy Explorer",
    "Regime Analysis",
    "Live Monitor",
    "Data Quality",
])


# ===================================================================
# PAGE 1 — Strategy Explorer
# ===================================================================

with tab_explorer:
    st.header("Strategy Explorer")

    # ---- sidebar filters ------------------------------------------------
    with st.sidebar:
        st.subheader("Filters")

        all_symbols = _query(
            "SELECT DISTINCT symbol FROM trades_log ORDER BY symbol"
        )["symbol"].tolist() or settings.symbols

        sel_symbols = st.multiselect(
            "Symbol", all_symbols, default=all_symbols, key="se_sym"
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            dt_from = st.date_input(
                "From", value=date(2021, 1, 1), key="se_from"
            )
        with col_d2:
            dt_to = st.date_input(
                "To", value=date.today(), key="se_to"
            )

        min_trades = st.slider(
            "Min trades", 10, 500, 30, step=10, key="se_min"
        )

        regime_funding = st.selectbox(
            "Regime: funding",
            ["all", "negative", "neutral", "positive"],
            key="se_rf",
        )
        regime_vol = st.selectbox(
            "Regime: volatility",
            ["all", "low", "medium", "high"],
            key="se_rv",
        )
        regime_session = st.selectbox(
            "Regime: session",
            ["all", "asia", "london", "new_york", "off_hours"],
            key="se_rs",
        )

        data_source = st.radio(
            "Data source",
            ["Both", "Backtest only", "Live only"],
            key="se_src",
        )

    # ---- build query ----------------------------------------------------
    where_clauses = [
        "entry_time >= :dt_from",
        "entry_time <= :dt_to",
        "win_loss IS NOT NULL",
    ]
    params: dict = {
        "dt_from": str(dt_from),
        "dt_to": str(dt_to),
    }

    if sel_symbols:
        where_clauses.append("symbol = ANY(:symbols)")
        params["symbols"] = sel_symbols

    if regime_funding != "all":
        where_clauses.append("regime_funding = :rf")
        params["rf"] = regime_funding
    if regime_vol != "all":
        where_clauses.append("regime_volatility = :rv")
        params["rv"] = regime_vol
    if regime_session != "all":
        where_clauses.append("regime_time_of_day = :rs")
        params["rs"] = regime_session

    if data_source == "Backtest only":
        where_clauses.append("is_backtest = TRUE")
    elif data_source == "Live only":
        where_clauses.append("is_backtest = FALSE")

    where_sql = " AND ".join(where_clauses)

    combo_sql = f"""
        SELECT
            strategy_combo,
            symbol,
            COUNT(*)                                        AS trades,
            ROUND(AVG(win_loss::int)::numeric * 100, 1)     AS win_rate_pct,
            ROUND(AVG(pnl_pct)::numeric * 100, 4)           AS avg_pnl_pct,
            ROUND(
                (AVG(CASE WHEN win_loss THEN pnl_pct END)
                 * AVG(win_loss::int)
                 + AVG(CASE WHEN NOT win_loss THEN pnl_pct END)
                 * (1 - AVG(win_loss::int)))::numeric,
                6
            )                                               AS expectancy,
            ROUND((AVG(pnl_pct) /
                   NULLIF(STDDEV_POP(pnl_pct), 0))::numeric,
                  2)                                        AS sharpe,
            ROUND(MIN(pnl_pct)::numeric * 100, 2)           AS max_dd_pct
        FROM trades_log
        WHERE {where_sql}
        GROUP BY strategy_combo, symbol
        HAVING COUNT(*) >= :min_trades
        ORDER BY expectancy DESC
        LIMIT 50
    """
    params["min_trades"] = min_trades

    df_combos = _query(combo_sql, params)

    if df_combos.empty:
        st.info("No strategy combos match current filters.")
    else:
        df_combos["strategy_combo_str"] = df_combos["strategy_combo"].apply(
            lambda x: " + ".join(x) if isinstance(x, list) else str(x)
        )
        st.dataframe(
            df_combos[[
                "strategy_combo_str", "symbol", "trades",
                "win_rate_pct", "avg_pnl_pct", "expectancy",
                "sharpe", "max_dd_pct",
            ]].rename(columns={"strategy_combo_str": "strategy_combo"}),
            width="stretch",
            hide_index=True,
        )

        # ---- drill-down selection ---------------------------------------
        combo_options = df_combos["strategy_combo_str"].tolist()
        selected_combo = st.selectbox(
            "Drill into combo", combo_options, key="se_drill"
        )
        if selected_combo:
            combo_list = selected_combo.split(" + ")
            drill_params = {
                "combo": combo_list,
                "dt_from": str(dt_from),
                "dt_to": str(dt_to),
            }
            bt_clause = ""
            if data_source == "Backtest only":
                bt_clause = " AND is_backtest = TRUE"
            elif data_source == "Live only":
                bt_clause = " AND is_backtest = FALSE"

            drill_sql = f"""
                SELECT entry_time, exit_time, direction, leverage,
                       entry_price, exit_price, pnl_pct, pnl_usd,
                       funding_paid_usd, win_loss, exit_reason,
                       regime_volatility, regime_funding, regime_time_of_day
                FROM trades_log
                WHERE strategy_combo @> :combo
                  AND win_loss IS NOT NULL
                  AND entry_time >= :dt_from
                  AND entry_time <= :dt_to
                  {bt_clause}
                ORDER BY entry_time
            """
            df_trades = _query(drill_sql, drill_params)

            if not df_trades.empty:
                # --- equity curve ----------------------------------------
                df_trades["cum_pnl"] = df_trades["pnl_usd"].astype(float).cumsum()
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=df_trades["entry_time"],
                    y=df_trades["cum_pnl"],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#00d084", width=2),
                    name="Cumulative PnL (USD)",
                ))
                fig_eq.update_layout(
                    title="Equity Curve",
                    xaxis_title="Time",
                    yaxis_title="Cumulative PnL (USD)",
                    template="plotly_dark",
                    height=380,
                )
                st.plotly_chart(fig_eq, width="stretch")

                # --- win-rate heatmap by hour × day-of-week ---------------
                df_trades["hour"] = pd.to_datetime(
                    df_trades["entry_time"]
                ).dt.hour
                df_trades["dow"] = pd.to_datetime(
                    df_trades["entry_time"]
                ).dt.day_name()

                heatmap = df_trades.groupby(["hour", "dow"]).agg(
                    wr=("win_loss", "mean"),
                    n=("win_loss", "count"),
                ).reset_index()
                heatmap["wr"] = (heatmap["wr"] * 100).round(1)

                day_order = [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday",
                ]
                heatmap["dow"] = pd.Categorical(
                    heatmap["dow"], categories=day_order, ordered=True,
                )

                if not heatmap.empty:
                    pivot = heatmap.pivot(
                        index="hour", columns="dow", values="wr",
                    )
                    fig_hm = px.imshow(
                        pivot,
                        color_continuous_scale="RdYlGn",
                        labels=dict(color="Win Rate %"),
                        aspect="auto",
                    )
                    fig_hm.update_layout(
                        title="Win Rate by Hour × Day",
                        template="plotly_dark",
                        height=420,
                    )
                    st.plotly_chart(fig_hm, width="stretch")

                # --- trade list ------------------------------------------
                st.subheader("Trade List")
                st.dataframe(df_trades, width="stretch", hide_index=True)


# ===================================================================
# PAGE 2 — Regime Analysis
# ===================================================================

with tab_regime:
    st.header("Regime Analysis")

    combo_list_ra = _query(
        "SELECT DISTINCT strategy_combo FROM trades_log "
        "WHERE win_loss IS NOT NULL ORDER BY strategy_combo"
    )
    if combo_list_ra.empty:
        st.info("No closed trades in trades_log yet.")
    else:
        combo_strs_ra = combo_list_ra["strategy_combo"].apply(
            lambda x: " + ".join(x) if isinstance(x, list) else str(x)
        ).tolist()

        sel_ra = st.selectbox("Strategy Combo", combo_strs_ra, key="ra_combo")
        combo_arr = sel_ra.split(" + ") if sel_ra else []

        if combo_arr:
            regime_df = _query(
                """
                SELECT
                    regime_volatility,
                    regime_funding,
                    regime_time_of_day,
                    COUNT(*)                                     AS trades,
                    ROUND(AVG(win_loss::int)::numeric * 100, 1)  AS win_rate_pct,
                    ROUND(AVG(pnl_pct)::numeric * 100, 4)        AS avg_pnl_pct,
                    ROUND(SUM(pnl_usd)::numeric, 2)              AS total_pnl_usd
                FROM trades_log
                WHERE strategy_combo @> :combo
                  AND win_loss IS NOT NULL
                GROUP BY regime_volatility, regime_funding, regime_time_of_day
                ORDER BY win_rate_pct DESC
                """,
                {"combo": combo_arr},
            )

            if regime_df.empty:
                st.info("No regime data for this combo.")
            else:
                def _color_wr(val):
                    if pd.isna(val):
                        return ""
                    if val > 65:
                        return "background-color: #2ecc71; color: #000"
                    if val < 40:
                        return "background-color: #e74c3c; color: #fff"
                    return ""

                styled = regime_df.style.map(
                    _color_wr, subset=["win_rate_pct"]
                )
                st.dataframe(styled, width="stretch", hide_index=True)

                # Pivot: funding × volatility, averaged over sessions
                pivot_data = regime_df.groupby(
                    ["regime_funding", "regime_volatility"]
                ).agg(
                    wr=("win_rate_pct", "mean"),
                    n=("trades", "sum"),
                ).reset_index()

                if not pivot_data.empty:
                    piv = pivot_data.pivot(
                        index="regime_funding",
                        columns="regime_volatility",
                        values="wr",
                    )
                    fig_rg = px.imshow(
                        piv,
                        color_continuous_scale="RdYlGn",
                        labels=dict(color="Win Rate %"),
                        aspect="auto",
                        text_auto=".1f",
                    )
                    fig_rg.update_layout(
                        title="Win Rate: Funding × Volatility",
                        template="plotly_dark",
                        height=350,
                    )
                    st.plotly_chart(fig_rg, width="stretch")


# ===================================================================
# PAGE 3 — Live Monitor
# ===================================================================

with tab_live:
    st.header("Live Monitor")

    col_l1, col_l2 = st.columns(2)

    # ---- open positions -------------------------------------------------
    with col_l1:
        st.subheader("Open Positions")
        try:
            open_pos = _query(
                """
                SELECT symbol, direction, leverage, entry_price,
                       position_size_usd, entry_time, strategy_combo,
                       funding_paid_usd
                FROM trades_log
                WHERE exit_time IS NULL
                  AND win_loss IS NULL
                  AND is_backtest = FALSE
                ORDER BY entry_time DESC
                """
            )
            if open_pos.empty:
                st.caption("No open positions.")
            else:
                st.dataframe(open_pos, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Could not load positions: {exc}")

    # ---- today's PnL chart ----------------------------------------------
    with col_l2:
        st.subheader("Today's PnL")
        today_iso = datetime.now(timezone.utc).date().isoformat()
        try:
            pnl_today = _query(
                """
                SELECT exit_time, pnl_usd
                FROM trades_log
                WHERE is_backtest = FALSE
                  AND exit_time::date = :today
                  AND win_loss IS NOT NULL
                ORDER BY exit_time
                """,
                {"today": today_iso},
            )
            if pnl_today.empty:
                st.caption("No closed trades today.")
            else:
                pnl_today["cum_pnl"] = pnl_today["pnl_usd"].astype(float).cumsum()
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=pnl_today["exit_time"],
                    y=pnl_today["cum_pnl"],
                    mode="lines+markers",
                    line=dict(color="#00d084", width=2),
                ))
                fig_pnl.update_layout(
                    template="plotly_dark", height=280,
                    xaxis_title="Time", yaxis_title="Cum PnL (USD)",
                )
                st.plotly_chart(fig_pnl, width="stretch")
        except Exception as exc:
            st.error(f"Could not load PnL: {exc}")

    # ---- last 20 signals ------------------------------------------------
    st.subheader("Last 20 Signals (generated / blocked / filled)")
    try:
        signals = _query(
            """
            SELECT entry_time, symbol, direction, strategy_combo,
                   exit_reason, win_loss, pnl_usd, leverage
            FROM trades_log
            WHERE is_backtest = FALSE
            ORDER BY entry_time DESC
            LIMIT 20
            """
        )
        if signals.empty:
            st.caption("No live signals yet.")
        else:
            st.dataframe(signals, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Could not load signals: {exc}")

    # ---- funding rates --------------------------------------------------
    st.subheader("Latest Funding Rates")
    try:
        funding = _query(
            """
            SELECT DISTINCT ON (symbol)
                symbol, timestamp, funding_rate, predicted_rate
            FROM funding_history
            ORDER BY symbol, timestamp DESC
            """
        )
        if funding.empty:
            st.caption("No funding data.")
        else:
            st.dataframe(funding, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Could not load funding: {exc}")


# ===================================================================
# PAGE 4 — Data Quality
# ===================================================================

with tab_quality:
    st.header("Data Quality")

    # ---- row counts & date ranges ---------------------------------------
    st.subheader("Table Overview")
    tables = [
        "candles_5m", "candles_1m", "indicators_5m", "indicators_15m",
        "trades_log", "strategy_performance", "funding_history",
        "liquidations", "raw_trades",
    ]
    rows_data = []
    for tbl in tables:
        try:
            info = _query(f"""
                SELECT
                    COUNT(*)                              AS row_count,
                    MIN(timestamp)::text                  AS min_ts,
                    MAX(timestamp)::text                  AS max_ts
                FROM {tbl}
            """)
            rows_data.append({
                "table": tbl,
                "rows": int(info["row_count"].iloc[0]),
                "earliest": info["min_ts"].iloc[0],
                "latest": info["max_ts"].iloc[0],
            })
        except Exception:
            rows_data.append({
                "table": tbl,
                "rows": 0,
                "earliest": "-",
                "latest": "-",
            })
    st.dataframe(
        pd.DataFrame(rows_data),
        width="stretch",
        hide_index=True,
    )

    # ---- gap detection --------------------------------------------------
    st.subheader("5m Candle Gaps")
    for sym in settings.symbols:
        try:
            gaps = _query(
                """
                SELECT COUNT(*) AS gaps FROM (
                    SELECT timestamp,
                           LEAD(timestamp) OVER (ORDER BY timestamp)
                           - timestamp AS gap
                    FROM candles_5m
                    WHERE symbol = :sym
                ) t
                WHERE gap > INTERVAL '5 minutes 30 seconds'
                """,
                {"sym": sym},
            )
            gap_count = int(gaps["gaps"].iloc[0])
            if gap_count == 0:
                st.success(f"{sym}: no gaps detected")
            else:
                st.warning(f"{sym}: {gap_count} gap(s) > 5.5 min")
        except Exception as exc:
            st.error(f"{sym}: {exc}")

    # ---- quality flags --------------------------------------------------
    st.subheader("Quality Flags")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        try:
            zero_vol = _query(
                "SELECT COUNT(*) AS n FROM candles_5m WHERE volume = 0"
            )
            n = int(zero_vol["n"].iloc[0])
            if n == 0:
                st.success("No candles with volume = 0")
            else:
                st.warning(f"{n} candle(s) with volume = 0")
        except Exception as exc:
            st.error(str(exc))

    with col_q2:
        try:
            null_ema = _query(
                """
                SELECT COUNT(*) AS n FROM indicators_5m
                WHERE ema_9 IS NULL
                  AND timestamp > (
                      SELECT MIN(timestamp) + INTERVAL '200 * 5 minutes'
                      FROM indicators_5m
                  )
                """
            )
            n = int(null_ema["n"].iloc[0])
            if n == 0:
                st.success("No NULL ema_9 after warmup")
            else:
                st.warning(f"{n} row(s) with NULL ema_9 after warmup")
        except Exception as exc:
            st.error(str(exc))

    # ---- last sync timestamps -------------------------------------------
    st.subheader("Sync Timestamps")
    try:
        last_candle = _query(
            "SELECT MAX(timestamp)::text AS ts FROM candles_5m"
        )
        st.metric(
            "Last candle_5m timestamp",
            last_candle["ts"].iloc[0] or "N/A",
        )
    except Exception:
        st.metric("Last candle_5m timestamp", "N/A")

    try:
        last_indicator = _query(
            "SELECT MAX(timestamp)::text AS ts FROM indicators_5m"
        )
        st.metric(
            "Last indicators_5m timestamp",
            last_indicator["ts"].iloc[0] or "N/A",
        )
    except Exception:
        st.metric("Last indicators_5m timestamp", "N/A")

    try:
        last_funding = _query(
            "SELECT MAX(timestamp)::text AS ts FROM funding_history"
        )
        st.metric(
            "Last funding_history timestamp",
            last_funding["ts"].iloc[0] or "N/A",
        )
    except Exception:
        st.metric("Last funding_history timestamp", "N/A")
