"""
Mapping functions: Tardis Machine normalized records → ORM-compatible dicts.

Each function takes a raw JSON record (dict) from the tardis-machine
replay-normalized NDJSON stream and returns a dict ready for
SQLAlchemy pg_insert().values().
"""

from datetime import datetime
from decimal import Decimal


def _dec(value) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def map_trade_bar(record: dict) -> dict:
    """Tardis trade_bar → candles row (works for both 1m and 5m).

    Tardis fields: timestamp, open, high, low, close, volume,
                   buyVolume, sellVolume, vwap, trades, symbol
    """
    buy_vol = Decimal(str(record["buyVolume"]))
    sell_vol = Decimal(str(record["sellVolume"]))
    volume = Decimal(str(record["volume"]))
    vwap = Decimal(str(record["vwap"]))

    return {
        "symbol": record["symbol"],
        "timestamp": _ts(record["timestamp"]),
        "open": Decimal(str(record["open"])),
        "high": Decimal(str(record["high"])),
        "low": Decimal(str(record["low"])),
        "close": Decimal(str(record["close"])),
        "volume": volume,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "volume_delta": buy_vol - sell_vol,
        "quote_volume": volume * vwap,
        "trade_count": record.get("trades"),
    }


def map_derivative_ticker_to_funding(record: dict) -> dict:
    """Tardis derivative_ticker → funding_history row.

    Called by the downloader only when a funding event is detected
    (fundingTimestamp transitions to a new value).

    Tardis fields: fundingTimestamp, fundingRate, markPrice,
                   indexPrice, predictedFundingRate, symbol
    """
    return {
        "symbol": record["symbol"],
        "timestamp": _ts(record["fundingTimestamp"]),
        "funding_rate": Decimal(str(record["fundingRate"])),
        "mark_price": _dec(record.get("markPrice")),
        "index_price": _dec(record.get("indexPrice")),
        "predicted_rate": _dec(record.get("predictedFundingRate")),
    }


def map_liquidation(record: dict) -> dict:
    """Tardis liquidation → liquidations row.

    Tardis fields: timestamp, symbol, side, amount, price
    """
    qty = Decimal(str(record["amount"]))
    price = Decimal(str(record["price"]))
    return {
        "symbol": record["symbol"],
        "timestamp": _ts(record["timestamp"]),
        "side": record["side"],
        "qty": qty,
        "price": price,
        "value_usd": qty * price,
    }


def map_trade(record: dict) -> dict:
    """Tardis trade → raw_trades row.

    Tardis fields: timestamp, symbol, id, price, amount, side
    """
    trade_id = record.get("id") or ""
    if not trade_id:
        import hashlib

        raw = f"{record['timestamp']}:{record['price']}:{record['amount']}:{record['side']}"
        trade_id = hashlib.sha256(raw.encode()).hexdigest()[:40]

    return {
        "symbol": record["symbol"],
        "timestamp": _ts(record["timestamp"]),
        "price": Decimal(str(record["price"])),
        "qty": Decimal(str(record["amount"])),
        "side": record["side"],
        "trade_id": trade_id,
    }
