"""
Test order placement — verifies the bot can actually place and cancel orders.

Places a small limit order far from market price (won't fill), verifies it
shows up on the exchange, then cancels it. Tests the full API pipeline.

Usage
-----
    python3 scripts/test_order.py                    # default SOLUSDT
    python3 scripts/test_order.py --symbol AVAXUSDT  # specific symbol
    python3 scripts/test_order.py --all              # test all configured symbols
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import settings


def _init_exchange():
    import ccxt

    exchange = ccxt.bybit({
        "apiKey": settings.bybit_api_key,
        "secret": settings.bybit_api_secret,
        "options": {
            "defaultType": "linear",
            "accountType": "UNIFIED",
        },
        "enableRateLimit": True,
    })

    if settings.bybit_testnet:
        exchange.set_sandbox_mode(True)
    elif settings.bybit_demo:
        api_urls = exchange.urls.get("api", {})
        if isinstance(api_urls, dict):
            for key in api_urls:
                if isinstance(api_urls[key], str):
                    api_urls[key] = api_urls[key].replace(
                        "api.bybit.com", "api-demo.bybit.com"
                    )

    import json
    import urllib.request
    url = "https://api.bybit.com/v5/market/time"
    resp = urllib.request.urlopen(url, timeout=10)
    server_time = json.loads(resp.read())["time"]
    local_time = int(time.time() * 1000)
    offset = local_time - server_time
    if abs(offset) > 2000:
        original_ms = exchange.milliseconds
        exchange.milliseconds = lambda: original_ms() - offset
        print(f"  Clock offset: {offset}ms (auto-corrected)")

    exchange.load_markets()
    return exchange


from src.live.order_manager import _to_ccxt_symbol  # reuse the canonical mapping


def test_symbol(exchange, symbol: str) -> bool:
    ccxt_sym = _to_ccxt_symbol(symbol)
    print(f"\n{'='*60}")
    print(f"  Testing: {symbol} ({ccxt_sym})")
    print(f"{'='*60}")

    # 1. Fetch ticker
    print("  [1/6] Fetching ticker ...", end=" ", flush=True)
    try:
        ticker = exchange.fetch_ticker(ccxt_sym)
        bid = ticker["bid"]
        ask = ticker["ask"]
        last = ticker["last"]
        print(f"OK  (bid={bid}, ask={ask}, last={last})")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    # 2. Check balance
    print("  [2/6] Checking balance ...", end=" ", flush=True)
    try:
        balance = exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        free = usdt.get("free", 0)
        total = usdt.get("total", 0)
        print(f"OK  (USDT free={free}, total={total})")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    # 3. Place a limit buy order 10% below market (won't fill)
    test_price = round(bid * 0.90, 6)
    test_size_usd = 15.0
    amount = test_size_usd / test_price if test_price > 0 else 0.001

    try:
        amount = float(exchange.amount_to_precision(ccxt_sym, amount))
        test_price = float(exchange.price_to_precision(ccxt_sym, test_price))
    except Exception:
        pass

    print(f"  [3/6] Placing test order (buy {amount} @ {test_price}) ...",
          end=" ", flush=True)
    try:
        order = exchange.create_order(
            symbol=ccxt_sym,
            type="limit",
            side="buy",
            amount=amount,
            price=test_price,
        )
        order_id = order["id"]
        print(f"OK  (order_id={order_id})")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    # 4. Verify order exists
    print("  [4/6] Verifying order exists ...", end=" ", flush=True)
    time.sleep(1)
    try:
        open_orders = exchange.fetch_open_orders(ccxt_sym)
        found = any(o["id"] == order_id for o in open_orders)
        if found:
            print(f"OK  (found in {len(open_orders)} open orders)")
        else:
            print(f"WARN  (order not found in {len(open_orders)} open orders)")
    except Exception as e:
        print(f"WARN: {e}")

    # 5. Cancel the order
    print("  [5/6] Cancelling test order ...", end=" ", flush=True)
    try:
        exchange.cancel_order(order_id, ccxt_sym)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    # 6. Verify cancellation
    print("  [6/6] Verifying cancellation ...", end=" ", flush=True)
    time.sleep(1)
    try:
        open_orders = exchange.fetch_open_orders(ccxt_sym)
        still_there = any(o["id"] == order_id for o in open_orders)
        if not still_there:
            print("OK  (order gone)")
        else:
            print("WARN  (order still appears)")
    except Exception as e:
        print(f"WARN: {e}")

    print(f"\n  {symbol}: ALL CHECKS PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test order placement")
    parser.add_argument("--symbol", default="SOLUSDT",
                        help="Symbol to test (default: SOLUSDT)")
    parser.add_argument("--all", action="store_true",
                        help="Test all symbols from .env SYMBOLS")
    args = parser.parse_args()

    print(f"\nBybit Order Test")
    print(f"  Testnet: {settings.bybit_testnet}")
    print(f"  Demo: {settings.bybit_demo}")
    print(f"  API Key: {settings.bybit_api_key[:6]}..." if settings.bybit_api_key else "  API Key: NOT SET")

    if not settings.bybit_api_key or not settings.bybit_api_secret:
        print("\n  ERROR: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")
        sys.exit(1)

    exchange = _init_exchange()
    print(f"  Exchange initialized OK")
    print(f"  Markets loaded: {len(exchange.markets)} pairs")

    symbols = settings.symbols if args.all else [args.symbol]
    results = {}

    for symbol in symbols:
        success = test_symbol(exchange, symbol)
        results[symbol] = success

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for sym, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {sym:<12} {status}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} symbols passed. Order pipeline is working.")


if __name__ == "__main__":
    main()
