#!/usr/bin/env python3
"""Quick debug script to check Kalshi API responses."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from trading.kalshi import KalshiAPI

def main():
    print("=== Kalshi API Debug ===\n")

    # Check credentials
    api_key = os.getenv('KALSHI_API_KEY_ID')
    private_key = os.getenv('KALSHI_PRIVATE_KEY')
    key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')

    print(f"API Key ID: {'Set' if api_key else 'NOT SET'}")
    print(f"Private Key (env): {'Set' if private_key else 'NOT SET'}")
    print(f"Private Key Path: {key_path or 'NOT SET'}")

    # Try loading from path if not in env
    if not private_key and key_path:
        key_file = Path(key_path)
        if not key_file.is_absolute():
            key_file = Path(__file__).parent / key_path
        if key_file.exists():
            private_key = key_file.read_text()
            os.environ['KALSHI_PRIVATE_KEY'] = private_key
            print(f"  -> Loaded from {key_file}")

    if not api_key or not private_key:
        print("\nERROR: Missing credentials")
        return

    print("\n--- Initializing API ---")
    try:
        api = KalshiAPI({'market_id': 'KXTSAW'})
        print("API initialized successfully")
    except Exception as e:
        print(f"ERROR initializing API: {e}")
        return

    print("\n--- Testing Balance ---")
    try:
        balance = api.get_balance()
        print(f"Raw response: {balance}")
        print(f"Response keys: {list(balance.keys()) if isinstance(balance, dict) else 'N/A'}")
        # Try to extract balance value
        if isinstance(balance, dict):
            for key in ['balance', 'available_balance', 'portfolio_value', 'cash_balance']:
                if key in balance:
                    val = balance[key]
                    print(f"  {key}: {val} (as dollars: ${val/100:.2f} if in cents)")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n--- Testing Events by Series ---")
    try:
        events = api.get_events_by_series('KXTSAW')
        print(f"Found {len(events)} total events")

        # Filter to KXTSAW-* only
        kxtsaw_events = [e for e in events if e.get('event_ticker', '').startswith('KXTSAW-')]
        print(f"Found {len(kxtsaw_events)} KXTSAW-* events")

        if kxtsaw_events:
            # Show most recent 5
            print("Most recent KXTSAW events:")
            for e in kxtsaw_events[:5]:
                print(f"  - {e.get('event_ticker')}: {e.get('strike_period')}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n--- Testing Current TSA Event ---")
    try:
        # Enable logging to see debug output
        import logging
        logging.basicConfig(level=logging.INFO)

        event = api.get_current_tsa_event()
        print(f"Current event: {event}")

        if event:
            # Show markets for this event
            print(f"\nMarkets for {event}:")
            result = api._request('GET', f'/markets?event_ticker={event}')
            markets = result.get('markets', [])
            for m in markets[:5]:
                print(f"  - {m.get('ticker')}: status={m.get('status')}, last_price={m.get('last_price')}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n--- Testing Market Data ---")
    try:
        market = api.get_market_data()
        print(f"Raw response: {market}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
