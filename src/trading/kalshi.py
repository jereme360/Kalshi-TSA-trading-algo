"""
Kalshi API integration for TSA prediction trading.
Uses RSA-PSS signature authentication per Kalshi's current API spec.
"""
import requests
import time
from typing import Dict, List, Optional
from base64 import b64encode
import logging
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

load_dotenv()
logger = logging.getLogger(__name__)


class KalshiAPI:
    """Interface for Kalshi exchange API with RSA-PSS authentication."""

    def __init__(self, config: Dict):
        self.api_key_id = os.getenv('KALSHI_API_KEY_ID')
        self.private_key_pem = os.getenv('KALSHI_PRIVATE_KEY')
        self.base_url = config.get('base_url', 'https://api.elections.kalshi.com/trade-api/v2')
        self.market_id = config.get('market_id', 'KXTSAW')

        if not self.api_key_id or not self.private_key_pem:
            raise ValueError("Missing Kalshi API credentials")

        # Load private key
        self.private_key = serialization.load_pem_private_key(
            self.private_key_pem.encode(),
            password=None
        )

        self.session = requests.Session()
        self.last_request_time = 0
        logger.info("Kalshi API client initialized")

    def _sign(self, message: str) -> str:
        """Sign message using RSA-PSS with SHA256."""
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size
            ),
            hashes.SHA256()
        )
        return b64encode(signature).decode()

    def _request_headers(self, method: str, path: str) -> Dict[str, str]:
        """Generate authentication headers for request."""
        timestamp = str(int(time.time() * 1000))
        # Remove query params for signature, but include full API path
        path_for_sig = path.split('?')[0]
        # Kalshi requires full path including /trade-api/v2 prefix
        full_path = '/trade-api/v2' + path_for_sig
        message = timestamp + method.upper() + full_path
        signature = self._sign(message)

        return {
            'KALSHI-ACCESS-KEY': self.api_key_id,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        }

    def _rate_limit(self):
        """Ensure minimum 100ms between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self.last_request_time = time.time()

    def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated API request."""
        self._rate_limit()
        url = f"{self.base_url}{path}"
        headers = self._request_headers(method, path)

        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, json=data)
            elif method == 'DELETE':
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json() if response.text else {}

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def get_balance(self) -> Dict:
        """Get account balance."""
        return self._request('GET', '/portfolio/balance')

    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        result = self._request('GET', '/portfolio/positions')
        return result.get('market_positions', [])

    def get_events_by_series(self, series_ticker: str) -> List[Dict]:
        """Get events for a series (e.g., KXTSAW for TSA markets)."""
        result = self._request('GET', f'/events?series_ticker={series_ticker}')
        return result.get('events', [])

    def get_current_tsa_event(self) -> Optional[str]:
        """Find the current/upcoming TSA weekly event ticker."""
        events = self.get_events_by_series('KXTSAW')
        if not events:
            return None

        # Filter to only KXTSAW-* events (not old TSAW-* format)
        kxtsaw_events = [e for e in events if e.get('event_ticker', '').startswith('KXTSAW-')]
        if not kxtsaw_events:
            logger.warning("No KXTSAW events found")
            return None

        # Parse date from event ticker (format: KXTSAW-26JAN18 -> Jan 18, 2026)
        def parse_event_date(ticker: str) -> Optional[str]:
            """Extract sortable date string from ticker like KXTSAW-26JAN18."""
            try:
                # Extract date part: 26JAN18 -> year=26, month=JAN, day=18
                date_part = ticker.split('-')[1]  # "26JAN18"
                year = int(date_part[:2])  # 26
                month_str = date_part[2:5]  # "JAN"
                day = int(date_part[5:])  # 18
                # Convert to sortable format: 2026-01-18
                months = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                          'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                          'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
                return f"20{year:02d}-{months.get(month_str, '01')}-{day:02d}"
            except Exception:
                return None

        # Sort by parsed date descending (most recent first)
        kxtsaw_events.sort(key=lambda e: parse_event_date(e.get('event_ticker', '')) or '', reverse=True)

        # Check the most recent events for open markets
        for event in kxtsaw_events[:5]:  # Check top 5 most recent
            event_ticker = event.get('event_ticker')
            if not event_ticker:
                continue

            # Get markets for this event to check if any are open
            try:
                result = self._request('GET', f'/markets?event_ticker={event_ticker}')
                markets = result.get('markets', [])
                # Look for open/active markets
                open_markets = [m for m in markets if m.get('status') in ('open', 'active')]
                if open_markets:
                    logger.info(f"Found open market in event {event_ticker}")
                    return event_ticker
                else:
                    statuses = [m.get('status') for m in markets]
                    logger.debug(f"Event {event_ticker} market statuses: {statuses}")
            except Exception as e:
                logger.warning(f"Error checking markets for {event_ticker}: {e}")
                continue

        # Fallback: return most recent event even if no open markets found
        logger.warning("No open markets found, returning most recent KXTSAW event")
        return kxtsaw_events[0].get('event_ticker') if kxtsaw_events else None

    def get_market_data(self, market_id: Optional[str] = None) -> Dict:
        """Get market data. If market_id is a series ticker, finds current event."""
        market_id = market_id or self.market_id

        # If this looks like a series ticker (uppercase, no date suffix), find current event
        if market_id.upper() == 'KXTSAW' and '-' not in market_id:
            event_ticker = self.get_current_tsa_event()
            if event_ticker:
                # Get markets for this event
                result = self._request('GET', f'/markets?event_ticker={event_ticker}')
                markets = result.get('markets', [])
                if markets:
                    return {'market': markets[0], 'event_ticker': event_ticker}
            return {'error': 'No active TSA markets found', 'event_ticker': None}

        return self._request('GET', f'/markets/{market_id}')

    def get_order_book(self, market_id: Optional[str] = None) -> Dict:
        """Get order book for market."""
        market_id = market_id or self.market_id
        return self._request('GET', f'/markets/{market_id}/orderbook')

    def place_order(self, side: str, size: int, price: float,
                    market_id: Optional[str] = None) -> Dict:
        """Place a limit order."""
        market_id = market_id or self.market_id
        order_data = {
            'ticker': market_id,
            'action': 'buy' if side.lower() in ['buy', 'yes'] else 'sell',
            'side': 'yes',  # Kalshi uses yes/no for contract side
            'count': size,
            'type': 'limit',
            'yes_price': int(price * 100)  # Convert to cents
        }
        return self._request('POST', '/portfolio/orders', order_data)

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        return self._request('DELETE', f'/portfolio/orders/{order_id}')


if __name__ == "__main__":
    config = {'market_id': 'KXTSAW'}
    try:
        api = KalshiAPI(config)
        balance = api.get_balance()
        print("Balance:", balance)

        # Test TSA event lookup
        event = api.get_current_tsa_event()
        print(f"Current TSA event: {event}")

        market_data = api.get_market_data()
        print(f"Market data: {market_data}")
    except Exception as e:
        print(f"Error: {e}")
