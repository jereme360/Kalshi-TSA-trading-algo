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
        self.market_id = config.get('market_id', 'kxtsaw')

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
        # Remove query params for signature
        path_for_sig = path.split('?')[0]
        message = timestamp + method.upper() + path_for_sig
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

    def get_market_data(self, market_id: Optional[str] = None) -> Dict:
        """Get market data."""
        market_id = market_id or self.market_id
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
    config = {'market_id': 'kxtsaw'}
    try:
        api = KalshiAPI(config)
        balance = api.get_balance()
        print("Balance:", balance)
    except Exception as e:
        print(f"Error: {e}")
