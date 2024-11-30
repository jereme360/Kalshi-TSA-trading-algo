"""
Kalshi API integration for TSA prediction trading.
Handles market data retrieval and order execution.
"""
import requests
import json
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from base64 import b64decode
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class KalshiAPI:
    """Interface for Kalshi exchange API."""
    
    def __init__(self, config: Dict):
        """
        Initialize Kalshi API client.
        
        Args:
            config: Configuration dictionary containing:
                - api_key_id: Kalshi API key ID
                - private_key_path: Path to private key file
                - base_url: API base URL (default: production)
                - market_id: TSA weekly check-in market ID
        """
        self.api_key_id = os.getenv('KALSHI_API_KEY_ID')
        self.private_key = os.getenv('KALSHI_PRIVATE_KEY')
        self.base_url = config.get('base_url', 'https://trading-api.kalshi.com/v1')
        self.market_id = config.get('market_id', 'kxtsaw')  # TSA weekly check-ins
        
        # Load and validate credentials
        if not self.api_key_id or not self.private_key:
            raise ValueError("Missing Kalshi API credentials")
        
        # Initialize session
        self.session = requests.Session()
        self.token = None
        self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Kalshi API using JWT."""
        try:
            # Create JWT token
            payload = {
                'sub': self.api_key_id,
                'iat': int(time.time()),
                'exp': int(time.time()) + 3600  # 1 hour expiry
            }
            
            # Load private key
            private_key = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None
            )
            
            # Sign token
            self.token = jwt.encode(
                payload,
                private_key,
                algorithm='RS256'
            )
            
            # Update session headers
            self.session.headers.update({
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json'
            })
            
            logger.info("Successfully authenticated with Kalshi API")
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
            
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict] = None) -> Dict:
        """
        Make API request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Dict: API response
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/{endpoint}"
                
                if method == 'GET':
                    response = self.session.get(url)
                elif method == 'POST':
                    response = self.session.post(url, json=data)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"API request failed after {max_retries} attempts: {str(e)}")
                    raise
                    
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                
                # Re-authenticate if token expired
                if response.status_code == 401:
                    self._authenticate()
                    
    def get_market_data(self, market_id: Optional[str] = None) -> Dict:
        """
        Get market data for TSA check-in contract.
        
        Args:
            market_id: Optional specific market ID
            
        Returns:
            Dict containing market data
        """
        market_id = market_id or self.market_id
        return self._make_request('GET', f'markets/{market_id}')
        
    def get_order_book(self, market_id: Optional[str] = None) -> Dict:
        """
        Get order book for market.
        
        Args:
            market_id: Optional specific market ID
            
        Returns:
            Dict containing order book data
        """
        market_id = market_id or self.market_id
        return self._make_request('GET', f'markets/{market_id}/orderbook')
        
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        return self._make_request('GET', 'portfolio/positions')
        
    def place_order(self,
                   side: str,
                   size: int,
                   price: float,
                   market_id: Optional[str] = None) -> Dict:
        """
        Place new order.
        
        Args:
            side: 'buy' or 'sell'
            size: Number of contracts
            price: Limit price
            market_id: Optional specific market ID
            
        Returns:
            Dict containing order details
        """
        market_id = market_id or self.market_id
        
        order_data = {
            'market_id': market_id,
            'side': side.upper(),
            'size': size,
            'price': int(price * 100),  # Convert to cents
            'type': 'limit'
        }
        
        return self._make_request('POST', 'orders', order_data)
        
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict containing cancellation details
        """
        return self._make_request('POST', f'orders/{order_id}/cancel')
        
    def get_market_history(self, 
                          market_id: Optional[str] = None,
                          days: int = 30) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            market_id: Optional specific market ID
            days: Number of days of history
            
        Returns:
            pd.DataFrame: Historical market data
        """
        market_id = market_id or self.market_id
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        params = {
            'market_id': market_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        data = self._make_request('GET', 'markets/history', params)
        return pd.DataFrame(data['history'])
        
    def execute_trade(self,
                     side: str,
                     size: int,
                     max_price: float,
                     timeout: int = 60) -> Dict:
        """
        Execute trade with smart order routing.
        
        Args:
            side: 'buy' or 'sell'
            size: Number of contracts
            max_price: Maximum acceptable price
            timeout: Order timeout in seconds
            
        Returns:
            Dict containing execution details
        """
        try:
            # Get current order book
            book = self.get_order_book()
            
            # Check if we can get filled at desired price
            if side.lower() == 'buy':
                best_ask = float(book['asks'][0]['price']) / 100
                if best_ask > max_price:
                    raise ValueError(f"Best ask ({best_ask}) above max price ({max_price})")
            else:
                best_bid = float(book['bids'][0]['price']) / 100
                if best_bid < max_price:
                    raise ValueError(f"Best bid ({best_bid}) below min price ({max_price})")
            
            # Place order
            order = self.place_order(side, size, max_price)
            
            # Wait for fill
            start_time = time.time()
            while time.time() - start_time < timeout:
                order_status = self._make_request('GET', f"orders/{order['id']}")
                if order_status['status'] == 'filled':
                    return order_status
                    
                time.sleep(1)
            
            # Cancel if not filled
            self.cancel_order(order['id'])
            raise TimeoutError("Order not filled within timeout")
            
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            raise

class KalshiMarketMaker:
    """Market making functionality for Kalshi markets."""
    
    def __init__(self, api: KalshiAPI, config: Dict):
        """
        Initialize market maker.
        
        Args:
            api: KalshiAPI instance
            config: Market making configuration
        """
        self.api = api
        self.config = config
        self.spread = config.get('spread', 0.02)  # 2 cents default spread
        self.position_limit = config.get('position_limit', 100)
        self.order_size = config.get('order_size', 10)
        
    def update_quotes(self, fair_value: float):
        """
        Update quotes based on fair value.
        
        Args:
            fair_value: Estimated fair value of contract
        """
        try:
            # Get current positions
            positions = self.api.get_positions()
            current_position = sum(p['size'] for p in positions)
            
            # Adjust spread based on position
            position_factor = current_position / self.position_limit
            adjusted_spread = self.spread * (1 + abs(position_factor))
            
            # Calculate bid/ask
            bid_price = fair_value - adjusted_spread/2
            ask_price = fair_value + adjusted_spread/2
            
            # Adjust sizes based on position
            bid_size = self.order_size * (1 - position_factor)
            ask_size = self.order_size * (1 + position_factor)
            
            # Place orders
            self.api.place_order('buy', int(bid_size), bid_price)
            self.api.place_order('sell', int(ask_size), ask_price)
            
        except Exception as e:
            logger.error(f"Error updating quotes: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    config = {
        'base_url': 'https://trading-api.kalshi.com/v1',
        'market_id': 'kxtsaw'
    }
    
    # Initialize API
    api = KalshiAPI(config)
    
    # Get market data
    market_data = api.get_market_data()
    print("\nMarket Data:")
    print(json.dumps(market_data, indent=2))
    
    # Get order book
    order_book = api.get_order_book()
    print("\nOrder Book:")
    print(json.dumps(order_book, indent=2))
    
    # Get current positions
    positions = api.get_positions()
    print("\nCurrent Positions:")
    print(json.dumps(positions, indent=2))
    
    # Example market making
    mm_config = {
        'spread': 0.02,
        'position_limit': 100,
        'order_size': 10
    }
    
    market_maker = KalshiMarketMaker(api, mm_config)
    
    # Update quotes based on fair value
    fair_value = 0.65  # Example fair value
    market_maker.update_quotes(fair_value)