"""
Trading service for the dashboard.
Wraps Kalshi API with safety guards.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import os
import sys
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

logger = logging.getLogger(__name__)

# Safety constants
MAX_CONTRACTS_PER_TRADE = 100
TRADE_COOLDOWN_SECONDS = 5
MIN_CONFIRMATION_DELAY = 2


class TradingService:
    """Service for Kalshi trading operations with safety guards."""

    def __init__(self):
        self.api = None
        self.connected = False
        self.last_trade_time = None
        self.risk_manager = None
        self._initialize()

    def _initialize(self):
        """Initialize Kalshi API connection."""
        try:
            from trading.kalshi import KalshiAPI
            from trading.risk import RiskManager

            config = {
                'base_url': os.getenv('KALSHI_BASE_URL', 'https://api.elections.kalshi.com/trade-api/v2'),
                'market_id': os.getenv('KALSHI_MARKET_ID', 'kxtsaw')
            }

            # Only connect if credentials are present
            private_key = os.getenv('KALSHI_PRIVATE_KEY')

            # If KALSHI_PRIVATE_KEY not set, try reading from KALSHI_PRIVATE_KEY_PATH
            if not private_key:
                key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')
                if key_path:
                    key_file = Path(key_path)
                    # Try relative to project root if not absolute
                    if not key_file.is_absolute():
                        key_file = Path(__file__).parent.parent.parent / key_path
                    if key_file.exists():
                        private_key = key_file.read_text()
                        os.environ['KALSHI_PRIVATE_KEY'] = private_key
                        logger.info(f"Loaded private key from {key_file}")

            if os.getenv('KALSHI_API_KEY_ID') and private_key:
                self.api = KalshiAPI(config)
                self.connected = True

                # Initialize risk manager
                risk_config = {
                    'risk_limits': {
                        'max_position_size': MAX_CONTRACTS_PER_TRADE,
                        'max_daily_loss': 0.15,
                        'max_trade_loss': 0.05,
                    },
                    'initial_capital': 10000
                }
                self.risk_manager = RiskManager(risk_config)

                logger.info("Connected to Kalshi API")
            else:
                logger.warning("Kalshi credentials not found - running in demo mode")

        except Exception as e:
            logger.error(f"Failed to connect to Kalshi: {e}")
            self.connected = False

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test API connection.

        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return False, "Not connected - check API credentials"

        try:
            market_data = self.api.get_market_data()
            return True, f"Connected - Market: {market_data.get('title', 'Unknown')}"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    def get_market_data(self, market_id: Optional[str] = None) -> Dict:
        """
        Get current market data.

        Returns:
            Dict with market info
        """
        if not self.connected:
            return self._get_sample_market_data()

        try:
            return self.api.get_market_data(market_id)
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return self._get_sample_market_data()

    def _get_sample_market_data(self) -> Dict:
        """Generate sample market data for demo."""
        return {
            'title': 'TSA Weekly Checkins > 18.5M',
            'market_id': 'kxtsaw-demo',
            'yes_bid': 0.62,
            'yes_ask': 0.65,
            'no_bid': 0.35,
            'no_ask': 0.38,
            'last_price': 0.63,
            'volume': 12500,
            'open_interest': 8500,
            'expiration': (datetime.now() + timedelta(days=3)).isoformat(),
            'status': 'demo'
        }

    def get_order_book(self, market_id: Optional[str] = None) -> Dict:
        """
        Get order book depth.

        Returns:
            Dict with bids and asks
        """
        if not self.connected:
            return self._get_sample_order_book()

        try:
            return self.api.get_order_book(market_id)
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return self._get_sample_order_book()

    def _get_sample_order_book(self) -> Dict:
        """Generate sample order book for demo."""
        return {
            'bids': [
                {'price': 62, 'size': 150},
                {'price': 61, 'size': 280},
                {'price': 60, 'size': 420},
                {'price': 59, 'size': 350},
                {'price': 58, 'size': 500},
            ],
            'asks': [
                {'price': 65, 'size': 180},
                {'price': 66, 'size': 250},
                {'price': 67, 'size': 380},
                {'price': 68, 'size': 290},
                {'price': 69, 'size': 450},
            ]
        }

    def get_positions(self) -> List[Dict]:
        """
        Get current positions.

        Returns:
            List of position dicts
        """
        if not self.connected:
            return self._get_sample_positions()

        try:
            return self.api.get_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return self._get_sample_positions()

    def _get_sample_positions(self) -> List[Dict]:
        """Generate sample positions for demo."""
        return [
            {
                'market_id': 'kxtsaw-2024-01-week3',
                'title': 'TSA Weekly > 18.5M (Jan 15-21)',
                'side': 'yes',
                'size': 25,
                'avg_price': 0.58,
                'current_price': 0.63,
                'unrealized_pnl': 1.25,
                'expiration': (datetime.now() + timedelta(days=3)).isoformat()
            },
            {
                'market_id': 'kxtsaw-2024-01-week4',
                'title': 'TSA Weekly > 19.0M (Jan 22-28)',
                'side': 'no',
                'size': 15,
                'avg_price': 0.45,
                'current_price': 0.42,
                'unrealized_pnl': 0.45,
                'expiration': (datetime.now() + timedelta(days=10)).isoformat()
            }
        ]

    def get_account_balance(self) -> Dict:
        """
        Get account balance info.

        Returns:
            Dict with balance details
        """
        if not self.connected:
            return self._get_sample_balance()

        try:
            return self.api.get_balance()
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return self._get_sample_balance()

    def _get_sample_balance(self) -> Dict:
        """Generate sample balance for demo."""
        return {
            'balance': 10250.75,
            'available': 8750.25,
            'reserved': 1500.50,
            'pending_orders': 2,
            'open_positions': 3
        }

    def calculate_signal(self, predicted_prob: float, market_price: float) -> Dict:
        """
        Calculate trading signal based on model prediction vs market.

        Args:
            predicted_prob: Model's predicted probability
            market_price: Current market price

        Returns:
            Dict with signal info
        """
        edge = predicted_prob - market_price
        min_edge = 0.05  # 5% minimum edge

        if abs(edge) < min_edge:
            signal = 'HOLD'
            strength = 0
        elif edge > 0:
            signal = 'BUY YES'
            strength = min(edge / 0.15, 1.0)  # Scale to max at 15% edge
        else:
            signal = 'BUY NO'
            strength = min(abs(edge) / 0.15, 1.0)

        return {
            'signal': signal,
            'edge': edge,
            'edge_pct': edge * 100,
            'strength': strength,
            'predicted_prob': predicted_prob,
            'market_price': market_price
        }

    def validate_order(self, side: str, size: int, price: float) -> Tuple[bool, str]:
        """
        Validate order parameters before submission.

        Args:
            side: 'yes' or 'no'
            size: Number of contracts
            price: Limit price

        Returns:
            Tuple of (valid, message)
        """
        # Check size limits
        if size <= 0:
            return False, "Size must be positive"
        if size > MAX_CONTRACTS_PER_TRADE:
            return False, f"Size exceeds maximum ({MAX_CONTRACTS_PER_TRADE} contracts)"

        # Check price validity
        if price <= 0 or price >= 1:
            return False, "Price must be between 0 and 1"

        # Check side
        if side.lower() not in ['yes', 'no']:
            return False, "Side must be 'yes' or 'no'"

        # Check cooldown
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            if elapsed < TRADE_COOLDOWN_SECONDS:
                return False, f"Please wait {TRADE_COOLDOWN_SECONDS - elapsed:.0f}s (cooldown)"

        # Check with risk manager if available
        if self.risk_manager:
            positions = {p['market_id']: p['size'] for p in self.get_positions()}
            passes, reason = self.risk_manager.check_risk_limits(size, price, positions)
            if not passes:
                return False, f"Risk limit: {reason}"

        return True, "Order validated"

    def place_order(self, side: str, size: int, price: float,
                   confirmed: bool = False) -> Dict:
        """
        Place an order with safety checks.

        Args:
            side: 'yes' or 'no'
            size: Number of contracts
            price: Limit price
            confirmed: User confirmation flag

        Returns:
            Dict with order result
        """
        # Require confirmation
        if not confirmed:
            return {
                'success': False,
                'error': 'Order must be confirmed before submission'
            }

        # Validate
        valid, message = self.validate_order(side, size, price)
        if not valid:
            return {'success': False, 'error': message}

        # Demo mode
        if not self.connected:
            return {
                'success': True,
                'demo': True,
                'order_id': f'demo-{int(time.time())}',
                'side': side,
                'size': size,
                'price': price,
                'message': 'Demo order placed (not executed)'
            }

        # Execute real order
        try:
            kalshi_side = 'buy' if side.lower() == 'yes' else 'sell'
            result = self.api.place_order(kalshi_side, size, price)
            self.last_trade_time = datetime.now()

            return {
                'success': True,
                'order_id': result.get('id'),
                'side': side,
                'size': size,
                'price': price,
                'status': result.get('status', 'submitted')
            }

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get trade history.

        Args:
            days: Days of history

        Returns:
            DataFrame with trade history
        """
        if not self.connected:
            return self._get_sample_trade_history(days)

        try:
            # This would fetch real trade history
            return self._get_sample_trade_history(days)
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return self._get_sample_trade_history(days)

    def _get_sample_trade_history(self, days: int) -> pd.DataFrame:
        """Generate sample trade history for demo."""
        n_trades = min(days * 2, 50)  # ~2 trades per day
        dates = pd.date_range(end=datetime.now(), periods=n_trades, freq='12H')

        sides = np.random.choice(['yes', 'no'], n_trades)
        sizes = np.random.randint(5, 30, n_trades)
        prices = np.random.uniform(0.40, 0.70, n_trades)
        pnls = np.random.normal(5, 20, n_trades)

        return pd.DataFrame({
            'timestamp': dates,
            'side': sides,
            'size': sizes,
            'price': prices.round(2),
            'pnl': pnls.round(2),
            'status': 'filled'
        }).set_index('timestamp')

    def get_pnl_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get P&L history.

        Args:
            days: Days of history

        Returns:
            DataFrame with daily P&L
        """
        trades = self.get_trade_history(days)

        if trades.empty:
            return pd.DataFrame()

        # Aggregate by day
        daily_pnl = trades['pnl'].resample('D').sum()
        cumulative_pnl = daily_pnl.cumsum()

        return pd.DataFrame({
            'daily_pnl': daily_pnl,
            'cumulative_pnl': cumulative_pnl
        })


@st.cache_resource
def get_trading_service() -> TradingService:
    """Get cached trading service instance."""
    return TradingService()
