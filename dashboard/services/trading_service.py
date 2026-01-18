"""
Trading service for the dashboard.
Wraps Kalshi API with safety guards.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
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
MAX_CONTRACTS_PER_TRADE = 1000
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
                'market_id': os.getenv('KALSHI_MARKET_ID', 'KXTSAW')
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

            # Handle series lookup response format
            if 'error' in market_data:
                return True, f"Connected - {market_data.get('error', 'No active TSA markets')}"

            if 'market' in market_data:
                market = market_data['market']
                event_ticker = market_data.get('event_ticker', '')
                title = market.get('title', market.get('ticker', 'Unknown'))
                return True, f"Connected - Event: {event_ticker}, Market: {title}"

            # Direct market lookup response
            return True, f"Connected - Market: {market_data.get('title', market_data.get('ticker', 'Unknown'))}"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    def get_market_data(self, market_id: Optional[str] = None) -> Dict:
        """
        Get current market data.

        Returns:
            Dict with market info (empty dict with status='not_connected' if not connected)
        """
        if not self.connected:
            return {'status': 'not_connected', 'error': 'Not connected to Kalshi API'}

        try:
            result = self.api.get_market_data(market_id)

            # Handle series lookup response - extract market data
            if 'market' in result:
                market = result['market']
                market['event_ticker'] = result.get('event_ticker')

                # Use ticker as market_id if market_id not present
                if 'market_id' not in market and 'ticker' in market:
                    market['market_id'] = market['ticker']

                # Convert price fields from cents to dollars
                price_fields = ['yes_bid', 'yes_ask', 'no_bid', 'no_ask', 'last_price', 'previous_price']
                for field in price_fields:
                    if field in market and market[field] is not None:
                        market[field] = market[field] / 100

                return market
            elif 'error' in result:
                logger.warning(result.get('error'))
                return {'status': 'no_markets', 'error': result.get('error')}

            # Direct market response (no wrapper) - also convert prices
            if 'market_id' not in result and 'ticker' in result:
                result['market_id'] = result['ticker']

            price_fields = ['yes_bid', 'yes_ask', 'no_bid', 'no_ask', 'last_price', 'previous_price']
            for field in price_fields:
                if field in result and result[field] is not None:
                    result[field] = result[field] / 100

            return result
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_order_book(self, market_id: Optional[str] = None) -> Dict:
        """
        Get order book depth.

        Returns:
            Dict with bids and asks (empty if not connected)
        """
        if not self.connected:
            return {'bids': [], 'asks': [], 'status': 'not_connected'}

        try:
            return self.api.get_order_book(market_id)
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return {'bids': [], 'asks': [], 'status': 'error', 'error': str(e)}

    def get_positions(self) -> List[Dict]:
        """
        Get current positions.

        Returns:
            List of position dicts (empty if not connected)
        """
        if not self.connected:
            return []

        try:
            return self.api.get_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_account_balance(self) -> Dict:
        """
        Get account balance info.

        Returns:
            Dict with balance details (zeros if not connected)
        """
        if not self.connected:
            return {
                'balance': 0,
                'available': 0,
                'status': 'not_connected'
            }

        try:
            raw = self.api.get_balance()
            # Kalshi returns balance in cents, convert to dollars
            return {
                'balance': raw.get('balance', 0) / 100,
                'available': raw.get('balance', 0) / 100,  # Kalshi uses 'balance' for available
                'portfolio_value': raw.get('portfolio_value', 0) / 100,
                'status': 'connected'
            }
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {
                'balance': 0,
                'available': 0,
                'status': 'error',
                'error': str(e)
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
            DataFrame with trade history (empty if not connected)
        """
        if not self.connected:
            return pd.DataFrame()

        try:
            # TODO: Implement real trade history fetch from Kalshi API
            # For now return empty - no sample data
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return pd.DataFrame()

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

    def get_backtest_results(self, initial_capital: float = 1000, weeks: int = 52) -> Dict:
        """
        Run backtest simulation for weekly Monday trading.

        Args:
            initial_capital: Starting investment amount
            weeks: Number of weeks to backtest

        Returns:
            Dict with backtest results (total_return, trades, equity_curve)
        """
        try:
            from backtesting.weekly_backtest import WeeklyBacktestEngine
            from dashboard.services.data_service import load_tsa_data

            # Load TSA data for backtest
            tsa_data = load_tsa_data(days=1500)  # ~4 years

            if tsa_data.empty:
                return {'error': 'No TSA data available for backtest'}

            # Run backtest
            engine = WeeklyBacktestEngine(
                initial_capital=initial_capital,
                bet_size=min(initial_capital, 100)  # Bet up to $100 per week
            )

            results = engine.run(tsa_data, start_date='2022-01-03')

            # Format for UI
            return {
                'initial_capital': results.get('initial_capital', initial_capital),
                'final_equity': results.get('final_equity', initial_capital),
                'total_return': results.get('total_profit', 0),
                'total_return_pct': (results.get('total_profit', 0) / initial_capital * 100) if initial_capital > 0 else 0,
                'num_trades': results.get('num_weeks', 0),
                'win_rate': results.get('win_rate', 0),
                'avg_profit_per_trade': results.get('avg_profit', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'equity_curve': results.get('equity_curve', []),
                'trades': results.get('weekly_profits', [])
            }
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {'error': str(e)}

    def get_available_contracts(self) -> List[Dict]:
        """
        Get current TSA contracts with thresholds and prices.

        Returns:
            List of dicts with keys: ticker, threshold, yes_price, no_price
        """
        if not self.connected:
            return []

        try:
            event_ticker = self.api.get_current_tsa_event()
            if not event_ticker:
                logger.warning("No current TSA event found")
                return []

            result = self.api._request('GET', f'/markets?event_ticker={event_ticker}')
            markets = result.get('markets', [])

            contracts = []
            for market in markets:
                ticker = market.get('ticker', '')
                threshold = self._parse_threshold(ticker)

                if threshold > 0:
                    contracts.append({
                        'ticker': ticker,
                        'threshold': threshold,
                        'yes_price': market.get('yes_bid', 50) / 100,  # Convert cents to dollars
                        'no_price': market.get('no_bid', 50) / 100,
                        'yes_ask': market.get('yes_ask', 50) / 100,
                        'no_ask': market.get('no_ask', 50) / 100,
                        'volume': market.get('volume', 0),
                        'status': market.get('status', 'unknown')
                    })

            # Sort by threshold
            contracts.sort(key=lambda x: x['threshold'])
            return contracts

        except Exception as e:
            logger.error(f"Error getting available contracts: {e}")
            return []

    def _parse_threshold(self, ticker: str) -> int:
        """
        Parse threshold from Kalshi ticker.

        Examples:
            KXTSAW-26JAN18-A2.80 -> 2800000 (Above 2.80M daily)
            KXTSAW-26JAN18-B2.15 -> 2150000 (Below 2.15M daily)
            KXTSAW-26JAN18-T17500000 -> 17500000 (old format)
        """
        try:
            # New format: -A or -B followed by decimal millions (e.g., A2.80 = 2.80M)
            if '-A' in ticker:
                threshold_str = ticker.split('-A')[-1]
                # Convert decimal millions to integer (2.80 -> 2800000)
                threshold_millions = float(threshold_str)
                return int(threshold_millions * 1_000_000)
            elif '-B' in ticker and '.' in ticker.split('-B')[-1]:
                # New format with B (Below)
                threshold_str = ticker.split('-B')[-1]
                threshold_millions = float(threshold_str)
                return int(threshold_millions * 1_000_000)
            # Old format: -T or -B followed by full number
            elif '-T' in ticker:
                threshold_str = ticker.split('-T')[-1]
                threshold_str = ''.join(c for c in threshold_str if c.isdigit())
                return int(threshold_str) if threshold_str else 0
            elif '-B' in ticker:
                threshold_str = ticker.split('-B')[-1]
                threshold_str = ''.join(c for c in threshold_str if c.isdigit())
                return int(threshold_str) if threshold_str else 0
            else:
                return 0

        except (ValueError, IndexError):
            return 0


@st.cache_resource
def get_trading_service() -> TradingService:
    """Get cached trading service instance."""
    return TradingService()
