"""
Backtesting engine for TSA prediction strategies.
Handles time-series cross validation and performance analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
from src.models.ensemble import EnsembleModel
from src.trading.strategies import TSAVolumeStrategy
from src.utils.metrics import calculate_prediction_metrics
import json

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting engine for TSA prediction strategies.
    Supports walk-forward optimization and multiple market environments.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtesting engine.
        
        Args:
            config: Configuration dictionary containing:
                - initial_capital: Starting capital
                - transaction_cost: Cost per trade
                - slippage: Slippage assumption
                - risk_free_rate: Annual risk-free rate
                - test_start: Start of testing period
                - test_end: End of testing period
                - walk_forward_window: Days for walk-forward testing
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 10000)
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 10 bps
        self.slippage = config.get('slippage', 0.001)  # 10 bps
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        
        # Test period settings
        self.test_start = pd.Timestamp(config['test_start'])
        self.test_end = pd.Timestamp(config['test_end'])
        self.walk_forward_window = config.get('walk_forward_window', 30)  # days
        
        # Performance tracking
        self.positions = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.performance_metrics = {}
        
    def simulate_market_price(self, 
                            true_probability: float,
                            noise_level: float = 0.05) -> float:
        """
        Simulate Kalshi market price with noise.
        
        Args:
            true_probability: True probability of event
            noise_level: Level of market noise
            
        Returns:
            float: Simulated market price
        """
        noise = np.random.normal(0, noise_level)
        price = true_probability + noise
        return np.clip(price, 0.01, 0.99)  # Keep within valid range
        
    def calculate_transaction_costs(self, price: float, size: int) -> float:
        """
        Calculate total transaction costs including slippage.
        
        Args:
            price: Trade price
            size: Position size
            
        Returns:
            float: Total transaction costs
        """
        base_cost = abs(size) * price * self.transaction_cost
        slippage_cost = abs(size) * price * self.slippage
        return base_cost + slippage_cost
        
    def execute_trade(self, 
                     current_position: int,
                     target_position: int,
                     price: float,
                     timestamp: pd.Timestamp) -> Dict:
        """
        Execute trade and calculate PnL.
        
        Args:
            current_position: Current position size
            target_position: Target position size
            price: Current price
            timestamp: Trade timestamp
            
        Returns:
            Dict with trade details
        """
        size = target_position - current_position
        if size == 0:
            return None
            
        costs = self.calculate_transaction_costs(price, size)
        
        trade = {
            'timestamp': timestamp,
            'size': size,
            'price': price,
            'costs': costs,
            'value': abs(size) * price + costs
        }
        
        return trade
        
    def run_backtest(self,
                    model: EnsembleModel,
                    strategy: TSAVolumeStrategy,
                    data: pd.DataFrame,
                    actual_outcomes: pd.Series) -> Dict:
        """
        Run backtest using walk-forward optimization.
        
        Args:
            model: Prediction model
            strategy: Trading strategy
            data: Feature DataFrame
            actual_outcomes: Series of actual outcomes
            
        Returns:
            Dict containing backtest results
        """
        current_time = self.test_start
        positions = []
        trades = []
        equity_curve = []
        current_position = 0
        capital = self.initial_capital
        
        while current_time <= self.test_end:
            # Get training data up to current time
            train_data = data[data.index < current_time]
            
            if len(train_data) < 30:  # Minimum required history
                current_time += timedelta(days=1)
                continue
                
            # Retrain model on available data
            train_y = actual_outcomes[actual_outcomes.index < current_time]
            model.train(train_data, train_y)
            
            # Get current features
            current_features = data.loc[current_time:current_time]
            if len(current_features) == 0:
                current_time += timedelta(days=1)
                continue
                
            # Get model prediction and uncertainty
            prediction, uncertainty = model.predict_with_uncertainty(current_features)
            confidence = model.get_prediction_confidence(current_features)
            
            # Simulate market price
            market_price = self.simulate_market_price(
                prediction[0],
                noise_level=0.05
            )
            
            # Prepare market data for strategy
            market_data = pd.Series({
                'price': market_price,
                'volume': 1000,  # Simulated volume
                'days_to_expiry': 7,  # Weekly contracts
                'timestamp': current_time
            })
            
            # Get strategy signal
            position_data = {
                'size': current_position,
                'entry_price': market_price,
                'position_pnl': 0,  # To be calculated
            }
            
            signal = strategy.generate_signal(
                current_features,
                market_data,
                position_data
            )
            
            # Execute trade if needed
            if signal['action'] in ['buy', 'sell']:
                size = signal['size'] if signal['action'] == 'buy' else -signal['size']
                trade = self.execute_trade(
                    current_position,
                    size,
                    market_price,
                    current_time
                )
                
                if trade:
                    trades.append(trade)
                    current_position = size
                    capital -= trade['value']
            
            # Record position
            position = {
                'timestamp': current_time,
                'position': current_position,
                'price': market_price,
                'prediction': prediction[0],
                'uncertainty': uncertainty[0],
                'confidence': confidence.iloc[0],
                'capital': capital
            }
            positions.append(position)
            
            # Update equity curve
            equity = capital + current_position * market_price
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity
            })
            
            current_time += timedelta(days=1)
        
        # Convert to DataFrames
        self.positions = pd.DataFrame(positions).set_index('timestamp')
        self.trades = pd.DataFrame(trades).set_index('timestamp')
        equity_curve = pd.DataFrame(equity_curve).set_index('timestamp')
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(equity_curve)
        
        return {
            'positions': self.positions,
            'trades': self.trades,
            'equity_curve': equity_curve,
            'metrics': self.performance_metrics
        }
        
    def _calculate_performance_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = equity_curve['equity'].pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility
        
        # Drawdown analysis
        rolling_max = equity_curve['equity'].expanding().max()
        drawdowns = equity_curve['equity'] / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Trading metrics
        win_rate = (self.trades['value'] > 0).mean() if len(self.trades) > 0 else 0
        avg_trade = self.trades['value'].mean() if len(self.trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'n_trades': len(self.trades),
            'final_equity': equity_curve['equity'].iloc[-1]
        }
        
    def plot_results(self, save_path: Optional[Path] = None):
        """Plot backtest results."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Equity curve
        equity_curve = self.positions['capital'] + self.positions['position'] * self.positions['price']
        equity_curve.plot(ax=ax1, title='Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        
        # Position sizes
        self.positions['position'].plot(ax=ax2, title='Position Size')
        ax2.set_ylabel('Contracts')
        
        # Model confidence
        self.positions['confidence'].plot(ax=ax3, title='Model Confidence')
        ax3.set_ylabel('Confidence Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def save_results(self, path: Path):
        """Save backtest results."""
        results = {
            'positions': self.positions.to_dict(),
            'trades': self.trades.to_dict(),
            'metrics': self.performance_metrics,
            'config': self.config
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Example usage
    from src.models.ensemble import EnsembleModel
    from src.trading.strategies import TSAVolumeStrategy
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Features
    data = pd.DataFrame({
        'weather_severity': np.random.normal(0, 1, n_samples),
        'airline_prices': np.random.normal(100, 10, n_samples),
        'is_holiday': np.random.binomial(1, 0.1, n_samples),
        'day_of_week': dates.dayofweek,
        'month': dates.month
    }, index=dates)
    
    # Actual outcomes (binary for Kalshi contracts)
    actual_outcomes = pd.Series(
        np.random.binomial(1, 0.6, n_samples),
        index=dates
    )
    
    # Initialize components
    config = {
        'initial_capital': 10000,
        'transaction_cost': 0.001,
        'test_start': '2023-06-01',
        'test_end': '2023-12-31'
    }
    
    engine = BacktestEngine(config)
    
    # Run backtest
    model = EnsembleModel("ensemble", {})  # Configure as needed
    strategy = TSAVolumeStrategy({})  # Configure as needed
    
    results = engine.run_backtest(model, strategy, data, actual_outcomes)
    
    # Print results
    print("\nBacktest Results:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")