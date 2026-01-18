"""
Risk management system for TSA prediction trading.
Handles position sizing, exposure limits, and risk metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_size: int = 1000  # Maximum position size in contracts
    max_daily_loss: float = 0.15  # Maximum daily loss as fraction of capital
    max_trade_loss: float = 0.05  # Maximum loss per trade as fraction of capital
    max_leverage: float = 1.0    # Maximum allowed leverage
    position_limit: int = 1000   # Maximum total position across all contracts
    concentration_limit: float = 0.25  # Maximum exposure to single contract
    min_prob_edge: float = 0.05  # Minimum probability edge required to trade

class RiskManager:
    """
    Risk management system for TSA trading.
    Manages position sizing and risk limits.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration dictionary
        """
        self.limits = RiskLimits(**config.get('risk_limits', {}))
        self.initial_capital = config.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        
        # Performance tracking
        self.daily_pnl = pd.Series()
        self.positions = pd.DataFrame()
        self.trades = pd.DataFrame()
        
        # Risk metrics
        self.var_window = config.get('var_window', 20)  # Days for VaR calculation
        self.var_confidence = config.get('var_confidence', 0.95)
        
    def calculate_position_size(self,
                              predicted_prob: float,
                              market_price: float,
                              model_confidence: float,
                              current_positions: Dict) -> int:
        """
        Calculate appropriate position size based on edge and risk limits.
        
        Args:
            predicted_prob: Model's predicted probability
            market_price: Current market price
            model_confidence: Model's confidence score (0-1)
            current_positions: Current portfolio positions
            
        Returns:
            int: Recommended position size (positive for long, negative for short)
        """
        try:
            # Calculate edge
            edge = predicted_prob - market_price
            
            # Check minimum edge requirement
            if abs(edge) < self.limits.min_prob_edge:
                return 0
            
            # Base size on edge and confidence
            raw_size = (abs(edge) / self.limits.min_prob_edge) * \
                      self.limits.max_position_size * \
                      model_confidence
            
            # Apply position limits
            total_exposure = sum(abs(pos) for pos in current_positions.values())
            remaining_capacity = self.limits.position_limit - total_exposure
            
            max_new_position = min(
                remaining_capacity,
                self.limits.max_position_size,
                self.current_capital * self.limits.max_leverage / market_price
            )
            
            # Scale position size
            position_size = min(int(raw_size), int(max_new_position))
            
            # Direction based on edge
            return position_size if edge > 0 else -position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def check_risk_limits(self,
                         position_size: int,
                         price: float,
                         current_positions: Dict) -> Tuple[bool, str]:
        """
        Check if trade complies with risk limits.
        
        Args:
            position_size: Proposed position size
            price: Current price
            current_positions: Current portfolio positions
            
        Returns:
            Tuple[bool, str]: (passes_checks, reason)
        """
        # Check daily loss limit
        last_pnl = self.daily_pnl.iloc[-1] if len(self.daily_pnl) > 0 else 0
        if last_pnl < -self.limits.max_daily_loss * self.initial_capital:
            return False, "Daily loss limit reached"
            
        # Check leverage
        new_exposure = sum(abs(pos) for pos in current_positions.values()) + abs(position_size)
        if new_exposure * price > self.current_capital * self.limits.max_leverage:
            return False, "Leverage limit exceeded"
            
        # Check concentration
        if abs(position_size) * price > self.current_capital * self.limits.concentration_limit:
            return False, "Concentration limit exceeded"
            
        return True, "Passes risk checks"
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Returns:
            Dict containing risk metrics
        """
        # Calculate returns
        returns = self.daily_pnl / self.current_capital
        
        # Value at Risk
        var = np.percentile(returns.tail(self.var_window), 
                          (1 - self.var_confidence) * 100)
        
        # Expected Shortfall
        es = returns[returns < var].mean()
        
        # Volatility
        vol = returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'value_at_risk': var,
            'expected_shortfall': es,
            'volatility': vol,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252)
        }
    
    def update_state(self,
                    pnl: float,
                    timestamp: datetime,
                    positions: Dict) -> None:
        """
        Update risk manager state with new PnL and positions.
        
        Args:
            pnl: Period PnL
            timestamp: Update timestamp
            positions: Current positions
        """
        # Update capital
        self.current_capital += pnl
        
        # Update daily PnL
        self.daily_pnl[timestamp] = pnl
        
        # Update positions
        self.positions.loc[timestamp] = pd.Series(positions)
        
    def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report.
        
        Returns:
            Dict containing risk report
        """
        risk_metrics = self.calculate_risk_metrics()
        
        # Position statistics
        position_stats = {
            'total_exposure': self.positions.abs().sum().sum(),
            'largest_position': self.positions.abs().max().max(),
            'net_exposure': self.positions.sum().sum()
        }
        
        # Performance metrics
        perf_metrics = {
            'total_pnl': self.daily_pnl.sum(),
            'daily_avg_pnl': self.daily_pnl.mean(),
            'pnl_volatility': self.daily_pnl.std(),
            'win_rate': (self.daily_pnl > 0).mean()
        }
        
        return {
            'risk_metrics': risk_metrics,
            'position_stats': position_stats,
            'performance_metrics': perf_metrics,
            'current_capital': self.current_capital,
            'capital_usage': position_stats['total_exposure'] / self.current_capital
        }
    
    def adjust_for_market_conditions(self,
                                   base_size: int,
                                   market_conditions: Dict) -> int:
        """
        Adjust position size based on market conditions.
        
        Args:
            base_size: Base position size
            market_conditions: Dict of market condition indicators
            
        Returns:
            int: Adjusted position size
        """
        # Volatility adjustment
        vol_scalar = 1.0
        if 'volatility' in market_conditions:
            vol_scalar = 1.0 / (1.0 + market_conditions['volatility'])
            
        # Liquidity adjustment
        liq_scalar = 1.0
        if 'liquidity' in market_conditions:
            liq_scalar = min(1.0, market_conditions['liquidity'])
            
        # Market stress adjustment
        stress_scalar = 1.0
        if 'market_stress' in market_conditions:
            stress_scalar = 1.0 - market_conditions['market_stress'] * 0.5
            
        # Calculate final size
        adjusted_size = int(base_size * vol_scalar * liq_scalar * stress_scalar)
        
        return adjusted_size

if __name__ == "__main__":
    # Example usage
    config = {
        'risk_limits': {
            'max_position_size': 1000,
            'max_daily_loss': 0.15,
            'max_trade_loss': 0.05,
            'max_leverage': 1.0,
            'position_limit': 1000,
            'concentration_limit': 0.25,
            'min_prob_edge': 0.05
        },
        'initial_capital': 10000,
        'var_window': 20,
        'var_confidence': 0.95
    }
    
    risk_manager = RiskManager(config)
    
    # Example position sizing
    predicted_prob = 0.65
    market_price = 0.60
    model_confidence = 0.8
    current_positions = {'contract1': 50, 'contract2': -30}
    
    size = risk_manager.calculate_position_size(
        predicted_prob,
        market_price,
        model_confidence,
        current_positions
    )
    
    print(f"\nRecommended position size: {size}")
    
    # Check risk limits
    passes_checks, reason = risk_manager.check_risk_limits(
        size,
        market_price,
        current_positions
    )
    
    print(f"Passes risk checks: {passes_checks}")
    print(f"Reason: {reason}")
    
    # Example market condition adjustment
    market_conditions = {
        'volatility': 0.2,
        'liquidity': 0.8,
        'market_stress': 0.1
    }
    
    adjusted_size = risk_manager.adjust_for_market_conditions(
        size,
        market_conditions
    )
    
    print(f"Adjusted size: {adjusted_size}")
    
    # Generate risk report
    risk_manager.update_state(100, datetime.now(), current_positions)
    report = risk_manager.generate_risk_report()
    
    print("\nRisk Report:")
    print(json.dumps(report, indent=2))