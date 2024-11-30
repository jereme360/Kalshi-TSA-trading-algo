"""
Trading strategies for Kalshi TSA weekly check-in contracts.
Integrates prediction models with trading decisions.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm
import logging
from src.models.ensemble import EnsembleModel
from src.models.causal import CausalGraph

logger = logging.getLogger(__name__)

class ModelIntegratedStrategy(ABC):
    """Base strategy that integrates prediction models with trading decisions."""
    
    def __init__(self, config: Dict):
        """
        Initialize strategy with models and configuration.
        
        Args:
            config: Strategy configuration including:
                - prediction_threshold: TSA volume threshold for contract
                - min_edge: Minimum required edge to trade
                - max_position: Maximum position size
                - confidence_threshold: Minimum model confidence to trade
                - max_model_deviation: Maximum allowed deviation between models
                - risk_limits: Dictionary of risk limits
        """
        self.config = config
        
        # Trading parameters
        self.prediction_threshold = config['prediction_threshold']
        self.min_edge = config.get('min_edge', 0.05)
        self.max_position = config.get('max_position', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.max_model_deviation = config.get('max_model_deviation', 0.15)
        
        # Risk management
        self.risk_limits = config.get('risk_limits', {
            'max_daily_loss': 0.15,  # 15% max daily loss
            'max_position_loss': 0.10,  # 10% max loss per position
            'max_weekly_positions': 5,  # Maximum new positions per week
            'min_time_to_expiry': 2,  # Minimum days before expiry
            'max_correlation': 0.7  # Maximum correlation between positions
        })
        
        # Model integration
        self.ensemble_model = None
        self.causal_graph = CausalGraph()  # For understanding feature relationships
        self.model_weights = {}  # Dynamic model weights based on performance
        self.feature_importance = {}  # Track important features
        
        # Performance tracking
        self.prediction_accuracy = pd.Series()  # Track prediction accuracy
        self.trading_performance = pd.Series()  # Track trading performance
        
    def setup_models(self, ensemble_model: EnsembleModel):
        """Connect prediction models to strategy."""
        self.ensemble_model = ensemble_model
        self.model_weights = ensemble_model.get_model_weights()
        self.feature_importance = ensemble_model.get_feature_importance()
    
    def convert_volume_to_probability(self, 
                                    predicted_volume: float,
                                    prediction_std: float,
                                    historical_distribution: Optional[pd.Series] = None) -> float:
        """
        Convert TSA volume prediction to probability of exceeding threshold.
        Uses both normal approximation and historical distribution.
        """
        if historical_distribution is not None:
            # Use empirical CDF if we have historical data
            historical_prob = (historical_distribution > self.prediction_threshold).mean()
            # Blend with normal approximation
            normal_prob = 1 - norm.cdf(
                (self.prediction_threshold - predicted_volume) / prediction_std
            )
            return 0.7 * normal_prob + 0.3 * historical_prob
        else:
            # Fallback to normal approximation
            return 1 - norm.cdf(
                (self.prediction_threshold - predicted_volume) / prediction_std
            )
    
    def calculate_model_confidence(self,
                                 current_data: pd.DataFrame,
                                 predictions: Dict[str, float]) -> float:
        """
        Calculate confidence in current prediction.
        
        Args:
            current_data: Current feature values
            predictions: Dictionary of predictions from each model
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Check model agreement
        pred_values = list(predictions.values())
        model_std = np.std(pred_values)
        if model_std > self.max_model_deviation:
            return 0.0
        
        # Check feature reliability
        important_features = self.feature_importance[
            self.feature_importance > self.feature_importance.quantile(0.8)
        ].index
        
        feature_confidence = 1.0
        for feature in important_features:
            if feature not in current_data:
                feature_confidence *= 0.8
                
        # Consider historical accuracy
        if len(self.prediction_accuracy) > 0:
            historical_accuracy = self.prediction_accuracy.tail(30).mean()
        else:
            historical_accuracy = 0.8  # Default assumption
            
        # Combine factors
        confidence = (
            (1.0 - model_std/self.max_model_deviation) * 0.4 +
            feature_confidence * 0.3 +
            historical_accuracy * 0.3
        )
        
        return min(1.0, max(0.0, confidence))
        
    def size_position(self,
                     edge: float,
                     confidence: float,
                     market_data: pd.Series) -> int:
        """
        Calculate position size based on edge, confidence, and market conditions.
        
        Args:
            edge: Calculated edge
            confidence: Model confidence
            market_data: Current market data
        """
        # Base size on edge and confidence
        base_size = (abs(edge) / self.min_edge) * self.max_position * confidence
        
        # Scale by time to expiry
        days_to_expiry = market_data['days_to_expiry']
        time_scalar = min(1.0, (days_to_expiry - 1) / 5)
        
        # Scale by market liquidity
        liquidity_scalar = min(1.0, market_data['volume'] / 1000)
        
        # Scale by price extremes
        price = market_data['price']
        if price < 0.1 or price > 0.9:
            price_scalar = 0.5
        else:
            price_scalar = 1.0
            
        final_size = int(base_size * time_scalar * liquidity_scalar * price_scalar)
        return min(final_size, self.max_position)
        
    def manage_risk(self,
                   current_signal: Dict,
                   market_data: pd.Series,
                   position_data: Dict) -> Dict:
        """
        Apply risk management rules to trading signal.
        """
        # Check if we're hitting any risk limits
        if position_data.get('daily_pnl_pct', 0) < -self.risk_limits['max_daily_loss']:
            return {
                'action': 'close',
                'size': 0,
                'reason': 'Daily loss limit hit'
            }
            
        if market_data['days_to_expiry'] < self.risk_limits['min_time_to_expiry']:
            return {
                'action': 'close',
                'size': 0,
                'reason': 'Too close to expiry'
            }
            
        if position_data.get('position_pnl_pct', 0) < -self.risk_limits['max_position_loss']:
            return {
                'action': 'close',
                'size': 0,
                'reason': 'Position loss limit hit'
            }
            
        # Adjust position size based on portfolio constraints
        weekly_positions = position_data.get('weekly_positions', 0)
        if weekly_positions >= self.risk_limits['max_weekly_positions']:
            current_signal['size'] = 0
            current_signal['reason'] = 'Weekly position limit reached'
            
        return current_signal

class TSAVolumeStrategy(ModelIntegratedStrategy):
    """
    Strategy specifically for TSA volume prediction and trading.
    Integrates multiple models and TSA-specific factors.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.seasonal_factors = self._initialize_seasonal_factors()
        
    def _initialize_seasonal_factors(self) -> Dict:
        """Initialize seasonal adjustment factors."""
        return {
            # Month-specific factors (holiday seasons, summer travel, etc.)
            'monthly_factors': {
                1: 0.9,  # January (post-holiday lull)
                7: 1.2,  # July (summer peak)
                11: 1.1, # November (Thanksgiving)
                12: 1.1  # December (Christmas)
            },
            # Day-of-week factors
            'dow_factors': {
                0: 1.0,  # Monday
                4: 1.2,  # Friday
                6: 0.8   # Sunday
            }
        }
        
    def generate_signal(self,
                       current_data: pd.DataFrame,
                       market_data: pd.Series,
                       position_data: Optional[Dict] = None) -> Dict:
        """
        Generate trading signal using integrated model predictions.
        """
        try:
            # Get predictions from all models
            predictions = {}
            prediction_std = None
            
            if self.ensemble_model:
                # Get ensemble prediction and uncertainty
                predicted_volume = self.ensemble_model.predict(current_data)
                prediction_std = self.ensemble_model.predict_uncertainty(current_data)
                
                # Get individual model predictions for confidence calculation
                for model_name, model in self.ensemble_model.models.items():
                    predictions[model_name] = model.predict(current_data)
            else:
                logger.error("Models not properly initialized")
                return {'action': 'hold', 'size': 0, 'reason': 'Models not ready'}
                
            # Calculate probability and confidence
            predicted_probability = self.convert_volume_to_probability(
                predicted_volume,
                prediction_std,
                position_data.get('historical_distribution')
            )
            
            model_confidence = self.calculate_model_confidence(
                current_data,
                predictions
            )
            
            # Calculate edge
            market_probability = market_data['price']
            edge = predicted_probability - market_probability
            
            # Apply seasonal adjustments
            current_month = pd.Timestamp.now().month
            current_dow = pd.Timestamp.now().dayofweek
            
            edge *= self.seasonal_factors['monthly_factors'].get(current_month, 1.0)
            edge *= self.seasonal_factors['dow_factors'].get(current_dow, 1.0)
            
            # Generate base signal
            if abs(edge) > self.min_edge and model_confidence > self.confidence_threshold:
                size = self.size_position(edge, model_confidence, market_data)
                signal = {
                    'action': 'buy' if edge > 0 else 'sell',
                    'size': size,
                    'reason': f'Edge: {edge:.3f}, Confidence: {model_confidence:.2f}',
                    'edge': edge,
                    'confidence': model_confidence,
                    'predicted_probability': predicted_probability
                }
            else:
                signal = {
                    'action': 'hold',
                    'size': 0,
                    'reason': 'Insufficient edge or confidence',
                    'edge': edge,
                    'confidence': model_confidence
                }
                
            # Apply risk management
            final_signal = self.manage_risk(signal, market_data, position_data or {})
            
            # Update performance tracking
            self._update_performance_metrics(predicted_volume, 
                                          market_data, 
                                          final_signal)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return {'action': 'hold', 'size': 0, 'reason': f'Error: {str(e)}'}
            
    def _update_performance_metrics(self,
                                  predicted_volume: float,
                                  market_data: pd.Series,
                                  signal: Dict):
        """Update strategy performance metrics."""
        try:
            # Update prediction accuracy if we have actual values
            if 'actual_volume' in market_data:
                accuracy = 1 - abs(predicted_volume - market_data['actual_volume']) / market_data['actual_volume']
                self.prediction_accuracy[market_data.name] = accuracy
                
            # Update trading performance
            if signal['action'] in ['buy', 'sell'] and signal['size'] > 0:
                self.trading_performance[market_data.name] = signal['edge']
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

if __name__ == "__main__":
    # Example usage
    config = {
        'prediction_threshold': 2000000,  # 2M weekly travelers
        'min_edge': 0.05,
        'max_position': 100,
        'confidence_threshold': 0.7,
        'max_model_deviation': 0.15
    }
    
    strategy = TSAVolumeStrategy(config)
    
    # Create sample data
    current_data = pd.DataFrame({
        'weather_severity': [0.5],
        'airline_prices': [100],
        'is_holiday': [0],
        'day_of_week': [2]
    })
    
    market_data = pd.Series({
        'price': 0.65,
        'volume': 1000,
        'days_to_expiry': 4
    })
    
    position_data = {
        'weekly_positions': 2,
        'daily_pnl_pct': -0.05,
        'position_pnl_pct': -0.02
    }
    
    # Example ensemble model would be initialized here
    # strategy.setup_models(ensemble_model)
    
    # Generate signal
    signal = strategy.generate_signal(current_data, market_data, position_data)
    print("\nTrading Signal:")
    for key, value in signal.items():
        print(f"{key}: {value}")