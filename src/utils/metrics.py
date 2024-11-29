"""
Performance metrics for model evaluation and backtesting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_prediction_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict containing various metric scores
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics

def calculate_trading_metrics(returns: pd.Series,
                            positions: pd.Series,
                            risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate trading strategy performance metrics.
    
    Args:
        returns: Series of strategy returns
        positions: Series of position sizes
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dict containing various trading metrics
    """
    # Annualization factor (assuming daily data)
    annual_factor = 252
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (annual_factor/len(returns)) - 1
    volatility = returns.std() * np.sqrt(annual_factor)
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Position metrics
    avg_position = positions.abs().mean()
    position_changes = (positions.diff() != 0).sum()
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_position': avg_position,
        'position_changes': position_changes
    }
    return metrics

def calculate_kalshi_specific_metrics(predictions: pd.Series,
                                    actual_values: pd.Series,
                                    contract_prices: pd.Series) -> Dict[str, float]:
    """
    Calculate Kalshi-specific trading metrics.
    
    Args:
        predictions: Predicted probabilities
        actual_values: Actual outcomes
        contract_prices: Contract prices
        
    Returns:
        Dict containing Kalshi-specific metrics
    """
    # Calculate prediction accuracy
    accuracy = (predictions.round() == actual_values).mean()
    
    # Calculate theoretical edge
    edge = (predictions - contract_prices).mean()
    
    # Calculate realized PnL
    theoretical_pnl = ((predictions > contract_prices) * (actual_values - contract_prices)).sum()
    
    metrics = {
        'prediction_accuracy': accuracy,
        'average_edge': edge,
        'theoretical_pnl': theoretical_pnl,
        'edge_sharpe': edge.mean() / edge.std() if edge.std() != 0 else 0
    }
    return metrics