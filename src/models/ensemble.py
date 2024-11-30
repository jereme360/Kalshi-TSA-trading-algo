"""
Ensemble models for TSA prediction combining multiple base predictors.
"""
from src.models.base import BaseModel
from src.models.predictors import SARIMAXModel, SimpleExponentialModel, GBMModel, NeuralNetModel
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import HuberRegressor
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple base models.
    Uses robust regression to learn optimal weights for each model.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize ensemble model.
        
        Args:
            name: Model identifier
            config: Configuration dictionary containing:
                - model_configs: Dict of configurations for base models
                - weights_method: Method for combining predictions ('learned' or 'equal')
                - validation_window: Days of validation data for weight learning
        """
        super().__init__(name, config)
        self.model_configs = config.get('model_configs', {})
        self.weights_method = config.get('weights_method', 'learned')
        self.validation_window = config.get('validation_window', 30)
        self.models = {}
        self.weights = None
        self.prediction_history = pd.DataFrame()  # Track prediction performance
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all base models."""
        model_classes = {
            'sarimax': SARIMAXModel,
            'exponential': SimpleExponentialModel,
            'gbm': GBMModel,
            'neural': NeuralNetModel
        }
        
        for model_name, model_config in self.model_configs.items():
            model_type = model_config.pop('type')
            if model_type in model_classes:
                self.models[model_name] = model_classes[model_type](
                    name=f"{self.name}_{model_name}",
                    config=model_config
                )
            else:
                logger.warning(f"Unknown model type: {model_type}")
                
    def _learn_weights(self, predictions: pd.DataFrame, 
                      actuals: pd.Series) -> np.ndarray:
        """
        Learn optimal weights for combining model predictions.
        Uses Huber regression for robustness against outliers.
        
        Args:
            predictions: DataFrame of predictions from each model
            actuals: Series of actual values
            
        Returns:
            np.ndarray: Model weights
        """
        try:
            # Fit robust regression
            huber = HuberRegressor(epsilon=1.35)  # Default epsilon for 95% efficiency
            huber.fit(predictions, actuals)
            
            # Ensure weights are non-negative and sum to 1
            weights = np.maximum(huber.coef_, 0)
            weights = weights / weights.sum()
            
            return weights
            
        except Exception as e:
            logger.warning(f"Error learning weights: {str(e)}. Using equal weights.")
            return np.ones(len(predictions.columns)) / len(predictions.columns)
            
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train ensemble model.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        try:
            # Split data for validation
            validation_cutoff = X.index[-1] - timedelta(days=self.validation_window)
            train_X = X[:validation_cutoff]
            train_y = y[:validation_cutoff]
            val_X = X[validation_cutoff:]
            val_y = y[validation_cutoff:]
            
            # Train each base model
            val_predictions = pd.DataFrame(index=val_X.index)
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                model.train(train_X, train_y)
                val_predictions[name] = model.predict(val_X)
                
            # Learn weights if specified
            if self.weights_method == 'learned':
                self.weights = self._learn_weights(val_predictions, val_y)
                logger.info(f"Learned weights: {dict(zip(self.models.keys(), self.weights))}")
            else:
                # Equal weights
                self.weights = np.ones(len(self.models)) / len(self.models)
                
            # Retrain models on full dataset
            for model in self.models.values():
                model.train(X, y)
                
            logger.info(f"Successfully trained {self.name}")
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            np.ndarray: Weighted average predictions
        """
        try:
            # Get predictions from each model
            predictions = pd.DataFrame(index=X.index)
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
                
            # Combine predictions using weights
            ensemble_pred = predictions.values @ self.weights
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            raise
            
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, uncertainties)
        """
        try:
            # Get predictions from each model
            predictions = pd.DataFrame(index=X.index)
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
            
            # Calculate ensemble prediction
            ensemble_pred = predictions.values @ self.weights
            
            # Calculate uncertainty as weighted std dev of model predictions
            uncertainties = np.std(predictions.values, axis=1)
            
            return ensemble_pred, uncertainties
            
        except Exception as e:
            logger.error(f"Error predicting with uncertainty: {str(e)}")
            raise
            
    def get_prediction_confidence(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence score for predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            pd.Series: Confidence scores between 0 and 1
        """
        try:
            _, uncertainties = self.predict_with_uncertainty(X)
            
            # Transform uncertainty to confidence score (inverse relationship)
            confidence = 1 / (1 + uncertainties)
            
            return pd.Series(confidence, index=X.index)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            raise
            
    def track_prediction_accuracy(self, 
                                predictions: np.ndarray,
                                actuals: np.ndarray,
                                timestamp: datetime) -> None:
        """
        Track prediction accuracy over time.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            timestamp: Prediction timestamp
        """
        accuracy = 1 - abs(predictions - actuals) / actuals
        self.prediction_history.loc[timestamp, 'accuracy'] = accuracy.mean()
        self.prediction_history.loc[timestamp, 'uncertainty'] = np.std(predictions)
            
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from all base models that support it.
        
        Returns:
            Optional[pd.DataFrame]: Feature importance from each model
        """
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                importance_dict[name] = model.feature_importance
                
        if importance_dict:
            return pd.DataFrame(importance_dict)
        return None
        
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current model weights.
        
        Returns:
            Dict[str, float]: Model weights
        """
        return dict(zip(self.models.keys(), self.weights))
    
    def get_model_performances(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get performance metrics for each base model.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            pd.DataFrame: Performance metrics for each model
        """
        performances = {}
        
        for name, model in self.models.items():
            metrics = model.evaluate(X, y)
            performances[name] = metrics
            
        return pd.DataFrame(performances).T
    
    def get_recent_performance(self, 
                             lookback_days: int = 30) -> Dict[str, float]:
        """
        Get recent prediction performance metrics.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Dict with performance metrics
        """
        recent_history = self.prediction_history.last(f'{lookback_days}D')
        return {
            'mean_accuracy': recent_history['accuracy'].mean(),
            'accuracy_std': recent_history['accuracy'].std(),
            'mean_uncertainty': recent_history['uncertainty'].mean()
        }

if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Features
    X = pd.DataFrame({
        'weather_severity': np.random.normal(0, 1, n_samples),
        'airline_prices': np.random.normal(100, 10, n_samples),
        'is_holiday': np.random.binomial(1, 0.1, n_samples),
        'day_of_week': dates.dayofweek,
        'month': dates.month
    }, index=dates)
    
    # Target with multiple patterns
    y = pd.Series(
        5000 +  # Base volume
        np.sin(np.arange(n_samples) * 2 * np.pi / 7) * 1000 +  # Weekly seasonality
        np.sin(np.arange(n_samples) * 2 * np.pi / 30) * 500 +  # Monthly seasonality
        np.arange(n_samples) * 0.5 +  # Trend
        np.random.normal(0, 200, n_samples),  # Noise
        index=dates
    )
    
    # Create ensemble config
    config = {
        'model_configs': {
            'sarimax': {
                'type': 'sarimax',
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 7),
                'exog_vars': ['weather_severity', 'airline_prices', 'is_holiday']
            },
            'exp': {
                'type': 'exponential',
                'seasonal_periods': [7]
            },
            'gbm': {
                'type': 'gbm',
                'lgb_params': {
                    'n_estimators': 500,
                    'learning_rate': 0.05
                }
            }
        },
        'weights_method': 'learned',
        'validation_window': 30
    }
    
    # Initialize and train ensemble
    ensemble = EnsembleModel('ensemble_model', config)
    
    # Split data
    train_cutoff = '2023-10-01'
    X_train = X[:train_cutoff]
    X_test = X[train_cutoff:]
    y_train = y[:train_cutoff]
    y_test = y[train_cutoff:]
    
    # Train and evaluate
    ensemble.train(X_train, y_train)
    
    # Make predictions with uncertainty
    test_pred, test_uncertainty = ensemble.predict_with_uncertainty(X_test)
    test_confidence = ensemble.get_prediction_confidence(X_test)
    
    # Track performance
    ensemble.track_prediction_accuracy(test_pred, y_test.values, X_test.index[-1])
    
    # Print results
    print("\nEnsemble Model Results:")
    print("Model Weights:", ensemble.get_model_weights())
    print("\nPrediction Confidence (mean):", test_confidence.mean())
    print("Prediction Uncertainty (mean):", test_uncertainty.mean())
    print("\nRecent Performance:", ensemble.get_recent_performance())
    
    # Print individual model performances
    print("\nIndividual Model Performances:")
    print(ensemble.get_model_performances(X_test, y_test))