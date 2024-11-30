"""
Prediction models for TSA passenger volume forecasting with uncertainty estimation.
"""
from abc import ABC, abstractmethod
from src.models.base import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import itertools
from pathlib import Path

logger = logging.getLogger(__name__)

class TSABaseModel(BaseModel):
    """Base class for TSA-specific models with uncertainty estimation."""
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, uncertainty)
        """
        raise NotImplementedError("Subclasses must implement predict_with_uncertainty")
    
    def get_prediction_intervals(self, X: pd.DataFrame, 
                               conf_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.
        
        Args:
            X: Feature DataFrame
            conf_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (lower_bound, upper_bound)
        """
        raise NotImplementedError("Subclasses must implement get_prediction_intervals")

class SARIMAXModel(TSABaseModel):
    """SARIMAX model with uncertainty estimation."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.order = config.get('order', (1, 1, 1))
        self.seasonal_order = config.get('seasonal_order', (1, 1, 1, 7))
        self.exog_vars = config.get('exog_vars', [])
        self.enforce_stationarity = config.get('enforce_stationarity', True)
        self.enforce_invertibility = config.get('enforce_invertibility', True)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        exog = X[self.exog_vars] if self.exog_vars else None
        
        try:
            self.model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
            self.fitted_model = self.model.fit(disp=False)
            self.feature_names = self.exog_vars
            self.residual_std = np.std(self.fitted_model.resid)
            logger.info(f"Successfully trained {self.name}")
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.validate_features(X)
        exog = X[self.exog_vars] if self.exog_vars else None
        
        try:
            predictions = self.fitted_model.forecast(steps=len(X), exog=exog)
            return predictions.values
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            raise
            
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            predictions = self.predict(X)
            forecast = self.fitted_model.get_forecast(
                steps=len(X),
                exog=X[self.exog_vars] if self.exog_vars else None
            )
            uncertainty = forecast.std_err
            return predictions, uncertainty
            
        except Exception as e:
            logger.error(f"Error predicting with uncertainty: {str(e)}")
            raise
            
    def get_prediction_intervals(self, X: pd.DataFrame, 
                               conf_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        try:
            forecast = self.fitted_model.get_forecast(
                steps=len(X),
                exog=X[self.exog_vars] if self.exog_vars else None
            )
            conf_int = forecast.conf_int(alpha=1-conf_level)
            return conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
            
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {str(e)}")
            raise

class SimpleExponentialModel(TSABaseModel):
    """Exponential smoothing model with uncertainty estimation."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.seasonal_periods = config.get('seasonal_periods', [7])
        self.trend = config.get('trend', 'add')
        self.seasonal = config.get('seasonal', 'add')
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods[0],
                damped_trend=True
            )
            self.fitted_model = self.model.fit(optimized=True, remove_bias=True)
            self.residual_std = np.std(self.fitted_model.resid)
            logger.info(f"Successfully trained {self.name}")
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            forecast = self.fitted_model.forecast(len(X))
            return forecast.values
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            raise
            
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            predictions = self.predict(X)
            # Calculate uncertainty based on residual variance and forecast horizon
            uncertainty = np.array([
                self.residual_std * np.sqrt(h + 1) 
                for h in range(len(X))
            ])
            return predictions, uncertainty
            
        except Exception as e:
            logger.error(f"Error predicting with uncertainty: {str(e)}")
            raise
            
    def get_prediction_intervals(self, X: pd.DataFrame, 
                               conf_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        try:
            predictions = self.predict(X)
            z_value = stats.norm.ppf((1 + conf_level) / 2)
            _, uncertainty = self.predict_with_uncertainty(X)
            
            lower_bound = predictions - z_value * uncertainty
            upper_bound = predictions + z_value * uncertainty
            
            return lower_bound, upper_bound
            
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {str(e)}")
            raise

class GBMModel(TSABaseModel):
    """Gradient Boosting model with uncertainty estimation."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.lgb_params = config.get('lgb_params', {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'early_stopping_rounds': 50
        })
        self.scaler = StandardScaler()
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Train main model
            tscv = TimeSeriesSplit(n_splits=5)
            valid_sets = []
            
            for train_idx, valid_idx in tscv.split(X_scaled):
                valid_sets.append((
                    lgb.Dataset(X_scaled.iloc[train_idx], y.iloc[train_idx]),
                    lgb.Dataset(X_scaled.iloc[valid_idx], y.iloc[valid_idx])
                ))
            
            self.model = lgb.train(
                params=self.lgb_params,
                train_set=valid_sets[0][0],
                valid_sets=[vs[1] for vs in valid_sets],
                callbacks=[lgb.early_stopping(self.lgb_params['early_stopping_rounds'])]
            )
            
            # Train quantile models for uncertainty
            self.quantile_models = {}
            for q in [0.025, 0.975]:  # For 95% prediction intervals
                params = self.lgb_params.copy()
                params['objective'] = 'quantile'
                params['alpha'] = q
                
                self.quantile_models[q] = lgb.train(
                    params=params,
                    train_set=valid_sets[0][0],
                    valid_sets=[vs[1] for vs in valid_sets],
                    callbacks=[lgb.early_stopping(50)]
                )
            
            self.feature_names = X.columns.tolist()
            self.feature_importance = pd.Series(
                self.model.feature_importance(),
                index=self.feature_names
            )
            
            logger.info(f"Successfully trained {self.name}")
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.validate_features(X)
        try:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            raise
            
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            
            predictions = self.model.predict(X_scaled)
            
            # Use quantile predictions for uncertainty
            lower = self.quantile_models[0.025].predict(X_scaled)
            upper = self.quantile_models[0.975].predict(X_scaled)
            
            # Uncertainty as half the prediction interval width
            uncertainty = (upper - lower) / 4
            
            return predictions, uncertainty
            
        except Exception as e:
            logger.error(f"Error predicting with uncertainty: {str(e)}")
            raise
            
    def get_prediction_intervals(self, X: pd.DataFrame, 
                               conf_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            
            alpha = (1 - conf_level) / 2
            lower = self.quantile_models[alpha].predict(X_scaled)
            upper = self.quantile_models[1-alpha].predict(X_scaled)
            
            return lower, upper
            
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {str(e)}")
            raise

class NeuralNetModel(TSABaseModel):
    """Neural network model with uncertainty estimation using MC Dropout."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.num_layers = config.get('num_layers', 1)
        self.seq_length = config.get('seq_length', 14)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.epochs = config.get('epochs', 50)
        self.dropout = config.get('dropout', 0.3)
        self.mc_samples = config.get('mc_samples', 100)  # For uncertainty estimation
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.validate_features(X)
        try:
            self.model.train()  # Enable dropout for MC sampling
            X_scaled = self.scaler_X.transform(X)
            
            if len(X_scaled) < self.seq_length:
                raise ValueError(f"Need at least {self.seq_length} samples")
            
            # Prepare sequences
            sequences = []
            for i in range(len(X_scaled) - self.seq_length + 1):
                sequence = X_scaled[i:i + self.seq_length]
                sequences.append(sequence)
            
            sequences = torch.FloatTensor(sequences).to(self.device)
            
            # MC Dropout sampling
            mc_predictions = []
            for _ in range(self.mc_samples):
                with torch.no_grad():
                    pred = self.model(sequences)
                    mc_predictions.append(pred.cpu().numpy())
            
            mc_predictions = np.array(mc_predictions)
            
            # Calculate mean and uncertainty
            mean_pred = np.mean(mc_predictions, axis=0)
            uncertainty = np.std(mc_predictions, axis=0)
            
            # Inverse transform
            mean_pred = self.scaler_y.inverse_transform(
                mean_pred.reshape(-1, 1)).squeeze()
            uncertainty = uncertainty * self.scaler_y.scale_
            
            return mean_pred, uncertainty
        
        except Exception as e:
            logger.error(f"Error predicting with uncertainty: {str(e)}")
            raise
            
    def get_prediction_intervals(self, X: pd.DataFrame, 
                               conf_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        try:
            predictions, uncertainty = self.predict_with_uncertainty(X)
            z_value = stats.norm.ppf((1 + conf_level) / 2)
            
            lower_bound = predictions - z_value * uncertainty
            upper_bound = predictions + z_value * uncertainty
            
            return lower_bound, upper_bound
            
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {str(e)}")
            raise
            
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).squeeze()
            
            dataset = TSADataset(X_scaled, y_scaled, self.seq_length)
            train_size = int(0.8 * len(dataset))
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, len(dataset) - train_size])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size
            )
            
            self.model = TSANeuralNet(
                input_dim=X.shape[1],
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(self.epochs):
                self.model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = F.mse_loss(y_pred, y_batch)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        y_pred = self.model(X_batch)
                        val_loss += F.mse_loss(y_pred, y_batch).item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered")
                        break
            
            self.model.load_state_dict(torch.load('best_model.pt'))
            self.feature_names = X.columns.tolist()
            
            logger.info(f"Successfully trained {self.name}")
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Use mean prediction from MC Dropout for point estimates."""
        predictions, _ = self.predict_with_uncertainty(X)
        return predictions

if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Create synthetic features
    X = pd.DataFrame({
        'weather_severity': np.random.normal(0, 1, n_samples),
        'airline_prices': np.random.normal(100, 10, n_samples),
        'is_holiday': np.random.binomial(1, 0.1, n_samples),
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'economic_index': np.random.normal(50, 5, n_samples)
    }, index=dates)
    
    # Create synthetic target with multiple patterns
    y = pd.Series(
        5000 +  # Base volume
        np.sin(np.arange(n_samples) * 2 * np.pi / 7) * 1000 +  # Weekly seasonality
        np.sin(np.arange(n_samples) * 2 * np.pi / 30) * 500 +  # Monthly seasonality
        np.arange(n_samples) * 0.5 +  # Trend
        np.random.normal(0, 200, n_samples),  # Noise
        index=dates
    )
    
    # Test all models
    models = {
        'sarimax': SARIMAXModel(
            name='sarimax_model',
            config={
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 7),
                'exog_vars': ['weather_severity', 'airline_prices', 'is_holiday']
            }
        ),
        'exponential': SimpleExponentialModel(
            name='exp_model',
            config={'seasonal_periods': [7]}
        ),
        'gbm': GBMModel(
            name='gbm_model',
            config={
                'lgb_params': {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 5
                }
            }
        ),
        'neural': NeuralNetModel(
            name='neural_model',
            config={
                'hidden_dim': 32,
                'num_layers': 1,
                'seq_length': 14,
                'batch_size': 64,
                'mc_samples': 100
            }
        )
    }
    
    # Split data
    train_cutoff = '2023-10-01'
    X_train = X[:train_cutoff]
    X_test = X[train_cutoff:]
    y_train = y[:train_cutoff]
    y_test = y[train_cutoff:]
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Train model
            model.train(X_train, y_train)
            
            # Make predictions with uncertainty
            predictions, uncertainty = model.predict_with_uncertainty(X_test)
            lower, upper = model.get_prediction_intervals(X_test)
            
            # Calculate metrics
            results[name] = {
                'mse': np.mean((predictions - y_test.values) ** 2),
                'mean_uncertainty': np.mean(uncertainty),
                'coverage': np.mean((y_test.values >= lower) & (y_test.values <= upper))
            }
            
            print(f"{name} Results:")
            print(f"MSE: {results[name]['mse']:.2f}")
            print(f"Mean Uncertainty: {results[name]['mean_uncertainty']:.2f}")
            print(f"95% CI Coverage: {results[name]['coverage']:.2%}")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue
    
    print("\nModel Comparison Summary:")
    print("------------------------")
    for name, result in results.items():
        print(f"\n{name}:")
        for metric, value in result.items():
            print(f"{metric}: {value:.4f}")