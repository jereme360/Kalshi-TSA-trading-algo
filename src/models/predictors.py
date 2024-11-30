"""
Prediction models for TSA passenger volume forecasting.
"""
from src.models.base import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
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
import itertools
from pathlib import Path

logger = logging.getLogger(__name__)

class SARIMAXModel(BaseModel):
    """SARIMAX model for time series prediction."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize SARIMAX model."""
        super().__init__(name, config)
        self.order = config.get('order', (1, 1, 1))
        self.seasonal_order = config.get('seasonal_order', (1, 1, 1, 7))
        self.exog_vars = config.get('exog_vars', [])
        self.enforce_stationarity = config.get('enforce_stationarity', True)
        self.enforce_invertibility = config.get('enforce_invertibility', True)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train SARIMAX model with cross-validation."""
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
            logger.info(f"Successfully trained {self.name}")
            self.feature_names = self.exog_vars
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        self.validate_features(X)
        exog = X[self.exog_vars] if self.exog_vars else None
        
        try:
            predictions = self.fitted_model.forecast(steps=len(X), exog=exog)
            return predictions.values
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            raise

class SimpleExponentialModel(BaseModel):
    """Simple exponential smoothing model with multiple seasonality."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize exponential smoothing model."""
        super().__init__(name, config)
        self.seasonal_periods = config.get('seasonal_periods', [7])
        self.trend = config.get('trend', 'add')
        self.seasonal = config.get('seasonal', 'add')
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train exponential smoothing model."""
        try:
            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods[0],
                damped_trend=True
            )
            self.fitted_model = self.model.fit(optimized=True, remove_bias=True)
            logger.info(f"Successfully trained {self.name}")
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        try:
            forecast = self.fitted_model.forecast(len(X))
            return forecast.values
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            raise

class GBMModel(BaseModel):
    """Gradient Boosting model for TSA prediction."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize GBM model."""
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
        """Train GBM model."""
        try:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
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
        """Generate predictions."""
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

class TSADataset(Dataset):
    """Custom dataset for TSA passenger volume."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
        
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_length]
        y_target = self.y[idx + self.seq_length]
        return X_seq, y_target

class TSANeuralNet(nn.Module):
    """Neural network architecture with regularization."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc_layers(last_out)
        return out.squeeze()

class NeuralNetModel(BaseModel):
    """Neural network model for TSA prediction."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize neural network model."""
        super().__init__(name, config)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.num_layers = config.get('num_layers', 1)
        self.seq_length = config.get('seq_length', 14)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.epochs = config.get('epochs', 50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = config.get('dropout', 0.3)
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train neural network model."""
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
        """Generate predictions."""
        self.validate_features(X)
        try:
            self.model.eval()
            X_scaled = self.scaler_X.transform(X)
            
            if len(X_scaled) < self.seq_length:
                raise ValueError(f"Need at least {self.seq_length} samples for prediction")
            
            sequences = []
            for i in range(len(X_scaled) - self.seq_length + 1):
                sequence = X_scaled[i:i + self.seq_length]
                sequences.append(sequence)
            
            sequences = torch.FloatTensor(sequences).to(self.device)
            
            with torch.no_grad():
                predictions_scaled = self.model(sequences)
            
            predictions = self.scaler_y.inverse_transform(
                predictions_scaled.cpu().numpy().reshape(-1, 1)
            ).squeeze()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    import numpy as np
    
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
        # Base volume
        5000 +
        # Weekly seasonality
        np.sin(np.arange(n_samples) * 2 * np.pi / 7) * 1000 +
        # Monthly seasonality
        np.sin(np.arange(n_samples) * 2 * np.pi / 30) * 500 +
        # Trend
        np.arange(n_samples) * 0.5 +
        # Random noise
        np.random.normal(0, 200, n_samples),
        index=dates
    )
    
    # Split into train and test
    train_cutoff = '2023-10-01'
    X_train = X[:train_cutoff]
    X_test = X[train_cutoff:]
    y_train = y[:train_cutoff]
    y_test = y[train_cutoff:]
    
    # Initialize and test all models
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
            config={
                'seasonal_periods': [7]  # Weekly seasonality
            }
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
                'batch_size': 64
            }
        )
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Train model
            model.train(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            
            results[name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            
            print(f"{name} Results:")
            print(f"Train RMSE: {train_metrics['rmse']:.2f}")
            print(f"Test RMSE: {test_metrics['rmse']:.2f}")
            
            # Print feature importance if available
            if hasattr(model, 'feature_importance') and model.feature_importance is not None:
                print("\nFeature Importance:")
                print(model.feature_importance.sort_values(ascending=False))
                
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue
    
    # Print summary of all models
    print("\nModel Comparison Summary:")
    print("------------------------")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Train RMSE: {result['train_metrics']['rmse']:.2f}")
        print(f"Test RMSE: {result['test_metrics']['rmse']:.2f}")
        print(f"R-squared: {result['test_metrics']['r2']:.3f}")