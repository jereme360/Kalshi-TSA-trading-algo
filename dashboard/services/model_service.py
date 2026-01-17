"""
Model service for the dashboard.
Wraps the EnsembleModel for predictions and accuracy tracking.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model predictions and performance tracking."""

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the trained ensemble model."""
        try:
            from models.ensemble import EnsembleModel
            from models.base import BaseModel

            model_dir = Path(__file__).parent.parent.parent / 'models'
            self.model = BaseModel.load_latest(model_dir, 'ensemble')
            self.model_loaded = True
            logger.info("Model loaded successfully")

        except FileNotFoundError:
            logger.warning("No trained model found")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False

    def get_prediction(self, features: pd.DataFrame) -> Dict:
        """
        Get prediction for given features.

        Args:
            features: Feature DataFrame

        Returns:
            Dict with prediction, uncertainty, and confidence
        """
        if not self.model_loaded:
            return self._get_sample_prediction()

        try:
            predictions, uncertainties = self.model.predict_with_uncertainty(features)
            confidence = self.model.get_prediction_confidence(features)

            return {
                'prediction': predictions[-1] if len(predictions) > 0 else None,
                'uncertainty': uncertainties[-1] if len(uncertainties) > 0 else None,
                'confidence': confidence.iloc[-1] if len(confidence) > 0 else None,
                'lower_bound': predictions[-1] - 1.96 * uncertainties[-1] if len(predictions) > 0 else None,
                'upper_bound': predictions[-1] + 1.96 * uncertainties[-1] if len(predictions) > 0 else None,
            }
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return self._get_sample_prediction()

    def _get_sample_prediction(self) -> Dict:
        """Generate sample prediction for demo."""
        base = 18_500_000  # Weekly passengers
        return {
            'prediction': base + np.random.normal(0, 200_000),
            'uncertainty': 300_000,
            'confidence': 0.82,
            'lower_bound': base - 600_000,
            'upper_bound': base + 600_000,
        }

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current ensemble model weights.

        Returns:
            Dict of model name to weight
        """
        if not self.model_loaded:
            return self._get_sample_weights()

        try:
            return self.model.get_model_weights()
        except Exception as e:
            logger.error(f"Error getting model weights: {e}")
            return self._get_sample_weights()

    def _get_sample_weights(self) -> Dict[str, float]:
        """Generate sample weights for demo."""
        return {
            'SARIMAX': 0.35,
            'GBM': 0.30,
            'Exponential': 0.20,
            'Neural Net': 0.15
        }

    def get_accuracy_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical prediction accuracy.

        Args:
            days: Number of days of history

        Returns:
            DataFrame with accuracy metrics over time
        """
        if not self.model_loaded or self.model.prediction_history.empty:
            return self._get_sample_accuracy_history(days)

        try:
            history = self.model.prediction_history.last(f'{days}D')
            return history
        except Exception as e:
            logger.error(f"Error getting accuracy history: {e}")
            return self._get_sample_accuracy_history(days)

    def _get_sample_accuracy_history(self, days: int) -> pd.DataFrame:
        """Generate sample accuracy history for demo."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate realistic accuracy pattern
        base_accuracy = 0.85
        noise = np.random.normal(0, 0.05, days)
        trend = np.sin(np.arange(days) * 2 * np.pi / 30) * 0.03

        return pd.DataFrame({
            'accuracy': np.clip(base_accuracy + noise + trend, 0.5, 0.98),
            'uncertainty': np.abs(np.random.normal(0.15, 0.03, days)),
            'predicted': 18_500_000 + np.random.normal(0, 300_000, days),
            'actual': 18_500_000 + np.random.normal(0, 250_000, days),
        }, index=dates)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from ensemble model.

        Returns:
            DataFrame with feature importance scores
        """
        if not self.model_loaded:
            return self._get_sample_feature_importance()

        try:
            importance = self.model.get_feature_importance()
            if importance is not None:
                return importance
            return self._get_sample_feature_importance()
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return self._get_sample_feature_importance()

    def _get_sample_feature_importance(self) -> pd.DataFrame:
        """Generate sample feature importance for demo."""
        features = [
            'passengers_lag_7', 'passengers_lag_1', 'day_of_week',
            'is_holiday', 'days_to_holiday', 'month', 'weather_severity_index',
            'avg_temperature', 'is_weekend', 'passengers_lag_14'
        ]

        importance = np.random.dirichlet(np.ones(len(features)) * 2)
        importance = sorted(importance, reverse=True)

        return pd.DataFrame({
            'feature': features,
            'importance': importance
        }).set_index('feature')

    def get_recent_performance(self, lookback_days: int = 30) -> Dict:
        """
        Get recent model performance metrics.

        Args:
            lookback_days: Days to look back

        Returns:
            Dict with performance metrics
        """
        if not self.model_loaded:
            return self._get_sample_performance()

        try:
            return self.model.get_recent_performance(lookback_days)
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return self._get_sample_performance()

    def _get_sample_performance(self) -> Dict:
        """Generate sample performance metrics for demo."""
        return {
            'mean_accuracy': 0.847,
            'accuracy_std': 0.052,
            'mean_uncertainty': 0.148,
            'rmse': 285_000,
            'mae': 215_000,
            'hit_rate': 0.73
        }


@st.cache_resource
def get_model_service() -> ModelService:
    """Get cached model service instance."""
    return ModelService()
