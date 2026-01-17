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
        """Return empty prediction when no model loaded."""
        return {
            'prediction': None,
            'uncertainty': None,
            'confidence': None,
            'lower_bound': None,
            'upper_bound': None,
            'error': 'No trained model loaded'
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
        """Return empty weights when no model loaded."""
        return {}

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
        """Return empty DataFrame when no model loaded."""
        return pd.DataFrame()

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
        """Return empty DataFrame when no model loaded."""
        return pd.DataFrame()

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
        """Return empty performance when no model loaded."""
        return {}


@st.cache_resource
def get_model_service() -> ModelService:
    """Get cached model service instance."""
    return ModelService()
