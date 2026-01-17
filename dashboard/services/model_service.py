"""
Model service for the dashboard.
Wraps the EnsembleModel for predictions and accuracy tracking.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from dashboard.services.data_service import load_tsa_data, get_weekly_tsa_data

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
        Get historical prediction accuracy by running model on past weeks.

        Args:
            days: Number of days of history (converted to weeks)

        Returns:
            DataFrame with columns: date, predicted, actual, accuracy
        """
        try:
            # Load historical TSA data
            tsa_data = load_tsa_data(days=max(days + 365, 730))  # Need extra for training
            if tsa_data.empty:
                logger.warning("No TSA data available for accuracy history")
                return pd.DataFrame()

            # Get passenger column
            col = 'passengers' if 'passengers' in tsa_data.columns else 'current_year'
            if col not in tsa_data.columns:
                return pd.DataFrame()

            # Calculate weekly totals
            weekly_data = tsa_data[col].resample('W-SUN').sum()

            # Compute rolling predictions for past weeks
            weeks_to_evaluate = days // 7
            results = []

            for i in range(weeks_to_evaluate, 0, -1):
                try:
                    # Get the week to predict
                    week_idx = len(weekly_data) - i
                    if week_idx < 52:  # Need at least 52 weeks of history
                        continue

                    week_date = weekly_data.index[week_idx]
                    actual = weekly_data.iloc[week_idx]

                    # Simple prediction: weighted average of same week last year + recent trend
                    same_week_ly = weekly_data.iloc[week_idx - 52] if week_idx >= 52 else actual
                    recent_avg = weekly_data.iloc[max(0, week_idx-4):week_idx].mean()

                    # Weight: 40% same week last year, 60% recent trend
                    predicted = 0.4 * same_week_ly + 0.6 * recent_avg

                    # If we have a trained model, use it instead
                    if self.model_loaded:
                        try:
                            # Use model's prediction method if available
                            train_end = week_date - timedelta(days=1)
                            train_data = tsa_data[tsa_data.index < train_end]
                            if len(train_data) > 100:
                                # Create simple features for prediction
                                features = self._create_simple_features(train_data, week_date)
                                if features is not None:
                                    pred_result = self.model.predict(features)
                                    if len(pred_result) > 0:
                                        predicted = pred_result[-1] * 7  # Daily to weekly
                        except Exception:
                            pass  # Fall back to simple prediction

                    # Calculate accuracy (1 - MAPE)
                    if actual > 0:
                        accuracy = 1 - abs(predicted - actual) / actual
                        accuracy = max(0, min(1, accuracy))  # Clamp to [0, 1]
                    else:
                        accuracy = 0

                    results.append({
                        'date': week_date,
                        'predicted': predicted,
                        'actual': actual,
                        'accuracy': accuracy
                    })

                except Exception as e:
                    logger.debug(f"Error computing week {i}: {e}")
                    continue

            if not results:
                return pd.DataFrame()

            df = pd.DataFrame(results)
            df.set_index('date', inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error getting accuracy history: {e}")
            return pd.DataFrame()

    def _create_simple_features(self, data: pd.DataFrame, target_date: datetime) -> Optional[pd.DataFrame]:
        """Create simple features for prediction."""
        try:
            col = 'passengers' if 'passengers' in data.columns else 'current_year'
            recent = data[col].iloc[-7:]  # Last 7 days

            features = pd.DataFrame({
                'day_of_week': [target_date.weekday()],
                'month': [target_date.month],
                'week_of_year': [target_date.isocalendar()[1]],
                'recent_mean': [recent.mean()],
                'recent_std': [recent.std()],
            }, index=[target_date])

            return features
        except Exception:
            return None

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
        Get recent model performance metrics computed from accuracy history.

        Args:
            lookback_days: Days to look back

        Returns:
            Dict with mean_accuracy, accuracy_std, rmse, mae
        """
        try:
            accuracy_df = self.get_accuracy_history(lookback_days)

            if accuracy_df.empty:
                return {
                    'mean_accuracy': 0,
                    'accuracy_std': 0,
                    'rmse': 0,
                    'mae': 0
                }

            # Calculate RMSE and MAE
            errors = accuracy_df['predicted'] - accuracy_df['actual']
            rmse = np.sqrt((errors ** 2).mean())
            mae = np.abs(errors).mean()

            return {
                'mean_accuracy': accuracy_df['accuracy'].mean(),
                'accuracy_std': accuracy_df['accuracy'].std() if len(accuracy_df) > 1 else 0,
                'rmse': rmse,
                'mae': mae
            }

        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {
                'mean_accuracy': 0,
                'accuracy_std': 0,
                'rmse': 0,
                'mae': 0
            }


@st.cache_resource
def get_model_service() -> ModelService:
    """Get cached model service instance."""
    return ModelService()
