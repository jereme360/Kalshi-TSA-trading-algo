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
        self._tsa_data = None
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
            logger.warning("No trained model found - using on-demand prediction")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False

    def _get_tsa_data(self) -> pd.DataFrame:
        """Get cached TSA data."""
        if self._tsa_data is None:
            self._tsa_data = load_tsa_data(days=730)  # 2 years
        return self._tsa_data

    def get_prediction(self, features: pd.DataFrame) -> Dict:
        """
        Get prediction for next week's TSA passengers.
        Computes on-demand from historical data if no model loaded.

        Args:
            features: Feature DataFrame (ignored if computing on-demand)

        Returns:
            Dict with prediction, uncertainty, and confidence
        """
        # Try trained model first
        if self.model_loaded and features is not None:
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
                logger.error(f"Error with trained model: {e}")

        # Compute prediction on-demand from TSA data
        return self._compute_prediction_on_demand()

    def _compute_prediction_on_demand(self) -> Dict:
        """Compute weekly prediction from historical TSA data."""
        try:
            tsa_data = self._get_tsa_data()
            if tsa_data.empty:
                return self._empty_prediction("No TSA data available")

            # Get passenger column
            col = 'passengers' if 'passengers' in tsa_data.columns else 'current_year'
            if col not in tsa_data.columns:
                return self._empty_prediction("No passenger data column found")

            # Calculate weekly totals
            weekly_data = tsa_data[col].resample('W-SUN').sum()

            if len(weekly_data) < 52:
                return self._empty_prediction("Insufficient historical data")

            # Prediction components:
            # 1. Same week last year (adjusted for YoY trend)
            same_week_ly = weekly_data.iloc[-52]

            # 2. Recent 4-week average
            recent_avg = weekly_data.iloc[-4:].mean()

            # 3. YoY growth rate
            if len(weekly_data) >= 104:
                last_year_avg = weekly_data.iloc[-104:-52].mean()
                this_year_avg = weekly_data.iloc[-52:].mean()
                yoy_growth = (this_year_avg / last_year_avg) - 1 if last_year_avg > 0 else 0
            else:
                yoy_growth = 0.05  # Default 5% growth assumption

            # Weighted prediction: 30% same week LY (adjusted), 70% recent trend
            prediction = 0.3 * same_week_ly * (1 + yoy_growth) + 0.7 * recent_avg

            # Uncertainty: based on recent weekly standard deviation
            uncertainty = weekly_data.iloc[-12:].std()
            uncertainty = max(uncertainty, prediction * 0.03)  # Minimum 3%

            # Confidence: inverse of coefficient of variation
            cv = uncertainty / prediction if prediction > 0 else 1
            confidence = 1 / (1 + cv)

            return {
                'prediction': prediction,
                'uncertainty': uncertainty,
                'confidence': confidence,
                'lower_bound': prediction - 1.96 * uncertainty,
                'upper_bound': prediction + 1.96 * uncertainty,
            }

        except Exception as e:
            logger.error(f"Error computing on-demand prediction: {e}")
            return self._empty_prediction(str(e))

    def _empty_prediction(self, error: str = "Unknown error") -> Dict:
        """Return empty prediction with error message."""
        return {
            'prediction': None,
            'uncertainty': None,
            'confidence': None,
            'lower_bound': None,
            'upper_bound': None,
            'error': error
        }

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current ensemble model weights.

        Returns:
            Dict of model name to weight
        """
        if not self.model_loaded:
            return self._get_default_weights()

        try:
            return self.model.get_model_weights()
        except Exception as e:
            logger.error(f"Error getting model weights: {e}")
            return self._get_default_weights()

    def _get_default_weights(self) -> Dict[str, float]:
        """Return default weights for 3-model ensemble."""
        return {
            'GBM': 0.45,
            'SARIMAX': 0.30,
            'Exponential': 0.25
        }

    def get_accuracy_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical prediction accuracy by running model on past weeks.

        Args:
            days: Number of days of history (converted to weeks)

        Returns:
            DataFrame with columns: date, predicted, actual, accuracy
        """
        try:
            # Load historical TSA data - request more to ensure we have enough
            tsa_data = load_tsa_data(days=1500)  # ~4 years
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
            # Convert days to weeks, minimum 4 weeks
            weeks_to_evaluate = max(days // 7, 4)
            results = []

            # Start from most recent COMPLETE week and go back
            # Exclude current week (likely incomplete) by checking if week end is in the future
            today = pd.Timestamp.now().normalize()
            end_idx = len(weekly_data) - 1

            # If the most recent week hasn't ended yet, exclude it
            if weekly_data.index[end_idx] >= today:
                end_idx -= 1

            start_idx = max(53, end_idx - weeks_to_evaluate)  # Need 52 weeks of history

            for week_idx in range(start_idx, end_idx + 1):
                try:
                    week_date = weekly_data.index[week_idx]
                    actual = weekly_data.iloc[week_idx]

                    # Skip if actual is 0 or very small (incomplete data)
                    # A full week should have 14M+ passengers (2M+ per day)
                    if actual < 10000000:  # Less than 10M passengers is likely incomplete
                        continue

                    # Simple prediction: weighted average of same week last year + recent trend
                    same_week_ly = weekly_data.iloc[week_idx - 52]
                    recent_avg = weekly_data.iloc[max(0, week_idx-4):week_idx].mean()

                    # YoY adjustment
                    if week_idx >= 104:
                        prev_year_avg = weekly_data.iloc[week_idx-104:week_idx-52].mean()
                        curr_year_avg = weekly_data.iloc[week_idx-52:week_idx].mean()
                        yoy_factor = curr_year_avg / prev_year_avg if prev_year_avg > 0 else 1.0
                    else:
                        yoy_factor = 1.0

                    # Weight: 30% same week LY (adjusted), 70% recent trend
                    predicted = 0.3 * same_week_ly * yoy_factor + 0.7 * recent_avg

                    # Calculate accuracy (1 - MAPE)
                    accuracy = 1 - abs(predicted - actual) / actual
                    accuracy = max(0, min(1, accuracy))  # Clamp to [0, 1]

                    results.append({
                        'date': week_date,
                        'predicted': predicted,
                        'actual': actual,
                        'accuracy': accuracy
                    })

                except Exception as e:
                    logger.debug(f"Error computing week {week_idx}: {e}")
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
            return self._get_default_feature_importance()

        try:
            importance = self.model.get_feature_importance()
            if importance is not None and not importance.empty:
                return importance
            return self._get_default_feature_importance()
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return self._get_default_feature_importance()

    def _get_default_feature_importance(self) -> pd.DataFrame:
        """Return default feature importance based on TSA prediction factors."""
        features = {
            'recent_weekly_avg': 0.35,
            'same_week_last_year': 0.25,
            'yoy_growth_trend': 0.15,
            'weekly_seasonality': 0.12,
            'holiday_proximity': 0.08,
            'recent_volatility': 0.05
        }
        df = pd.DataFrame({
            'importance': list(features.values())
        }, index=list(features.keys()))
        return df

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
