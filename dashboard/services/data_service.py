"""
Data loading service for the dashboard.
Wraps TSA data collection and feature engineering.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import sys
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.collectors.tsa_collector import TSACollector
from data.features import FeatureEngineer

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_tsa_data(days: int = 365) -> pd.DataFrame:
    """
    Load TSA checkpoint data.

    Args:
        days: Number of days of historical data to load

    Returns:
        DataFrame with TSA passenger data
    """
    try:
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        collector = TSACollector(data_dir)

        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)

        df = collector.fetch_data(start_date, end_date)
        return df

    except Exception as e:
        logger.error(f"Error loading TSA data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame:
    """
    Load the latest feature snapshot.

    Returns:
        DataFrame with engineered features
    """
    try:
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'features'
        engineer = FeatureEngineer(data_dir)
        return engineer.load_latest_features()

    except FileNotFoundError:
        logger.warning("No feature snapshots found")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return pd.DataFrame()


def get_latest_tsa_value() -> Dict:
    """
    Get the most recent TSA checkpoint value.

    Returns:
        Dict with latest date and passenger count
    """
    df = load_tsa_data(days=30)

    if df.empty:
        return {'date': None, 'passengers': None, 'yoy_change': None}

    latest = df.iloc[-1]

    # Get passenger count - handle both column naming conventions
    if 'passengers' in df.columns:
        passengers = latest['passengers']
    elif 'current_year' in df.columns:
        passengers = latest['current_year']
    else:
        return {'date': df.index[-1], 'passengers': None, 'yoy_change': None}

    # Calculate YoY change if previous year data available
    yoy_change = None
    if 'previous_year' in df.columns and passengers > 0:
        yoy_change = (passengers - latest['previous_year']) / latest['previous_year']

    return {
        'date': df.index[-1],
        'passengers': passengers,
        'yoy_change': yoy_change
    }


def get_weekly_tsa_data() -> pd.DataFrame:
    """
    Get TSA data aggregated by week.

    Returns:
        DataFrame with weekly passenger counts
    """
    df = load_tsa_data(days=365)

    if df.empty:
        return pd.DataFrame()

    # Handle both column naming conventions
    col = 'passengers' if 'passengers' in df.columns else 'current_year'
    if col not in df.columns:
        return pd.DataFrame()

    weekly = df[col].resample('W').sum()
    return weekly.to_frame(name='passengers')


def get_data_freshness() -> Dict:
    """
    Check freshness of cached data.

    Returns:
        Dict with freshness info for each data source
    """
    freshness = {}

    # Check TSA data
    try:
        df = load_tsa_data(days=7)
        if not df.empty:
            latest_date = df.index[-1]
            days_old = (datetime.now() - latest_date).days
            freshness['tsa'] = {
                'latest_date': latest_date,
                'days_old': days_old,
                'status': 'fresh' if days_old <= 2 else 'stale'
            }
        else:
            freshness['tsa'] = {'status': 'unavailable'}
    except Exception:
        freshness['tsa'] = {'status': 'error'}

    # Check features
    try:
        features = load_features()
        if not features.empty:
            freshness['features'] = {
                'rows': len(features),
                'columns': len(features.columns),
                'status': 'available'
            }
        else:
            freshness['features'] = {'status': 'unavailable'}
    except Exception:
        freshness['features'] = {'status': 'error'}

    return freshness


def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Generate sample data for demo/testing when real data unavailable.

    Returns:
        Dict of sample DataFrames
    """
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')

    # Sample TSA data with weekly seasonality
    base = 2_500_000
    weekly_pattern = np.sin(np.arange(365) * 2 * np.pi / 7) * 200_000
    trend = np.arange(365) * 500
    noise = np.random.normal(0, 100_000, 365)

    tsa_df = pd.DataFrame({
        'current_year': base + weekly_pattern + trend + noise,
        'previous_year': base + weekly_pattern + noise * 0.8,
    }, index=dates)
    tsa_df = tsa_df.astype(int)

    # Sample predictions
    predictions_df = pd.DataFrame({
        'predicted': tsa_df['current_year'].shift(-7) + np.random.normal(0, 50_000, 365),
        'actual': tsa_df['current_year'],
        'confidence': np.random.uniform(0.7, 0.95, 365)
    }, index=dates)

    return {
        'tsa': tsa_df,
        'predictions': predictions_df
    }
