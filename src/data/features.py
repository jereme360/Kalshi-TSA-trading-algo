"""
Feature engineering for TSA prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import holidays
import logging
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineers features for TSA prediction model."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize feature engineer.
        
        Args:
            data_dir: Directory to save feature data
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/features")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.us_holidays = holidays.US()
        
    def _calculate_days_to_holiday(self, dates: pd.Series) -> pd.Series:
        """Calculate number of days to next holiday."""
        days_to_holiday = []
        
        for date in dates:
            next_holiday = min((h for h in self.us_holidays 
                              if h > date.date()), default=None)
            days = (next_holiday - date.date()).days if next_holiday else 365
            days_to_holiday.append(days)
            
        return pd.Series(days_to_holiday, index=dates.index)
        
    def create_calendar_features(self, df: pd.DataFrame, 
                               date_column: str = 'date') -> pd.DataFrame:
        """
        Create calendar-based features.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            
        Returns:
            pd.DataFrame: DataFrame with calendar features
        """
        features = pd.DataFrame(index=df.index)
        dates = df[date_column]
        
        # Basic date features
        features['day_of_week'] = dates.dt.dayofweek
        features['day_of_month'] = dates.dt.day
        features['month'] = dates.dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Holiday features
        features['is_holiday'] = dates.apply(lambda x: x in self.us_holidays).astype(int)
        features['days_to_holiday'] = self._calculate_days_to_holiday(dates)
        
        # Seasonal features
        features['is_summer'] = dates.dt.month.isin([6, 7, 8]).astype(int)
        features['is_winter'] = dates.dt.month.isin([12, 1, 2]).astype(int)
        features['is_shoulder_season'] = (~features['is_summer'] & ~features['is_winter']).astype(int)
        
        return features
        
    def create_weather_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-based features.
        
        Args:
            weather_df: Weather data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with weather features
        """
        features = pd.DataFrame(index=weather_df.index)
        
        # Aggregate across airports
        airport_groups = weather_df.groupby(['date'])
        
        # Weather severity features
        features['severe_weather_airports'] = (
            airport_groups['extreme_weather'].sum()
        )
        features['avg_temperature'] = airport_groups['temperature'].mean()
        features['max_wind_speed'] = airport_groups['wind_speed'].max()
        features['total_precipitation'] = airport_groups['precipitation'].sum()
        
        # Create weather severity index
        features['weather_severity_index'] = (
            features['severe_weather_airports'] * 0.4 +
            (features['max_wind_speed'] / features['max_wind_speed'].max()) * 0.3 +
            (features['total_precipitation'] / features['total_precipitation'].max()) * 0.3
        )
        
        return features
        
    def create_airline_features(self, airline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create airline-based features.
        
        Args:
            airline_df: Airline data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with airline features
        """
        features = pd.DataFrame(index=airline_df.index)
        
        # Aggregate across routes
        route_groups = airline_df.groupby(['date'])
        
        # Price features
        features['avg_price'] = route_groups['price'].mean()
        features['price_volatility'] = route_groups['price_zscore'].std()
        
        # Capacity features
        features['total_capacity'] = route_groups['capacity'].sum()
        features['avg_capacity_utilization'] = route_groups['capacity_utilization'].mean()
        
        return features
        
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str],
                           lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        features = pd.DataFrame(index=df.index)
        
        for col in columns:
            for lag in lags:
                features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return features
        
    def create_all_features(self, data: Dict[str, pd.DataFrame], 
                          timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """
        Create all features from processed data.
        
        Args:
            data: Dictionary of processed DataFrames
            timestamp: Timestamp for the feature snapshot
            
        Returns:
            pd.DataFrame: Combined feature DataFrame
        """
        # Create individual feature sets
        calendar_features = self.create_calendar_features(data['tsa'])
        weather_features = self.create_weather_features(data['weather'])
        airline_features = self.create_airline_features(data['airline'])
        
        # Create lag features for TSA data
        lag_features = self.create_lag_features(
            data['tsa'],
            columns=['passengers'],
            lags=[1, 7, 14, 28]  # Day-of-week and monthly patterns
        )
        
        # Combine all features
        features = pd.concat([
            calendar_features,
            weather_features,
            airline_features,
            lag_features
        ], axis=1)
        
        # Save feature snapshot
        if timestamp:
            self.save_feature_snapshot(features, timestamp)
        
        return features
        
    def save_feature_snapshot(self, features: pd.DataFrame, 
                            timestamp: datetime):
        """Save feature snapshot."""
        snapshot_dir = self.data_dir / "snapshots" / timestamp.strftime("%Y%m%d_%H%M%S")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = snapshot_dir / "features.parquet"
        features.to_parquet(file_path)
        logger.info(f"Saved feature snapshot to {file_path}")
        
    def load_latest_features(self) -> pd.DataFrame:
        """Load the most recent feature snapshot."""
        snapshot_dir = self.data_dir / "snapshots"
        if not snapshot_dir.exists():
            raise FileNotFoundError("No snapshots directory found")
            
        latest_snapshot = max(snapshot_dir.glob("*"), key=lambda x: x.name)
        return pd.read_parquet(latest_snapshot / "features.parquet")

if __name__ == "__main__":
    # Example usage
    from processor import DataProcessor
    
    # Load processed data
    processor = DataProcessor()
    data = processor.load_latest_snapshot()
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_all_features(data)
    print("Features created successfully:", features.columns.tolist())