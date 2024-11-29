"""
Time safety utilities to prevent look-ahead bias in data processing and modeling.
"""
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
import pandas as pd

class TimeSafetyManager:
    def __init__(self, reference_time: Optional[datetime] = None):
        """
        Initialize time safety manager.
        
        Args:
            reference_time: The current time to use as reference. If None, uses actual current time.
        """
        self.reference_time = reference_time or datetime.now()
        
        # Define data availability delays (in hours)
        self.data_delays = {
            'tsa': 24,  # TSA data available next day
            'weather': 1,  # Weather data available within hour
            'airlines': 48,  # Airline pricing data 2-day delay
            'hotels': 72,  # Hotel data 3-day delay
            'economic': 168,  # Economic data weekly delay
        }
    
    def is_data_available(self, data_type: str, data_timestamp: datetime) -> bool:
        """Check if specific data would have been available at reference_time."""
        if data_type not in self.data_delays:
            raise ValueError(f"Unknown data type: {data_type}")
            
        delay = timedelta(hours=self.data_delays[data_type])
        return data_timestamp + delay <= self.reference_time
    
    def get_available_data(self, df: pd.DataFrame, 
                          data_type: str,
                          timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Filter DataFrame to only include data that would have been available."""
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column {timestamp_col} not found in DataFrame")
            
        mask = df[timestamp_col].apply(lambda x: self.is_data_available(data_type, x))
        return df[mask].copy()
    
    def get_latest_available_timestamp(self, data_type: str) -> datetime:
        """Get the latest timestamp that would be available for a given data type."""
        delay = timedelta(hours=self.data_delays[data_type])
        return self.reference_time - delay

    def validate_feature_timeline(self, features_df: pd.DataFrame, 
                                feature_timestamps: Dict[str, str]) -> bool:
        """
        Validate that features are using appropriate historical data.
        
        Args:
            features_df: DataFrame containing features
            feature_timestamps: Dict mapping feature names to their timestamp columns
        
        Returns:
            bool: Whether all features respect time constraints
        """
        for feature, timestamp_col in feature_timestamps.items():
            if not all(self.is_data_available(feature, ts) 
                      for ts in features_df[timestamp_col]):
                return False
        return True