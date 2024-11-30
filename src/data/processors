"""
Data processing utilities for TSA prediction project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes raw data into clean, analysis-ready format."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data processor.
        
        Args:
            data_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/processed")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_tsa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw TSA checkpoint data.
        
        Args:
            df: Raw TSA data
            
        Returns:
            pd.DataFrame: Processed TSA data
        """
        processed = df.copy()
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(processed['date']):
            processed['date'] = pd.to_datetime(processed['date'])
            
        # Sort by date
        processed = processed.sort_values('date')
        
        # Calculate year-over-year growth
        processed['yoy_growth'] = (
            (processed['passengers'] - processed['passengers_prior_year']) 
            / processed['passengers_prior_year']
        )
        
        # Calculate 7-day moving average
        processed['passengers_7d_ma'] = processed['passengers'].rolling(7).mean()
        
        return processed
        
    def process_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw weather data.
        
        Args:
            df: Raw weather data
            
        Returns:
            pd.DataFrame: Processed weather data
        """
        processed = df.copy()
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(processed['date']):
            processed['date'] = pd.to_datetime(processed['date'])
            
        # Handle missing values
        processed = processed.fillna({
            'temperature': processed.groupby('airport')['temperature'].transform('mean'),
            'precipitation': 0,
            'snow': 0,
            'wind_speed': processed.groupby('airport')['wind_speed'].transform('mean')
        })
        
        # Calculate airport-specific metrics
        processed['extreme_weather'] = (
            (processed['precipitation'] > processed.groupby('airport')['precipitation'].transform('quantile', 0.95)) |
            (processed['snow'] > 0) |
            (processed['wind_speed'] > processed.groupby('airport')['wind_speed'].transform('quantile', 0.95))
        )
        
        return processed
        
    def process_airline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw airline data.
        
        Args:
            df: Raw airline data
            
        Returns:
            pd.DataFrame: Processed airline data
        """
        processed = df.copy()
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(processed['date']):
            processed['date'] = pd.to_datetime(processed['date'])
            
        # Calculate route-specific metrics
        processed['price_zscore'] = (
            processed.groupby('route')['price']
            .transform(lambda x: (x - x.mean()) / x.std())
        )
        
        # Calculate capacity utilization
        processed['capacity_utilization'] = (
            processed.groupby('route')['capacity']
            .transform(lambda x: x / x.max())
        )
        
        return processed
        
    def save_processed_data(self, data: Dict[str, pd.DataFrame], 
                          timestamp: Optional[datetime] = None):
        """
        Save processed data with timestamp.
        
        Args:
            data: Dictionary of processed DataFrames
            timestamp: Timestamp for the data snapshot
        """
        timestamp = timestamp or datetime.now()
        snapshot_dir = self.data_dir / "snapshots" / timestamp.strftime("%Y%m%d_%H%M%S")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            file_path = snapshot_dir / f"{name}.parquet"
            df.to_parquet(file_path)
            logger.info(f"Saved processed {name} data to {file_path}")
            
    def load_latest_snapshot(self) -> Dict[str, pd.DataFrame]:
        """
        Load the most recent data snapshot.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of processed DataFrames
        """
        snapshot_dir = self.data_dir / "snapshots"
        if not snapshot_dir.exists():
            raise FileNotFoundError("No snapshots directory found")
            
        # Get latest snapshot directory
        latest_snapshot = max(snapshot_dir.glob("*"), key=lambda x: x.name)
        
        data = {}
        for file_path in latest_snapshot.glob("*.parquet"):
            name = file_path.stem
            data[name] = pd.read_parquet(file_path)
            
        return data

if __name__ == "__main__":
    # Example usage
    from collectors import DataCollector
    from datetime import timedelta
    
    # Collect data
    collector = DataCollector()
    start_date = datetime.now() - timedelta(days=30)
    raw_data = collector.collect_all_data(start_date)
    
    # Process data
    processor = DataProcessor()
    processed_data = {
        'tsa': processor.process_tsa_data(raw_data['tsa']),
        'weather': processor.process_weather_data(raw_data['weather']),
        'airline': processor.process_airline_data(raw_data['airline'])
    }
    
    # Save processed data
    processor.save_processed_data(processed_data)