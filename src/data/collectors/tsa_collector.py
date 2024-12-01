"""Collector for TSA checkpoint travel numbers. Handles raw data collection only."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class TSACollector(BaseCollector):
    """Collector for raw TSA checkpoint travel numbers."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize TSA data collector."""
        super().__init__(data_dir)
        self.base_url = "https://www.tsa.gov/travel/passenger-volumes"
        self.session = requests.Session()
        self.data_dir = self.data_dir / "tsa"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(self,
                  start_date: datetime,
                  end_date: Optional[datetime] = None,
                  use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch raw TSA checkpoint data for date range.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection (defaults to yesterday)
            use_cache: Whether to use cached data if available
            
        Returns:
            pd.DataFrame with columns:
                - date (index): Date of checkpoint data
                - current_year: Current year passenger count
                - previous_year: Previous year passenger count
                - previous_2_years: Two years ago passenger count
        """
        end_date = end_date or datetime.now() - timedelta(days=1)
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        try:
            # Check cache first if requested
            if use_cache:
                cached_data = self._load_from_cache(start_date, end_date)
                if cached_data is not None:
                    return cached_data
            
            # Fetch data for each year needed
            years_needed = range(start_date.year, end_date.year + 1)
            all_data = []
            
            for year in years_needed:
                df = self._fetch_year_data(year)
                all_data.append(df)
                
                # Cache the year's data
                cache_file = self.data_dir / f"tsa_data_{year}.parquet"
                df.to_parquet(cache_file)
                
                time.sleep(1)  # Be nice to TSA's server
            
            # Combine and filter to requested date range
            combined_df = pd.concat(all_data)
            mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
            return combined_df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error fetching TSA data: {str(e)}")
            raise
    
    def _fetch_year_data(self, year: int) -> pd.DataFrame:
        """Fetch TSA checkpoint data for a specific year."""
        url = f"{self.base_url}/{year}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse the webpage
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')
            
            if not table:
                raise ValueError(f"No data table found for year {year}")
            
            # Parse table into DataFrame
            df = pd.read_html(str(table))[0]
            
            # Clean up column names
            df.columns = ['date', 'current_year', 'previous_year', 'previous_2_years']
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert passenger numbers to integers
            for col in ['current_year', 'previous_year', 'previous_2_years']:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            
            # Set date as index
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {year} data: {str(e)}")
            raise
    
    def _load_from_cache(self, start_date: datetime, 
                        end_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        try:
            years_needed = range(start_date.year, end_date.year + 1)
            all_data = []
            
            for year in years_needed:
                cache_file = self.data_dir / f"tsa_data_{year}.parquet"
                if not cache_file.exists():
                    return None
                
                # For current year, check if cache is too old
                if year == datetime.now().year:
                    cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if datetime.now() - cache_time > timedelta(days=1):
                        return None
                
                df = pd.read_parquet(cache_file)
                all_data.append(df)
            
            combined_df = pd.concat(all_data)
            mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
            return combined_df[mask].copy()
            
        except Exception:
            return None

if __name__ == "__main__":
    # Example usage
    collector = TSACollector()
    
    try:
        # Fetch last 30 days of data
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        
        data = collector.fetch_data(start_date, end_date)
        print("\nRaw TSA Checkpoint Data:")
        print(data.head())
        print("\nShape:", data.shape)
        print("\nColumns:", data.columns.tolist())
        
    except Exception as e:
        print(f"Error: {str(e)}")