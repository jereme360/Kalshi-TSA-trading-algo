"""
Data collectors for TSA prediction project.
Collects TSA checkpoint numbers and economic indicators from FRED.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from bs4 import BeautifulSoup
import time
from pathlib import Path
from fredapi import Fred
import os

logger = logging.getLogger(__name__)

class TSADataCollector:
    """Collector for TSA checkpoint travel numbers."""
    
    def __init__(self):
        """Initialize TSA data collector."""
        self.base_url = "https://www.tsa.gov/travel/passenger-volumes"
        self.session = requests.Session()
        
        # Create data directory if it doesn't exist
        self.data_dir = Path("data/raw/tsa")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _fetch_year_data(self, year: int) -> pd.DataFrame:
        """
        Fetch TSA checkpoint data for a specific year.
        
        Args:
            year: Year to fetch data for
            
        Returns:
            pd.DataFrame: TSA checkpoint data for the year
        """
        url = f"{self.base_url}/{year}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse the webpage
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the data table
            table = soup.find('table')
            if not table:
                raise ValueError(f"No data table found for year {year}")
            
            # Parse table into DataFrame
            df = pd.read_html(str(table))[0]
            
            # Clean up column names and data
            df.columns = ['date', 'current_year', 'previous_year', 'previous_2_years']
            
            # Convert date strings to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert passenger numbers (handling commas in numbers)
            numeric_cols = ['current_year', 'previous_year', 'previous_2_years']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            logger.info(f"Successfully fetched TSA data for {year}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching TSA data for {year}: {str(e)}")
            raise
    
    def fetch_historical_data(self, start_date: datetime, 
                            end_date: Optional[datetime] = None,
                            use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical TSA checkpoint data for date range.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection (defaults to yesterday)
            use_cache: Whether to use cached data if available
            
        Returns:
            pd.DataFrame: TSA checkpoint data with columns:
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
            
            # Determine which years we need to fetch
            years_needed = range(start_date.year, end_date.year + 1)
            
            # Fetch data for each year
            all_data = []
            for year in years_needed:
                df = self._fetch_year_data(year)
                all_data.append(df)
                
                # Cache the year's data
                self._save_to_cache(df, year)
                
                # Add delay between requests
                time.sleep(1)
            
            # Combine all years
            if not all_data:
                raise ValueError("No data collected")
                
            combined_df = pd.concat(all_data)
            
            # Filter to requested date range
            mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
            return combined_df[mask].copy()
            
        except Exception as e:
            logger.error(f"Error fetching historical TSA data: {str(e)}")
            raise
    
    def _save_to_cache(self, df: pd.DataFrame, year: int):
        """Save data to cache."""
        cache_file = self.data_dir / f"tsa_data_{year}.parquet"
        df.to_parquet(cache_file)
        
    def _load_from_cache(self, start_date: datetime, 
                        end_date: datetime) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not too old."""
        try:
            years_needed = range(start_date.year, end_date.year + 1)
            all_data = []
            
            for year in years_needed:
                cache_file = self.data_dir / f"tsa_data_{year}.parquet"
                if not cache_file.exists():
                    return None
                
                # Check if cache is too old (more than 1 day for current year)
                if year == datetime.now().year:
                    cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if datetime.now() - cache_time > timedelta(days=1):
                        return None
                
                df = pd.read_parquet(cache_file)
                all_data.append(df)
            
            if not all_data:
                return None
                
            combined_df = pd.concat(all_data)
            mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
            return combined_df[mask].copy()
            
        except Exception:
            return None

class FREDDataCollector:
    """Collector for economic data from FRED (Federal Reserve Economic Data)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data collector.
        
        Args:
            api_key: FRED API key. If None, looks for FRED_API_KEY in environment.
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key not found. Set FRED_API_KEY environment variable.")
            
        self.fred = Fred(api_key=self.api_key)
        self.data_dir = Path("data/raw/fred")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define series of interest
        self.series_ids = {
            'UNEMPLOYMENT': 'UNRATE',           # Unemployment Rate
            'CPI': 'CPIAUCSL',                 # Consumer Price Index
            'DISPOSABLE_INCOME': 'DSPIC96',    # Real Disposable Personal Income
            'CONSUMER_SENTIMENT': 'UMCSENT',    # Consumer Sentiment Index
            'AIR_REVENUE': 'A563RC1Q027SBEA',  # Air Transportation Revenue
            'RETAIL_SALES': 'RSXFS',           # Retail Sales
            'GDP': 'GDP',                      # Gross Domestic Product
            'JET_FUEL': 'POJETAUSDM',          # Jet Fuel Prices
        }
    
    def fetch_series(self, 
                    series_id: str,
                    start_date: datetime,
                    end_date: Optional[datetime] = None,
                    use_cache: bool = True) -> pd.Series:
        """
        Fetch a single series from FRED.
        """
        cache_file = self.data_dir / f"{series_id}.parquet"
        
        # Check cache first if requested
        if use_cache and cache_file.exists():
            cached_data = pd.read_parquet(cache_file)
            if cached_data.index.max() >= (end_date or datetime.now()):
                mask = (cached_data.index >= start_date) & (cached_data.index <= (end_date or datetime.now()))
                return cached_data[mask]
        
        try:
            # Fetch data from FRED
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            
            # Cache the data
            if use_cache:
                data.to_parquet(cache_file)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {str(e)}")
            raise
    
    def fetch_all_series(self,
                        start_date: datetime,
                        end_date: Optional[datetime] = None,
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch all configured economic series.
        """
        all_series = {}
        
        for name, series_id in self.series_ids.items():
            try:
                series = self.fetch_series(series_id, start_date, end_date, use_cache)
                all_series[name] = series
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({series_id}): {str(e)}")
                continue
        
        # Combine all series into a DataFrame
        df = pd.DataFrame(all_series)
        
        # Forward fill missing values (FRED series have different frequencies)
        df = df.fillna(method='ffill')
        
        return df

class DataCollector:
    """Main data collection coordinator."""
    
    def __init__(self):
        """Initialize data collectors."""
        self.tsa_collector = TSADataCollector()
        self.fred_collector = FREDDataCollector()
    
    def collect_all_data(self, start_date: datetime,
                        end_date: Optional[datetime] = None,
                        use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect all available data for the model.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            use_cache: Whether to use cached data if available
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of collected datasets
        """
        end_date = end_date or datetime.now() - timedelta(days=1)
        
        try:
            data = {
                'tsa': self.tsa_collector.fetch_historical_data(
                    start_date, end_date, use_cache=use_cache
                ),
                'economic': self.fred_collector.fetch_all_series(
                    start_date, end_date, use_cache=use_cache
                )
            }
            
            logger.info(f"Successfully collected data from {start_date} to {end_date}")
            return data
            
        except Exception as e:
            logger.error(f"Error in collect_all_data: {str(e)}")
            raise

def create_sample_data(days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    Create sample data for testing when real data collection not needed.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create sample TSA data
    tsa_data = pd.DataFrame(index=dates)
    
    # Add some seasonal and weekly patterns
    t = np.arange(len(dates))
    base_traffic = 2_000_000  # Base daily travelers
    
    # Weekly pattern (more on weekends)
    weekly = np.sin(2 * np.pi * t / 7) * 200_000
    
    # Yearly pattern (more in summer and holidays)
    yearly = np.sin(2 * np.pi * t / 365) * 500_000
    
    # Trend (growing over time)
    trend = t * 1000
    
    # Combine patterns and add noise
    current_year = base_traffic + weekly + yearly + trend + np.random.normal(0, 50_000, len(t))
    previous_year = current_year * 0.9 + np.random.normal(0, 50_000, len(t))
    previous_2_years = previous_year * 0.8 + np.random.normal(0, 50_000, len(t))
    
    tsa_data['current_year'] = current_year
    tsa_data['previous_year'] = previous_year
    tsa_data['previous_2_years'] = previous_2_years
    
    # Create sample economic data
    economic_data = pd.DataFrame(index=dates)
    economic_data['UNEMPLOYMENT'] = 5.5 + np.random.normal(0, 0.1, len(dates))
    economic_data['CPI'] = 200 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
    economic_data['CONSUMER_SENTIMENT'] = 80 + np.random.normal(0, 2, len(dates))
    
    return {
        'tsa': tsa_data,
        'economic': economic_data
    }

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Fetch last 30 days of data
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    
    try:
        # Real data collection
        print("\nFetching real data...")
        data = collector.collect_all_data(start_date, end_date)
        print("\nTSA Checkpoint Data:")
        print(data['tsa'].head())
        print("\nEconomic Data:")
        print(data['economic'].head())
        
        # Sample data generation
        print("\nGenerating sample data...")
        sample_data = create_sample_data(days=30)
        print("\nSample TSA Data:")
        print(sample_data['tsa'].head())
        print("\nSample Economic Data:")
        print(sample_data['economic'].head())
        
    except Exception as e:
        print(f"Error: {str(e)}")