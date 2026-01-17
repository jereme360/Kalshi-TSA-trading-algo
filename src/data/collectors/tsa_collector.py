"""Collector for TSA checkpoint travel numbers."""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path
from .base_collectors import BaseCollector

logger = logging.getLogger(__name__)


class TSACollector(BaseCollector):
    """Collector for TSA checkpoint passenger counts."""

    BASE_URL = "https://www.tsa.gov/travel/passenger-volumes"

    def __init__(self, data_dir: Optional[Path] = None):
        super().__init__(data_dir)
        self.data_dir = self.data_dir / "tsa"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, start_date: datetime, end_date: Optional[datetime] = None,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch TSA checkpoint data for date range.

        Returns DataFrame with 'passengers' column indexed by date.
        """
        end_date = end_date or datetime.now() - timedelta(days=1)

        if use_cache:
            cached = self._load_cache()
            if cached is not None and not cached.empty:
                # Check if cache covers the date range
                if cached.index.min() <= start_date and cached.index.max() >= end_date - timedelta(days=2):
                    return cached[(cached.index >= start_date) & (cached.index <= end_date)]

        # Fetch fresh data
        all_data = []
        current_year = datetime.now().year

        for year in range(start_date.year, end_date.year + 1):
            df = self._fetch_year(year, is_current=(year == current_year))
            if df is not None:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame(columns=['passengers'])

        combined = pd.concat(all_data).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]

        # Save to cache
        combined.to_parquet(self.data_dir / "tsa_data.parquet")

        return combined[(combined.index >= start_date) & (combined.index <= end_date)]

    def _fetch_year(self, year: int, is_current: bool = False) -> Optional[pd.DataFrame]:
        """Fetch data for a specific year."""
        # Current year has no suffix, previous years have /YYYY
        url = self.BASE_URL if is_current else f"{self.BASE_URL}/{year}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.find('table')
            if not table:
                logger.warning(f"No table found for year {year}")
                return None

            df = pd.read_html(str(table))[0]

            # Normalize column names (TSA uses 'Date' and 'Numbers')
            df.columns = ['date', 'passengers']
            df['date'] = pd.to_datetime(df['date'])
            df['passengers'] = pd.to_numeric(
                df['passengers'].astype(str).str.replace(',', ''),
                errors='coerce'
            )
            df = df.set_index('date').sort_index()

            return df

        except Exception as e:
            logger.error(f"Error fetching year {year}: {e}")
            return None

    def _load_cache(self) -> Optional[pd.DataFrame]:
        """Load cached data if fresh enough."""
        cache_file = self.data_dir / "tsa_data.parquet"
        if not cache_file.exists():
            return None

        # Check if cache is less than 1 day old
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age > timedelta(days=1):
            return None

        try:
            return pd.read_parquet(cache_file)
        except Exception:
            return None
