# data/collectors/base_collector.py
"""Base class for all data collectors."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import logging
import pickle
import pandas as pd

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Abstract base class for all data collectors."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize base collector.
        
        Args:
            data_dir: Directory for storing data. If None, uses default.
        """
        self.data_dir = data_dir or Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fetch_data(self,
                  start_date: datetime,
                  end_date: Optional[datetime] = None,
                  use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch data for the given date range.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            use_cache: Whether to use cached data
            
        Returns:
            Dict containing collected data
        """
        pass
    
    def _save_to_cache(self, data: Any, cache_file: Path) -> None:
        """Save data to cache."""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, pd.DataFrame):
                data.to_parquet(cache_file)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    def _load_from_cache(self, cache_file: Path) -> Optional[Any]:
        """Load data from cache if available."""
        try:
            if not cache_file.exists():
                return None
                
            if cache_file.suffix == '.parquet':
                return pd.read_parquet(cache_file)
            else:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return None