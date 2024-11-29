"""
Configuration management for the TSA prediction project.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data collection and processing configuration."""
    weather_api_key: str
    airline_api_key: str
    tsa_data_url: str
    data_directory: Path
    snapshot_directory: Path

@dataclass
class ModelConfig:
    """Model configuration."""
    prediction_horizon: int  # days
    feature_lookback: int   # days
    train_start_date: str
    test_start_date: str
    random_seed: int
    
@dataclass
class TradingConfig:
    """Trading and risk management configuration."""
    kalshi_api_key: str
    max_position_size: float
    risk_limit: float
    min_edge: float

class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path or Path("configs/config.yaml")
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Initialize config dataclasses
        self.data = DataConfig(**config['data'])
        self.model = ModelConfig(**config['model'])
        self.trading = TradingConfig(**config['trading'])
        
        # Additional configuration
        self.logging = config.get('logging', {})
        self.backtesting = config.get('backtesting', {})
        
    def save_config(self, path: Optional[str] = None):
        """
        Save current configuration to YAML file.
        
        Args:
            path: Path to save configuration file
        """
        path = path or self.config_path
        
        config = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'trading': self.trading.__dict__,
            'logging': self.logging,
            'backtesting': self.backtesting
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f)
            
    def update_config(self, section: str, updates: Dict[str, Any]):
        """
        Update configuration section.
        
        Args:
            section: Configuration section to update
            updates: Dictionary of updates to apply
        """
        if not hasattr(self, section):
            raise ValueError(f"Invalid config section: {section}")
            
        current = getattr(self, section)
        for key, value in updates.items():
            if not hasattr(current, key):
                raise ValueError(f"Invalid config key: {key} in section {section}")
            setattr(current, key, value)