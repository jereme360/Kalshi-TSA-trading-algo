"""
Base model class for TSA prediction models.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import joblib
from datetime import datetime
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all TSA prediction models."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            name: Model name/identifier
            config: Model configuration dictionary
        """
        self.name = name
        self.config = config
        self.model = None
        self.feature_importance: Optional[pd.Series] = None
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    def save(self, timestamp: Optional[datetime] = None) -> Path:
        """
        Save model and metadata.
        
        Args:
            timestamp: Optional timestamp for versioning
            
        Returns:
            Path: Path where model was saved
        """
        timestamp = timestamp or datetime.now()
        save_dir = self.model_dir / f"{self.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / "model.joblib"
        joblib.dump(self.model, model_path)
        
        # Save feature importance if available
        if self.feature_importance is not None:
            importance_path = save_dir / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path)
        
        # Save metadata
        metadata = {
            "name": self.name,
            "timestamp": timestamp.isoformat(),
            "config": self.config,
            "feature_names": self.feature_names if hasattr(self, 'feature_names') else None
        }
        
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved model to {save_dir}")
        return save_dir
    
    def load(self, model_path: Path) -> None:
        """
        Load model and metadata.
        
        Args:
            model_path: Path to model directory
        """
        # Load model
        self.model = joblib.load(model_path / "model.joblib")
        
        # Load feature importance if available
        importance_path = model_path / "feature_importance.csv"
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path, index_col=0).squeeze()
        
        # Load metadata
        metadata_path = model_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.config.update(metadata.get('config', {}))
            if 'feature_names' in metadata:
                self.feature_names = metadata['feature_names']
                
        logger.info(f"Loaded model from {model_path}")
    
    def validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate feature DataFrame.
        
        Args:
            X: Feature DataFrame to validate
            
        Raises:
            ValueError: If features are invalid
        """
        if hasattr(self, 'feature_names'):
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores if available.
        
        Returns:
            Optional[pd.Series]: Feature importance scores
        """
        return self.feature_importance
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True values
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        from src.utils.metrics import calculate_prediction_metrics
        
        predictions = self.predict(X)
        metrics = calculate_prediction_metrics(y.values, predictions)
        
        logger.info(f"Model {self.name} evaluation metrics: {metrics}")
        return metrics
    
    @staticmethod
    def load_latest(model_dir: Path, name: str) -> 'BaseModel':
        """
        Load the latest version of a model.
        
        Args:
            model_dir: Directory containing model versions
            name: Model name/identifier
            
        Returns:
            BaseModel: Loaded model instance
        """
        # Find latest model version
        model_versions = list(model_dir.glob(f"{name}_*"))
        if not model_versions:
            raise FileNotFoundError(f"No saved versions found for model {name}")
            
        latest_version = max(model_versions, key=lambda x: x.name)
        
        # Load metadata to determine model class
        metadata_path = latest_version / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Import and instantiate correct model class
        model_class = _get_model_class(metadata['config'].get('model_type', 'base'))
        model = model_class(name=name, config=metadata['config'])
        model.load(latest_version)
        
        return model

def _get_model_class(model_type: str) -> type:
    """Get model class by type string."""
    from src.models.predictors import SARIMAXModel, GBMModel
    
    model_classes = {
        'sarimax': SARIMAXModel,
        'gbm': GBMModel,
        # Add other model types as needed
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_classes[model_type]