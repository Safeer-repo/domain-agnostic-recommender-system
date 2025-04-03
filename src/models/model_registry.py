import logging
from typing import Dict, List, Type, Any, Optional
import os
import json

from src.models.base_model import BaseModel
from src.models.algorithms.als_model import ALSModel

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for all recommender models.
    Manages model initialization, discovery, and metadata.
    """
    
    def __init__(self, artifacts_dir: str):
        """
        Initialize the model registry.
        
        Args:
            artifacts_dir: Directory to store model artifacts
        """
        self.artifacts_dir = artifacts_dir
        self.models: Dict[str, Type[BaseModel]] = {}
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """Register all available models."""
        self.register_model("als", ALSModel)
        # Add additional models as they are implemented
        # self.register_model("ncf", NCFModel)
        # self.register_model("svd", SVDModel)
    
    def register_model(self, model_id: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class with the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_class: Model class (must inherit from BaseModel)
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class.__name__} must inherit from BaseModel")
        
        self.models[model_id] = model_class
        logger.info(f"Registered model: {model_id}")
    
    def get_model(self, model_id: str) -> BaseModel:
        """
        Get a new instance of a model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Instance of the requested model
        """
        if model_id not in self.models:
            raise ValueError(f"Unknown model: {model_id}. Available models: {list(self.models.keys())}")
        
        return self.models[model_id](model_name=model_id)
    
    def get_available_models(self) -> List[str]:
        """Get a list of all available model IDs."""
        return list(self.models.keys())
    
    def save_model(self, model: BaseModel, domain: str, dataset_name: str) -> str:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            domain: Domain name
            dataset_name: Dataset name
            
        Returns:
            Path to the saved model
        """
        if not model.fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Construct the path
        model_dir = os.path.join(self.artifacts_dir, "models", domain, dataset_name)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model.model_name}.pkl")
        model.save(model_path)
        
        # Save metadata separately for easier access
        metadata_path = os.path.join(model_dir, f"{model.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(model.metadata, f, indent=2)
        
        return model_path
    
    def load_model(self, model_id: str, domain: str, dataset_name: str) -> BaseModel:
        """
        Load a model from disk.
        
        Args:
            model_id: Model identifier
            domain: Domain name
            dataset_name: Dataset name
            
        Returns:
            Loaded model
        """
        # Construct the path
        model_path = os.path.join(self.artifacts_dir, "models", domain, dataset_name, f"{model_id}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create a new instance of the model
        model = self.get_model(model_id)
        
        # Load the model
        model.load(model_path)
        
        return model
    
    def list_trained_models(self, domain: Optional[str] = None, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all trained models with their metadata.
        
        Args:
            domain: Optional domain filter
            dataset_name: Optional dataset filter
            
        Returns:
            List of model metadata dictionaries
        """
        models_dir = os.path.join(self.artifacts_dir, "models")
        if not os.path.exists(models_dir):
            return []
        
        metadata_files = []
        
        # Navigate directory structure based on filters
        if domain is not None:
            domain_dir = os.path.join(models_dir, domain)
            if not os.path.exists(domain_dir):
                return []
            
            if dataset_name is not None:
                dataset_dir = os.path.join(domain_dir, dataset_name)
                if not os.path.exists(dataset_dir):
                    return []
                
                # Find metadata files in the dataset directory
                for file in os.listdir(dataset_dir):
                    if file.endswith("_metadata.json"):
                        metadata_files.append(os.path.join(dataset_dir, file))
            else:
                # Find metadata files in all dataset directories under the domain
                for dataset in os.listdir(domain_dir):
                    dataset_dir = os.path.join(domain_dir, dataset)
                    if os.path.isdir(dataset_dir):
                        for file in os.listdir(dataset_dir):
                            if file.endswith("_metadata.json"):
                                metadata_files.append(os.path.join(dataset_dir, file))
        else:
            # Find all metadata files in the models directory
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if file.endswith("_metadata.json"):
                        metadata_files.append(os.path.join(root, file))
        
        # Load metadata from files
        metadata_list = []
        for file_path in metadata_files:
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    
                    # Add domain and dataset info to metadata
                    parts = file_path.split(os.sep)
                    metadata['domain'] = parts[-3] if len(parts) >= 3 else None
                    metadata['dataset'] = parts[-2] if len(parts) >= 2 else None
                    metadata['file_path'] = file_path
                    
                    metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading metadata from {file_path}: {str(e)}")
        
        return metadata_list
