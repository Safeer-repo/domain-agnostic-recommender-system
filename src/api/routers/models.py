from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import numpy as np

from src.api.schemas.models import ModelInfo, ModelsResponse
from src.models.model_registry import ModelRegistry

router = APIRouter()

def get_model_registry():
    """Dependency for getting the model registry."""
    artifacts_dir = "./artifacts"  # Adjust path as needed
    return ModelRegistry(artifacts_dir=artifacts_dir)

# Helper function to convert NumPy types to native Python types
def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(i) for i in obj)
    else:
        return obj

@router.get("/", response_model=ModelsResponse)
async def list_models(
    domain: Optional[str] = None,
    dataset: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    List available trained models.
    """
    try:
        # Get trained models
        trained_models = registry.list_trained_models(domain, dataset)
        
        # Format model information and convert NumPy types
        models_info = []
        for model_data in trained_models:
            # Convert NumPy types in metrics and parameters
            metrics = convert_to_native_types(model_data.get("performance_metrics", {}))
            parameters = convert_to_native_types(model_data.get("hyperparameters", {}))
            
            model_info = ModelInfo(
                model_id=model_data.get("model_name", "unknown"),
                domain=model_data.get("domain", "unknown"),
                dataset=model_data.get("dataset", "unknown"),
                metrics=metrics,
                parameters=parameters
            )
            models_info.append(model_info)
        
        return {"models": models_info}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )

@router.get("/{model_id}/{domain}/{dataset}", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    domain: str,
    dataset: str,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Get information about a specific model.
    """
    try:
        # Load the model
        model = registry.load_model(model_id, domain, dataset)
        
        # Get model metadata and convert NumPy types
        metadata = model.metadata
        metrics = convert_to_native_types(metadata.get("performance", {}))
        parameters = convert_to_native_types(metadata.get("hyperparameters", {}))
        
        return ModelInfo(
            model_id=model_id,
            domain=domain,
            dataset=dataset,
            metrics=metrics,
            parameters=parameters
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_id}/{domain}/{dataset}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )
