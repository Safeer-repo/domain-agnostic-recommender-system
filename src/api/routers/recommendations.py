from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import numpy as np

from src.api.schemas.models import (
    RecommendationRequest, 
    RecommendationResponse, 
    SimilarItemsRequest, 
    SimilarItemsResponse
)
from src.models.model_registry import ModelRegistry

router = APIRouter()

def get_model_registry():
    """Dependency for getting the model registry."""
    # In a production system, this would use a connection pool or similar
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

@router.post("/user", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Get personalized recommendations for a user.
    """
    try:
        # Load the appropriate model
        model_id = request.model_id or "als"  # Default to ALS if not specified
        model = registry.load_model(model_id, request.domain, request.dataset)
        
        # Get recommendations
        recommendations = model.predict(request.user_id, n=request.n)
        
        # Format recommendations and convert NumPy types to Python types
        formatted_recommendations = [
            {"item_id": convert_to_native_types(item_id), "score": convert_to_native_types(score)}
            for item_id, score in recommendations
        ]
        
        return {
            "user_id": convert_to_native_types(request.user_id),
            "recommendations": formatted_recommendations,
            "model_id": model_id,
            "domain": request.domain,
            "dataset": request.dataset
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for {request.domain}/{request.dataset}/{model_id}"
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"User {request.user_id} not found in the dataset"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@router.post("/similar", response_model=SimilarItemsResponse)
async def get_similar_items(
    request: SimilarItemsRequest,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Get items similar to a given item.
    """
    try:
        # Load the appropriate model
        model_id = request.model_id or "als"  # Default to ALS if not specified
        model = registry.load_model(model_id, request.domain, request.dataset)
        
        # Get similar items
        similar_items = model.get_similar_items(request.item_id, n=request.n)
        
        # Format similar items and convert NumPy types to Python types
        formatted_items = [
            {"item_id": convert_to_native_types(item_id), "score": convert_to_native_types(score)}
            for item_id, score in similar_items
        ]
        
        return {
            "item_id": convert_to_native_types(request.item_id),
            "similar_items": formatted_items,
            "model_id": model_id,
            "domain": request.domain,
            "dataset": request.dataset
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for {request.domain}/{request.dataset}/{model_id}"
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Item {request.item_id} not found in the dataset"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar items: {str(e)}"
        )
