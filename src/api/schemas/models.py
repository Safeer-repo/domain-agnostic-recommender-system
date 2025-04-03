from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="ID of the user to get recommendations for")
    n: int = Field(10, description="Number of recommendations to return")
    domain: str = Field(..., description="Domain (entertainment, ecommerce, education)")
    dataset: str = Field(..., description="Dataset name")
    model_id: Optional[str] = Field(None, description="Specific model to use (defaults to best available)")

class SimilarItemsRequest(BaseModel):
    item_id: int = Field(..., description="ID of the item to find similar items for")
    n: int = Field(10, description="Number of similar items to return")
    domain: str = Field(..., description="Domain (entertainment, ecommerce, education)")
    dataset: str = Field(..., description="Dataset name")
    model_id: Optional[str] = Field(None, description="Specific model to use (defaults to best available)")

class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="ID of the user")
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommendations")
    model_id: str = Field(..., description="ID of the model used")
    domain: str = Field(..., description="Domain used")
    dataset: str = Field(..., description="Dataset used")

class SimilarItemsResponse(BaseModel):
    item_id: int = Field(..., description="ID of the item")
    similar_items: List[Dict[str, Any]] = Field(..., description="List of similar items")
    model_id: str = Field(..., description="ID of the model used")
    domain: str = Field(..., description="Domain used")
    dataset: str = Field(..., description="Dataset used")

class ModelInfo(BaseModel):
    model_id: str
    domain: str
    dataset: str
    metrics: Dict[str, float] = Field({}, description="Performance metrics")
    parameters: Dict[str, Any] = Field({}, description="Model parameters")

class ModelsResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="List of available models")
