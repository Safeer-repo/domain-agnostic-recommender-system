from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body, Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Any, Optional, Union
import os
import time
import json
import logging
from datetime import datetime, timedelta
import uuid
import jwt
from passlib.context import CryptContext

from src.utils.ratings_storage import RatingsStorage
from src.utils.user_manager import UserManager
from src.utils.dataset_updater import DatasetUpdater
from src.utils.model_retraining_manager import ModelRetrainingManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-jwt")  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Create FastAPI app
app = FastAPI(
    title="Domain-Agnostic Recommender API",
    description="API for a recommender system that works across multiple domains",
    version="1.0.0"
)

# Initialize components
data_dir = os.getenv("DATA_DIR", os.path.abspath("./data"))
ratings_storage = RatingsStorage(data_dir=data_dir)
user_manager = UserManager(data_dir=data_dir)
dataset_updater = DatasetUpdater(data_dir=data_dir, ratings_storage=ratings_storage)
# In src/api/app.py
model_manager = ModelRetrainingManager(data_dir=data_dir, models_dir="./artifacts/models")

# Authentication utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models for request/response
class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    domain_preferences: Optional[Dict[str, List[str]]] = None

class UserLogin(BaseModel):
    username: str
    password: str

class Rating(BaseModel):
    user_id: str
    item_id: int
    rating: float
    source: Optional[str] = "explicit"

class BatchRatings(BaseModel):
    domain: str
    dataset: str
    ratings: List[Rating]

class RecommendationFeedback(BaseModel):
    user_id: str
    item_id: int
    interaction_type: str  # click, add_to_cart, purchase, etc.
    timestamp: Optional[int] = None
    
class RecommendationRequest(BaseModel):
    user_id: str
    domain: str
    dataset: str
    count: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None

class SimilarItemsRequest(BaseModel):
    item_id: int
    domain: str
    dataset: str
    count: Optional[int] = 10

class UpdateDatasetRequest(BaseModel):
    domain: str
    dataset: str
    incremental: Optional[bool] = True

class RetrainRequest(BaseModel):
    domain: str
    dataset: str
    algorithm: Optional[str] = None
    force: Optional[bool] = False
    
# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = user_manager.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

async def get_current_admin(user: dict = Depends(get_current_user)):
    # You would typically check a role field in the user data
    # For simplicity, we're checking a metadata field
    if user.get("metadata", {}).get("is_admin", False) is not True:
        raise HTTPException(status_code=403, detail="Not authorized")
    return user

# API Endpoints

# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Find user by username
    user_id = user_manager.get_user_id(form_data.username)
    if not user_id:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Check password
    stored_password = user.get("metadata", {}).get("password")
    if not stored_password or not verify_password(form_data.password, stored_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id, "username": user.get("username")},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "username": user.get("username")
    }

# User endpoints
@app.post("/user/create", response_model=Dict[str, Any])
async def create_user(user_data: UserCreate):
    """Create a new user account"""
    # Check if user already exists
    if user_manager.user_exists(user_data.username) or user_manager.user_exists(user_data.email):
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    # Hash the password
    hashed_password = get_password_hash(user_data.password)
    
    # Create the user
    user_id = user_manager.create_user(
        username=user_data.username,
        email=user_data.email,
        domain_preferences=user_data.domain_preferences
    )

    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")

    # Add password to user metadata
    user = user_manager.get_user(user_id)

    # Set admin privileges if username is admin
    if user_data.username == "admin":
        metadata = user.get("metadata", {})
        metadata["is_admin"] = True
        user_manager.update_user(user_id, {"metadata": metadata})
        logger.info(f"Set admin privileges for user {user_id}")

    metadata = user.get("metadata", {})
    metadata["password"] = hashed_password
    user_manager.update_user(user_id, {"metadata": metadata})
    
    # Return user info (without password)
    user = user_manager.get_user(user_id)
    if "metadata" in user and "password" in user["metadata"]:
        del user["metadata"]["password"]
    
    return user

@app.post("/user/login")
async def login_user(login_data: UserLogin):
    """Login a user and return an access token"""
    user_id = user_manager.get_user_id(login_data.username)
    if not user_id:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Check password
    stored_password = user.get("metadata", {}).get("password")
    if not stored_password or not verify_password(login_data.password, stored_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id, "username": user.get("username")},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "username": user.get("username")
    }

@app.get("/user/me", response_model=Dict[str, Any])
async def get_user_info(user: dict = Depends(get_current_user)):
    """Get the current user's profile information"""
    # Remove password from response
    if "metadata" in user and "password" in user["metadata"]:
        user = user.copy()  # Make a copy to avoid modifying the original
        del user["metadata"]["password"]
    
    return user

@app.post("/user/rate", response_model=Dict[str, Any])
async def submit_rating(rating: Rating, user: dict = Depends(get_current_user)):
    """Submit a new rating for an item"""
    # Verify user_id matches authenticated user
    if rating.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="Cannot submit ratings for other users")
    
    # Determine domain and dataset from user preferences
    # This would be more sophisticated in a real system
    domain = "entertainment"  # Default
    dataset = "movielens"     # Default
    
    # Store the rating
    success = ratings_storage.store_rating(
    domain=domain,
    dataset=dataset,
    user_id=rating.user_id,  # No int() conversion
    item_id=rating.item_id,
    rating=rating.rating,
    source=rating.source
)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store rating")
    
    return {
        "status": "success",
        "message": "Rating submitted successfully",
        "rating": rating.dict()
    }

@app.post("/user/rate/batch", response_model=Dict[str, Any])
async def submit_batch_ratings(batch: BatchRatings, user: dict = Depends(get_current_user)):
    """Submit multiple ratings at once"""
    # Verify all ratings are for the authenticated user
    for rating in batch.ratings:
        if rating.user_id != user["user_id"]:
            raise HTTPException(status_code=403, detail="Cannot submit ratings for other users")
    
    # Prepare ratings in the format expected by ratings_storage
    ratings_data = [
    {
        "user_id": r.user_id,  
        "item_id": r.item_id,
        "rating": r.rating,
        "source": r.source
    }
    for r in batch.ratings
]
    
    # Store the batch of ratings
    count = ratings_storage.store_ratings_batch(
        domain=batch.domain,
        dataset=batch.dataset,
        ratings=ratings_data
    )
    
    if count == 0:
        raise HTTPException(status_code=500, detail="Failed to store ratings")
    
    return {
        "status": "success",
        "message": f"Successfully stored {count} ratings",
        "processed_count": count
    }

@app.post("/user/feedback", response_model=Dict[str, Any])
async def submit_recommendation_feedback(feedback: RecommendationFeedback, user: dict = Depends(get_current_user)):
    """Record user interaction with a recommendation"""
    # Verify user_id matches authenticated user
    if feedback.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="Cannot submit feedback for other users")
    
    # Set timestamp if not provided
    if feedback.timestamp is None:
        feedback.timestamp = int(time.time())
    
    # Determine domain and dataset from user preferences
    # This would be more sophisticated in a real system
    domain = "entertainment"  # Default
    dataset = "movielens"     # Default
    
    # Translate interaction_type to ratings
    rating_value = None
    if feedback.interaction_type == "click":
        rating_value = 3.0
    elif feedback.interaction_type == "add_to_cart":
        rating_value = 4.0
    elif feedback.interaction_type == "purchase":
        rating_value = 5.0
    elif feedback.interaction_type == "ignore":
        rating_value = 2.0
    
    success = False
    if rating_value is not None:
        success = ratings_storage.store_rating(
            domain=domain,
            dataset=dataset,
            user_id=feedback.user_id,
            item_id=feedback.item_id,
            rating=rating_value,  # Use the calculated rating value
            timestamp=feedback.timestamp,
            source="implicit"
        )
    
    # Additionally, could store the raw interaction in a separate system
    # For simplicity, we'll just use the ratings storage here
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store feedback")
    
    return {
        "status": "success",
        "message": "Feedback recorded successfully",
        "feedback": feedback.dict()
    }

# Recommendation endpoints
@app.post("/recommendations/user", response_model=Dict[str, Any])
async def get_user_recommendations(request: RecommendationRequest, user: dict = Depends(get_current_user)):
    """Get personalized recommendations for a user"""
    # Verify user_id matches authenticated user
    if request.user_id != user["user_id"]:
        raise HTTPException(status_code=403, detail="Cannot get recommendations for other users")
    
    # Check if this is a new user with no ratings
    user_has_ratings = ratings_storage.has_user_ratings(
        domain=request.domain,
        dataset=request.dataset,
        user_id=request.user_id
    )
    
    if not user_has_ratings:
        # For new users, return a special response
        return {
            "user_id": request.user_id,
            "domain": request.domain,
            "dataset": request.dataset,
            "model": {
                "algorithm": "none",
                "version": "none"
            },
            "recommendations": [],
            "status": "cold_start",
            "message": "Please rate some items to get personalized recommendations"
        }
    
    # Get the recommended model for this domain/dataset
    model_info = model_manager.get_recommended_model(
        domain=request.domain,
        dataset=request.dataset
    )
    
    if model_info.get("algorithm") is None:
        raise HTTPException(status_code=404, detail="No recommendation model available for this domain/dataset")
    
    # Generate recommendations
    recommendations = [
        {
            "item_id": 100 + i,
            "score": 0.9 - (i * 0.05),
            "reason": "Based on your viewing history"
        }
        for i in range(request.count)
    ]
    
    return {
        "user_id": request.user_id,
        "domain": request.domain,
        "dataset": request.dataset,
        "model": {
            "algorithm": model_info.get("algorithm"),
            "version": model_info.get("version")
        },
        "recommendations": recommendations
    }

@app.post("/recommendations/similar", response_model=Dict[str, Any])
async def get_similar_items(request: SimilarItemsRequest, _: dict = Depends(get_current_user)):
    """Get items similar to a given item"""
    # Get the recommended model for this domain/dataset
    model_info = model_manager.get_recommended_model(
        domain=request.domain,
        dataset=request.dataset
    )
    
    if model_info.get("algorithm") is None:
        raise HTTPException(status_code=404, detail="No recommendation model available for this domain/dataset")
    
    # In a real system, you would use the model to find similar items
    # For this example, we'll generate dummy similar items
    similar_items = [
        {
            "item_id": request.item_id + (1000 * i),
            "similarity": 0.9 - (i * 0.1),
            "reason": "Similar genre and rating pattern"
        }
        for i in range(request.count)
    ]
    
    return {
        "item_id": request.item_id,
        "domain": request.domain,
        "dataset": request.dataset,
        "model": {
            "algorithm": model_info.get("algorithm"),
            "version": model_info.get("version")
        },
        "similar_items": similar_items
    }

@app.get("/models/{domain}/{dataset}", response_model=Dict[str, Any])
async def get_model_info(
    domain: str = Path(..., description="Domain name"),
    dataset: str = Path(..., description="Dataset name")
):
    """Get information about the currently active model for a domain/dataset"""
    model_info = model_manager.get_recommended_model(domain=domain, dataset=dataset)
    
    if model_info.get("algorithm") is None:
        raise HTTPException(status_code=404, detail="No model found for this domain/dataset")
    
    # Get model metrics
    metrics = model_manager.get_model_metrics(
        domain=domain,
        dataset=dataset,
        algorithm=model_info.get("algorithm"),
        version=model_info.get("version")
    )
    
    return {
        "domain": domain,
        "dataset": dataset,
        "algorithm": model_info.get("algorithm"),
        "version": model_info.get("version"),
        "source": model_info.get("source"),
        "metrics": metrics
    }

@app.get("/models/{domain}/{dataset}/history", response_model=Dict[str, Any])
async def get_model_history(
    domain: str = Path(..., description="Domain name"),
    dataset: str = Path(..., description="Dataset name"),
    algorithm: str = Query(None, description="Filter by algorithm"),
    limit: int = Query(10, description="Maximum number of history entries to return")
):
    """Get the training history for models in a domain/dataset"""
    if algorithm:
        history = model_manager.get_model_history(
            domain=domain,
            dataset=dataset,
            algorithm=algorithm,
            limit=limit
        )
        
        return {
            "domain": domain,
            "dataset": dataset,
            "algorithm": algorithm,
            "history": history
        }
    else:
        # Get the algorithms available for this domain/dataset
        model_dir = os.path.join(data_dir, "models", domain, dataset)
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail="No models found for this domain/dataset")
        
        algorithms = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        all_history = {}
        
        for alg in algorithms:
            all_history[alg] = model_manager.get_model_history(
                domain=domain,
                dataset=dataset,
                algorithm=alg,
                limit=limit
            )
        
        return {
            "domain": domain,
            "dataset": dataset,
            "algorithms": algorithms,
            "history": all_history
        }

# Admin endpoints
@app.post("/admin/update-dataset", response_model=Dict[str, Any])
async def update_dataset(
    request: UpdateDatasetRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(get_current_admin)
):
    """Trigger the dataset update process to incorporate new ratings"""
    # Queue the update as a background task
    def process_update():
        logger.info(f"Starting dataset update for {request.domain}/{request.dataset}")
        report = dataset_updater.process_new_ratings(
            domain=request.domain,
            dataset=request.dataset,
            incremental=request.incremental
        )
        logger.info(f"Dataset update completed with status: {report.get('overall_status')}")
    
    background_tasks.add_task(process_update)
    
    return {
        "status": "processing",
        "message": f"Dataset update queued for {request.domain}/{request.dataset}",
        "incremental": request.incremental
    }

@app.post("/admin/retrain", response_model=Dict[str, Any])
async def retrain_model(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    _: dict = Depends(get_current_admin)
):
    """Trigger model retraining for a domain/dataset"""
    # First check if we should retrain
    if not request.force:
        retrain_info = model_manager.should_retrain(
            domain=request.domain,
            dataset=request.dataset
        )
        
        if not retrain_info["should_retrain"]:
            return {
                "status": "skipped",
                "message": "Retraining not needed based on current data",
                "details": retrain_info
            }
    
    # Queue the retraining as a background task
    def process_retraining():
        logger.info(f"Starting model retraining for {request.domain}/{request.dataset}")
        # In a real system, you would load the dataset and train the model here
        # For this example, we'll just simulate it
        time.sleep(5)  # Simulate training time
        
        # Generate dummy model data and metrics
        model_data = b"DUMMY MODEL DATA FOR " + request.domain.encode() + b"/" + request.dataset.encode()
        metrics = {
            "map_at_k": 0.35,
            "ndcg": 0.30,
            "precision": 0.25,
            "recall": 0.40,
            "training_time": 120,
            "hyperparameters": {
                "factors": 100,
                "regularization": 0.1,
                "iterations": 20
            }
        }
        
        # Register the new model
        algorithm = request.algorithm or "als"  # Default to ALS if not specified
        model_manager.register_trained_model(
            domain=request.domain,
            dataset=request.dataset,
            algorithm=algorithm,
            model_data=model_data,
            metrics=metrics
        )
        
        logger.info(f"Model retraining completed for {request.domain}/{request.dataset}")
    
    background_tasks.add_task(process_retraining)
    
    return {
        "status": "processing",
        "message": f"Model retraining queued for {request.domain}/{request.dataset}",
        "algorithm": request.algorithm or "default",
        "force": request.force
    }

@app.get("/admin/stats", response_model=Dict[str, Any])
async def get_system_stats(_: dict = Depends(get_current_admin)):
    """Get statistics about the recommender system"""
    # Get all domains and datasets
    models_dir = os.path.join(data_dir, "models")
    if not os.path.exists(models_dir):
        return {"status": "no_data", "message": "No models directory found"}
    
    domains = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    stats = {
        "domains": {},
        "users": {
            "total": len(user_manager.get_all_users())
        },
        "models": {
            "total": 0
        }
    }
    
    for domain in domains:
        domain_dir = os.path.join(models_dir, domain)
        datasets = [d for d in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, d))]
        
        stats["domains"][domain] = {
            "datasets": {}
        }
        
        for dataset in datasets:
            dataset_dir = os.path.join(domain_dir, dataset)
            algorithms = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            
            # Get dataset stats
            stats_path = os.path.join(data_dir, "dataset_stats", domain, dataset, "dataset_stats.json")
            dataset_stats = {}
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, 'r') as f:
                        dataset_stats = json.load(f)
                except:
                    pass
            
            # Check for new ratings
            new_ratings_count = 0
            try:
                new_ratings_count = ratings_storage.get_new_ratings_count(domain, dataset)
            except:
                pass
            
            stats["domains"][domain]["datasets"][dataset] = {
                "algorithms": algorithms,
                "models_count": 0,
                "new_ratings": new_ratings_count,
                "total_ratings": dataset_stats.get("total_ratings", 0),
                "total_users": dataset_stats.get("total_users", 0),
                "total_items": dataset_stats.get("total_items", 0)
            }
            
            # Count models for each algorithm
            for algorithm in algorithms:
                algorithm_dir = os.path.join(dataset_dir, algorithm)
                model_files = [f for f in os.listdir(algorithm_dir) if f.endswith(".model")]
                stats["domains"][domain]["datasets"][dataset]["models_count"] += len(model_files)
                stats["models"]["total"] += len(model_files)
    
    return stats

# Other utility endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Domain-Agnostic Recommender API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "running"
    }

# Run the API if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)