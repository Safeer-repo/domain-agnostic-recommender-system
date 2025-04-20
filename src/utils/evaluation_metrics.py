import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
import scipy.spatial.distance as distance

def calculate_coverage(recommended_items: List[Set[int]], catalog_size: int) -> float:
    """
    Calculate the coverage metric - percentage of items that the system recommends.
    
    Args:
        recommended_items: List of sets, where each set contains item IDs recommended to one user
        catalog_size: Total number of items in the catalog
        
    Returns:
        Coverage score between 0 and 1
    """
    # Flatten all recommended items into a single set
    all_recommended = set().union(*recommended_items) if recommended_items else set()
    
    # Calculate coverage
    coverage = len(all_recommended) / catalog_size if catalog_size > 0 else 0
    
    return coverage

def calculate_novelty(recommended_items: List[List[Tuple[int, float]]], 
                     item_popularity: Dict[int, int],
                     total_interactions: int) -> float:
    """
    Calculate novelty - measures how "unpopular" or "surprising" the recommendations are.
    
    Args:
        recommended_items: List of recommendation lists, where each recommendation is (item_id, score)
        item_popularity: Dictionary mapping item IDs to their popularity (number of interactions)
        total_interactions: Total number of interactions in the training data
        
    Returns:
        Novelty score (higher means more novel/unexpected recommendations)
    """
    if not recommended_items:
        return 0.0
    
    # Calculate self-information for each item: -log2(popularity)
    # More popular items have lower self-information
    self_info = {}
    for item_id, count in item_popularity.items():
        # Add smoothing to avoid division by zero
        probability = count / total_interactions
        self_info[item_id] = -np.log2(probability) if probability > 0 else 0
    
    # Calculate average self-information of recommended items for each user
    novelty_scores = []
    for user_recs in recommended_items:
        if not user_recs:
            continue
            
        user_items = [item_id for item_id, _ in user_recs]
        user_novelty = np.mean([self_info.get(item_id, 0) for item_id in user_items])
        novelty_scores.append(user_novelty)
    
    # Return average novelty across users
    return np.mean(novelty_scores) if novelty_scores else 0.0

def calculate_diversity(recommended_items: List[List[Tuple[int, float]]],
                       item_features: Dict[int, np.ndarray] = None) -> float:
    """
    Calculate diversity - measures how different the items in each recommendation list are.
    
    Args:
        recommended_items: List of recommendation lists, where each recommendation is (item_id, score)
        item_features: Optional dictionary mapping item IDs to feature vectors
        
    Returns:
        Diversity score (higher means more diverse recommendations)
    """
    if not recommended_items:
        return 0.0
    
    user_diversity = []
    
    for user_recs in recommended_items:
        if len(user_recs) < 2:  # Need at least 2 items to calculate diversity
            continue
            
        item_ids = [item_id for item_id, _ in user_recs]
        
        # If item features are provided, calculate feature-based diversity
        if item_features:
            # Get feature vectors for all items in the recommendation list
            vectors = [item_features.get(item_id) for item_id in item_ids if item_id in item_features]
            
            if len(vectors) < 2:  # Need at least 2 vectors
                continue
                
            # Calculate pairwise distances
            distances = []
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    # Cosine distance = 1 - cosine similarity
                    dist = distance.cosine(vectors[i], vectors[j])
                    distances.append(dist)
                    
            # Average distance
            avg_distance = np.mean(distances)
            user_diversity.append(avg_distance)
            
        else:
            # If no features, use item ID diversity (not as meaningful)
            # Use Intra-List Diversity (ILD) based on item IDs
            n_items = len(item_ids)
            n_unique = len(set(item_ids))
            ild = n_unique / n_items if n_items > 0 else 0
            user_diversity.append(ild)
    
    # Return average diversity across users
    return np.mean(user_diversity) if user_diversity else 0.0

def calculate_all_beyond_accuracy_metrics(
    recommendations: Dict[int, List[Tuple[int, float]]],
    train_data: pd.DataFrame,
    item_features: Dict[int, np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all beyond-accuracy metrics (coverage, novelty, diversity)
    
    Args:
        recommendations: Dictionary mapping user IDs to recommendation lists
        train_data: Training data DataFrame with user_id, item_id, rating columns
        item_features: Optional dictionary mapping item IDs to feature vectors
        
    Returns:
        Dictionary with metric names and values
    """
    results = {}
    
    # Convert recommendations to format needed by metrics functions
    recommended_items_list = list(recommendations.values())
    recommended_items_sets = [set(item_id for item_id, _ in recs) for recs in recommended_items_list]
    
    # Calculate catalog size
    catalog_size = train_data['item_id'].nunique()
    
    # Calculate item popularity
    item_popularity = train_data.groupby('item_id').size().to_dict()
    total_interactions = len(train_data)
    
    # Calculate metrics
    results['coverage'] = calculate_coverage(recommended_items_sets, catalog_size)
    results['novelty'] = calculate_novelty(recommended_items_list, item_popularity, total_interactions)
    results['diversity'] = calculate_diversity(recommended_items_list, item_features)
    
    return results
