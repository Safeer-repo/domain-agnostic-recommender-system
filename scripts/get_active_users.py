import pandas as pd
import os
import sys
import random

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Define paths
data_path = os.path.join(project_root, "data", "processed", "entertainment", "movielens", "train.csv")

def get_active_users(min_ratings=20, num_users=5):
    """
    Get a list of active users who have submitted at least min_ratings
    
    Args:
        min_ratings: Minimum number of ratings to qualify as an active user
        num_users: Number of users to select
        
    Returns:
        List of user IDs with their rating counts
    """
    # Load the ratings data
    print(f"Loading ratings data from {data_path}")
    ratings_df = pd.read_csv(data_path)
    
    # Count ratings per user
    user_rating_counts = ratings_df['user_id'].value_counts()
    
    # Filter users with at least min_ratings
    active_users = user_rating_counts[user_rating_counts >= min_ratings]
    
    print(f"Found {len(active_users)} users with at least {min_ratings} ratings")
    
    # Select random users if we have more than requested
    if len(active_users) > num_users:
        selected_users = random.sample(list(active_users.items()), num_users)
    else:
        selected_users = list(active_users.items())
    
    # Format the results
    results = []
    for user_id, rating_count in selected_users:
        results.append({
            "user_id": int(user_id),
            "rating_count": int(rating_count)
        })
    
    return results

if __name__ == "__main__":
    active_users = get_active_users(min_ratings=50, num_users=5)
    
    print("\nSelected active users:")
    for i, user in enumerate(active_users, 1):
        print(f"{i}. User ID: {user['user_id']} - {user['rating_count']} ratings")
    
    print("\nCode to create these users:")
    print("demo_users = [")
    for user in active_users:
        print(f"    {{\"original_id\": {user['user_id']}, \"username\": \"user_{user['user_id']}\", \"password\": \"password123\"}},")
    print("]")