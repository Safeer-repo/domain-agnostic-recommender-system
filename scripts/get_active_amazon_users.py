import pandas as pd
import os
import sys
import random

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Define the correct path for Amazon dataset
data_path = os.path.join(project_root, "data", "processed", "ecommerce", "amazon", "train.csv")

def get_active_users(min_ratings=10, num_users=5):
    """
    Get a list of active users who have submitted at least min_ratings
    
    Args:
        min_ratings: Minimum number of ratings to qualify as an active user
        num_users: Number of users to select
    
    Returns:
        List of user IDs with their rating counts
    """
    # Load the ratings data
    print(f"Loading Amazon ratings data from {data_path}")
    
    try:
        ratings_df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(ratings_df)} ratings")
        print(f"Columns: {list(ratings_df.columns)}")
        
        # Count ratings per user
        user_rating_counts = ratings_df['user_id'].value_counts()
        
        # Filter users with at least min_ratings
        active_users = user_rating_counts[user_rating_counts >= min_ratings]
        
        print(f"Found {len(active_users)} users with at least {min_ratings} ratings")
        
        # If no users found with min_ratings, lower the threshold
        if len(active_users) == 0:
            print(f"No users found with {min_ratings} ratings. Checking for users with fewer ratings...")
            all_users = user_rating_counts
            print(f"User rating distribution:")
            print(f"Max ratings per user: {all_users.max()}")
            print(f"Min ratings per user: {all_users.min()}")
            print(f"Mean ratings per user: {all_users.mean():.2f}")
            
            # Lower the threshold to get some users
            threshold = min(5, all_users.max())
            active_users = user_rating_counts[user_rating_counts >= threshold]
            print(f"Using threshold of {threshold} ratings instead")
        
        # Select random users if we have more than requested
        if len(active_users) > num_users:
            selected_users = random.sample(list(active_users.items()), num_users)
        else:
            selected_users = list(active_users.items())
        
        # Format the results
        results = []
        for user_id, rating_count in selected_users:
            results.append({
                "user_id": user_id,  # Amazon user IDs might be strings
                "rating_count": int(rating_count)
            })
        
        return results
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        # Try test.csv if train.csv doesn't work
        test_path = data_path.replace("train.csv", "test.csv")
        print(f"Trying test.csv at {test_path}")
        return get_active_users_from_file(test_path, min_ratings, num_users)
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_active_users_from_file(file_path, min_ratings=10, num_users=5):
    """Helper function to get active users from a specific file"""
    try:
        ratings_df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(ratings_df)} ratings from {os.path.basename(file_path)}")
        
        user_rating_counts = ratings_df['user_id'].value_counts()
        active_users = user_rating_counts[user_rating_counts >= min_ratings]
        
        if len(active_users) == 0:
            # If no users meet the criteria, just get the top users
            active_users = user_rating_counts.head(num_users)
        
        selected_users = list(active_users.items())[:num_users]
        
        results = []
        for user_id, rating_count in selected_users:
            results.append({
                "user_id": user_id,
                "rating_count": int(rating_count)
            })
        
        return results
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

if __name__ == "__main__":
    # Get active Amazon users
    active_users = get_active_users(min_ratings=10, num_users=5)
    
    if not active_users:
        # If no active users found with 10+ ratings, try with a lower threshold
        print("\nTrying with lower threshold...")
        active_users = get_active_users(min_ratings=5, num_users=5)
    
    if active_users:
        print("\nSelected active Amazon users:")
        for i, user in enumerate(active_users, 1):
            print(f"{i}. User ID: {user['user_id']} - {user['rating_count']} ratings")
        
        print("\nCode to create these users:")
        print("demo_users = [")
        for i, user in enumerate(active_users, 1):
            # Create usernames that indicate they're Amazon users
            username = f"amazon_user_{i}"
            print(f'    {{"original_id": "{user["user_id"]}", "username": "{username}", "password": "password123"}},')
        print("]")
    else:
        print("No active users found or data file not accessible")