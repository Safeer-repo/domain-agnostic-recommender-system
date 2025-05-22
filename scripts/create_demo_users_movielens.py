import os
import sys
from passlib.context import CryptContext

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import from src package
from src.utils.user_manager import UserManager

# Define password hashing function directly
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def get_password_hash(password):
    return pwd_context.hash(password)

# Initialize the user manager
data_dir = os.path.abspath("./data")
user_manager = UserManager(data_dir=data_dir)

# Define demo users from your active MovieLens users
demo_users = [
    {"original_id": 21, "username": "user_21", "password": "password123"},
    {"original_id": 546, "username": "user_546", "password": "password123"},
    {"original_id": 555, "username": "user_555", "password": "password123"},
    {"original_id": 748, "username": "user_748", "password": "password123"},
    {"original_id": 116, "username": "user_116", "password": "password123"},
]

# Function to create demo users
def create_demo_users():
    for user in demo_users:
        # Check if user already exists
        if user_manager.user_exists(user["username"]):
            print(f"User {user['username']} already exists. Skipping.")
            continue
            
        # Create the user
        user_id = user_manager.create_user(
            username=user["username"],
            email=f"{user['username']}@example.com",
            domain_preferences={"entertainment": ["movielens"]}
        )
        
        if user_id:
            # Add password and original ID to metadata
            user_data = user_manager.get_user(user_id)
            metadata = user_data.get("metadata", {})
            metadata["password"] = get_password_hash(user["password"])
            metadata["original_dataset_id"] = user["original_id"]
            user_manager.update_user(user_id, {"metadata": metadata})
            
            print(f"Created demo user: {user['username']} with ID {user_id}")
        else:
            print(f"Failed to create user: {user['username']}")

if __name__ == "__main__":
    create_demo_users()
    print("Demo user creation complete.")