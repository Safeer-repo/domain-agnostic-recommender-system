import os
import json
import time
import uuid
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class UserManager:
    """
    User management system for the recommender system.
    Handles user registration, retrieval, and verification.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the user manager.
        
        Args:
            data_dir: Root directory for data storage
        """
        self.data_dir = data_dir
        self.users_dir = os.path.join(data_dir, "users")
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self) -> None:
        """Create necessary directory structure if it doesn't exist."""
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir, exist_ok=True)
            logger.info(f"Created users directory at {self.users_dir}")
            
    def _get_user_file_path(self, user_id: str) -> str:
        """
        Get the path to a user's JSON file.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Path to the user's JSON file
        """
        return os.path.join(self.users_dir, f"{user_id}.json")
    
    def _get_users_index_path(self) -> str:
        """
        Get the path to the users index file.
        
        Returns:
            Path to the users index file
        """
        return os.path.join(self.users_dir, "users_index.json")
    
    def _load_users_index(self) -> Dict[str, str]:
        """
        Load the index of users (maps usernames/emails to user_ids).
        
        Returns:
            Dictionary mapping usernames/emails to user_ids
        """
        index_path = self._get_users_index_path()
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading users index: {str(e)}")
                return {}
        else:
            # Create empty index if it doesn't exist
            empty_index = {}
            self._save_users_index(empty_index)
            return empty_index
            
    def _save_users_index(self, index: Dict[str, str]) -> bool:
        """
        Save the users index to disk.
        
        Args:
            index: Dictionary mapping usernames/emails to user_ids
            
        Returns:
            True if successful, False otherwise
        """
        index_path = self._get_users_index_path()
        try:
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving users index: {str(e)}")
            return False
    
    def user_exists(self, identifier: str) -> bool:
        """
        Check if a user exists by user_id, username, or email.
        
        Args:
            identifier: User ID, username, or email
            
        Returns:
            True if the user exists, False otherwise
        """
        # Check if it's a direct user_id
        if os.path.exists(self._get_user_file_path(identifier)):
            return True
            
        # Check if it's a username or email in the index
        users_index = self._load_users_index()
        return identifier in users_index
    
    def get_user_id(self, identifier: str) -> Optional[str]:
        """
        Get a user's ID from their username or email.
        
        Args:
            identifier: Username or email
            
        Returns:
            User ID if found, None otherwise
        """
        # If the identifier is already a user_id, check if it exists
        if os.path.exists(self._get_user_file_path(identifier)):
            return identifier
            
        # Look up in the index
        users_index = self._load_users_index()
        return users_index.get(identifier)
    
    def create_user(self, username: str, email: str, 
                   domain_preferences: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
        """
        Create a new user.
        
        Args:
            username: Unique username
            email: User's email address
            domain_preferences: Optional dictionary mapping domains to lists of dataset names
                               the user is interested in
            
        Returns:
            User ID if created successfully, None otherwise
        """
        # Check if username or email already exists
        users_index = self._load_users_index()
        if username in users_index or email in users_index:
            logger.warning(f"User with username '{username}' or email '{email}' already exists")
            return None
            
        # Generate a unique user ID
        user_id = str(uuid.uuid4())
        
        # Create user data
        timestamp = int(time.time())
        user_data = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "created_at": timestamp,
            "last_updated": timestamp,
            "domain_preferences": domain_preferences or {},
            "metadata": {}
        }
        
        # Save user data
        try:
            user_path = self._get_user_file_path(user_id)
            with open(user_path, 'w') as f:
                json.dump(user_data, f, indent=2)
                
            # Update the index
            users_index[username] = user_id
            users_index[email] = user_id
            self._save_users_index(users_index)
            
            logger.info(f"Created new user: {username} ({user_id})")
            return user_id
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user's information.
        
        Args:
            user_id: User ID
            
        Returns:
            User data dictionary if found, None otherwise
        """
        user_path = self._get_user_file_path(user_id)
        if not os.path.exists(user_path):
            logger.warning(f"User {user_id} not found")
            return None
            
        try:
            with open(user_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user {user_id}: {str(e)}")
            return None
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a user's information.
        
        Args:
            user_id: User ID
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        user_data = self.get_user(user_id)
        if not user_data:
            return False
            
        # Update the user data
        for key, value in updates.items():
            if key == "user_id":  # Don't allow changing the user_id
                continue
                
            if key == "username" or key == "email":
                # Update the index if username or email is changing
                old_value = user_data.get(key)
                if old_value != value:
                    users_index = self._load_users_index()
                    if old_value in users_index:
                        del users_index[old_value]
                    users_index[value] = user_id
                    self._save_users_index(users_index)
            
            user_data[key] = value
            
        # Update the last_updated timestamp
        user_data["last_updated"] = int(time.time())
        
        # Save the updated user data
        try:
            user_path = self._get_user_file_path(user_id)
            with open(user_path, 'w') as f:
                json.dump(user_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {str(e)}")
            return False
    
    def add_domain_preference(self, user_id: str, domain: str, dataset: str) -> bool:
        """
        Add a domain preference for a user.
        
        Args:
            user_id: User ID
            domain: Domain name (e.g., entertainment, ecommerce)
            dataset: Dataset name within the domain
            
        Returns:
            True if successful, False otherwise
        """
        user_data = self.get_user(user_id)
        if not user_data:
            return False
            
        # Initialize domain preferences if not exists
        if "domain_preferences" not in user_data:
            user_data["domain_preferences"] = {}
            
        # Initialize domain if not exists
        if domain not in user_data["domain_preferences"]:
            user_data["domain_preferences"][domain] = []
            
        # Add dataset if not already in preferences
        if dataset not in user_data["domain_preferences"][domain]:
            user_data["domain_preferences"][domain].append(dataset)
            
        # Update the user data
        return self.update_user(user_id, {"domain_preferences": user_data["domain_preferences"]})
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        Get a list of all users.
        
        Returns:
            List of user data dictionaries
        """
        users = []
        
        # Iterate through all user files
        for filename in os.listdir(self.users_dir):
            if filename.endswith(".json") and filename != "users_index.json":
                user_id = filename[:-5]  # Remove .json extension
                user_data = self.get_user(user_id)
                if user_data:
                    users.append(user_data)
                    
        return users
    
    def get_users_by_domain_interest(self, domain: str, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get users interested in a specific domain/dataset.
        
        Args:
            domain: Domain name
            dataset: Optional dataset name within the domain
            
        Returns:
            List of user data dictionaries for users interested in the domain/dataset
        """
        all_users = self.get_all_users()
        matching_users = []
        
        for user in all_users:
            preferences = user.get("domain_preferences", {})
            
            # Check if user is interested in this domain
            if domain in preferences:
                # If dataset is specified, check if user is interested in this dataset
                if dataset is None or dataset in preferences[domain]:
                    matching_users.append(user)
                    
        return matching_users