# tests/unit/preprocessing/test_data_loader.py
import os
import unittest
import logging
from src.preprocessing.data_loader import DataLoader

# Configure basic logging
logging.basicConfig(level=logging.INFO)

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a data loader instance with the project's data directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        self.data_dir = os.path.join(self.project_root, 'data')
        self.data_loader = DataLoader(self.data_dir)
    
    def test_initialization(self):
        """Test if the DataLoader initializes correctly."""
        self.assertEqual(self.data_loader.data_dir, self.data_dir)
    
    def test_load_dataset_invalid_domain(self):
        """Test if DataLoader raises appropriate error for invalid domain."""
        with self.assertRaises(ValueError):
            self.data_loader.load_dataset("invalid_domain", "invalid_dataset")
    
    def test_load_movielens_missing_file(self):
        """Test if DataLoader raises FileNotFoundError when dataset is missing."""
        # This should fail if the dataset hasn't been downloaded yet
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_dataset("entertainment", "movielens")
    
if __name__ == '__main__':
    unittest.main()