import logging
import os
import sys
from datetime import datetime

def setup_logging(config, log_filename=None):
    """
    Set up logging configuration.
    
    Args:
        config: Config object with logs_dir attribute
        log_filename: Optional custom log filename
    """
    # Create logs directory if it doesn't exist
    if not hasattr(config, 'logs_dir'):
        config.logs_dir = './logs'
        
    os.makedirs(config.logs_dir, exist_ok=True)
    
    # Generate log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"recommender_{timestamp}.log"
    
    log_path = os.path.join(config.logs_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logger
    logger = logging.getLogger()
    logger.info(f"Logging initialized: {log_path}")
    
    return logger
