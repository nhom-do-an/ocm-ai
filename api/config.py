"""
Configuration management for AI Training Service
Loads settings from environment variables with fallback defaults
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    """Application configuration"""
    
    # Database
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5433))
    DB_NAME = os.getenv('DB_NAME', 'testocm')
    DB_USER = os.getenv('DB_USER', 'zalolog')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '123456')
    
    @property
    def DB_CONFIG(self):
        return {
            'host': self.DB_HOST,
            'port': self.DB_PORT,
            'database': self.DB_NAME,
            'user': self.DB_USER,
            'password': self.DB_PASSWORD
        }
    
    # Flask
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5001))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_SAVE_DIR = BASE_DIR / os.getenv('MODEL_SAVE_DIR', 'results')
    
    # Cache
    CACHE_EXPIRY_DAYS = int(os.getenv('CACHE_EXPIRY_DAYS', 7))
    
    # Recommendation Model
    REC_EPOCHS = int(os.getenv('RECOMMENDATION_EPOCHS', 20))
    REC_BATCH_SIZE = int(os.getenv('RECOMMENDATION_BATCH_SIZE', 256))
    REC_LEARNING_RATE = float(os.getenv('RECOMMENDATION_LEARNING_RATE', 0.001))
    REC_EMBED_DIM = int(os.getenv('RECOMMENDATION_EMBED_DIM', 32))
    REC_HIDDEN_LAYERS = [int(x) for x in os.getenv('RECOMMENDATION_HIDDEN_LAYERS', '64,32,16').split(',')]
    
    # Interaction Weights
    WEIGHT_ORDER = float(os.getenv('WEIGHT_ORDER', 1.0))
    WEIGHT_CART = float(os.getenv('WEIGHT_CART', 0.3))
    
    # Trending Model
    TREND_MAX_ROUNDS = int(os.getenv('TRENDING_MAX_ROUNDS', 1000))
    TREND_EARLY_STOPPING = int(os.getenv('TRENDING_EARLY_STOPPING', 50))
    TREND_LEARNING_RATE = float(os.getenv('TRENDING_LEARNING_RATE', 0.05))
    TREND_MAX_DEPTH = int(os.getenv('TRENDING_MAX_DEPTH', 6))
    TREND_NUM_LEAVES = int(os.getenv('TRENDING_NUM_LEAVES', 31))
    
    # Next Item Model
    NEXT_ITEM_ORDER = int(os.getenv('NEXT_ITEM_ORDER', 1))
    NEXT_ITEM_MIN_SUPPORT = int(os.getenv('NEXT_ITEM_MIN_SUPPORT', 2))
    NEXT_ITEM_SMOOTHING = float(os.getenv('NEXT_ITEM_SMOOTHING', 1.0))
    
    # Data
    TIME_SERIES_DAYS_BACK = int(os.getenv('TIME_SERIES_DAYS_BACK', 180))
    
    MIN_ORDERS_REQUIRED = int(os.getenv('MIN_ORDERS_REQUIRED', 100))
    MIN_USERS_REQUIRED = int(os.getenv('MIN_USERS_REQUIRED', 20))
    MIN_ITEMS_REQUIRED = int(os.getenv('MIN_ITEMS_REQUIRED', 10))
    MIN_DAYS_REQUIRED = int(os.getenv('MIN_DAYS_REQUIRED', 30))
    MIN_TIME_SERIES_RECORDS = int(os.getenv('MIN_TIME_SERIES_RECORDS', 20))

# Global config instance
config = Config()
