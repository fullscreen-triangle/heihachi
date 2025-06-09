"""
Configuration for Heihachi REST API
"""

import os
from datetime import timedelta


class APIConfig:
    """Base configuration for the API"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'heihachi-audio-analysis-api-key')
    
    # File upload configuration
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE', 500 * 1024 * 1024))  # 500MB default
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'aac', 'm4a', 'ogg'}
    
    # Processing configuration
    PROCESSING_TIMEOUT = int(os.environ.get('PROCESSING_TIMEOUT', 1800))  # 30 minutes
    MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 5))
    
    # Results configuration
    RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', 'results')
    RESULTS_RETENTION_DAYS = int(os.environ.get('RESULTS_RETENTION_DAYS', 7))
    
    # Heihachi configuration
    HEIHACHI_CONFIG_PATH = os.environ.get('HEIHACHI_CONFIG_PATH', 'configs/default.yaml')
    CACHE_DIR = os.environ.get('CACHE_DIR', 'cache')
    
    # HuggingFace configuration
    HUGGINGFACE_ENABLED = os.environ.get('HUGGINGFACE_ENABLED', 'true').lower() == 'true'
    HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')
    
    # Rate limiting
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'true').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per hour')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    @staticmethod
    def allowed_file(filename):
        """Check if a file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in APIConfig.ALLOWED_EXTENSIONS


class DevelopmentConfig(APIConfig):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(APIConfig):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Use more restrictive settings in production
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    PROCESSING_TIMEOUT = 900  # 15 minutes
    MAX_CONCURRENT_JOBS = 3


class TestingConfig(APIConfig):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB for testing 