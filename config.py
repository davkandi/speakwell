import os
from datetime import timedelta

# Load environment variables from .env file in development
# In production, these should be set as system environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available in production, rely on system env vars
    pass

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://speakwell:speakwell123@localhost/speakwell'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-change-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION') or 'us-east-1'
    S3_BUCKET = os.environ.get('S3_BUCKET')
    REDIS_HOST = os.environ.get('REDIS_HOST') or 'localhost'
    REDIS_PORT = os.environ.get('REDIS_PORT') or 6379
    REDIS_DB = os.environ.get('REDIS_DB') or 0
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'
    
    # Celery timeout settings for long-running video analysis tasks
    task_soft_time_limit = 1800  # 30 minutes soft limit (allows graceful cleanup)
    task_time_limit = 2400       # 40 minutes hard limit (kills task)
    worker_prefetch_multiplier = 1  # Process one task at a time for memory efficiency
    
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
