# backend/app/config.py
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Application
    PROJECT_NAME: str = "ClearSat"
    PROJECT_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        if ENVIRONMENT == "production":
            raise ValueError("SECRET_KEY must be set in production")
        SECRET_KEY = "dev-secret-key-change-in-production"
    
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    
    # Database
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "clearsat")
    MONGODB_MAX_POOL_SIZE: int = int(os.getenv("MONGODB_MAX_POOL_SIZE", "100"))
    MONGODB_MIN_POOL_SIZE: int = int(os.getenv("MONGODB_MIN_POOL_SIZE", "10"))
    
    # External Services
    EARTH_ENGINE_CREDENTIALS: str = os.getenv("EARTH_ENGINE_CREDENTIALS", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Guest user limitations
    GUEST_ANALYSIS_LIMIT: int = int(os.getenv("GUEST_ANALYSIS_LIMIT", "1"))
    GUEST_SESSION_HOURS: int = int(os.getenv("GUEST_SESSION_HOURS", "24"))
    
    # Analysis Limits
    MAX_ANALYSIS_BUFFER_KM: int = 50
    MAX_ANALYSIS_DAYS: int = 365
    
    # Validation
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

settings = Settings()