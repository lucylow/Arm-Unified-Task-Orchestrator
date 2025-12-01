import os
from typing import List, Optional

class Config:
    """Configuration management for AutoRL application"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Qdrant Vector Database
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # Appium Server Configuration
    APPIUM_HOST: str = os.getenv("APPIUM_HOST", "localhost")
    APPIUM_PORT: int = int(os.getenv("APPIUM_PORT", "4723"))
    APPIUM_SERVER_URL: str = f"http://{APPIUM_HOST}:{APPIUM_PORT}/wd/hub"
    
    # Device Configuration
    DEVICE_PLATFORM: str = os.getenv("DEVICE_PLATFORM", "Android")
    DEVICE_MODE: str = os.getenv("DEVICE_MODE", "appium")
    
    # Application Configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Security Configuration
    JWT_SECRET: str = os.getenv("JWT_SECRET", "your_jwt_secret_here")
    ENABLE_PII_MASKING: bool = os.getenv("ENABLE_PII_MASKING", "true").lower() == "true"
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./autorl.db")
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Monitoring Configuration
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    
    # WebSocket Configuration
    WEBSOCKET_HEARTBEAT_INTERVAL: int = int(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", "30"))
    WEBSOCKET_TIMEOUT: int = int(os.getenv("WEBSOCKET_TIMEOUT", "60"))
    
    # Task Configuration
    MAX_PARALLEL_TASKS: int = int(os.getenv("MAX_PARALLEL_TASKS", "5"))
    TASK_TIMEOUT_SECONDS: int = int(os.getenv("TASK_TIMEOUT_SECONDS", "300"))
    MAX_RETRY_ATTEMPTS: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    
    # Logging Configuration
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "./logs/autorl.log")
    LOG_ROTATION_SIZE: str = os.getenv("LOG_ROTATION_SIZE", "10MB")
    LOG_RETENTION_DAYS: int = int(os.getenv("LOG_RETENTION_DAYS", "7"))
    
    # Development Configuration
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    
    # CORS Configuration
    @property
    def CORS_ORIGINS(self) -> List[str]:
        cors_str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
        return [origin.strip() for origin in cors_str.split(",")]
    
    # Tesseract OCR Configuration
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
    TESSDATA_PREFIX: str = os.getenv("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")
    
    # Required environment variables validation
    REQUIRED_VARS = [
        "OPENAI_API_KEY",
        "QDRANT_URL",
        "APPIUM_HOST",
        "APPIUM_PORT"
    ]
    
    @classmethod
    def validate_required_vars(cls) -> List[str]:
        """Validate that all required environment variables are set"""
        missing_vars = []
        for var in cls.REQUIRED_VARS:
            if not os.getenv(var):
                missing_vars.append(var)
        return missing_vars
    
    @classmethod
    def get_required_env_example(cls) -> str:
        """Get example environment configuration"""
        return """
# AutoRL Environment Configuration
# Copy this to .env file and fill in your actual values

# Required Configuration
OPENAI_API_KEY=your_actual_api_key_here
QDRANT_URL=http://localhost:6333
APPIUM_HOST=localhost
APPIUM_PORT=4723

# Optional Configuration
DEVICE_PLATFORM=Android
PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO
JWT_SECRET=your_jwt_secret_here
ENABLE_PII_MASKING=true
DEBUG=true
RELOAD=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
MAX_PARALLEL_TASKS=5
TASK_TIMEOUT_SECONDS=300
MAX_RETRY_ATTEMPTS=3
WEBSOCKET_HEARTBEAT_INTERVAL=30
WEBSOCKET_TIMEOUT=60
TESSERACT_CMD=/usr/bin/tesseract
"""

# Global config instance
config = Config()




