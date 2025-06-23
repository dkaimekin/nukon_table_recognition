import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Model paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "table_classifier.h5")
    CLASS_INDICES_PATH: str = os.getenv("CLASS_INDICES_PATH", "class_indices.json")
    
    # API settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Limits
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "10"))
    
    # Security
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        config = cls()
        
        # Check if model files exist
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {config.MODEL_PATH}")
        
        if not os.path.exists(config.CLASS_INDICES_PATH):
            raise FileNotFoundError(f"Class indices file not found: {config.CLASS_INDICES_PATH}")
        
        return config