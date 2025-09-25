"""Configuration management for the AI Resume Checker."""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging


class Config:
    """Configuration manager for API keys and settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self._load_env_config()
        
        # Default settings
        self.settings = {
            # API configurations
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'groq_model': os.getenv('GROQ_MODEL', 'llama3-8b-8192'),
            
            # Similarity model settings
            'similarity_model': os.getenv('SIMILARITY_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.5')),
            
            # Text processing settings
            'max_text_length': int(os.getenv('MAX_TEXT_LENGTH', '10000')),
            'remove_personal_info': os.getenv('REMOVE_PERSONAL_INFO', 'true').lower() == 'true',
            
            # Review generation settings
            'max_review_length': int(os.getenv('MAX_REVIEW_LENGTH', '2000')),
            'include_suggestions': os.getenv('INCLUDE_SUGGESTIONS', 'true').lower() == 'true',
            'suggestion_count': int(os.getenv('SUGGESTION_COUNT', '5')),
            
            # Logging settings
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': os.getenv('LOG_FILE', 'resume_checker.log'),
        }
        
        # Load additional config from file if provided
        if config_file:
            self._load_file_config(config_file)
    
    def _load_env_config(self):
        """Load environment variables from .env file if available."""
        try:
            from dotenv import load_dotenv
            
            # Look for .env file in current directory or parent directories
            env_path = Path('.env')
            if not env_path.exists():
                env_path = Path(__file__).parent.parent.parent / '.env'
            
            if env_path.exists():
                load_dotenv(env_path)
                self.logger.info(f"Loaded environment variables from {env_path}")
            else:
                self.logger.info("No .env file found, using system environment variables")
                
        except ImportError:
            self.logger.warning("python-dotenv not installed, using system environment variables only")
    
    def _load_file_config(self, config_file: str):
        """Load additional configuration from a file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            if config_path.suffix.lower() == '.json':
                import json
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    self.settings.update(file_config)
                    self.logger.info(f"Loaded configuration from {config_file}")
            else:
                self.logger.warning(f"Unsupported configuration file format: {config_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Error loading configuration file {config_file}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.settings[key] = value
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate that required API keys are available.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'groq_api_key': bool(self.settings.get('groq_api_key'))
        }
        
        return validation
    
    def get_groq_config(self) -> Dict[str, Any]:
        """Get Groq API configuration."""
        return {
            'api_key': self.settings.get('groq_api_key'),
            'model': self.settings.get('groq_model'),
            'max_tokens': self.settings.get('max_review_length', 2000),
        }
    
    def get_similarity_config(self) -> Dict[str, Any]:
        """Get similarity calculation configuration."""
        return {
            'model_name': self.settings.get('similarity_model'),
            'threshold': self.settings.get('similarity_threshold'),
            'max_length': self.settings.get('max_text_length')
        }
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get text preprocessing configuration."""
        return {
            'remove_personal_info': self.settings.get('remove_personal_info'),
            'max_length': self.settings.get('max_text_length')
        }
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.settings.get('log_level', 'INFO').upper())
        log_file = self.settings.get('log_file')
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def __str__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        safe_settings = self.settings.copy()
        
        # Mask sensitive information
        sensitive_keys = ['groq_api_key']
        for key in sensitive_keys:
            if key in safe_settings and safe_settings[key]:
                safe_settings[key] = '***masked***'
        
        return f"Config({safe_settings})"