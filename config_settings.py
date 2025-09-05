from pydantic import BaseSettings, Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Configuration settings for Bot Quantum Max."""
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    postgres_user: str = Field(..., env="POSTGRES_USER")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")
    postgres_db: str = Field(..., env="POSTGRES_DB")
    
    # API Keys
    football_data_api_key: str = Field(..., env="FOOTBALL_DATA_API_KEY")
    understat_api_key: Optional[str] = Field(None, env="UNDERSTAT_API_KEY")
    openweather_api_key: str = Field(..., env="OPENWEATHER_API_KEY")
    odds_api_key: str = Field(..., env="ODDS_API_KEY")
    
    # Telegram
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")
    
    # Email
    smtp_server: Optional[str] = Field(None, env="SMTP_SERVER")
    smtp_port: int = Field(587, env="SMTP_PORT")
    email_user: Optional[str] = Field(None, env="EMAIL_USER")
    email_password: Optional[str] = Field(None, env="EMAIL_PASSWORD")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Model Configuration
    max_monte_carlo_runs: int = Field(10000, env="MAX_MONTE_CARLO_RUNS")
    confidence_threshold: float = Field(0.65, env="CONFIDENCE_THRESHOLD")
    min_edge: float = Field(0.05, env="MIN_EDGE")
    
    # Simulation Parameters
    simulation_seasons: int = 5
    agent_based_iterations: int = 1000
    
    # Portfolio Parameters
    max_kelly_fraction: float = 0.25
    min_odds: float = 1.5
    max_odds: float = 10.0
    max_correlation: float = 0.7
    
    # Feature Engineering
    lookback_days: int = 180
    min_matches_for_prediction: int = 5
    
    # Supported Leagues
    supported_leagues: List[str] = [
        "PL",  # Premier League
        "FL1", # Ligue 1
        "BL1", # Bundesliga
        "SA",  # Serie A
        "PD",  # La Liga
        "CL",  # Champions League
        "EL",  # Europa League
    ]
    
    # Bet Types
    bet_types: List[str] = [
        "1X2",
        "OVER_UNDER",
        "BTTS",
        "DOUBLE_CHANCE",
        "EXACT_SCORE",
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.log_level,
            "formatter": "default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.log_level,
            "formatter": "detailed",
            "filename": "logs/quantum_bot.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": settings.log_level,
            "propagate": False,
        },
    },
}