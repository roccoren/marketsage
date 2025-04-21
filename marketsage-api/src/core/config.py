#!/usr/bin/env python3
"""
Configuration settings for the MarketSage API.
"""

import os
import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, PostgresDsn, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API configuration settings."""

    # API General Settings
    PROJECT_NAME: str = "MarketSage API"
    PROJECT_DESCRIPTION: str = "API for MarketSage ecosystem - stock data analysis, predictions, and portfolio management"
    PROJECT_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security Settings
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost", "http://localhost:3000", "http://localhost:8000"]

    # Database Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "marketsage"
    POSTGRES_PORT: int = 5432
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.data.get("POSTGRES_USER"),
            password=values.data.get("POSTGRES_PASSWORD"),
            host=values.data.get("POSTGRES_SERVER"),
            port=values.data.get("POSTGRES_PORT"),
            path=f"{values.data.get('POSTGRES_DB') or ''}",
        )

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CACHE_EXPIRE_SECONDS: int = 3600  # 1 hour default cache expiry

    # Component Integration Settings
    DATAHUB_URL: str = "http://localhost:8001"
    MARKETSAGE_CORE_URL: str = "http://localhost:8002"
    
    # Time Series Data Service Settings
    TSDS_HOST: str = "localhost"
    TSDS_PORT: int = 50051
    TSDS_TIMEOUT: int = 30  # seconds

    # RabbitMQ Settings
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    RABBITMQ_VHOST: str = "/"
    
    # Exchange and Queue Settings
    DATA_EXCHANGE: str = "data.exchange"
    ANALYSIS_EXCHANGE: str = "analysis.exchange"
    SYSTEM_EXCHANGE: str = "system.exchange"
    API_NOTIFICATIONS_QUEUE: str = "api.notifications"

    # Email Settings for Notifications
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = 587
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # SMS Settings for Notifications
    SMS_PROVIDER: Optional[str] = None
    SMS_API_KEY: Optional[str] = None

    # Scheduler Settings
    SCHEDULER_TIMEZONE: str = "UTC"
    SCHEDULER_MAX_INSTANCES: int = 10
    SCHEDULER_MISFIRE_GRACE_TIME: int = 60  # seconds

    # Data Retention Settings
    PREDICTION_RETENTION_DAYS: int = 365  # 1 year

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "env_nested_delimiter": None,
        "extra": "ignore",
    }


# Create settings instance
settings = Settings()