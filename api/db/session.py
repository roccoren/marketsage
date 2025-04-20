#!/usr/bin/env python3
"""
Database session management for the StockScan API.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from api.core.config import settings

# Create SQLAlchemy engine
engine = create_engine(
    str(settings.SQLALCHEMY_DATABASE_URI),
    pool_pre_ping=True,  # Test connections before using them
    echo=settings.DEBUG,  # Log SQL queries if in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()