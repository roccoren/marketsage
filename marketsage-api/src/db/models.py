#!/usr/bin/env python3
"""
Database models for the MarketSage API.

This module defines SQLAlchemy models for the MarketSage API database.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String, Table, Text
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from marketsage_api.db.session import Base


# Association table for many-to-many relationships
user_watchlist = Table(
    "user_watchlist",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True),
    Column("stock_id", UUID(as_uuid=True), ForeignKey("stocks.id"), primary_key=True),
)


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # User roles: user, analyst, admin
    role = Column(String(64), default="user")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    watchlist = relationship("Stock", secondary=user_watchlist, back_populates="watched_by")
    predictions = relationship("Prediction", back_populates="user")
    schedules = relationship("Schedule", back_populates="user")


class Stock(Base):
    """Stock model for storing stock information."""
    
    __tablename__ = "stocks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(16), unique=True, index=True, nullable=False)
    name = Column(String(255))
    exchange = Column(String(64))
    sector = Column(String(128))
    industry = Column(String(128))
    country = Column(String(64))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    watched_by = relationship("User", secondary=user_watchlist, back_populates="watchlist")
    predictions = relationship("Prediction", back_populates="stock")
    schedules = relationship("Schedule", back_populates="stock")


class Prediction(Base):
    """Prediction model for storing prediction results."""
    
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Prediction type (e.g., price, trend, signal)
    prediction_type = Column(String(64), nullable=False)
    
    # Prediction parameters
    parameters = Column(JSONB)
    
    # Prediction results
    results = Column(JSONB)
    
    # Prediction status
    status = Column(
        Enum("pending", "running", "completed", "failed", name="prediction_status"),
        default="pending",
        nullable=False,
    )
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    stock = relationship("Stock", back_populates="predictions")
    user = relationship("User", back_populates="predictions")


class Schedule(Base):
    """Schedule model for storing scheduled tasks."""
    
    __tablename__ = "schedules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Schedule name and description
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Schedule type (e.g., data_acquisition, prediction, alert)
    schedule_type = Column(String(64), nullable=False)
    
    # Schedule parameters
    parameters = Column(JSONB)
    
    # Schedule expression (cron format)
    cron_expression = Column(String(64), nullable=False)
    
    # Schedule status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    
    # Relationships
    stock = relationship("Stock", back_populates="schedules")
    user = relationship("User", back_populates="schedules")


class ApiKey(Base):
    """API key model for API authentication."""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # API key name and value
    name = Column(String(255), nullable=False)
    key = Column(String(64), unique=True, index=True, nullable=False)
    
    # API key scopes
    scopes = Column(JSONB, default=lambda: ["user"])
    
    # API key status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User")