#!/usr/bin/env python3
"""
Database models for the StockScan API.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Table, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship

from api.db.session import Base


# Association tables for many-to-many relationships
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True),
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id"), primary_key=True),
)


class BaseModel(Base):
    """Base model class with common fields and methods."""

    __abstract__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower() + "s"

    def dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result


class User(BaseModel):
    """User model for authentication and authorization."""

    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    schedules = relationship("Schedule", back_populates="user", cascade="all, delete-orphan")


class Role(BaseModel):
    """Role model for role-based access control."""

    name = Column(String(50), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary="role_permissions", back_populates="roles")


class Permission(BaseModel):
    """Permission model for fine-grained access control."""

    name = Column(String(100), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary="role_permissions", back_populates="permissions")


# Association table for roles and permissions
role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id"), primary_key=True),
    Column("permission_id", UUID(as_uuid=True), ForeignKey("permissions.id"), primary_key=True),
)


class RefreshToken(BaseModel):
    """Refresh token model for JWT authentication."""

    token = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")


# These model classes will be implemented as we progress
# They are commented out for now to avoid errors

"""
class Portfolio(BaseModel):
    # Portfolio model for portfolio management
    
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_simulation = Column(Boolean, default=False, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")


class Position(BaseModel):
    # Position model for portfolio positions
    
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_date = Column(DateTime, nullable=False)
    notes = Column(Text, nullable=True)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")


class Alert(BaseModel):
    # Alert model for stock alerts
    
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)  # price_target, prediction_signal, technical_indicator
    symbol = Column(String(20), nullable=False)
    conditions = Column(JSON, nullable=False)
    notification_channels = Column(ARRAY(String), nullable=False)
    status = Column(String(20), default="active", nullable=False)
    last_triggered = Column(DateTime, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="alerts")


class Schedule(BaseModel):
    # Schedule model for scheduled tasks
    
    name = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)  # data_refresh, prediction, backtest
    cron_expression = Column(String(100), nullable=True)
    interval_seconds = Column(Integer, nullable=True)
    params = Column(JSON, nullable=False)
    status = Column(String(20), default="active", nullable=False)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="schedules")
"""