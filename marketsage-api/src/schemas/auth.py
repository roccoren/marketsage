#!/usr/bin/env python3
"""
Authentication schemas for the MarketSage API.

This module defines Pydantic models for authentication and user management.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator


class Token(BaseModel):
    """Token schema for authentication response."""
    
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token data schema for decoded JWT tokens."""
    
    username: Optional[str] = None
    scopes: List[str] = []


class UserBase(BaseModel):
    """Base user schema with common fields."""
    
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    role: str = "user"


class UserCreate(UserBase):
    """User creation schema with password."""
    
    password: str = Field(..., min_length=8)
    
    @validator("password")
    def password_complexity(cls, v):
        """Validate password complexity."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """User update schema."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None
    
    @validator("password")
    def password_complexity(cls, v):
        """Validate password complexity if provided."""
        if v is None:
            return v
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserInDBBase(UserBase):
    """Base user schema for database models."""
    
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        
        from_attributes = True


class User(UserInDBBase):
    """User schema for API responses."""
    
    pass


class UserInDB(UserInDBBase):
    """User schema for database operations with hashed password."""
    
    hashed_password: str
    is_superuser: bool = False


class TokenRequest(BaseModel):
    """Token request schema for login."""
    
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    
    token: str
    new_password: str = Field(..., min_length=8)
    
    @validator("new_password")
    def password_complexity(cls, v):
        """Validate password complexity."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class ApiKeyCreate(BaseModel):
    """API key creation schema."""
    
    name: str
    scopes: List[str] = ["user"]
    expires_at: Optional[datetime] = None


class ApiKeyResponse(BaseModel):
    """API key response schema."""
    
    id: uuid.UUID
    name: str
    key: str
    scopes: List[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        
        from_attributes = True