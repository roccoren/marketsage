#!/usr/bin/env python3
"""
Authentication schemas for the StockScan API.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator


class Token(BaseModel):
    """Token schema."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Token payload schema."""
    
    sub: Optional[str] = None
    exp: int = 0
    type: str = "access"


class UserBase(BaseModel):
    """Base user schema."""
    
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True


class UserCreate(UserBase):
    """User creation schema."""
    
    password: str = Field(..., min_length=8)
    
    @validator("password")
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        return v


class UserUpdate(BaseModel):
    """User update schema."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    
    @validator("password")
    def password_strength(cls, v):
        """Validate password strength."""
        if v is None:
            return v
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        return v


class UserInDBBase(UserBase):
    """User in DB base schema."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class User(UserInDBBase):
    """User schema (response model)."""
    
    roles: List[str] = []


class UserInDB(UserInDBBase):
    """User in DB schema (internal use only)."""
    
    hashed_password: str


class Login(BaseModel):
    """Login schema."""
    
    username: str
    password: str


class RefreshToken(BaseModel):
    """Refresh token schema."""
    
    refresh_token: str


class PasswordReset(BaseModel):
    """Password reset schema."""
    
    email: EmailStr


class PasswordChange(BaseModel):
    """Password change schema."""
    
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator("new_password")
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        return v