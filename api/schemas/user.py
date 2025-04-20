#!/usr/bin/env python3
"""
User schemas for the StockScan API.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field

from api.schemas.auth import UserBase


class RoleBase(BaseModel):
    """Base role schema."""
    
    name: str
    description: Optional[str] = None


class RoleCreate(RoleBase):
    """Role creation schema."""
    
    pass


class RoleUpdate(BaseModel):
    """Role update schema."""
    
    name: Optional[str] = None
    description: Optional[str] = None


class Role(RoleBase):
    """Role schema (response model)."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class PermissionBase(BaseModel):
    """Base permission schema."""
    
    name: str
    description: Optional[str] = None


class PermissionCreate(PermissionBase):
    """Permission creation schema."""
    
    pass


class PermissionUpdate(BaseModel):
    """Permission update schema."""
    
    name: Optional[str] = None
    description: Optional[str] = None


class Permission(PermissionBase):
    """Permission schema (response model)."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class UserWithRoles(UserBase):
    """User with roles schema."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    roles: List[Role] = []
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class UserPreferences(BaseModel):
    """User preferences schema."""
    
    default_prediction_period: str = "30d"
    notification_preferences: dict = Field(default_factory=lambda: {
        "email": True,
        "sms": False,
        "in_app": True
    })


class UserPreferencesUpdate(BaseModel):
    """User preferences update schema."""
    
    default_prediction_period: Optional[str] = None
    notification_preferences: Optional[dict] = None


class UserRoleUpdate(BaseModel):
    """User role update schema."""
    
    roles: List[str]


class UserList(BaseModel):
    """User list schema."""
    
    users: List[UserWithRoles]
    total: int
    page: int
    page_size: int
    pages: int


class UserStats(BaseModel):
    """User statistics schema."""
    
    total_users: int
    active_users: int
    inactive_users: int
    admin_users: int
    analyst_users: int
    regular_users: int