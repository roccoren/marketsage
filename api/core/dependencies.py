#!/usr/bin/env python3
"""
Dependencies for the StockScan API.
Contains common dependencies used across API endpoints.
"""

from typing import Annotated, Generator, List, Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.core.security import get_current_user
from api.db.session import SessionLocal

# Database dependency
def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Current user dependency
async def get_current_active_user(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Get the current active user.
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        Current user information
        
    Raises:
        HTTPException: If user is inactive
    """
    # In a real implementation, we would check if the user is active
    # For now, we'll just return the user
    return current_user


# Role-based access control dependencies
async def get_current_admin_user(
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """
    Get the current user with admin role.
    
    Args:
        current_user: Current user
        
    Returns:
        Current user information
        
    Raises:
        HTTPException: If user is not an admin
    """
    # In a real implementation, we would check the user's roles from the database
    # For now, we'll just raise an exception
    # This is a placeholder that will be updated once we implement the user model
    is_admin = False  # This would be a database lookup
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user


async def get_current_analyst_user(
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """
    Get the current user with analyst role.
    
    Args:
        current_user: Current user
        
    Returns:
        Current user information
        
    Raises:
        HTTPException: If user is not an analyst
    """
    # In a real implementation, we would check the user's roles from the database
    # For now, we'll just raise an exception
    # This is a placeholder that will be updated once we implement the user model
    is_analyst = False  # This would be a database lookup
    if not is_analyst:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user


# Permission checking dependency
async def check_permissions(
    required_permissions: List[str],
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """
    Check if the current user has the required permissions.
    
    Args:
        required_permissions: List of required permissions
        current_user: Current user
        
    Returns:
        Current user information
        
    Raises:
        HTTPException: If user doesn't have the required permissions
    """
    # In a real implementation, we would check the user's permissions from the database
    # For now, we'll just raise an exception
    # This is a placeholder that will be updated once we implement the permission model
    user_permissions = []  # This would be a database lookup
    has_permission = any(perm in user_permissions for perm in required_permissions)
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have the required permissions",
        )
    return current_user


# Type annotations for common dependencies
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[dict, Depends(get_current_active_user)]
AdminUser = Annotated[dict, Depends(get_current_admin_user)]
AnalystUser = Annotated[dict, Depends(get_current_analyst_user)]