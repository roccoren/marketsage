#!/usr/bin/env python3
"""
Security utilities for the StockScan API.
Handles password hashing, JWT token creation and verification.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import ValidationError

from api.core.config import settings
from api.schemas.auth import TokenPayload

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_PREFIX}/auth/login"
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        subject: Subject of the token (usually user ID)
        expires_delta: Token expiration time
        
    Returns:
        JWT token as string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        subject: Subject of the token (usually user ID)
        expires_delta: Token expiration time
        
    Returns:
        JWT token as string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Get the current user from a JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        User information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        # Check token type and expiration
        if token_data.type != "access":
            raise credentials_exception
        if datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise credentials_exception
            
    except (JWTError, ValidationError):
        raise credentials_exception
    
    # In a real implementation, we would fetch the user from the database here
    # For now, we'll just return the user ID
    return {"user_id": token_data.sub}


def get_token_payload(token: str) -> Dict[str, Any]:
    """
    Decode a JWT token and return its payload.
    
    Args:
        token: JWT token
        
    Returns:
        Token payload
        
    Raises:
        JWTError: If token is invalid
    """
    return jwt.decode(
        token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
    )