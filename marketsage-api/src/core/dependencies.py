#!/usr/bin/env python3
"""
Dependency functions for the MarketSage API.

This module provides reusable dependencies for FastAPI routes, including:
- Database session management
- Authentication and authorization
- Integration with other MarketSage components
"""

import logging
from typing import Annotated, Generator, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from marketsage_api.core.config import settings
from marketsage_api.db.session import SessionLocal
from marketsage_api.schemas.auth import TokenData, UserInDB
from marketsage_api.services.auth import get_user_by_username

# Configure logging
logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_PREFIX}/auth/token",
    scopes={
        "user": "Standard user access",
        "analyst": "Analyst access with additional permissions",
        "admin": "Administrator access with all permissions",
    },
)


def get_db() -> Generator[Session, None, None]:
    """
    Create a new database session for a request.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Type alias for database session dependency
DbSession = Annotated[Session, Depends(get_db)]


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserInDB:
    """
    Get the current authenticated user based on the JWT token.
    
    Args:
        security_scopes: Security scopes required for the endpoint
        token: JWT token from the request
        db: Database session
        
    Returns:
        UserInDB: Current user information
        
    Raises:
        HTTPException: If authentication fails
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
            
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=token_scopes)
        
    except (JWTError, ValidationError):
        logger.error("Failed to decode JWT token", exc_info=True)
        raise credentials_exception
        
    # Get user from database
    user = get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
        
    # Check if user has required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )
            
    return user


async def get_current_active_user(
    current_user: Annotated[UserInDB, Depends(get_current_user)]
) -> UserInDB:
    """
    Get the current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        UserInDB: Current active user information
        
    Raises:
        HTTPException: If the user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


# Type aliases for user role dependencies
CurrentUser = Annotated[UserInDB, Security(get_current_active_user, scopes=["user"])]
AnalystUser = Annotated[UserInDB, Security(get_current_active_user, scopes=["analyst"])]
AdminUser = Annotated[UserInDB, Security(get_current_active_user, scopes=["admin"])]


# Service client dependencies
def get_datahub_client():
    """
    Get a client for the DataHub service.
    
    Returns:
        DataHubClient: Client for interacting with the DataHub service
    """
    from marketsage_api.services.datahub import DataHubClient
    
    return DataHubClient(base_url=settings.DATAHUB_URL)


def get_marketsage_core_client():
    """
    Get a client for the MarketSage Core service.
    
    Returns:
        MarketSageCoreClient: Client for interacting with the MarketSage Core service
    """
    from marketsage_api.services.marketsage_core import MarketSageCoreClient
    
    return MarketSageCoreClient(base_url=settings.MARKETSAGE_CORE_URL)


def get_tsds_client():
    """
    Get a client for the Time Series Data Service.
    
    Returns:
        TSDSClient: Client for interacting with the Time Series Data Service
    """
    from marketsage_api.services.tsds import TSDSClient
    
    return TSDSClient(
        host=settings.TSDS_HOST,
        port=settings.TSDS_PORT,
        timeout=settings.TSDS_TIMEOUT,
    )


def get_rabbitmq_producer():
    """
    Get a RabbitMQ producer for sending messages.
    
    Returns:
        RabbitMQProducer: Producer for sending messages to RabbitMQ
    """
    from marketsage_api.services.rabbitmq import RabbitMQProducer
    
    return RabbitMQProducer(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_PORT,
        username=settings.RABBITMQ_USER,
        password=settings.RABBITMQ_PASSWORD,
        virtual_host=settings.RABBITMQ_VHOST,
    )


# Type aliases for service client dependencies
DataHubClient = Annotated[object, Depends(get_datahub_client)]
MarketSageCoreClient = Annotated[object, Depends(get_marketsage_core_client)]
TSDSClient = Annotated[object, Depends(get_tsds_client)]
RabbitMQProducer = Annotated[object, Depends(get_rabbitmq_producer)]