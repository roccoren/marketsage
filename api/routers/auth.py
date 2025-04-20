#!/usr/bin/env python3
"""
Authentication router for the StockScan API.
"""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from api.core.config import settings
from api.core.dependencies import DbSession, CurrentUser
from api.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    get_token_payload,
    verify_password,
)
from api.db.models import RefreshToken, User
from api.schemas.auth import (
    Login,
    PasswordChange,
    PasswordReset,
    RefreshToken as RefreshTokenSchema,
    Token,
    UserCreate,
    UserInDB,
)

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(DbSession),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    # Find user by username
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(user.id)
    
    # Create refresh token
    refresh_token_str = create_refresh_token(user.id)
    
    # Store refresh token in database
    refresh_token = RefreshToken(
        token=refresh_token_str,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(refresh_token)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token_str,
        "token_type": "bearer",
    }


@router.post("/login/alt", response_model=Token)
async def login_alt(
    login_data: Login,
    db: Session = Depends(DbSession),
) -> Any:
    """
    Alternative login endpoint that accepts JSON instead of form data.
    """
    # Find user by username
    user = db.query(User).filter(User.username == login_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(user.id)
    
    # Create refresh token
    refresh_token_str = create_refresh_token(user.id)
    
    # Store refresh token in database
    refresh_token = RefreshToken(
        token=refresh_token_str,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(refresh_token)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token_str,
        "token_type": "bearer",
    }


@router.post("/register", response_model=Token)
async def register(
    user_data: UserCreate,
    db: Session = Depends(DbSession),
) -> Any:
    """
    Register a new user.
    """
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create new user
    user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create access token
    access_token = create_access_token(user.id)
    
    # Create refresh token
    refresh_token_str = create_refresh_token(user.id)
    
    # Store refresh token in database
    refresh_token = RefreshToken(
        token=refresh_token_str,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(refresh_token)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token_str,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token_data: RefreshTokenSchema,
    db: Session = Depends(DbSession),
) -> Any:
    """
    Refresh access token.
    """
    # Find refresh token in database
    refresh_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == refresh_token_data.refresh_token)
        .first()
    )
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if refresh token is expired
    if refresh_token.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if refresh token is revoked
    if refresh_token.is_revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token payload
    try:
        payload = get_token_payload(refresh_token_data.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id = payload.get("sub")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Find user
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access token
    access_token = create_access_token(user.id)
    
    # Create new refresh token
    new_refresh_token_str = create_refresh_token(user.id)
    
    # Store new refresh token in database
    new_refresh_token = RefreshToken(
        token=new_refresh_token_str,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(new_refresh_token)
    
    # Revoke old refresh token
    refresh_token.is_revoked = True
    
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token_str,
        "token_type": "bearer",
    }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    refresh_token_data: RefreshTokenSchema,
    db: Session = Depends(DbSession),
) -> None:
    """
    Logout user by revoking refresh token.
    """
    # Find refresh token in database
    refresh_token = (
        db.query(RefreshToken)
        .filter(RefreshToken.token == refresh_token_data.refresh_token)
        .first()
    )
    if refresh_token:
        # Revoke refresh token
        refresh_token.is_revoked = True
        db.commit()


@router.post("/reset-password", status_code=status.HTTP_202_ACCEPTED)
async def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(DbSession),
) -> Any:
    """
    Request password reset.
    """
    # Find user by email
    user = db.query(User).filter(User.email == reset_data.email).first()
    if not user:
        # Return success even if user not found to prevent email enumeration
        return {"message": "If your email is registered, you will receive a password reset link"}
    
    # In a real implementation, we would send a password reset email here
    
    return {"message": "If your email is registered, you will receive a password reset link"}


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    change_data: PasswordChange,
    current_user: dict = CurrentUser,
    db: Session = Depends(DbSession),
) -> Any:
    """
    Change user password.
    """
    # Find user
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Verify current password
    if not verify_password(change_data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password",
        )
    
    # Update password
    user.hashed_password = get_password_hash(change_data.new_password)
    db.commit()
    
    # Revoke all refresh tokens
    db.query(RefreshToken).filter(RefreshToken.user_id == user.id).update(
        {"is_revoked": True}
    )
    db.commit()
    
    return {"message": "Password changed successfully"}


@router.get("/me", response_model=UserInDB)
async def get_current_user_info(
    current_user: dict = CurrentUser,
    db: Session = Depends(DbSession),
) -> Any:
    """
    Get current user info.
    """
    # Find user
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return user