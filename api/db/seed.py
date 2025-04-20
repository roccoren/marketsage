#!/usr/bin/env python3
"""
Seed script for the StockScan API database.
This script creates default roles, permissions, and an admin user.
"""

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from api.core.security import get_password_hash
from api.db.models import Permission, Role, User
from api.db.session import SessionLocal


def create_roles(db: Session) -> dict:
    """Create default roles."""
    roles = {
        "admin": Role(
            id=uuid.uuid4(),
            name="admin",
            description="Administrator role with all permissions",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "analyst": Role(
            id=uuid.uuid4(),
            name="analyst",
            description="Analyst role with permissions to manage predictions and data",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "user": Role(
            id=uuid.uuid4(),
            name="user",
            description="Regular user role with basic permissions",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
    }
    
    for role in roles.values():
        db.add(role)
    
    db.commit()
    
    for role in roles.values():
        db.refresh(role)
    
    return roles


def create_permissions(db: Session) -> dict:
    """Create default permissions."""
    permissions = {
        # User management permissions
        "user:read": Permission(
            id=uuid.uuid4(),
            name="user:read",
            description="Read user information",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "user:create": Permission(
            id=uuid.uuid4(),
            name="user:create",
            description="Create users",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "user:update": Permission(
            id=uuid.uuid4(),
            name="user:update",
            description="Update users",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "user:delete": Permission(
            id=uuid.uuid4(),
            name="user:delete",
            description="Delete users",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        
        # Stock data permissions
        "stock:read": Permission(
            id=uuid.uuid4(),
            name="stock:read",
            description="Read stock data",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "stock:refresh": Permission(
            id=uuid.uuid4(),
            name="stock:refresh",
            description="Refresh stock data",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        
        # Prediction permissions
        "prediction:read": Permission(
            id=uuid.uuid4(),
            name="prediction:read",
            description="Read predictions",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "prediction:create": Permission(
            id=uuid.uuid4(),
            name="prediction:create",
            description="Create predictions",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        
        # Schedule permissions
        "schedule:read": Permission(
            id=uuid.uuid4(),
            name="schedule:read",
            description="Read schedules",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "schedule:create": Permission(
            id=uuid.uuid4(),
            name="schedule:create",
            description="Create schedules",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "schedule:update": Permission(
            id=uuid.uuid4(),
            name="schedule:update",
            description="Update schedules",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
        "schedule:delete": Permission(
            id=uuid.uuid4(),
            name="schedule:delete",
            description="Delete schedules",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ),
    }
    
    for permission in permissions.values():
        db.add(permission)
    
    db.commit()
    
    for permission in permissions.values():
        db.refresh(permission)
    
    return permissions


def assign_permissions_to_roles(db: Session, roles: dict, permissions: dict) -> None:
    """Assign permissions to roles."""
    # Admin role gets all permissions
    admin_role = roles["admin"]
    for permission in permissions.values():
        admin_role.permissions.append(permission)
    
    # Analyst role gets stock and prediction permissions
    analyst_role = roles["analyst"]
    analyst_permissions = [
        "stock:read", "stock:refresh",
        "prediction:read", "prediction:create",
        "schedule:read", "schedule:create", "schedule:update", "schedule:delete",
    ]
    for perm_name in analyst_permissions:
        analyst_role.permissions.append(permissions[perm_name])
    
    # User role gets basic read permissions
    user_role = roles["user"]
    user_permissions = [
        "user:read", "stock:read", "prediction:read", "schedule:read",
    ]
    for perm_name in user_permissions:
        user_role.permissions.append(permissions[perm_name])
    
    db.commit()


def create_admin_user(db: Session, roles: dict) -> None:
    """Create an admin user."""
    admin_user = User(
        id=uuid.uuid4(),
        username="admin",
        email="admin@stockscan.com",
        full_name="Admin User",
        hashed_password=get_password_hash("admin123"),  # In production, use a secure password
        is_active=True,
        is_superuser=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    db.add(admin_user)
    db.commit()
    db.refresh(admin_user)
    
    # Assign admin role to admin user
    admin_user.roles.append(roles["admin"])
    db.commit()


def seed_database() -> None:
    """Seed the database with initial data."""
    db = SessionLocal()
    try:
        # Check if database is already seeded
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        if admin_role:
            print("Database already seeded.")
            return
        
        print("Seeding database...")
        
        # Create roles
        roles = create_roles(db)
        print("Created roles.")
        
        # Create permissions
        permissions = create_permissions(db)
        print("Created permissions.")
        
        # Assign permissions to roles
        assign_permissions_to_roles(db, roles, permissions)
        print("Assigned permissions to roles.")
        
        # Create admin user
        create_admin_user(db, roles)
        print("Created admin user.")
        
        print("Database seeded successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()