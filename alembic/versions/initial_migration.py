"""initial migration

Revision ID: initial_migration
Revises: 
Create Date: 2025-04-20

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision = 'initial_migration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create roles table
    op.create_table(
        'roles',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(50), unique=True, nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    
    # Create permissions table
    op.create_table(
        'permissions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('username', sa.String(50), unique=True, nullable=False),
        sa.Column('email', sa.String(100), unique=True, nullable=False),
        sa.Column('full_name', sa.String(100), nullable=True),
        sa.Column('hashed_password', sa.String(100), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    
    # Create refresh_tokens table
    op.create_table(
        'refresh_tokens',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('token', sa.String(255), unique=True, nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, default=False),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    
    # Create association tables
    op.create_table(
        'user_roles',
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('role_id', UUID(as_uuid=True), sa.ForeignKey('roles.id'), primary_key=True),
    )
    
    op.create_table(
        'role_permissions',
        sa.Column('role_id', UUID(as_uuid=True), sa.ForeignKey('roles.id'), primary_key=True),
        sa.Column('permission_id', UUID(as_uuid=True), sa.ForeignKey('permissions.id'), primary_key=True),
    )
    
    # Create indexes
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_index(op.f('ix_roles_name'), 'roles', ['name'], unique=True)
    op.create_index(op.f('ix_permissions_name'), 'permissions', ['name'], unique=True)
    op.create_index(op.f('ix_refresh_tokens_token'), 'refresh_tokens', ['token'], unique=True)


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f('ix_refresh_tokens_token'), table_name='refresh_tokens')
    op.drop_index(op.f('ix_permissions_name'), table_name='permissions')
    op.drop_index(op.f('ix_roles_name'), table_name='roles')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    
    # Drop association tables
    op.drop_table('role_permissions')
    op.drop_table('user_roles')
    
    # Drop main tables
    op.drop_table('refresh_tokens')
    op.drop_table('users')
    op.drop_table('permissions')
    op.drop_table('roles')