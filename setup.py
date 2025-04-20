#!/usr/bin/env python3
"""
Setup script for the StockScan API.
This script initializes the project by:
1. Creating the database
2. Running migrations
3. Seeding the database with initial data
"""

import argparse
import os
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup the StockScan API")
    
    parser.add_argument(
        "--skip-db-creation",
        action="store_true",
        help="Skip database creation step",
    )
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Skip running migrations",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Skip seeding the database",
    )
    
    return parser.parse_args()


def create_database():
    """Create the PostgreSQL database."""
    db_name = os.getenv("POSTGRES_DB", "stockscan")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db_host = os.getenv("POSTGRES_SERVER", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    
    # Check if database already exists
    check_cmd = [
        "psql",
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres",
        "-c", f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'",
        "-t"
    ]
    
    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        if result.stdout.strip() == "1":
            print(f"Database '{db_name}' already exists.")
            return
    except Exception as e:
        print(f"Error checking if database exists: {e}")
        sys.exit(1)
    
    # Create database
    create_cmd = [
        "psql",
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres",
        "-c", f"CREATE DATABASE {db_name}"
    ]
    
    try:
        print(f"Creating database '{db_name}'...")
        subprocess.run(create_cmd, check=True)
        print(f"Database '{db_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating database: {e}")
        sys.exit(1)


def run_migrations():
    """Run database migrations using Alembic."""
    try:
        print("Running database migrations...")
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        print("Migrations completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running migrations: {e}")
        sys.exit(1)


def seed_database():
    """Seed the database with initial data."""
    try:
        print("Seeding database...")
        subprocess.run(["python", "-m", "api.db.seed"], check=True)
        print("Database seeded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error seeding database: {e}")
        sys.exit(1)


def main():
    """Run the setup process."""
    args = parse_args()
    
    print("Setting up StockScan API...")
    
    if not args.skip_db_creation:
        create_database()
    else:
        print("Skipping database creation.")
    
    if not args.skip_migrations:
        run_migrations()
    else:
        print("Skipping migrations.")
    
    if not args.skip_seed:
        seed_database()
    else:
        print("Skipping database seeding.")
    
    print("\nSetup completed successfully!")
    print("\nYou can now run the API server with:")
    print("  python run_api.py")
    print("\nAPI documentation will be available at:")
    print("  http://localhost:8000/api/docs")


if __name__ == "__main__":
    main()