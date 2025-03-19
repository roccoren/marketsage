#!/usr/bin/env python3
"""
StockScan Root Directory Cleanup Script

This script moves files from the root directory to appropriate locations in the new directory structure.
It moves or cleans up the following types of files:
1. Backtesting related scripts -> scripts/
2. Test scripts -> tests/
3. Data processing scripts -> scripts/
4. Documentation -> docs/
5. Docker related files -> kept in the root directory

Usage: python scripts/cleanup_root.py
"""

import os
import shutil
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='StockScan Root Directory Cleanup Tool')
    parser.add_argument('--dry-run', action='store_true', help='Do not perform actions, only print what would be done')
    parser.add_argument('--remove-src', action='store_true', help='Delete src directory after moving files')
    parser.add_argument('--remove-backup', action='store_true', help='Delete backup directory')
    return parser.parse_args()

def move_file(src_path, dest_path, dry_run=False):
    """Move file to target directory"""
    if not src_path.exists():
        logger.warning(f"Source file does not exist: {src_path}")
        return False
    
    # Ensure target directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        logger.info(f"[DRY RUN] Would move: {src_path} -> {dest_path}")
        return True
    
    try:
        shutil.move(src_path, dest_path)
        logger.info(f"Moved: {src_path} -> {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Error moving file: {src_path} -> {dest_path}, Error: {e}")
        return False

def copy_file(src_path, dest_path, dry_run=False):
    """Copy file to target directory"""
    if not src_path.exists():
        logger.warning(f"Source file does not exist: {src_path}")
        return False
    
    # Ensure target directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        logger.info(f"[DRY RUN] Would copy: {src_path} -> {dest_path}")
        return True
    
    try:
        shutil.copy2(src_path, dest_path)
        logger.info(f"Copied: {src_path} -> {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying file: {src_path} -> {dest_path}, Error: {e}")
        return False

def update_run_backtest(dry_run=False):
    """Update import paths in run_backtest.py file"""
    file_path = PROJECT_ROOT / "run_backtest.py"
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return
    
    if dry_run:
        logger.info(f"[DRY RUN] Would update file import paths: {file_path}")
        return
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Update import paths
        updated_content = content.replace('from src.', 'from stockscan.')
        updated_content = updated_content.replace('import src.', 'import stockscan.')
        
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Updated file import paths: {file_path}")
    except Exception as e:
        logger.error(f"Error updating file import paths: {file_path}, Error: {e}")

def cleanup_root_directory(args):
    """Clean up root directory, move files to appropriate locations"""
    # Move backtesting and optimization related scripts to scripts directory
    script_files = {
        "run_optimized_backtest.py": "scripts/run_optimized_backtest.py",
        "optimize_strategy.py": "scripts/optimize_strategy.py",
        "nyse_backtest_system.py": "scripts/nyse_backtest_system.py",
        "run_with_pythonpath.sh": "scripts/run_with_pythonpath.sh",
        "download_market_data.py": "scripts/download_market_data.py",
    }
    
    # Move test scripts to tests directory
    test_files = {
        "test_qlib_data.py": "tests/data/test_qlib_data.py",
        "inspect_data.py": "tests/data/inspect_data.py",
    }
    
    # Move documentation files to docs directory
    doc_files = {
        "directory_structure.md": "docs/directory_structure.md",
    }
    
    # Migrate requirements.txt file
    reqs_file = PROJECT_ROOT / "requirements.txt"
    if reqs_file.exists():
        base_reqs_file = PROJECT_ROOT / "requirements/base.txt"
        if not base_reqs_file.exists() or os.path.getsize(base_reqs_file) == 0:
            copy_file(reqs_file, base_reqs_file, args.dry_run)
    
    # Move script files
    for src, dest in script_files.items():
        move_file(PROJECT_ROOT / src, PROJECT_ROOT / dest, args.dry_run)
    
    # Move test files
    for src, dest in test_files.items():
        move_file(PROJECT_ROOT / src, PROJECT_ROOT / dest, args.dry_run)
    
    # Move documentation files
    for src, dest in doc_files.items():
        move_file(PROJECT_ROOT / src, PROJECT_ROOT / dest, args.dry_run)
    
    # Update import paths in run_backtest.py
    update_run_backtest(args.dry_run)
    
    # If specified, delete src directory
    if args.remove_src:
        src_dir = PROJECT_ROOT / "src"
        if src_dir.exists():
            if args.dry_run:
                logger.info(f"[DRY RUN] Would delete directory: {src_dir}")
            else:
                try:
                    # First ensure files in migration_map have been migrated
                    migration_map = {
                        "src/utils/exchange_data_loader.py": "stockscan/core/data_loader.py",
                        "src/strategies/base.py": "stockscan/strategies/base.py",
                        "src/strategies/backtester.py": "stockscan/strategies/backtester.py",
                        "src/strategies/optimizer.py": "stockscan/strategies/optimizer.py",
                        "src/strategies/macd.py": "stockscan/strategies/technical/macd.py",
                        "src/strategies/ma_crossover_strategy.py": "stockscan/strategies/technical/ma_crossover.py",
                        "src/strategies/multi_indicator_strategy.py": "stockscan/strategies/technical/multi_indicator.py",
                        "src/utils/technical_indicators.py": "stockscan/utils/technical_indicators.py",
                        "src/utils/backtest_formatter.py": "stockscan/utils/backtest_formatter.py",
                        "src/scanner.py": "stockscan/scanner/stock_scanner.py",
                    }
                    
                    all_migrated = True
                    for src_file, dest_file in migration_map.items():
                        if not (PROJECT_ROOT / dest_file).exists():
                            # If target file doesn't exist, attempt one last migration
                            src_path = PROJECT_ROOT / src_file
                            dest_path = PROJECT_ROOT / dest_file
                            if src_path.exists():
                                dest_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src_path, dest_path)
                                logger.info(f"Migrated: {src_file} -> {dest_file}")
                            else:
                                all_migrated = False
                                logger.warning(f"Both source and target files don't exist: {src_file} -> {dest_file}")
                    
                    if all_migrated:
                        shutil.rmtree(src_dir)
                        logger.info(f"Deleted src directory")
                    else:
                        logger.warning("Could not delete src directory because some files haven't been fully migrated")
                except Exception as e:
                    logger.error(f"Error deleting src directory: {e}")
    
    # If specified, delete backup directory
    if args.remove_backup:
        backup_dir = PROJECT_ROOT / "stockscan_backup"
        if backup_dir.exists():
            if args.dry_run:
                logger.info(f"[DRY RUN] Would delete backup directory: {backup_dir}")
            else:
                try:
                    shutil.rmtree(backup_dir)
                    logger.info(f"Deleted backup directory")
                except Exception as e:
                    logger.error(f"Error deleting backup directory: {e}")

def main():
    """Main function"""
    args = parse_args()
    
    logger.info("Starting cleanup of StockScan root directory")
    
    # Clean up root directory
    cleanup_root_directory(args)
    
    logger.info("Root directory cleanup complete")
    if args.dry_run:
        logger.info("This was a dry run, no actual operations were performed")

if __name__ == "__main__":
    main() 