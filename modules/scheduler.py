#!/usr/bin/env python3
"""
Scheduler module for the Qlib-based quantitative analysis project.
This module provides scheduling capabilities for recurring tasks
such as daily data updates.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from modules.config_manager import get_config
from modules.logging_manager import get_logger
from modules.event_dispatcher import dispatch_event, EventTypes, EventPriority


# Configure logger
logger = get_logger(__name__)


class SchedulerManager:
    """
    Scheduler manager for recurring tasks.
    """
    
    def __init__(self):
        """Initialize scheduler manager."""
        self.scheduler = None
        self.initialized = False
        self.jobs = {}
    
    def initialize(self) -> None:
        """Initialize the scheduler."""
        if self.initialized:
            return
        
        # Configure job stores and executors
        job_stores = {
            'default': MemoryJobStore()
        }
        
        max_threads = get_config('async.max_threads', 10)
        executors = {
            'default': ThreadPoolExecutor(max_threads)
        }
        
        # Create scheduler
        self.scheduler = BackgroundScheduler(
            jobstores=job_stores,
            executors=executors,
            timezone='UTC'
        )
        
        # Add default jobs
        self._add_default_jobs()
        
        self.initialized = True
        logger.info("Scheduler initialized")
    
    def _add_default_jobs(self) -> None:
        """Add default scheduled jobs."""
        # Schedule data updates if enabled
        update_schedule = get_config('data_acquisition.update_schedule', '0 0 * * *')
        if update_schedule:
            self.add_job(
                'data_update',
                self._data_update_job,
                update_schedule,
                replace_existing=True
            )
    
    def _data_update_job(self) -> None:
        """Job for updating stock data."""
        try:
            logger.info("Starting scheduled data update")
            
            # Import here to avoid circular imports
            from modules.data_acquisition import DataAcquisitionManager
            from modules.storage import StorageManager
            from modules.stock_manager import load_stock_symbols
            
            # Load configuration
            symbols_file = get_config('data_acquisition.symbols_file', 'stock_symbols.txt')
            start_date = get_config('data_acquisition.start_date', '2020-01-01')
            
            # Load symbols
            if os.path.exists(symbols_file):
                symbols = load_stock_symbols(symbols_file)
            else:
                # Default symbols if file doesn't exist
                symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']
            
            # Create managers
            storage_manager = StorageManager()
            data_manager = DataAcquisitionManager(storage_manager=storage_manager)
            
            # Fetch and store data
            results = data_manager.fetch_multiple_stocks(symbols, start_date=start_date)
            
            # Log results
            success_count = len(results)
            logger.info(f"Data update completed: {success_count}/{len(symbols)} symbols updated")
            
            # Dispatch event
            dispatch_event(
                EventTypes.DATA_UPDATED,
                {
                    'symbols': list(results.keys()),
                    'success_count': success_count,
                    'total_count': len(symbols)
                }
            )
        except Exception as e:
            logger.error(f"Error in data update job: {e}")
            
            # Dispatch error event
            dispatch_event(
                EventTypes.ERROR_OCCURRED,
                {
                    'source': 'scheduler',
                    'job': 'data_update',
                    'error': str(e)
                },
                EventPriority.HIGH
            )
    
    def add_job(self, job_id: str, func: Callable, schedule: str, replace_existing: bool = False, **kwargs) -> str:
        """
        Add a scheduled job.
        
        Args:
            job_id (str): Job identifier
            func (Callable): Function to execute
            schedule (str): Cron schedule expression
            replace_existing (bool): Whether to replace existing job with same ID
            **kwargs: Additional arguments for the job
            
        Returns:
            str: Job ID
        """
        if not self.initialized:
            self.initialize()
        
        # Create trigger
        trigger = CronTrigger.from_crontab(schedule)
        
        # Add job
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=replace_existing,
            **kwargs
        )
        
        self.jobs[job_id] = job
        logger.info(f"Added job '{job_id}' with schedule '{schedule}'")
        
        return job_id
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a scheduled job.
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            bool: True if job was removed, False otherwise
        """
        if not self.initialized or job_id not in self.jobs:
            return False
        
        self.scheduler.remove_job(job_id)
        del self.jobs[job_id]
        
        logger.info(f"Removed job '{job_id}'")
        return True
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.initialized:
            self.initialize()
        
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self.initialized and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")
    
    def get_jobs(self) -> Dict[str, Any]:
        """
        Get all scheduled jobs.
        
        Returns:
            Dict[str, Any]: Dictionary of job information
        """
        if not self.initialized:
            return {}
        
        jobs_info = {}
        for job_id, job in self.jobs.items():
            jobs_info[job_id] = {
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.strftime('%Y-%m-%d %H:%M:%S') if job.next_run_time else None
            }
        
        return jobs_info


# Default scheduler manager
_default_scheduler_manager = None


def get_scheduler_manager() -> SchedulerManager:
    """
    Get the default scheduler manager.
    
    Returns:
        SchedulerManager: Default scheduler manager
    """
    global _default_scheduler_manager
    
    if _default_scheduler_manager is None:
        _default_scheduler_manager = SchedulerManager()
    
    return _default_scheduler_manager


def add_scheduled_job(job_id: str, func: Callable, schedule: str, replace_existing: bool = False, **kwargs) -> str:
    """
    Add a scheduled job to the default scheduler.
    
    Args:
        job_id (str): Job identifier
        func (Callable): Function to execute
        schedule (str): Cron schedule expression
        replace_existing (bool): Whether to replace existing job with same ID
        **kwargs: Additional arguments for the job
        
    Returns:
        str: Job ID
    """
    return get_scheduler_manager().add_job(job_id, func, schedule, replace_existing, **kwargs)


def remove_scheduled_job(job_id: str) -> bool:
    """
    Remove a scheduled job from the default scheduler.
    
    Args:
        job_id (str): Job identifier
        
    Returns:
        bool: True if job was removed, False otherwise
    """
    return get_scheduler_manager().remove_job(job_id)


def start_scheduler() -> None:
    """Start the default scheduler."""
    get_scheduler_manager().start()


def stop_scheduler() -> None:
    """Stop the default scheduler."""
    get_scheduler_manager().stop()


def get_scheduled_jobs() -> Dict[str, Any]:
    """
    Get all scheduled jobs from the default scheduler.
    
    Returns:
        Dict[str, Any]: Dictionary of job information
    """
    return get_scheduler_manager().get_jobs()