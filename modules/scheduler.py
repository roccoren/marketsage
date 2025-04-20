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


class Scheduler:
    """
    Scheduler class for the StockScan API.
    This class provides a simplified interface to the SchedulerManager.
    """
    
    def __init__(self):
        """Initialize scheduler."""
        self.manager = get_scheduler_manager()
        self.schedules = {}
        self.executions = {}
    
    def calculate_next_run(self, cron_expression: Optional[str] = None, interval_seconds: Optional[int] = None) -> datetime:
        """
        Calculate next run time based on cron expression or interval.
        
        Args:
            cron_expression (str, optional): Cron expression
            interval_seconds (int, optional): Interval in seconds
            
        Returns:
            datetime: Next run time
        """
        if cron_expression:
            # Use APScheduler's CronTrigger to calculate next run time
            trigger = CronTrigger.from_crontab(cron_expression)
            next_run = trigger.get_next_fire_time(None, datetime.utcnow())
        elif interval_seconds:
            # Calculate next run time based on interval
            next_run = datetime.utcnow() + timedelta(seconds=interval_seconds)
        else:
            # Default to now
            next_run = datetime.utcnow()
        
        return next_run
    
    def add_schedule(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a schedule.
        
        Args:
            schedule (Dict[str, Any]): Schedule data
            
        Returns:
            Dict[str, Any]: Added schedule
        """
        # Store schedule
        self.schedules[schedule["id"]] = schedule
        
        # Add job to scheduler if schedule is active
        if schedule["status"] == "active":
            if "cron_expression" in schedule and schedule["cron_expression"]:
                self.manager.add_job(
                    f"schedule_{schedule['id']}",
                    lambda: self._execute_schedule(schedule["id"]),
                    schedule["cron_expression"],
                    replace_existing=True,
                )
            elif "interval_seconds" in schedule and schedule["interval_seconds"]:
                # For interval-based schedules, we need to create a cron expression
                # that runs every N seconds
                seconds = schedule["interval_seconds"]
                if seconds < 60:
                    cron = f"*/{seconds} * * * * *"  # Every N seconds
                elif seconds < 3600:
                    minutes = seconds // 60
                    cron = f"0 */{minutes} * * * *"  # Every N minutes
                else:
                    hours = seconds // 3600
                    cron = f"0 0 */{hours} * * *"  # Every N hours
                
                self.manager.add_job(
                    f"schedule_{schedule['id']}",
                    lambda: self._execute_schedule(schedule["id"]),
                    cron,
                    replace_existing=True,
                )
        
        return schedule
    
    def update_schedule(self, schedule_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a schedule.
        
        Args:
            schedule_id (str): Schedule ID
            update_data (Dict[str, Any]): Update data
            
        Returns:
            Dict[str, Any]: Updated schedule
        """
        # Check if schedule exists
        if schedule_id not in self.schedules:
            raise ValueError(f"Schedule with ID {schedule_id} not found")
        
        # Update schedule
        schedule = self.schedules[schedule_id]
        schedule.update(update_data)
        
        # Update job if schedule is active
        if schedule["status"] == "active":
            if "cron_expression" in schedule and schedule["cron_expression"]:
                self.manager.add_job(
                    f"schedule_{schedule_id}",
                    lambda: self._execute_schedule(schedule_id),
                    schedule["cron_expression"],
                    replace_existing=True,
                )
            elif "interval_seconds" in schedule and schedule["interval_seconds"]:
                # For interval-based schedules, we need to create a cron expression
                # that runs every N seconds
                seconds = schedule["interval_seconds"]
                if seconds < 60:
                    cron = f"*/{seconds} * * * * *"  # Every N seconds
                elif seconds < 3600:
                    minutes = seconds // 60
                    cron = f"0 */{minutes} * * * *"  # Every N minutes
                else:
                    hours = seconds // 3600
                    cron = f"0 0 */{hours} * * *"  # Every N hours
                
                self.manager.add_job(
                    f"schedule_{schedule_id}",
                    lambda: self._execute_schedule(schedule_id),
                    cron,
                    replace_existing=True,
                )
        else:
            # Remove job if schedule is not active
            self.manager.remove_job(f"schedule_{schedule_id}")
        
        return schedule
    
    def delete_schedule(self, schedule_id: str) -> None:
        """
        Delete a schedule.
        
        Args:
            schedule_id (str): Schedule ID
        """
        # Check if schedule exists
        if schedule_id not in self.schedules:
            raise ValueError(f"Schedule with ID {schedule_id} not found")
        
        # Remove job from scheduler
        self.manager.remove_job(f"schedule_{schedule_id}")
        
        # Remove schedule
        del self.schedules[schedule_id]
    
    def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a schedule.
        
        Args:
            schedule_id (str): Schedule ID
            
        Returns:
            Dict[str, Any]: Schedule data or None if not found
        """
        return self.schedules.get(schedule_id)
    
    def get_schedules(self, user_id: Optional[str] = None, task_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get schedules with optional filtering.
        
        Args:
            user_id (str, optional): Filter by user ID
            task_type (str, optional): Filter by task type
            status (str, optional): Filter by status
            
        Returns:
            List[Dict[str, Any]]: List of schedules
        """
        schedules = list(self.schedules.values())
        
        # Apply filters
        if user_id:
            schedules = [s for s in schedules if s.get("user_id") == user_id]
        if task_type:
            schedules = [s for s in schedules if s.get("task_type") == task_type]
        if status:
            schedules = [s for s in schedules if s.get("status") == status]
        
        return schedules
    
    def execute_task(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task_type (str): Task type
            params (Dict[str, Any]): Task parameters
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Execute task based on type
        if task_type == "data_refresh":
            return self._execute_data_refresh_task(params)
        elif task_type == "prediction":
            return self._execute_prediction_task(params)
        elif task_type == "backtest":
            return self._execute_backtest_task(params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _execute_data_refresh_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data refresh task.
        
        Args:
            params (Dict[str, Any]): Task parameters
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Import here to avoid circular imports
        from modules.data_acquisition import DataAcquisitionManager
        from modules.storage import StorageManager
        
        # Get parameters
        symbols = params.get("symbols", [])
        start_date = params.get("start_date", "2020-01-01")
        interval = params.get("interval", "1d")
        
        # Create managers
        storage_manager = StorageManager()
        data_manager = DataAcquisitionManager(storage_manager=storage_manager)
        
        # Fetch and store data
        results = {}
        for symbol in symbols:
            try:
                data = data_manager.fetch_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    store=True,
                )
                results[symbol] = "success"
            except Exception as e:
                results[symbol] = f"error: {str(e)}"
        
        return {
            "task_type": "data_refresh",
            "symbols": symbols,
            "results": results,
            "success_count": sum(1 for r in results.values() if r == "success"),
            "total_count": len(symbols),
        }
    
    def _execute_prediction_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prediction task.
        
        Args:
            params (Dict[str, Any]): Task parameters
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Import here to avoid circular imports
        from modules.forecasting import ForecastingManager
        
        # Get parameters
        symbols = params.get("symbols", [])
        model_type = params.get("model_type", "ensemble")
        prediction_period = params.get("prediction_period", "30d")
        
        # Create forecasting manager
        forecasting = ForecastingManager()
        
        # Generate predictions
        results = {}
        for symbol in symbols:
            try:
                prediction = forecasting.generate_forecast(
                    symbol=symbol,
                    model_type=model_type,
                )
                results[symbol] = {
                    "status": "success",
                    "forecast": prediction.get("forecast_values", []),
                    "trend": prediction.get("trend"),
                    "volatility": prediction.get("volatility"),
                    "overall_action": prediction.get("overall_action"),
                }
            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "error": str(e),
                }
        
        return {
            "task_type": "prediction",
            "symbols": symbols,
            "model_type": model_type,
            "prediction_period": prediction_period,
            "results": results,
            "success_count": sum(1 for r in results.values() if r.get("status") == "success"),
            "total_count": len(symbols),
        }
    
    def _execute_backtest_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute backtest task.
        
        Args:
            params (Dict[str, Any]): Task parameters
            
        Returns:
            Dict[str, Any]: Task result
        """
        # Import here to avoid circular imports
        from modules.backtesting import BacktestManager
        
        # Get parameters
        symbols = params.get("symbols", [])
        start_date = params.get("start_date", "2020-01-01")
        end_date = params.get("end_date")
        strategy = params.get("strategy", "simple")
        
        # Create backtest manager
        backtest = BacktestManager()
        
        # Run backtests
        results = {}
        for symbol in symbols:
            try:
                result = backtest.run_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                )
                results[symbol] = {
                    "status": "success",
                    "returns": result.get("returns"),
                    "sharpe_ratio": result.get("sharpe_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                }
            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "error": str(e),
                }
        
        return {
            "task_type": "backtest",
            "symbols": symbols,
            "strategy": strategy,
            "start_date": start_date,
            "end_date": end_date,
            "results": results,
            "success_count": sum(1 for r in results.values() if r.get("status") == "success"),
            "total_count": len(symbols),
        }
    
    def _execute_schedule(self, schedule_id: str) -> None:
        """
        Execute a schedule.
        
        Args:
            schedule_id (str): Schedule ID
        """
        # Check if schedule exists
        if schedule_id not in self.schedules:
            return
        
        # Get schedule
        schedule = self.schedules[schedule_id]
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        execution = {
            "id": execution_id,
            "schedule_id": schedule_id,
            "start_time": datetime.utcnow(),
            "status": "running",
        }
        
        try:
            # Execute task
            result = self.execute_task(
                task_type=schedule["task_type"],
                params=schedule["params"],
            )
            
            # Update execution
            execution["end_time"] = datetime.utcnow()
            execution["status"] = "completed"
            execution["result"] = result
            
        except Exception as e:
            # Update execution with error
            execution["end_time"] = datetime.utcnow()
            execution["status"] = "failed"
            execution["error"] = str(e)
        
        # Store execution
        self.store_execution(execution)
    
    def store_execution(self, execution: Dict[str, Any]) -> None:
        """
        Store execution record.
        
        Args:
            execution (Dict[str, Any]): Execution data
        """
        self.executions[execution["id"]] = execution
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution record.
        
        Args:
            execution_id (str): Execution ID
            
        Returns:
            Dict[str, Any]: Execution data or None if not found
        """
        return self.executions.get(execution_id)
    
    def get_executions(self, schedule_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get execution records with optional filtering.
        
        Args:
            schedule_id (str, optional): Filter by schedule ID
            status (str, optional): Filter by status
            
        Returns:
            List[Dict[str, Any]]: List of execution records
        """
        executions = list(self.executions.values())
        
        # Apply filters
        if schedule_id:
            executions = [e for e in executions if e.get("schedule_id") == schedule_id]
        if status:
            executions = [e for e in executions if e.get("status") == status]
        
        # Sort by start time (newest first)
        executions.sort(key=lambda e: e.get("start_time", datetime.min), reverse=True)
        
        return executions