#!/usr/bin/env python3
"""
Schedule schemas for the StockScan API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ScheduleParams(BaseModel):
    """Base class for schedule parameters."""
    pass


class DataRefreshParams(ScheduleParams):
    """Data refresh schedule parameters."""
    
    symbols: List[str]
    interval: str = "1d"  # 1d, 5m, etc.
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    @validator("interval")
    def validate_interval(cls, v):
        """Validate interval."""
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval: {v}. Must be one of {valid_intervals}")
        return v


class PredictionParams(ScheduleParams):
    """Prediction schedule parameters."""
    
    symbols: List[str]
    model_type: str = "lstm"
    prediction_period: str = "30d"
    
    @validator("prediction_period")
    def validate_prediction_period(cls, v):
        """Validate prediction period."""
        valid_periods = ["1d", "5d", "10d", "30d", "90d", "180d", "220d"]
        if v not in valid_periods:
            raise ValueError(f"Invalid prediction period: {v}. Must be one of {valid_periods}")
        return v
    
    @validator("model_type")
    def validate_model_type(cls, v):
        """Validate model type."""
        valid_models = ["lstm", "ensemble", "linear", "arima", "prophet"]
        if v not in valid_models:
            raise ValueError(f"Invalid model type: {v}. Must be one of {valid_models}")
        return v


class BacktestParams(ScheduleParams):
    """Backtest schedule parameters."""
    
    symbols: List[str]
    strategy: str
    strategy_params: Dict[str, Union[str, int, float, bool]]
    start_date: str
    end_date: str


class ScheduleCreate(BaseModel):
    """Schedule creation schema."""
    
    name: str
    task_type: str  # data_refresh, prediction, backtest
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    params: Union[DataRefreshParams, PredictionParams, BacktestParams]
    
    @validator("task_type")
    def validate_task_type(cls, v):
        """Validate task type."""
        valid_types = ["data_refresh", "prediction", "backtest"]
        if v not in valid_types:
            raise ValueError(f"Invalid task type: {v}. Must be one of {valid_types}")
        return v
    
    @validator("cron_expression", "interval_seconds")
    def validate_timing(cls, v, values):
        """Validate that either cron_expression or interval_seconds is provided."""
        if "cron_expression" in values and "interval_seconds" in values:
            if values["cron_expression"] is None and values["interval_seconds"] is None:
                raise ValueError("Either cron_expression or interval_seconds must be provided")
        return v


class ScheduleUpdate(BaseModel):
    """Schedule update schema."""
    
    name: Optional[str] = None
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    params: Optional[Union[DataRefreshParams, PredictionParams, BacktestParams]] = None
    status: Optional[str] = None
    
    @validator("status")
    def validate_status(cls, v):
        """Validate status."""
        valid_statuses = ["active", "inactive"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v


class ScheduleExecution(BaseModel):
    """Schedule execution schema."""
    
    id: str
    schedule_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str  # pending, running, completed, failed
    result: Optional[Dict[str, Union[str, int, float, bool, List, Dict]]] = None
    error: Optional[str] = None


class Schedule(BaseModel):
    """Schedule schema."""
    
    id: str
    name: str
    task_type: str
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    params: Union[DataRefreshParams, PredictionParams, BacktestParams]
    user_id: str
    status: str  # active, inactive
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class ScheduleList(BaseModel):
    """Schedule list schema."""
    
    schedules: List[Schedule]
    total: int
    page: int
    page_size: int
    pages: int


class ScheduleExecutionList(BaseModel):
    """Schedule execution list schema."""
    
    executions: List[ScheduleExecution]
    total: int
    page: int
    page_size: int
    pages: int