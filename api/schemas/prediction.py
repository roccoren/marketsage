#!/usr/bin/env python3
"""
Prediction schemas for the StockScan API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Prediction request schema."""
    
    symbol: str
    model_type: str = "lstm"  # lstm, ensemble, etc.
    prediction_period: str = "30d"  # 1d, 5d, 10d, 30d, 90d, 180d, 220d
    
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


class PredictionBatchRequest(BaseModel):
    """Prediction batch request schema."""
    
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


class PredictionResults(BaseModel):
    """Prediction results schema."""
    
    predicted_prices: List[float]
    confidence: float
    signals: List[str]  # buy, sell, hold
    forecast_change_percent: float
    trend: float
    volatility: float


class PredictionMetadata(BaseModel):
    """Prediction metadata schema."""
    
    model_version: str
    features_used: List[str]
    training_date: Optional[datetime] = None
    training_accuracy: Optional[float] = None
    training_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    validation_loss: Optional[float] = None


class Prediction(BaseModel):
    """Prediction schema."""
    
    id: str
    symbol: str
    model_type: str
    prediction_period: str
    created_at: datetime
    user_id: Optional[str] = None
    status: str  # pending, running, completed, failed
    results: Optional[PredictionResults] = None
    metadata: Optional[PredictionMetadata] = None
    error: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True


class PredictionList(BaseModel):
    """Prediction list schema."""
    
    predictions: List[Prediction]
    total: int
    page: int
    page_size: int
    pages: int


class PredictionHistoryPoint(BaseModel):
    """Prediction history point schema."""
    
    date: datetime
    predicted_price: float
    actual_price: Optional[float] = None
    error: Optional[float] = None
    signal: str  # buy, sell, hold


class PredictionHistory(BaseModel):
    """Prediction history schema."""
    
    symbol: str
    model_type: str
    prediction_period: str
    history: List[PredictionHistoryPoint]
    accuracy: Optional[float] = None
    mean_error: Optional[float] = None
    start_date: datetime
    end_date: datetime


class PredictionPerformance(BaseModel):
    """Prediction performance schema."""
    
    symbol: str
    model_type: str
    prediction_period: str
    accuracy: float
    mean_error: float
    correct_signals: int
    incorrect_signals: int
    signal_accuracy: float
    profit_if_followed: float
    start_date: datetime
    end_date: datetime


class ModelComparison(BaseModel):
    """Model comparison schema."""
    
    symbol: str
    prediction_period: str
    models: List[str]
    accuracies: Dict[str, float]
    mean_errors: Dict[str, float]
    signal_accuracies: Dict[str, float]
    profits_if_followed: Dict[str, float]
    best_model: str
    start_date: datetime
    end_date: datetime