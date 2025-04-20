#!/usr/bin/env python3
"""
Predictions router for the StockScan API.
"""

import uuid
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.core.dependencies import (
    AnalystUser,
    CurrentUser,
    DbSession,
    get_current_active_user,
)
from api.schemas.prediction import (
    ModelComparison,
    Prediction,
    PredictionBatchRequest,
    PredictionHistory,
    PredictionList,
    PredictionPerformance,
    PredictionRequest,
)

# Import existing modules for forecasting
from modules.forecasting import ForecastingManager as Forecasting
from modules.storage import StorageManager as Storage

router = APIRouter()


@router.post("/", response_model=Prediction)
async def create_prediction(
    prediction_request: PredictionRequest,
    current_user: dict = Depends(get_current_active_user),
    db: Session = Depends(DbSession),
) -> Any:
    """
    Create a new prediction.
    
    Args:
        prediction_request: Prediction request
        current_user: Current user
        db: Database session
        
    Returns:
        Created prediction
    """
    # Initialize forecasting module
    forecasting = Forecasting()
    
    # Initialize storage
    storage = Storage()
    
    # Generate prediction ID
    prediction_id = str(uuid.uuid4())
    
    # Create prediction record
    prediction = {
        "id": prediction_id,
        "symbol": prediction_request.symbol,
        "model_type": prediction_request.model_type,
        "prediction_period": prediction_request.prediction_period,
        "created_at": datetime.utcnow(),
        "user_id": current_user["user_id"],
        "status": "pending",
    }
    
    try:
        # Run prediction
        prediction["status"] = "running"
        
        # Get prediction results
        results = forecasting.predict(
            symbol=prediction_request.symbol,
            model_type=prediction_request.model_type,
            prediction_period=prediction_request.prediction_period,
        )
        
        # Update prediction record
        prediction["status"] = "completed"
        prediction["results"] = {
            "predicted_prices": results["predicted_prices"],
            "confidence": results["confidence"],
            "signals": results["signals"],
            "forecast_change_percent": results["forecast_change_percent"],
            "trend": results["trend"],
            "volatility": results["volatility"],
        }
        
        # Add metadata if available
        if "metadata" in results:
            prediction["metadata"] = results["metadata"]
        
        # Store prediction
        storage.store_prediction(prediction)
        
    except Exception as e:
        # Update prediction record with error
        prediction["status"] = "failed"
        prediction["error"] = str(e)
        
        # Store prediction
        storage.store_prediction(prediction)
        
        # Re-raise exception
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating prediction: {str(e)}",
        )
    
    return prediction


@router.get("/{prediction_id}", response_model=Prediction)
async def get_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Get prediction details.
    
    Args:
        prediction_id: Prediction ID
        current_user: Current user
        
    Returns:
        Prediction details
    """
    # Initialize storage
    storage = Storage()
    
    # Get prediction
    prediction = storage.get_prediction(prediction_id)
    
    # Check if prediction exists
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID {prediction_id} not found",
        )
    
    # Check if user has access to prediction
    if prediction.get("user_id") != current_user["user_id"]:
        # In a real implementation, we would check if the user has access to the prediction
        # For now, we'll just allow access if the user is an admin or analyst
        is_admin_or_analyst = False  # This would be a database lookup
        if not is_admin_or_analyst:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this prediction",
            )
    
    return prediction


@router.get("/", response_model=PredictionList)
async def list_predictions(
    symbol: Optional[str] = None,
    model_type: Optional[str] = None,
    prediction_period: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    List predictions.
    
    Args:
        symbol: Filter by symbol
        model_type: Filter by model type
        prediction_period: Filter by prediction period
        status: Filter by status
        page: Page number
        page_size: Page size
        current_user: Current user
        
    Returns:
        List of predictions
    """
    # Initialize storage
    storage = Storage()
    
    # Get predictions
    predictions = storage.list_predictions(
        user_id=current_user["user_id"],
        symbol=symbol,
        model_type=model_type,
        prediction_period=prediction_period,
        status=status,
    )
    
    # Apply pagination
    total = len(predictions)
    pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paginated_predictions = predictions[start:end]
    
    return {
        "predictions": paginated_predictions,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


@router.post("/batch", response_model=List[Prediction])
async def create_batch_predictions(
    batch_request: PredictionBatchRequest,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Create batch predictions.
    
    Args:
        batch_request: Batch prediction request
        current_user: Current user
        
    Returns:
        Created predictions
    """
    # Initialize forecasting module
    forecasting = Forecasting()
    
    # Initialize storage
    storage = Storage()
    
    # Create predictions
    predictions = []
    
    for symbol in batch_request.symbols:
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Create prediction record
        prediction = {
            "id": prediction_id,
            "symbol": symbol,
            "model_type": batch_request.model_type,
            "prediction_period": batch_request.prediction_period,
            "created_at": datetime.utcnow(),
            "user_id": current_user["user_id"],
            "status": "pending",
        }
        
        try:
            # Run prediction
            prediction["status"] = "running"
            
            # Get prediction results
            results = forecasting.predict(
                symbol=symbol,
                model_type=batch_request.model_type,
                prediction_period=batch_request.prediction_period,
            )
            
            # Update prediction record
            prediction["status"] = "completed"
            prediction["results"] = {
                "predicted_prices": results["predicted_prices"],
                "confidence": results["confidence"],
                "signals": results["signals"],
                "forecast_change_percent": results["forecast_change_percent"],
                "trend": results["trend"],
                "volatility": results["volatility"],
            }
            
            # Add metadata if available
            if "metadata" in results:
                prediction["metadata"] = results["metadata"]
            
            # Store prediction
            storage.store_prediction(prediction)
            
        except Exception as e:
            # Update prediction record with error
            prediction["status"] = "failed"
            prediction["error"] = str(e)
            
            # Store prediction
            storage.store_prediction(prediction)
        
        # Add prediction to list
        predictions.append(prediction)
    
    return predictions


@router.get("/history", response_model=PredictionHistory)
async def get_prediction_history(
    symbol: str,
    model_type: Optional[str] = "lstm",
    prediction_period: Optional[str] = "30d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Get prediction history.
    
    Args:
        symbol: Stock symbol
        model_type: Model type
        prediction_period: Prediction period
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        current_user: Current user
        
    Returns:
        Prediction history
    """
    # Initialize storage
    storage = Storage()
    
    # Get prediction history
    history = storage.get_prediction_history(
        symbol=symbol,
        model_type=model_type,
        prediction_period=prediction_period,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Check if history exists
    if not history or len(history) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction history found for symbol {symbol}",
        )
    
    # Calculate accuracy and mean error
    accuracy = None
    mean_error = None
    
    if all("actual_price" in point for point in history):
        correct_predictions = sum(
            1 for point in history
            if point["actual_price"] is not None and point["error"] is not None and abs(point["error"]) < 0.05
        )
        accuracy = correct_predictions / len(history)
        
        errors = [
            abs(point["error"])
            for point in history
            if point["actual_price"] is not None and point["error"] is not None
        ]
        if errors:
            mean_error = sum(errors) / len(errors)
    
    # Get start and end dates
    start_date = min(point["date"] for point in history)
    end_date = max(point["date"] for point in history)
    
    return {
        "symbol": symbol,
        "model_type": model_type,
        "prediction_period": prediction_period,
        "history": history,
        "accuracy": accuracy,
        "mean_error": mean_error,
        "start_date": start_date,
        "end_date": end_date,
    }


@router.get("/performance/{symbol}", response_model=PredictionPerformance)
async def get_prediction_performance(
    symbol: str,
    model_type: Optional[str] = "lstm",
    prediction_period: Optional[str] = "30d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Get prediction performance.
    
    Args:
        symbol: Stock symbol
        model_type: Model type
        prediction_period: Prediction period
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        current_user: Current user
        
    Returns:
        Prediction performance
    """
    # Initialize storage
    storage = Storage()
    
    # Get prediction history
    history = storage.get_prediction_history(
        symbol=symbol,
        model_type=model_type,
        prediction_period=prediction_period,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Check if history exists
    if not history or len(history) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction history found for symbol {symbol}",
        )
    
    # Calculate performance metrics
    correct_predictions = sum(
        1 for point in history
        if point["actual_price"] is not None and point["error"] is not None and abs(point["error"]) < 0.05
    )
    accuracy = correct_predictions / len(history)
    
    errors = [
        abs(point["error"])
        for point in history
        if point["actual_price"] is not None and point["error"] is not None
    ]
    mean_error = sum(errors) / len(errors) if errors else None
    
    correct_signals = sum(
        1 for point in history
        if point["signal"] is not None and point["actual_signal"] is not None and point["signal"] == point["actual_signal"]
    )
    incorrect_signals = sum(
        1 for point in history
        if point["signal"] is not None and point["actual_signal"] is not None and point["signal"] != point["actual_signal"]
    )
    signal_accuracy = correct_signals / (correct_signals + incorrect_signals) if (correct_signals + incorrect_signals) > 0 else None
    
    # Calculate profit if followed
    profit_if_followed = 0.0
    for point in history:
        if point["signal"] == "buy" and point["actual_price"] is not None and point["predicted_price"] is not None:
            profit_if_followed += (point["actual_price"] - point["predicted_price"]) / point["predicted_price"]
        elif point["signal"] == "sell" and point["actual_price"] is not None and point["predicted_price"] is not None:
            profit_if_followed += (point["predicted_price"] - point["actual_price"]) / point["predicted_price"]
    
    # Get start and end dates
    start_date = min(point["date"] for point in history)
    end_date = max(point["date"] for point in history)
    
    return {
        "symbol": symbol,
        "model_type": model_type,
        "prediction_period": prediction_period,
        "accuracy": accuracy,
        "mean_error": mean_error,
        "correct_signals": correct_signals,
        "incorrect_signals": incorrect_signals,
        "signal_accuracy": signal_accuracy,
        "profit_if_followed": profit_if_followed,
        "start_date": start_date,
        "end_date": end_date,
    }


@router.get("/compare/{symbol}", response_model=ModelComparison)
async def compare_models(
    symbol: str,
    prediction_period: Optional[str] = "30d",
    models: List[str] = Query(["lstm", "ensemble", "linear"]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Compare different prediction models.
    
    Args:
        symbol: Stock symbol
        prediction_period: Prediction period
        models: List of models to compare
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        current_user: Current user
        
    Returns:
        Model comparison
    """
    # Initialize storage
    storage = Storage()
    
    # Get performance for each model
    accuracies = {}
    mean_errors = {}
    signal_accuracies = {}
    profits_if_followed = {}
    
    for model_type in models:
        # Get prediction history
        history = storage.get_prediction_history(
            symbol=symbol,
            model_type=model_type,
            prediction_period=prediction_period,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Check if history exists
        if not history or len(history) == 0:
            continue
        
        # Calculate performance metrics
        correct_predictions = sum(
            1 for point in history
            if point["actual_price"] is not None and point["error"] is not None and abs(point["error"]) < 0.05
        )
        accuracy = correct_predictions / len(history)
        accuracies[model_type] = accuracy
        
        errors = [
            abs(point["error"])
            for point in history
            if point["actual_price"] is not None and point["error"] is not None
        ]
        mean_error = sum(errors) / len(errors) if errors else None
        mean_errors[model_type] = mean_error
        
        correct_signals = sum(
            1 for point in history
            if point["signal"] is not None and point["actual_signal"] is not None and point["signal"] == point["actual_signal"]
        )
        incorrect_signals = sum(
            1 for point in history
            if point["signal"] is not None and point["actual_signal"] is not None and point["signal"] != point["actual_signal"]
        )
        signal_accuracy = correct_signals / (correct_signals + incorrect_signals) if (correct_signals + incorrect_signals) > 0 else None
        signal_accuracies[model_type] = signal_accuracy
        
        # Calculate profit if followed
        profit_if_followed = 0.0
        for point in history:
            if point["signal"] == "buy" and point["actual_price"] is not None and point["predicted_price"] is not None:
                profit_if_followed += (point["actual_price"] - point["predicted_price"]) / point["predicted_price"]
            elif point["signal"] == "sell" and point["actual_price"] is not None and point["predicted_price"] is not None:
                profit_if_followed += (point["predicted_price"] - point["actual_price"]) / point["predicted_price"]
        profits_if_followed[model_type] = profit_if_followed
    
    # Check if any models were found
    if not accuracies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction history found for symbol {symbol} with the specified models",
        )
    
    # Determine best model
    best_model = max(accuracies, key=accuracies.get)
    
    # Get start and end dates from the best model's history
    history = storage.get_prediction_history(
        symbol=symbol,
        model_type=best_model,
        prediction_period=prediction_period,
        start_date=start_date,
        end_date=end_date,
    )
    start_date = min(point["date"] for point in history)
    end_date = max(point["date"] for point in history)
    
    return {
        "symbol": symbol,
        "prediction_period": prediction_period,
        "models": list(accuracies.keys()),
        "accuracies": accuracies,
        "mean_errors": mean_errors,
        "signal_accuracies": signal_accuracies,
        "profits_if_followed": profits_if_followed,
        "best_model": best_model,
        "start_date": start_date,
        "end_date": end_date,
    }