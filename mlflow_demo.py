#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLflow enhanced integration demonstration for StockScan project
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json
import mlflow
from typing import Dict, Any, Optional, List, Union, Tuple

from src.utils import (
    setup_mlflow, 
    create_nested_run, 
    log_extended_params, 
    log_dataset, 
    log_model_with_signature,
    log_prediction_results, 
    log_backtest_metrics, 
    find_best_runs, 
    compare_runs,
    create_comparison_chart
)

from src.config import QLIB_CONFIG, MLFLOW_CONFIG, MODEL_CONFIG


def generate_sample_data(
    num_stocks: int = 30,
    dates: int = 100,
    starting_date: str = "2023-01-01"
) -> pd.DataFrame:
    """
    Generate synthetic stock data for demonstration
    """
    print(f"Generating sample data with {num_stocks} stocks and {dates} days")
    
    # Generate dates
    date_range = pd.date_range(starting_date, periods=dates)
    
    # Generate stock symbols
    stocks = [f"STOCK_{i}" for i in range(1, num_stocks + 1)]
    
    # Create a DataFrame with synthetic prices
    data = {}
    for stock in stocks:
        # Random starting price between 10 and 100
        price = np.random.uniform(10, 100)
        # Create a price series with random walk
        prices = [price]
        for _ in range(1, len(date_range)):
            # Random percent change (-2% to +2%)
            pct_change = np.random.normal(0.0005, 0.015)
            price = price * (1 + pct_change)
            prices.append(price)
        
        data[stock] = prices
    
    # Create DataFrame
    df = pd.DataFrame(data, index=date_range)
    return df


def simulate_model_prediction(data: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate model predictions for stocks
    """
    print("Simulating model predictions")
    
    # Create a DataFrame with same index and columns as data
    predictions = pd.DataFrame(index=data.index, columns=data.columns)
    
    # Generate predictions (random values between -1 and 1)
    for col in data.columns:
        # Create a series following a sine wave with noise
        t = np.linspace(0, 4*np.pi, len(data))
        base = np.sin(t + np.random.uniform(0, np.pi))
        noise = np.random.normal(0, 0.2, len(data))
        predictions[col] = base + noise
    
    # Normalize predictions
    for col in predictions.columns:
        predictions[col] = (predictions[col] - predictions[col].mean()) / predictions[col].std()
    
    return predictions


def simulate_backtest(
    predictions: pd.DataFrame, 
    data: pd.DataFrame, 
    topk: int = 10
) -> Dict[str, Any]:
    """
    Simulate a backtest based on predictions
    """
    print(f"Simulating backtest with top-{topk} strategy")
    
    # Initialize portfolio with $100,000
    initial_capital = 100000
    portfolio_value = pd.Series(index=predictions.index, dtype=float)
    portfolio_value.iloc[0] = initial_capital
    
    # Initialize positions DataFrame
    positions = pd.DataFrame(0, index=predictions.index, columns=predictions.columns)
    
    # Initialize trades list
    trades = []
    
    # Simulate trading based on predictions
    for i in range(1, len(predictions.index)):
        current_date = predictions.index[i]
        previous_date = predictions.index[i-1]
        
        # Get top k stocks based on predictions
        top_stocks = predictions.iloc[i-1].nlargest(topk).index
        
        # Update positions
        for stock in predictions.columns:
            positions.loc[current_date, stock] = 1 if stock in top_stocks else 0
        
        # Calculate portfolio returns (simplified)
        if i > 0:
            # Get price changes for all stocks
            price_changes = data.iloc[i] / data.iloc[i-1] - 1
            
            # Calculate weighted return based on previous positions
            prev_positions = positions.iloc[i-1]
            # Ensure equal weight for each position
            if prev_positions.sum() > 0:
                weights = prev_positions / prev_positions.sum()
                portfolio_return = (price_changes * weights).sum()
            else:
                portfolio_return = 0
                
            # Update portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + portfolio_return)
        
        # Track trades (when positions change)
        if i > 0:
            for stock in predictions.columns:
                if positions.iloc[i, positions.columns.get_loc(stock)] != positions.iloc[i-1, positions.columns.get_loc(stock)]:
                    direction = 'buy' if positions.iloc[i, positions.columns.get_loc(stock)] > 0 else 'sell'
                    trades.append({
                        'date': current_date,
                        'stock': stock,
                        'direction': direction,
                        'price': float(data.iloc[i, data.columns.get_loc(stock)]),  # Convert to float
                        'position': int(positions.iloc[i, positions.columns.get_loc(stock)])  # Convert to int
                    })
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df.set_index('date', inplace=True)
    
    # Calculate returns
    returns = portfolio_value.pct_change().dropna()
    
    # Calculate metrics and ensure they are native Python types
    total_return = float((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1)
    annual_return = float(total_return * (252 / len(portfolio_value)))  # Annualize
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    max_drawdown = float(((portfolio_value / portfolio_value.cummax()) - 1).min())
    
    backtest_results = {
        "portfolio_value": portfolio_value,
        "positions": positions,
        "trades": trades_df,
        "metrics": {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": float(portfolio_value.iloc[-1])
        }
    }
    
    return backtest_results


def run_end_to_end_workflow(
    experiment_name: str = "mlflow_enhanced_demo",
    num_runs: int = 3,
    num_stocks: int = 30,
    num_days: int = 100
) -> None:
    """
    Run an end-to-end workflow demonstration using enhanced MLflow capabilities
    """
    print("\n===== STARTING ENHANCED MLFLOW DEMONSTRATION =====\n")
    
    # Configure MLflow
    mlflow_config = MLFLOW_CONFIG.copy()
    mlflow_config["experiment_name"] = experiment_name
    mlflow_config["run_name"] = f"e2e_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MLflow with enhanced functionality
    experiment_id, parent_run_id = setup_mlflow(mlflow_config)
    print(f"Parent run ID: {parent_run_id}")
    
    try:
        # Log extended parameters
        complete_config = {
            "qlib_config": QLIB_CONFIG,
            "model_config": MODEL_CONFIG,
            "workflow_params": {
                "num_runs": num_runs,
                "num_stocks": num_stocks,
                "num_days": num_days,
                "strategies": ["topk", "weighted", "momentum"],
                "complex_setting": {
                    "enable_advanced_features": True,
                    "risk_constraints": {"max_position": 0.1, "sector_limits": {"tech": 0.3, "healthcare": 0.2}}
                }
            }
        }
        
        log_extended_params(complete_config)
        print("Logged extended configuration parameters")
        
        # Generate sample data
        data = generate_sample_data(num_stocks=num_stocks, dates=num_days)
        
        # Log dataset with enhanced metadata
        log_dataset(
            data, 
            "stock_prices", 
            description="Synthetic stock price data for demonstration",
            metadata={
                "source": "synthetic",
                "num_stocks": num_stocks,
                "date_range": f"{data.index.min()} to {data.index.max()}",
                "frequency": "daily"
            }
        )
        print("Logged dataset with enhanced metadata")
        
        # Dictionary to store run IDs for comparison
        run_ids = []
        
        # Run multiple model configurations as nested runs
        models = ["Linear", "LGBM", "DEnsemble"]
        topk_values = [5, 10, 15]
        
        # End the parent run to avoid conflicts when creating child runs
        mlflow.end_run()
        
        for i, (model, topk) in enumerate(zip(models, topk_values)):
            # Create a nested run for this model configuration with proper nesting
            print(f"\n----- Starting nested run for {model} model with top-{topk} strategy -----")
            run_name = f"{model}_top{topk}"
            
            # Start a new run for each model, using the same experiment
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                current_run_id = run.info.run_id
                
                # Set run-specific tags
                mlflow.set_tag("model_type", model)
                mlflow.set_tag("strategy", f"top{topk}")
                mlflow.set_tag("run_index", i+1)
                mlflow.set_tag("parent_workflow", parent_run_id)  # Link to the parent workflow
                
                # Log model-specific parameters
                mlflow.log_params({
                    "model": model,
                    "topk": topk,
                    "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                })
                
                # Simulate model predictions
                predictions = simulate_model_prediction(data)
                
                # Log prediction results with visualizations
                log_prediction_results(predictions, f"{model}_predictions")
                print(f"Logged predictions for {model} model")
                
                # Run backtest
                backtest_results = simulate_backtest(predictions, data, topk=topk)
                
                # Log backtest metrics and visualizations
                log_backtest_metrics(
                    backtest_results["metrics"],
                    backtest_results["portfolio_value"],
                    backtest_results["positions"],
                    backtest_results["trades"],
                    f"{model}_top{topk}"
                )
                print(f"Logged backtest results for {model} with top-{topk} strategy")
                
                # Store run ID for later comparison
                run_ids.append(current_run_id)
                
                print(f"Completed run for {model} model with run ID: {current_run_id}")
        
        # Start a final run for comparison analysis
        with mlflow.start_run(experiment_id=experiment_id, run_name="run_comparison") as comparison_run:
            comparison_run_id = comparison_run.info.run_id
            
            # Tag this run as a comparison run
            mlflow.set_tag("run_type", "comparison")
            mlflow.set_tag("parent_workflow", parent_run_id)
            
            # Log the run IDs being compared
            mlflow.log_param("compared_runs", ",".join(run_ids))
            
            # Find best runs based on Sharpe ratio
            print("\n----- Finding best runs based on Sharpe ratio -----")
            best_runs = find_best_runs(
                experiment_name=experiment_name,
                metric_name="sharpe",
                max_results=3,
                higher_is_better=True
            )
            
            print(f"Best runs by Sharpe ratio:")
            for i, run in enumerate(best_runs):
                print(f"{i+1}. Run: {run['run_name']}, Sharpe: {run['sharpe']:.4f}")
            
            # Compare runs
            print("\n----- Comparing all runs -----")
            metrics_to_compare = ["total_return", "annual_return", "sharpe", "max_drawdown"]
            comparison_df = compare_runs(run_ids, metrics_to_compare)
            
            # Save comparison results
            comparison_path = "run_comparison.csv"
            comparison_df.to_csv(comparison_path)
            mlflow.log_artifact(comparison_path)
            print(f"Saved run comparison to {comparison_path}")
            
            # Create comparison charts
            for metric in metrics_to_compare:
                chart_path = f"comparison_{metric}.png"
                create_comparison_chart(comparison_df, metric, save_path=chart_path)
                mlflow.log_artifact(chart_path)
                print(f"Created comparison chart for {metric}")
            
            print("\n===== MLFLOW ENHANCED DEMONSTRATION COMPLETED SUCCESSFULLY =====")
            print(f"MLflow UI: mlflow ui --backend-store-uri {mlflow_config['tracking_uri']}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensure all MLflow runs are closed
        mlflow.end_run()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MLflow Enhanced StockScan Demonstration")
    
    parser.add_argument("--experiment-name", type=str, default="mlflow_enhanced_demo",
                       help="MLflow experiment name")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs to execute")
    parser.add_argument("--stocks", type=int, default=30,
                       help="Number of stocks in synthetic data")
    parser.add_argument("--days", type=int, default=100,
                       help="Number of days in synthetic data")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_end_to_end_workflow(
        experiment_name=args.experiment_name,
        num_runs=args.runs,
        num_stocks=args.stocks,
        num_days=args.days
    ) 