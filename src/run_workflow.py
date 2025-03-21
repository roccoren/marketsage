#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bridge script to connect run_stockscan.py with the standalone workflow implementation
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import subprocess

# Add parent directory to path to import standalone_workflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import QLIB_CONFIG, MLFLOW_CONFIG, MODEL_CONFIG, FETCH_CONFIG

def parse_args():
    """
    Define command line arguments for the workflow
    """
    parser = argparse.ArgumentParser(description="Run stock analysis workflow")
    
    # Basic arguments
    parser.add_argument("--region", type=str, default="cn", choices=["cn", "us"], 
                        help="Region to run workflow for (cn or us)")
    parser.add_argument("--market", type=str, default="csi300", 
                        help="Market to fetch data for (e.g., csi300, sp500)")
    parser.add_argument("--start-date", type=str, default="2020-01-01", 
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=None, 
                        help="End date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--model", type=str, default="Linear", 
                        choices=["Linear", "LGBM", "DEnsemble"],
                        help="Model to use for prediction")
    parser.add_argument("--strategy", type=str, default="TopkDropoutStrategy", 
                        help="Strategy to use for backtesting")
    parser.add_argument("--topk", type=int, default=10, 
                        help="Top-k parameter for strategy")
    parser.add_argument("--n-drop", type=int, default=2, 
                        help="Number of stocks to drop for TopkDropoutStrategy")
    
    # Workflow control
    parser.add_argument("--skip-data", action="store_true", default=False,
                        help="Skip data fetching step")
    parser.add_argument("--skip-prediction", action="store_true", default=False,
                        help="Skip prediction step")
    parser.add_argument("--skip-backtest", action="store_true", default=False,
                        help="Skip backtesting step")
    
    # Advanced options
    parser.add_argument("--qlib-dir", type=str, default=None, 
                        help="Directory with Qlib data (will fetch if not provided)")
    parser.add_argument("--result-dir", type=str, default=None, 
                        help="Directory to save results")
    parser.add_argument("--fine-tune", action="store_true", default=False,
                        help="Run fine-tuning on the model")
    parser.add_argument("--multi-strategy", action="store_true", default=False,
                        help="Run multiple strategies for comparison")
    parser.add_argument("--multi", action="store_true",
                        help="Run multiple models and compare")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["Linear", "LGBM", "DEnsemble"],
                        help="List of models to run when using --multi")
    parser.add_argument("--no_mlflow", action="store_true",
                        help="Disable MLflow tracking")
    
    # Time-related arguments for data segments
    parser.add_argument("--train_start", type=str, default="2020-01-01",
                        help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train_end", type=str, default="2020-06-30",
                        help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--valid_start", type=str, default="2020-07-01",
                        help="Validation start date (YYYY-MM-DD)")
    parser.add_argument("--valid_end", type=str, default="2020-09-30",
                        help="Validation end date (YYYY-MM-DD)")
    parser.add_argument("--test_start", type=str, default="2020-10-01",
                        help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test_end", type=str, default="2020-12-31",
                        help="Test end date (YYYY-MM-DD)")
    
    # Prediction and backtest specific date ranges
    parser.add_argument("--predict_start", type=str, default=None,
                        help="Prediction start date (YYYY-MM-DD). Defaults to test_start if not specified.")
    parser.add_argument("--predict_end", type=str, default=None,
                        help="Prediction end date (YYYY-MM-DD). Defaults to test_end if not specified.")
    parser.add_argument("--backtest_start", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD). Defaults to test_start if not specified.")
    parser.add_argument("--backtest_end", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD). Defaults to test_end if not specified.")
    
    # Data generation parameters
    parser.add_argument("--num_stocks", type=int, default=50,
                        help="Number of synthetic stocks to generate")
    parser.add_argument("--price_volatility", type=float, default=0.01,
                        help="Price volatility for synthetic data generation")
    
    return parser.parse_args()


def run_workflow(args):
    """
    Bridge function to run the standalone workflow with the provided arguments
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with results
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting workflow bridge to standalone implementation")
    
    # Prepare command for standalone workflow
    standalone_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "standalone_workflow.py")
    
    # Ensure standalone_workflow.py is executable
    if not os.access(standalone_path, os.X_OK):
        logger.info("Making standalone_workflow.py executable")
        os.chmod(standalone_path, 0o755)
    
    # Build command based on arguments
    cmd = [standalone_path]
    
    # Handle model selection
    if args.multi:
        cmd.append("--multi")
        if hasattr(args, "models") and args.models:
            cmd.append("--models")
            cmd.extend(args.models)
    else:
        cmd.extend(["--model", args.model])
    
    # Add other parameters
    cmd.extend(["--topk", str(args.topk)])
    cmd.extend(["--n_drop", str(args.n_drop)])
    
    # Set date ranges for data segments
    if hasattr(args, "train_start"):
        cmd.extend(["--train_start", args.train_start])
    if hasattr(args, "train_end"):
        cmd.extend(["--train_end", args.train_end])
    if hasattr(args, "valid_start"):
        cmd.extend(["--valid_start", args.valid_start])
    if hasattr(args, "valid_end"):
        cmd.extend(["--valid_end", args.valid_end])
    if hasattr(args, "test_start"):
        cmd.extend(["--test_start", args.test_start])
    if hasattr(args, "test_end"):
        cmd.extend(["--test_end", args.test_end])
    
    # Add prediction and backtest specific date ranges if provided
    if hasattr(args, "predict_start") and args.predict_start:
        cmd.extend(["--predict_start", args.predict_start])
    if hasattr(args, "predict_end") and args.predict_end:
        cmd.extend(["--predict_end", args.predict_end])
    if hasattr(args, "backtest_start") and args.backtest_start:
        cmd.extend(["--backtest_start", args.backtest_start])
    if hasattr(args, "backtest_end") and args.backtest_end:
        cmd.extend(["--backtest_end", args.backtest_end])
    
    # Add data generation parameters
    if hasattr(args, "num_stocks"):
        cmd.extend(["--num_stocks", str(args.num_stocks)])
    if hasattr(args, "price_volatility"):
        cmd.extend(["--price_volatility", str(args.price_volatility)])
    
    # Set output directory if specified
    if args.result_dir:
        cmd.extend(["--output_dir", args.result_dir])
    
    # Handle MLflow configuration
    if hasattr(args, "no_mlflow") and args.no_mlflow:
        cmd.append("--no_mlflow")
    
    # Handle workflow step flags
    if args.skip_data:
        cmd.append("--skip_data")
    if args.skip_prediction:
        cmd.append("--skip_prediction")
    if args.skip_backtest:
        cmd.append("--skip_backtest")
    
    # Convert command to strings
    cmd = [str(c) for c in cmd]
    
    # Log the command being executed
    logger.info(f"Running standalone workflow with command: {' '.join(cmd)}")
    
    # Execute the standalone workflow
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Standalone workflow completed with return code {result.returncode}")
        logger.info(f"Output: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        
        # Return a results dictionary similar to what would be expected
        # We don't have direct access to the objects, but we can provide the output directory
        output_dir = args.result_dir if args.result_dir else "Results saved to standard location"
        return {
            "status": "success",
            "output_dir": output_dir,
            "command": ' '.join(cmd)
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Standalone workflow failed with return code {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        raise RuntimeError(f"Standalone workflow execution failed: {str(e)}")


def main():
    """
    Main entry point for running the workflow from command line
    """
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up result directory if not provided
    if args.result_dir is None:
        args.result_dir = os.path.join(os.getcwd(), "results", f"stock_analysis_{timestamp}")
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    try:
        results = run_workflow(args)
        print(f"Workflow completed successfully")
        print(f"Results saved to: {results['output_dir']}")
        return 0
    
    except Exception as e:
        print(f"Workflow failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 