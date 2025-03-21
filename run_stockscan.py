#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line script to run StockScan workflow
"""

import argparse
import os
import sys
from datetime import datetime

# Import the bridge functions from our run_workflow implementation
from src.run_workflow import run_workflow, parse_args


def main():
    """
    Main entry point for the stockscan workflow
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run StockScan workflow")
    parser.add_argument("--region", type=str, default="cn", choices=["cn", "us"], 
                        help="Region to run workflow for (cn or us)")
    parser.add_argument("--market", type=str, default="csi300", 
                        help="Market to fetch data for (e.g., csi300, sp500)")
    parser.add_argument("--start-date", type=str, default="2015-01-01", 
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
    
    # Quick run options
    parser.add_argument("--quick", action="store_true", default=False,
                        help="Run a quick workflow with minimal data")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Run the full workflow with all steps")
    parser.add_argument("--data-only", action="store_true", default=False,
                        help="Only fetch and prepare data")
    parser.add_argument("--predict-only", action="store_true", default=False,
                        help="Only run prediction (assumes data is prepared)")
    parser.add_argument("--backtest-only", action="store_true", default=False,
                        help="Only run backtesting (assumes prediction is done)")
    parser.add_argument("--compare-strategies", action="store_true", default=False,
                        help="Compare multiple strategies")
    
    # Advanced options
    parser.add_argument("--qlib-dir", type=str, default=None, 
                        help="Directory with Qlib data (will fetch if not provided)")
    parser.add_argument("--result-dir", type=str, default=None, 
                        help="Directory to save results")
    parser.add_argument("--fine-tune", action="store_true", default=False,
                        help="Run fine-tuning on the model")
    parser.add_argument("--no_mlflow", action="store_true", default=False,
                        help="Disable MLflow tracking")
    parser.add_argument("--multi", action="store_true", default=False,
                        help="Run multiple models and compare")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["Linear", "LGBM", "DEnsemble"],
                        help="List of models to run when using --multi")
    
    # Time-related arguments
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
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle quick run mode
    if args.quick:
        print("Running quick workflow with minimal data...")
        args.start_date = "2022-01-01"  # Use a shorter timeframe
        args.market = "csi300" if args.region == "cn" else "sp500"
        # Set a shorter test period for quick runs
        args.test_start = "2022-01-01"
        args.test_end = "2022-03-31"
        args.num_stocks = 20  # Fewer stocks for quicker processing
        
    # Handle full run mode
    if args.full:
        print("Running full workflow with all steps...")
        args.fine_tune = True
        args.compare_strategies = True
        args.num_stocks = 100  # More stocks for comprehensive testing
        
    # Set up result directory if not provided
    if args.result_dir is None:
        args.result_dir = os.path.join(os.getcwd(), "results", f"stockscan_{timestamp}")
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Configure skip flags based on run mode
    args.skip_data = False
    args.skip_prediction = False
    args.skip_backtest = False
    
    if args.data_only:
        args.skip_prediction = True
        args.skip_backtest = True
    elif args.predict_only:
        args.skip_data = True
        args.skip_backtest = True
    elif args.backtest_only:
        args.skip_data = True
        args.skip_prediction = True
    
    # Set multi_strategy flag based on compare_strategies
    args.multi_strategy = args.compare_strategies
    
    # Print run configuration
    print(f"StockScan Workflow Configuration:")
    print(f"- Region: {args.region}")
    print(f"- Market: {args.market}")
    print(f"- Date Range: {args.start_date} to {args.end_date or 'today'}")
    print(f"- Model: {args.model}")
    print(f"- Strategy: {args.strategy} (topk={args.topk}, n_drop={args.n_drop})")
    print(f"- Data Configuration: {args.num_stocks} stocks with volatility {args.price_volatility}")
    
    # Print date ranges
    print(f"- Training period: {args.train_start} to {args.train_end}")
    print(f"- Validation period: {args.valid_start} to {args.valid_end}")
    print(f"- Test period: {args.test_start} to {args.test_end}")
    
    if args.predict_start or args.predict_end:
        print(f"- Prediction period: {args.predict_start or args.test_start} to {args.predict_end or args.test_end}")
    
    if args.backtest_start or args.backtest_end:
        print(f"- Backtest period: {args.backtest_start or args.test_start} to {args.backtest_end or args.test_end}")
    
    print(f"- Fine-tune: {'Yes' if args.fine_tune else 'No'}")
    print(f"- Compare Strategies: {'Yes' if args.compare_strategies else 'No'}")
    print(f"- Results Directory: {args.result_dir}")
    
    # Run the workflow
    try:
        print(f"\nStarting StockScan workflow at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...\n")
        results = run_workflow(args)
        print(f"\nWorkflow completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    except Exception as e:
        print(f"\nWorkflow failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 