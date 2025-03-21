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
    
    # New optimization options
    optimization_group = parser.add_argument_group('Optimization Options')
    optimization_group.add_argument("--optimize", action="store_true", default=False,
                                   help="Run optimization process")
    optimization_group.add_argument("--optimize-mode", type=str, choices=["model", "strategy", "feature", "ensemble", "all"],
                                   default="model", help="Optimization mode")
    optimization_group.add_argument("--n-trials", type=int, default=50,
                                   help="Number of optimization trials")
    optimization_group.add_argument("--opt-metric", type=str, default="IC",
                                   help="Metric to optimize (IC, ICIR for models; sharpe, total_return, etc. for strategies)")
    optimization_group.add_argument("--top-k-features", type=int, default=50,
                                   help="Number of top features to select in feature importance analysis")
    optimization_group.add_argument("--model-paths", type=str, nargs="+",
                                   help="Paths to trained models for ensemble")
    optimization_group.add_argument("--model-weights", type=float, nargs="+",
                                   help="Weights for models in ensemble")
    optimization_group.add_argument("--prediction-path", type=str,
                                   help="Path to prediction file or directory for strategy optimization")
    
    # Data filtering options
    parser.add_argument("--num-stocks", type=int, default=100,
                        help="Number of stocks to process")
    parser.add_argument("--price-volatility", type=float, default=0.1,
                        help="Price volatility threshold for stock filtering")
    parser.add_argument("--predict-start", type=str, default=None,
                        help="Prediction start date (YYYY-MM-DD)")
    parser.add_argument("--predict-end", type=str, default=None,
                        help="Prediction end date (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--backtest-end", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default values for missing arguments
    if not hasattr(args, "num_stocks"):
        args.num_stocks = 100
    if not hasattr(args, "price_volatility"):
        args.price_volatility = 0.1
    
    # Handle quick run mode
    if args.quick:
        print("Running quick workflow with minimal data...")
        args.start_date = "2022-01-01"  # Use a shorter timeframe
        args.market = "csi300" if args.region == "cn" else "sp500"
        args.test_start = "2022-01-01"
        args.test_end = "2022-03-31"
        args.num_stocks = 20  # Fewer stocks for quicker processing
        args.price_volatility = 0.15  # Slightly higher threshold for quick runs
    
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
    
    # Print run configuration with error handling
    print(f"StockScan Workflow Configuration:")
    print(f"- Region: {args.region}")
    print(f"- Market: {args.market}")
    print(f"- Date Range: {args.start_date} to {args.end_date or 'today'}")
    print(f"- Model: {args.model}")
    print(f"- Strategy: {args.strategy} (topk={args.topk}, n_drop={args.n_drop})")
    
    # Print data configuration only if not in optimization mode
    if not args.optimize:
        print(f"- Data Configuration: {args.num_stocks} stocks")
        if hasattr(args, "price_volatility"):
            print(f"  - Volatility threshold: {args.price_volatility}")
    
    # Print optimization configuration if in optimization mode
    if args.optimize:
        print(f"- Optimization Mode: {args.optimize_mode}")
        print(f"- Number of trials: {args.n_trials}")
        print(f"- Optimization metric: {args.opt_metric}")
        
        if args.optimize_mode == "feature":
            print(f"- Top K features: {args.top_k_features}")
        elif args.optimize_mode == "ensemble" and args.model_paths:
            print(f"- Number of models in ensemble: {len(args.model_paths)}")
    
    # Print date ranges with error handling
    if hasattr(args, "train_start") and hasattr(args, "train_end"):
        print(f"- Training period: {args.train_start} to {args.train_end}")
    if hasattr(args, "valid_start") and hasattr(args, "valid_end"):
        print(f"- Validation period: {args.valid_start} to {args.valid_end}")
    if hasattr(args, "test_start") and hasattr(args, "test_end"):
        print(f"- Test period: {args.test_start} to {args.test_end}")
    
    if hasattr(args, "predict_start") or hasattr(args, "predict_end"):
        print(f"- Prediction period: {args.predict_start or args.test_start} to {args.predict_end or args.test_end}")
    
    if hasattr(args, "backtest_start") or hasattr(args, "backtest_end"):
        print(f"- Backtest period: {args.backtest_start or args.test_start} to {args.backtest_end or args.test_end}")
    
    print(f"- Fine-tune: {'Yes' if args.fine_tune else 'No'}")
    print(f"- Compare Strategies: {'Yes' if args.compare_strategies else 'No'}")
    print(f"- Results Directory: {args.result_dir}")
    
    # Check if running optimization mode
    if args.optimize:
        print(f"Starting optimization in mode: {args.optimize_mode}")
        
        # Import optimization module
        from optimize_qlib import (
            optimize_model,
            optimize_strategy,
            analyze_feature_importance,
            create_ensemble
        )
        
        # Create a namespace with the arguments for optimization
        from types import SimpleNamespace
        opt_args = SimpleNamespace()
        
        if args.optimize_mode == "model" or args.optimize_mode == "all":
            print("Running model hyperparameter optimization...")
            opt_args.model = args.model
            opt_args.n_trials = args.n_trials
            opt_args.timeout = None
            opt_args.metric = args.opt_metric
            opt_args.train_start = args.train_start
            opt_args.train_end = args.train_end
            opt_args.valid_start = args.valid_start
            opt_args.valid_end = args.valid_end
            opt_args.test_start = args.test_start
            opt_args.test_end = args.test_end
            opt_args.output_dir = os.path.join(args.result_dir, "model_optimization")
            
            best_params = optimize_model(opt_args)
            print(f"Best model parameters: {best_params}")
        
        if args.optimize_mode == "strategy" or args.optimize_mode == "all":
            print("Running strategy parameter optimization...")
            if args.prediction_path is None and args.optimize_mode == "strategy":
                print("Error: --prediction-path is required for strategy optimization")
                return 1
            
            opt_args.prediction_path = args.prediction_path
            opt_args.strategy = args.strategy
            opt_args.n_trials = args.n_trials
            opt_args.metric = args.opt_metric if args.opt_metric in ["sharpe", "total_return", "annual_return", "max_drawdown"] else "sharpe"
            opt_args.output_dir = os.path.join(args.result_dir, "strategy_optimization")
            
            best_params = optimize_strategy(opt_args)
            print(f"Best strategy parameters: {best_params}")
        
        if args.optimize_mode == "feature" or args.optimize_mode == "all":
            print("Running feature importance analysis...")
            if not hasattr(args, "model_path") and args.optimize_mode == "feature":
                print("Error: --model-path is required for feature importance analysis")
                return 1
            
            opt_args.model_path = getattr(args, "model_path", "")
            opt_args.top_k = args.top_k_features
            opt_args.output_dir = os.path.join(args.result_dir, "feature_importance")
            
            feature_results = analyze_feature_importance(opt_args)
            print(f"Feature importance analysis completed")
        
        if args.optimize_mode == "ensemble" or args.optimize_mode == "all":
            print("Creating model ensemble...")
            if not args.model_paths and args.optimize_mode == "ensemble":
                print("Error: --model-paths is required for ensemble creation")
                return 1
            
            opt_args.model_paths = args.model_paths
            opt_args.weights = args.model_weights
            opt_args.output_dir = os.path.join(args.result_dir, "ensemble")
            
            ensemble_results = create_ensemble(opt_args)
            print(f"Ensemble creation completed")
        
        print(f"Optimization completed. Results saved to {args.result_dir}")
        return 0
    
    # Run regular workflow if not in optimization mode
    results = run_workflow(args)
    
    print("\nWorkflow completed!")
    print(f"Results saved to: {args.result_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 