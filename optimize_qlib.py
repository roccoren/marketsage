#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line tool for Qlib model and strategy optimization
"""

import os
import argparse
from typing import Dict, Any, Optional
import mlflow

from src.utils.optimize import (
    run_model_optimization,
    feature_importance_analysis,
    model_ensemble,
    StrategyOptimizer
)

def optimize_model(args) -> Dict[str, Any]:
    """
    Run model hyperparameter optimization
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Dict[str, Any]
        Best parameters and optimization results
    """
    # Create Qlib config
    qlib_config = {
        "provider_uri": os.getenv("QLIB_DATA_PATH", "~/.qlib/qlib_data/cn_data"),
        "region": getattr(args, "region", "cn")
    }
    
    # Create dataset config
    dataset_kwargs = {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": args.train_start,
                "end_time": args.test_end,
                "fit_start_time": args.train_start,
                "fit_end_time": args.train_end,
                "instruments": getattr(args, "market", "csi300")
            }
        }
    }
    
    # Run optimization
    results = run_model_optimization(
        qlib_config=qlib_config,
        model_name=args.model,
        dataset_kwargs=dataset_kwargs,
        n_trials=args.n_trials,
        timeout=getattr(args, "timeout", None),
        metric=args.metric,
        resume=True,
        base_dir=args.output_dir
    )
    
    return results["best_params"]

def optimize_strategy(args) -> Dict[str, Any]:
    """
    Run strategy parameter optimization
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Dict[str, Any]
        Best parameters
    """
    optimizer = StrategyOptimizer(
        prediction_path=args.prediction_path,
        n_trials=args.n_trials
    )
    
    if args.strategy == "TopkDropoutStrategy":
        return optimizer.optimize_topk_strategy(metric=args.metric)
    else:
        raise ValueError(f"Strategy {args.strategy} not supported for optimization")

def analyze_feature_importance(args) -> Dict[str, Any]:
    """
    Run feature importance analysis
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    return feature_importance_analysis(
        model_path=args.model_path,
        top_k=args.top_k,
        save_path=args.output_dir
    )

def create_ensemble(args) -> Dict[str, Any]:
    """
    Create model ensemble
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Dict[str, Any]
        Ensemble results
    """
    return model_ensemble(
        model_paths=args.model_paths,
        weights=args.weights,
        save_path=args.output_dir
    )

def main():
    """
    Main entry point for optimization command-line tool
    """
    parser = argparse.ArgumentParser(description="Qlib model and strategy optimization tool")
    subparsers = parser.add_subparsers(dest="command", help="Optimization command")
    
    # Model optimization parser
    model_parser = subparsers.add_parser("optimize-model", help="Optimize model hyperparameters")
    model_parser.add_argument("--model", type=str, required=True,
                            choices=["Linear", "LGBM", "DEnsemble"],
                            help="Model to optimize")
    model_parser.add_argument("--n-trials", type=int, default=50,
                            help="Number of optimization trials")
    model_parser.add_argument("--metric", type=str, default="IC",
                            help="Metric to optimize")
    model_parser.add_argument("--region", type=str, default="cn",
                            choices=["cn", "us"], help="Region")
    model_parser.add_argument("--market", type=str, default="csi300",
                            help="Market")
    model_parser.add_argument("--train-start", type=str, required=True,
                            help="Training start date")
    model_parser.add_argument("--train-end", type=str, required=True,
                            help="Training end date")
    model_parser.add_argument("--valid-start", type=str, required=True,
                            help="Validation start date")
    model_parser.add_argument("--valid-end", type=str, required=True,
                            help="Validation end date")
    model_parser.add_argument("--test-start", type=str, required=True,
                            help="Test start date")
    model_parser.add_argument("--test-end", type=str, required=True,
                            help="Test end date")
    model_parser.add_argument("--output-dir", type=str, default="results/optimize",
                            help="Output directory")
    
    # Strategy optimization parser
    strategy_parser = subparsers.add_parser("optimize-strategy", help="Optimize strategy parameters")
    strategy_parser.add_argument("--prediction-path", type=str, required=True,
                               help="Path to prediction file")
    strategy_parser.add_argument("--strategy", type=str, default="TopkDropoutStrategy",
                               help="Strategy to optimize")
    strategy_parser.add_argument("--n-trials", type=int, default=30,
                               help="Number of optimization trials")
    strategy_parser.add_argument("--metric", type=str, default="sharpe",
                               choices=["sharpe", "total_return", "annual_return", "max_drawdown"],
                               help="Metric to optimize")
    strategy_parser.add_argument("--output-dir", type=str, default="results/optimize",
                               help="Output directory")
    
    # Feature importance parser
    feature_parser = subparsers.add_parser("feature-importance", help="Analyze feature importance")
    feature_parser.add_argument("--model-path", type=str, required=True,
                              help="Path to trained model")
    feature_parser.add_argument("--top-k", type=int, default=50,
                              help="Number of top features to select")
    feature_parser.add_argument("--output-dir", type=str, default="results/optimize",
                              help="Output directory")
    
    # Ensemble parser
    ensemble_parser = subparsers.add_parser("model-ensemble", help="Create model ensemble")
    ensemble_parser.add_argument("--model-paths", type=str, nargs="+", required=True,
                               help="Paths to trained models")
    ensemble_parser.add_argument("--weights", type=float, nargs="+",
                               help="Model weights")
    ensemble_parser.add_argument("--output-dir", type=str, default="results/optimize",
                               help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    if hasattr(args, "output_dir"):
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.command == "optimize-model":
            results = optimize_model(args)
            print(f"\nBest parameters found:")
            for param, value in results.items():
                print(f"  {param}: {value}")
                
        elif args.command == "optimize-strategy":
            results = optimize_strategy(args)
            print(f"\nBest strategy parameters found:")
            for param, value in results.items():
                print(f"  {param}: {value}")
                
        elif args.command == "feature-importance":
            results = analyze_feature_importance(args)
            print(f"\nFeature importance analysis completed")
            print(f"Results saved to: {args.output_dir}")
            
        elif args.command == "model-ensemble":
            results = create_ensemble(args)
            print(f"\nModel ensemble created")
            print(f"Results saved to: {args.output_dir}")
            
        else:
            parser.print_help()
            return 1
            
        return 0
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 