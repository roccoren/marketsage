#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the workflow implementation
"""

import os
import sys
import argparse
from datetime import datetime

from src.run_workflow import run_full_workflow


def main():
    """
    Run a simplified test of the stock analysis workflow
    """
    parser = argparse.ArgumentParser(description="Test stock analysis workflow")
    
    # Basic arguments
    parser.add_argument("--model", type=str, default="Linear", 
                      help="Model to use (Linear, LGBM, DEnsemble)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save results")
    parser.add_argument("--no_mlflow", action="store_true",
                      help="Disable MLflow tracking")
    
    args = parser.parse_args()
    
    # Use a small dataset for testing
    test_segments = {
        "train": ("2020-01-01", "2020-06-30"),
        "valid": ("2020-07-01", "2020-09-30"),
        "test": ("2020-10-01", "2020-12-31")
    }
    
    print("Running test workflow with simplified implementation...")
    
    # Create a default output directory if needed
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(os.getcwd(), "results", f"test_{args.model}_{timestamp}")
    
    try:
        # Run the workflow with a smaller dataset
        results = run_full_workflow(
            model_name=args.model,
            dataset_segments=test_segments,
            topk=10,  # Use smaller topk for faster testing
            n_drop=2,
            output_dir=args.output_dir,
            track_mlflow=not args.no_mlflow,
            fetch_data=False  # Use existing data
        )
        
        print(f"Test successful! Results saved to {results['output_dir']}")
        return 0
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 