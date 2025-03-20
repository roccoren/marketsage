#!/usr/bin/env python
"""
Script to run Qlib strategy
"""

import os
import sys
import argparse
import logging
import yaml
import qlib
from qlib.config import C
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_qlib_strategy')

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_strategy(config_path):
    """Run Qlib strategy with the given configuration"""
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Initialize Qlib
    provider_uri = config['qlib_init']['provider_uri']
    region = config['qlib_init']['region']
    logger.info(f"Initializing Qlib with provider_uri={provider_uri}, region={region}")
    qlib.init(provider_uri=provider_uri, region=region)
    
    # Create dataset handler
    handler_config = config['data_handler_config']
    handler = Alpha158(**handler_config)
    
    # Create dataset
    dataset_config = config['dataset']
    dataset = DatasetH(handler, **dataset_config['kwargs'])
    
    # Create and train model
    model_config = config['model']
    model = LGBModel(**model_config['kwargs'])
    
    logger.info("Training model...")
    model.fit(dataset)
    
    # Generate predictions
    logger.info("Generating predictions...")
    pred_df = model.predict(dataset)
    
    # Save predictions
    pred_path = "output/predictions.csv"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pred_df.to_csv(pred_path)
    logger.info(f"Predictions saved to {pred_path}")
    
    # Analyze predictions
    logger.info("Analyzing predictions...")
    latest_preds = pred_df.groupby(level=0).last()
    top_stocks = latest_preds.nlargest(10)
    
    logger.info("\nTop 10 stocks by latest prediction:")
    for stock, score in top_stocks.items():
        logger.info(f"{stock}: {score:.4f}")
    
    logger.info("Strategy execution completed!")

def main():
    parser = argparse.ArgumentParser(description="Run Qlib strategy")
    parser.add_argument("--config", default="config/qlib_strategy.yaml", help="Path to strategy configuration file")
    
    args = parser.parse_args()
    run_strategy(args.config)

if __name__ == "__main__":
    main() 