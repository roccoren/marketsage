#!/usr/bin/env python
"""
Azure OpenAI integration example - Demonstrates using Azure OpenAI with RDAgent for stock predictions
"""
import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from stockscan.core.data_manager import get_stock_data
from stockscan.core.model_factory import create_model
from stockscan.core.prediction import predict_stock_movement
from stockscan.utils.env_setup import setup_environment
from stockscan.evolution.rdagent_manager import RDAgentManager
from stockscan.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('azure_openai_example')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Azure OpenAI StockScan Example')
    
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,NVDA',
                      help='Comma-separated list of stock symbols to analyze')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                      help='Start date for historical data')
    parser.add_argument('--end_date', type=str, default='2023-01-01',
                      help='End date for historical data')
    parser.add_argument('--output_dir', type=str, default='./output/azure_openai',
                      help='Directory for output files')
    parser.add_argument('--azure_endpoint', type=str, required=True,
                      help='Azure OpenAI endpoint URL')
    parser.add_argument('--azure_deployment', type=str, required=True,
                      help='Azure OpenAI deployment name')
    parser.add_argument('--azure_api_version', type=str, default='2023-05-15',
                      help='Azure OpenAI API version')
    parser.add_argument('--rdagent_ui', action='store_true',
                      help='Start the RDAgent UI')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return 1
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    logger.info(f"Starting Azure OpenAI StockScan example with {len(symbols)} symbols")
    logger.info(f"Using Azure OpenAI deployment: {args.azure_deployment}")
    
    # Create RDAgent manager with Azure OpenAI configuration
    rdagent_manager = RDAgentManager(
        log_dir=os.path.join(args.output_dir, 'logs'),
        api_key=api_key,
        chat_model='gpt-4',  # This will be mapped to your Azure deployment
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        azure_api_version=args.azure_api_version
    )
    
    # Create pipeline
    pipeline = Pipeline(
        provider_uri='~/.qlib/qlib_data/us_data',
        region='us',
        output_dir=args.output_dir,
        log_dir=os.path.join(args.output_dir, 'logs'),
        rdagent_api_key=api_key,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        azure_api_version=args.azure_api_version,
        chat_model='gpt-4'  # This will be mapped to your Azure deployment
    )
    
    # Start RDAgent UI if requested
    if args.rdagent_ui:
        rdagent_manager.start_ui()
    
    # Prepare data
    pipeline.prepare_data(
        instruments=symbols,
        start_time=args.start_date,
        end_time=args.end_date
    )
    
    # Run pipeline with feature evolution
    results = pipeline.run_pipeline(
        max_iterations=3,
        stopping_threshold=0.01
    )
    
    # Print final results
    logger.info("=== Final Results ===")
    logger.info(f"Iterations completed: {results['iterations']}")
    logger.info(f"Best model: {results['best_model']}")
    logger.info(f"Best strategy: {results['best_strategy']}")
    logger.info("Best metrics:")
    for key, value in results['best_metrics'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 