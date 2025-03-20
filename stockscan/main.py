"""
StockScan Main Script - Examples of how to use the system
"""
import os
import argparse
from dotenv import load_dotenv
from stockscan.pipeline import Pipeline

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stock prediction pipeline')
    
    # Data source arguments
    parser.add_argument('--provider_uri', type=str, default='~/.qlib/qlib_data/cn_data',
                      help='URI for data provider (qlib)')
    parser.add_argument('--region', type=str, default='cn',
                      help='Region code for data [cn, us]')
    parser.add_argument('--instruments', type=str, default=None,
                      help='Stock symbols or market index')
    parser.add_argument('--start_time', type=str, default='2020-01-01',
                      help='Start date for data')
    parser.add_argument('--end_time', type=str, default='2023-01-01',
                      help='End date for data')
                      
    # Pipeline control arguments
    parser.add_argument('--max_iterations', type=int, default=10,
                      help='Maximum iterations for self-evolution')
    parser.add_argument('--stopping_threshold', type=float, default=0.01,
                      help='Improvement threshold below which to stop iterations')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Directory for output files')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory for log files')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='lightgbm',
                      help='Model type [lightgbm, xgboost, catboost]')
    parser.add_argument('--model_kwargs', type=str, default='{}',
                      help='Model kwargs as JSON string')
                      
    # Strategy arguments
    parser.add_argument('--strategy', type=str, default='topk',
                      help='Strategy type [topk, enhanced_indexing]')
    parser.add_argument('--topk', type=int, default=50,
                      help='Number of stocks to buy (for topk strategy)')
    parser.add_argument('--n_drop', type=int, default=5,
                      help='Number of stocks to sell (for topk strategy)')
    parser.add_argument('--risk_degree', type=float, default=0.95,
                      help='Risk degree (for enhanced_indexing strategy)')
    
    # RDAgent arguments
    parser.add_argument('--rdagent_ui', action='store_true',
                      help='Start the RDAgent UI')
    parser.add_argument('--rdagent_port', type=int, default=19899,
                      help='Port for the RDAgent UI')
    parser.add_argument('--chat_model', type=str, default='gpt-4-turbo',
                      help='LLM model to use for RDAgent')
    
    # Azure OpenAI arguments
    parser.add_argument('--azure_endpoint', type=str, default=None,
                      help='Azure OpenAI endpoint URL')
    parser.add_argument('--azure_deployment', type=str, default=None,
                      help='Azure OpenAI deployment name')
    parser.add_argument('--azure_api_version', type=str, default='2023-05-15',
                      help='Azure OpenAI API version')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Get API key from environment
    rdagent_api_key = os.environ.get('OPENAI_API_KEY')
    
    # Create pipeline
    pipeline = Pipeline(
        provider_uri=args.provider_uri,
        region=args.region,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        rdagent_api_key=rdagent_api_key,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        azure_api_version=args.azure_api_version,
        chat_model=args.chat_model
    )
    
    # Check Docker for RDAgent
    pipeline.rdagent_manager.check_docker()
    
    # Start RDAgent UI if requested
    if args.rdagent_ui:
        pipeline.rdagent_manager.start_ui(port=args.rdagent_port)
    
    # Prepare data
    pipeline.prepare_data(
        instruments=args.instruments,
        start_time=args.start_time,
        end_time=args.end_time
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(
        max_iterations=args.max_iterations,
        stopping_threshold=args.stopping_threshold
    )
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Iterations completed: {results['iterations']}")
    print(f"Best model: {results['best_model']}")
    print(f"Best strategy: {results['best_strategy']}")
    print("\nBest metrics:")
    for key, value in results['best_metrics'].items():
        print(f"  {key}: {value:.4f}")

def example_manual_workflow():
    """Example of how to use the system components manually"""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    rdagent_api_key = os.environ.get('OPENAI_API_KEY')
    
    # Create pipeline components
    from stockscan.core.qlib_manager import QlibManager
    from stockscan.core.feature_manager import FeatureManager
    from stockscan.models.model_manager import ModelManager
    from stockscan.strategies.strategy_manager import StrategyManager
    from stockscan.evolution.rdagent_manager import RDAgentManager
    
    # Initialize Qlib
    qlib_manager = QlibManager(provider_uri='~/.qlib/qlib_data/cn_data')
    qlib_manager.init()
    
    # Filter instruments
    instruments = qlib_manager.filter_instruments_by_name('SH.*')
    
    # Load features
    feature_manager = FeatureManager()
    feature_manager.load_alpha158_features()
    
    # Create dataset
    handler = qlib_manager.create_dataset_handler(
        instruments=instruments,
        start_time='2010-01-01',
        end_time='2021-12-31',
        fields=feature_manager.get_feature_list()
    )
    
    segments = {
        'train': ('2010-01-01', '2018-12-31'),
        'valid': ('2019-01-01', '2019-12-31'),
        'test': ('2020-01-01', '2021-12-31')
    }
    
    dataset = qlib_manager.prepare_dataset(handler, segments)
    
    # Train model
    model_manager = ModelManager()
    model = model_manager.create_lgb_model(
        'lgb_model',
        loss='mse',
        colsample_bytree=0.8879,
        learning_rate=0.0421, 
        max_depth=8,
        num_leaves=210
    )
    
    train_score, valid_score = model_manager.train_model('lgb_model', dataset)
    print(f"Model trained with scores - Train: {train_score}, Valid: {valid_score}")
    
    # Generate predictions
    predictions = model_manager.predict('lgb_model', dataset, 'test')
    
    # Create strategy
    strategy_manager = StrategyManager()
    strategy = strategy_manager.create_topk_strategy('topk_strategy', topk=50, n_drop=5)
    
    # Generate signals
    signals = strategy_manager.generate_signals('topk_strategy', predictions)
    
    # Run backtest
    backtest_result = strategy_manager.backtest_strategy('topk_strategy', predictions)
    
    # Evaluate results
    metrics = strategy_manager.evaluate_strategy(backtest_result)
    print("\nBacktest results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evolve factors using RDAgent
    rdagent_manager = RDAgentManager(log_dir='./logs/rdagent', api_key=rdagent_api_key)
    
    evolved_factors = rdagent_manager.evolve_factors(
        data_path='./output/data',
        output_path='./output/evolved_factors',
        existing_factors=feature_manager.custom_features,
        backtest_results=metrics
    )
    
    # Optimize model using RDAgent
    optimized_config = rdagent_manager.optimize_model(
        model_config=model_manager.model_configs.get('lgb_model'),
        data_path='./output/data',
        output_path='./output/model_optimization',
        backtest_results=metrics
    )

if __name__ == '__main__':
    main() 