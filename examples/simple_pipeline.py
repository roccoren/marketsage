"""
Simple pipeline example - Shows how to use the StockScan pipeline
"""
import os
from dotenv import load_dotenv
import pandas as pd
from stockscan.pipeline import Pipeline

# Load environment variables
load_dotenv()

# Get API key from environment
rdagent_api_key = os.environ.get('OPENAI_API_KEY')

# Create pipeline
pipeline = Pipeline(
    provider_uri='~/.qlib/qlib_data/cn_data',
    region='cn',
    output_dir='./output',
    log_dir='./logs',
    rdagent_api_key=rdagent_api_key
)

# Check Docker for RDAgent
docker_available = pipeline.rdagent_manager.check_docker()
if not docker_available:
    print("Warning: Docker is not available. RDAgent's evolution features will not work.")

# Initialize Qlib
pipeline.initialize_qlib()

# Prepare data for CSI300 stocks from 2019 to 2021
pipeline.prepare_data(
    instruments='csi300',
    start_time='2019-01-01',
    end_time='2021-12-31'
)

# Start with a basic LightGBM model for the first iteration
model_name, scores = pipeline.train_model(
    model_type='lgb', 
    loss='mse',
    colsample_bytree=0.8879,
    learning_rate=0.0421, 
    max_depth=8,
    num_leaves=210
)

print(f"Model {model_name} trained with scores:")
print(f"  Train score: {scores[0]:.4f}")
print(f"  Valid score: {scores[1]:.4f}")

# Generate signals with a TopK strategy
signals, predictions = pipeline.generate_signals(
    strategy_type='topk',
    topk=50,
    n_drop=5
)

print(f"Generated signals for {len(predictions)} instruments")

# Run backtest
backtest_result, metrics = pipeline.run_backtest(predictions)

print("Backtest results:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

# Evolve factors if Docker is available
if docker_available:
    print("\nEvolving factors...")
    evolved_factors = pipeline.evolve_factors()
    print(f"Evolved {len(evolved_factors)} factors")
    
    # Optimize model if Docker is available
    print("\nOptimizing model...")
    optimized_model_name, opt_scores = pipeline.optimize_model()
    print(f"Optimized model {optimized_model_name} trained with scores:")
    print(f"  Train score: {opt_scores[0]:.4f}")
    print(f"  Valid score: {opt_scores[1]:.4f}")
    
    # Generate signals with the optimized model
    signals, predictions = pipeline.generate_signals(
        strategy_type='topk',
        topk=50,
        n_drop=5
    )
    
    # Run backtest with the optimized model
    backtest_result, metrics = pipeline.run_backtest(predictions)
    
    print("\nBacktest results with optimized model:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

# For a full evolutionary pipeline, you would run:
# results = pipeline.run_pipeline(max_iterations=5, stopping_threshold=0.01)
# print("\nFinal results after evolution:")
# print(f"Iterations completed: {results['iterations']}")
# print(f"Best model: {results['best_model']}")
# print(f"Best strategy: {results['best_strategy']}")
# print("\nBest metrics:")
# for key, value in results['best_metrics'].items():
#     print(f"  {key}: {value:.4f}")

print("\nExample pipeline completed") 