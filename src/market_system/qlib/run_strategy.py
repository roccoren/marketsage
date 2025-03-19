"""
Main script for running strategies using Qlib.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import yaml
import pandas as pd

from .data.data_loader import QlibDataLoader
from .strategies.macd_strategy import MACDStrategy
from .backtesting.backtest_engine import BacktestEngine
from .config import QLIB_CONFIG, DATA_HANDLER_CONFIG

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_strategy(config: Dict, output_dir: Optional[str] = None):
    """Run a strategy with the given configuration.
    
    Args:
        config (dict): Strategy configuration
        output_dir (str, optional): Directory for output files. Defaults to None.
    """
    # Initialize data loader
    data_loader = QlibDataLoader()
    
    # Load dataset
    dataset = data_loader.load_dataset(
        instruments=config.get("instruments"),
        start_time=config.get("start_time"),
        end_time=config.get("end_time")
    )
    
    # Initialize strategy
    strategy_config = config.get("strategy", {})
    strategy = MACDStrategy(**strategy_config)
    
    # Initialize backtest engine
    backtest_config = config.get("backtest", {})
    engine = BacktestEngine(strategy, backtest_config)
    
    # Run backtest
    results = engine.run(
        dataset,
        start_time=config["start_time"],
        end_time=config["end_time"]
    )
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([results["analysis"]])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        
        # Save portfolio metrics
        results["portfolio_metrics"].to_csv(output_dir / "portfolio.csv")
        
        # Plot results
        engine.plot_results(results, str(output_dir / "performance.png"))
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run trading strategy using Qlib")
    parser.add_argument("--config", type=str, required=True, help="Path to strategy config file")
    parser.add_argument("--output", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load and run strategy
    config = load_config(args.config)
    results = run_strategy(config, args.output)
    
    # Print summary
    print("\nStrategy Results:")
    print("-" * 40)
    for metric, value in results["analysis"].items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
    
if __name__ == "__main__":
    main() 