import pandas as pd
import json
import gzip
from pathlib import Path
from .strategies.multi_indicator_strategy import MultiIndicatorStrategy
from .strategies.optimizer import StrategyOptimizer
from .strategies.backtester import Backtester
from .utils.technical_indicators import TechnicalIndicators
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(symbol: str) -> pd.DataFrame:
    """Load test data from JSON file"""
    data_path = Path("test_data") / "HKEX" / f"{symbol}.json.gz"
    
    with gzip.open(data_path, 'rt') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'history' in data:
        df = pd.DataFrame(data['history'])
    else:
        df = pd.DataFrame(data)
    
    # Convert date column (try different possible column names)
    date_columns = ['date', 'Date', 'timestamp', 'Timestamp']
    for col in date_columns:
        if col in df.columns:
            df['date'] = pd.to_datetime(df[col])
            break
    else:
        raise ValueError(f"No date column found. Available columns: {df.columns.tolist()}")
    
    # Set the date as index and ensure column names are capitalized
    df.set_index('date', inplace=True)
    df.columns = df.columns.str.capitalize()
    
    return df.sort_index()

def main():
    try:
        # Load test data
        symbol = "0700.HK"  # Tencent as example
        data = load_test_data(symbol)
        logger.info(f"\nLoaded data for {symbol}")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Available columns: {', '.join(data.columns)}")
        logger.info(f"Data shape: {data.shape}")

        # Print sample of the data
        logger.info("\nSample of loaded data:")
        print(data.head())

        # Define parameter grid for optimization (reduced periods for testing)
        param_grid = {
            # Technical indicator parameters (reduced periods)
            'sma_short_period': [20],   # Reduced from 50
            'sma_long_period': [50],    # Reduced from 200
            'ema_short_period': [20],   # Reduced from 50
            'ema_long_period': [50],    # Reduced from 200
            'rsi_period': [14],
            'rsi_oversold': [30],
            'rsi_overbought': [70],
            'bb_period': [20],
            'bb_std': [2.0],
            'macd_fast': [12],
            'macd_slow': [26],
            'macd_signal': [9],
            
            # Strategy weights (fixed to ensure they sum to 1.0)
            'trend_weight': [0.4],
            'momentum_weight': [0.2],
            'volatility_weight': [0.2],
            'macd_weight': [0.2]
        }

        logger.info("\nInitializing strategy optimizer...")
        optimizer = StrategyOptimizer(
            strategy_class=MultiIndicatorStrategy,
            param_grid=param_grid,
            data=data,
            train_size=0.8,
            metric='sharpe_ratio'
        )

        logger.info("\nRunning optimization...")
        results = optimizer.optimize(initial_capital=100000)

        if results['best_params'] is None:
            logger.error("Optimization failed to find valid parameters")
            return

        # Print optimization results
        logger.info("\nOptimization Results:")
        logger.info(f"Best Parameters: {results['best_params']}")
        logger.info(f"Best Training Sharpe Ratio: {results['best_train_score']:.2f}")
        logger.info(f"Out-of-Sample Sharpe Ratio: {results['best_test_score']:.2f}")

        # Generate detailed report
        report = optimizer.get_optimization_report()
        logger.info("\nParameter Combinations Tested:")
        print(report)

        # Save optimization results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        report.to_csv(output_dir / f"{symbol}_optimization_results.csv")
        
        with open(output_dir / f"{symbol}_best_params.json", 'w') as f:
            json.dump(results['best_params'], f, indent=4)

        # Run backtest with optimal parameters
        logger.info("\nRunning backtest with optimal parameters...")
        strategy = MultiIndicatorStrategy(**results['best_params'])
        backtester = Backtester(strategy, initial_capital=100000)
        backtest_result = backtester.run(data)

        # Print backtest metrics
        logger.info("\nBacktest Results:")
        metrics = backtest_result.metrics
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: ${metrics['max_drawdown']:.2f}")
        print(f"Total P&L: ${metrics['total_pnl']:.2f}")

        # Save backtest results
        with open(output_dir / f"{symbol}_backtest_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

    except Exception as e:
        logger.exception("An error occurred during backtesting")
        raise

if __name__ == "__main__":
    main()