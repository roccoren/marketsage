#!/usr/bin/env python3
"""
Stock prediction and backtesting script with initial capital of $10,000.
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
from modules.forecasting import ForecastingManager
from modules.backtesting import BacktestingManager
from modules.data_acquisition import DataAcquisitionManager
from modules.mlflow_integration import log_experiment

def parse_args():
    parser = argparse.ArgumentParser(description='Stock prediction and backtesting script')
    parser.add_argument('--symbols', type=str, required=True,
                      help='Comma-separated list of stock symbols (e.g., AAPL,MSFT)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                      help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                      default=datetime.now().strftime('%Y-%m-%d'),
                      help='End date for analysis (YYYY-MM-DD)')
    return parser.parse_args()

def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]

    # Initialize managers
    data_manager = DataAcquisitionManager()
    forecast_manager = ForecastingManager(forecast_horizon=10)  # 10-day forecast
    backtest_manager = BacktestingManager()

    # Fetch data for all symbols
    stock_data = data_manager.fetch_multiple_stocks(
        symbols,
        start_date=args.start_date,
        end_date=args.end_date
    )

    all_metrics = {}
    all_forecasts = {}

    for symbol, data in stock_data.items():
        if data is not None and not data.empty:
            print(f"\nProcessing {symbol}...")
            
            # Generate forecast
            forecast_result = forecast_manager.generate_forecast(symbol, data)
            all_forecasts[symbol] = forecast_result['forecast_values']
            
            print(f"\nForecast Results for {symbol}:")
            print(f"Last Price: ${forecast_result['last_price']:.2f}")
            print(f"Predicted Price Changes (%):")
            for i, change in enumerate(forecast_result['daily_changes'], 1):
                print(f"Day {i}: {change:.2f}%")
            print(f"Overall Action: {forecast_result['overall_action']}")

            # Run backtest
            backtest_result = backtest_manager.run_backtest(
                symbol=symbol,
                data=data,
                forecasts=forecast_result['forecast_values']
            )

            # Store and print metrics
            metrics = backtest_result.get_metrics()
            all_metrics[symbol] = metrics
            
            print(f"\nBacktest Results for {symbol} (Initial Capital: $10,000):")
            print(f"Total Return: {metrics['total_return']*100:.2f}%")
            print(f"Annual Return: {metrics['annual_return']*100:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"Win Rate: {metrics['win_rate']*100:.2f}%")

            # Plot portfolio performance
            backtest_result.plot_portfolio()

    # Log experiment results to MLflow for all symbols
    log_experiment(
        data=stock_data,
        forecasts=all_forecasts,
        backtest_results=all_metrics
    )

if __name__ == "__main__":
    main()