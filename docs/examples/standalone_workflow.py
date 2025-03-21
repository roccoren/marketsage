#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import logging
import mlflow
import copy
import traceback

# Constants for configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Default configurations
DEFAULT_CONFIG = {
    "model": "Linear",
    "topk": 10,
    "n_drop": 2,
    "train_start": "2020-01-01",
    "train_end": "2020-06-30",
    "valid_start": "2020-07-01",
    "valid_end": "2020-09-30",
    "test_start": "2020-10-01",
    "test_end": "2020-12-31",
    "mlflow_tracking_uri": os.path.join(BASE_DIR, "mlruns"),
    "mlflow_experiment": "stock_analysis"
}

# Function to set up logging
def setup_logger(log_file=None):
    """Set up the logger with both file and console handlers."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


class SimpleModel:
    """Simple model implementation for testing."""
    
    def __init__(self, model_type="Linear"):
        """Initialize the model."""
        self.model_type = model_type
        self.fitted = False
    
    def fit(self, data):
        """Fit the model to the data."""
        # For simplicity, we just mark the model as fitted
        self.fitted = True
        return self
    
    def predict(self, data):
        """
        Generate predictions based on model type.
        
        Returns a DataFrame with stocks as columns and dates as index.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract dates and stocks
        dates = data.index.unique() if isinstance(data.index, pd.MultiIndex) else data.index
        
        if isinstance(data.index, pd.MultiIndex):
            stocks = data.index.get_level_values('stock').unique()
        else:
            # If we have a DataFrame with dates as index and stocks as columns
            stocks = data.columns
        
        # Create a DataFrame for predictions
        predictions = pd.DataFrame(index=dates, columns=stocks)
        
        # Fill with synthetic predictions based on model type
        if self.model_type == "Linear":
            # Simple linear trend with small random noise
            for i, stock in enumerate(stocks):
                trend = np.linspace(0, 1, len(dates))
                noise = np.random.normal(0, 0.1, len(dates))
                predictions[stock] = trend + noise
        
        elif self.model_type == "LGBM":
            # More volatile predictions
            for i, stock in enumerate(stocks):
                base = np.sin(np.linspace(0, 4*np.pi, len(dates)))
                noise = np.random.normal(0, 0.2, len(dates))
                predictions[stock] = base + noise
        
        elif self.model_type == "DEnsemble":
            # Complex predictions with sine waves
            for i, stock in enumerate(stocks):
                base = np.sin(np.linspace(0, 4*np.pi, len(dates))) * np.cos(np.linspace(0, 2*np.pi, len(dates)))
                noise = np.random.normal(0, 0.15, len(dates))
                predictions[stock] = base + noise
        
        # Normalize predictions
        for stock in stocks:
            predictions[stock] = (predictions[stock] - predictions[stock].mean()) / predictions[stock].std()
        
        return predictions


class SimpleBacktester:
    """
    Simple backtester to evaluate predictions
    """
    def __init__(self, pred_df, topk=10, n_drop=2, initial_capital=100000):
        self.pred_df = pred_df
        self.topk = topk
        self.n_drop = n_drop
        self.initial_capital = initial_capital
        
        # Ensure pred_df has the right format
        if not isinstance(self.pred_df.index, pd.DatetimeIndex):
            # Try to convert index to datetime if it's not already
            try:
                self.pred_df.index = pd.to_datetime(self.pred_df.index)
            except:
                raise ValueError("Prediction dataframe index must be convertible to datetime")
                
        # Sort index
        self.pred_df = self.pred_df.sort_index()
    
    def run_backtest(self):
        """
        Run a backtest on the predictions
        """
        # Portfolio value over time
        portfolio_values = pd.Series(index=self.pred_df.index, dtype=float)
        portfolio_values.iloc[0] = self.initial_capital
        
        # Current positions
        positions = pd.DataFrame(0, index=self.pred_df.index, columns=self.pred_df.columns)
        
        # Track metrics
        daily_returns = []
        turnover_rates = []
        
        # Run backtest day by day
        for i, date in enumerate(self.pred_df.index):
            if i == 0:
                # Initial portfolio allocation
                top_stocks = self.pred_df.iloc[i].nlargest(self.topk).index
                # Fix: Assign positions one by one instead of as a series
                for stock in top_stocks:
                    positions.loc[date, stock] = 1
                continue
            
            # Get top stocks for current date
            top_stocks = self.pred_df.iloc[i].nlargest(self.topk).index
            
            # Calculate turnover
            prev_positions = positions.iloc[i-1]
            current_positions = pd.Series(0, index=self.pred_df.columns)
            current_positions[top_stocks] = 1
            
            # Calculate turnover rate
            turnover = (current_positions != prev_positions).sum() / (2 * self.topk)
            turnover_rates.append(turnover)
            
            # Update positions
            # Fix: Assign positions one by one instead of as a series
            for stock in self.pred_df.columns:
                positions.loc[date, stock] = 1 if stock in top_stocks else 0
            
            # Calculate daily return (random for demonstration)
            # In a real implementation, you would use actual price data
            daily_return = np.random.normal(0.0005, 0.01)  # Mean 0.05%, std 1%
            daily_returns.append(daily_return)
            
            # Update portfolio value
            portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_return)
        
        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = total_return / (len(self.pred_df) / 252)
        
        daily_returns_series = pd.Series(daily_returns)
        sharpe = daily_returns_series.mean() / daily_returns_series.std() * np.sqrt(252)
        max_drawdown = ((portfolio_values / portfolio_values.cummax()) - 1).min()
        
        # Prepare results
        results = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "turnover": np.mean(turnover_rates) if turnover_rates else 0,
            "portfolio_value": portfolio_values,
            "positions": positions
        }
        
        return results
    
    def plot_portfolio_value(self, save_path=None):
        """
        Plot portfolio value over time
        """
        results = self.run_backtest()
        portfolio_value = results["portfolio_value"]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value.index, portfolio_value.values)
        plt.title("Portfolio Value")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            return save_path
        
        # Otherwise show plot
        plt.show()
        plt.close()
        return None


def generate_stock_data(num_stocks=50, start_date='2020-01-01', end_date='2020-12-31', freq='D', price_volatility=0.01):
    """
    Generate synthetic stock data for testing
    
    Parameters
    ----------
    num_stocks : int
        Number of stocks to generate
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    freq : str
        Frequency of data points (D for daily, W for weekly, etc.)
    price_volatility : float
        Volatility parameter for price generation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic stock data
    """
    # Convert dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate date range
    dates = pd.date_range(start=start, end=end, freq=freq)
    
    # Generate stock symbols
    stocks = [f"stock_{i:03d}" for i in range(1, num_stocks+1)]
    
    # Create empty dataframe with DatetimeIndex
    df = pd.DataFrame(index=dates, columns=stocks)
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    for stock in stocks:
        # Generate initial price between 50 and 150
        initial_price = np.random.uniform(50, 150)
        
        # Generate daily returns with specified volatility
        daily_returns = np.random.normal(0.0005, price_volatility, len(dates))
        
        # Add some trend component
        trend = np.linspace(-0.1, 0.1, len(dates))
        seasonal = 0.05 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns + trend + seasonal)
        
        # Calculate prices
        prices = initial_price * cumulative_returns
        
        # Add to dataframe
        df[stock] = prices
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    return df


def init_mlflow(tracking_uri, experiment_name):
    """
    Initialize MLflow tracking
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Start a new run
    run = mlflow.start_run(experiment_id=experiment_id)
    
    return run.info.run_id


def close_mlflow():
    """
    End MLflow tracking
    """
    mlflow.end_run()


def run_workflow(
    model_type,
    dataset_segments=None,
    topk=10,
    n_drop=2,
    output_dir=None,
    track_mlflow=True,
    num_stocks=50,
    price_volatility=0.01,
    predict_start=None,
    predict_end=None,
    backtest_start=None,
    backtest_end=None,
    skip_data=False,
    skip_prediction=False,
    skip_backtest=False,
    data=None
):
    """
    Run the complete workflow for a single model
    
    Parameters
    ----------
    model_type : str
        Type of model to use
    dataset_segments : Dict[str, Tuple[str, str]]
        Dictionary with dataset segments
    topk : int
        Number of top stocks to select
    n_drop : int
        Number of stocks to drop at each rebalance
    output_dir : str
        Directory to save outputs
    track_mlflow : bool
        Whether to track with MLflow
    num_stocks : int
        Number of stocks to simulate
    price_volatility : float
        Volatility parameter for price generation
    predict_start : str
        Start date for prediction (default: test_start)
    predict_end : str
        End date for prediction (default: test_end)
    backtest_start : str
        Start date for backtesting (default: test_start)
    backtest_end : str
        End date for backtesting (default: test_end)
    skip_data : bool
        Whether to skip data generation step
    skip_prediction : bool
        Whether to skip prediction step
    skip_backtest : bool
        Whether to skip backtesting step
    data : pd.DataFrame
        Pre-generated data (if provided, skip_data will be set to True)
        
    Returns
    -------
    Dict
        Dictionary with results
    """
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default dataset segments if not provided
    if dataset_segments is None:
        dataset_segments = {
            "train": ("2020-01-01", "2020-06-30"),
            "valid": ("2020-07-01", "2020-09-30"),
            "test": ("2020-10-01", "2020-12-31")
        }
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, f"stockscan_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "workflow.log")
    setup_logger(log_file)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting workflow for model: {model_type}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset segments: {dataset_segments}")
    
    # Create subdirectories
    data_dir = os.path.join(output_dir, "data")
    model_dir = os.path.join(output_dir, "model")
    backtest_dir = os.path.join(output_dir, "backtest")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(backtest_dir, exist_ok=True)
    
    # Initialize MLflow if tracking is enabled
    run_id = None
    if track_mlflow:
        run_id = init_mlflow(DEFAULT_CONFIG["mlflow_tracking_uri"], DEFAULT_CONFIG["mlflow_experiment"])
        mlflow.log_params({
            "model_type": model_type,
            "topk": topk,
            "n_drop": n_drop,
            "train_start": dataset_segments["train"][0],
            "train_end": dataset_segments["train"][1],
            "valid_start": dataset_segments["valid"][0],
            "valid_end": dataset_segments["valid"][1],
            "test_start": dataset_segments["test"][0],
            "test_end": dataset_segments["test"][1],
            "predict_start": predict_start or dataset_segments["test"][0],
            "predict_end": predict_end or dataset_segments["test"][1],
            "backtest_start": backtest_start or dataset_segments["test"][0],
            "backtest_end": backtest_end or dataset_segments["test"][1]
        })
    
    try:
        # STEP 1: Generate or load data
        if data is not None:
            # Use the provided data
            logger.info("Using provided data")
            skip_data = True
        else:
            data = None
        
        if not skip_data:
            logger.info("Generating synthetic stock data")
            data_file = os.path.join(data_dir, "stock_data.pkl")
            
            data = generate_stock_data(
                start_date=dataset_segments["train"][0],
                end_date=dataset_segments["test"][1],
                num_stocks=num_stocks,
                price_volatility=price_volatility
            )
            
            with open(data_file, "wb") as f:
                pickle.dump(data, f)
            
            logger.info(f"Data generated and saved to {data_file}")
            
            if track_mlflow:
                mlflow.log_param("num_stocks", data.shape[1])
                mlflow.log_param("date_range", f"{data.index[0]} to {data.index[-1]}")
        elif data is None:
            # If skipping data generation and no data provided, try to load from file
            data_file = os.path.join(data_dir, "stock_data.pkl")
            if os.path.exists(data_file):
                logger.info(f"Loading data from {data_file}")
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"No data provided and no data file found at {data_file}")
        
        # Log data structure for debugging
        logger.info(f"Data type: {type(data)}")
        logger.info(f"Data index type: {type(data.index)}")
        if isinstance(data.index, pd.MultiIndex):
            logger.info(f"Data has MultiIndex with levels: {data.index.names}")
        else:
            logger.info(f"Data has single-level index: {data.index.name or 'unnamed'}")
        logger.info(f"First few index values: {data.index[:5]}")
        
        # Ensure data has DatetimeIndex if it's not a MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.info("Converting data index to DatetimeIndex")
                data.index = pd.to_datetime(data.index)
        
        # STEP 2: Train model and make predictions
        predictions = None
        
        if not skip_prediction:
            logger.info(f"Training {model_type} model and making predictions")
            
            # Use the specified prediction period or default to test period
            if predict_start is not None and predict_end is not None:
                pred_start_date = pd.Timestamp(predict_start)
                pred_end_date = pd.Timestamp(predict_end)
            else:
                pred_start_date = pd.Timestamp(dataset_segments["test"][0])
                pred_end_date = pd.Timestamp(dataset_segments["test"][1])
            
            logger.info(f"Prediction period: {pred_start_date.strftime('%Y-%m-%d')} to {pred_end_date.strftime('%Y-%m-%d')}")
            
            # Filter data for prediction period based on index type
            if isinstance(data.index, pd.MultiIndex):
                # For MultiIndex, filter based on the first level if it's the date level
                date_level_idx = data.index.names.index('date') if 'date' in data.index.names else 0
                prediction_dates = data.index.get_level_values(date_level_idx)
                prediction_data = data[
                    (prediction_dates >= pred_start_date) & 
                    (prediction_dates <= pred_end_date)
                ]
            else:
                # For regular DatetimeIndex
                prediction_data = data[
                    (data.index >= pred_start_date) & 
                    (data.index <= pred_end_date)
                ]
                
            logger.info(f"Filtered prediction data shape: {prediction_data.shape}")
            
            # Train and predict with the specified model
            model = SimpleModel(model_type=model_type)
            model.fit(prediction_data)
            predictions = model.predict(prediction_data)
            
            # Save predictions
            predictions_file = os.path.join(model_dir, "predictions.pkl")
            predictions.to_pickle(predictions_file)
            
            csv_file = os.path.join(model_dir, "predictions.csv")
            predictions.to_csv(csv_file)
            
            logger.info(f"Predictions generated and saved to {predictions_file}")
            logger.info(f"Predictions index type: {type(predictions.index)}")
            logger.info(f"Predictions shape: {predictions.shape}")
            
            if track_mlflow:
                mlflow.log_artifact(csv_file)
        else:
            logger.info("Skipping prediction step")
        
        # STEP 3: Run backtest
        backtest_results = None
        
        if not skip_backtest and predictions is not None:
            logger.info(f"Running backtest with topk={topk}, n_drop={n_drop}")
            
            # Filter predictions for backtest period if specified
            if backtest_start is not None and backtest_end is not None:
                # Ensure dates are timestamps
                bt_start_date = pd.Timestamp(backtest_start)
                bt_end_date = pd.Timestamp(backtest_end)
                
                # Ensure predictions index is DatetimeIndex
                if not isinstance(predictions.index, pd.DatetimeIndex):
                    logger.info(f"Converting predictions index from {type(predictions.index)} to DatetimeIndex")
                    predictions.index = pd.to_datetime(predictions.index)
                
                logger.info(f"Filtering predictions for backtest from {bt_start_date} to {bt_end_date}")
                logger.info(f"Predictions index range: {predictions.index.min()} to {predictions.index.max()}")
                
                # Filter predictions based on backtest date range
                backtest_predictions = predictions[
                    (predictions.index >= bt_start_date) & 
                    (predictions.index <= bt_end_date)
                ]
                
                if backtest_predictions.empty:
                    raise ValueError(f"No predictions found within backtest date range: {backtest_start} to {backtest_end}")
                
                logger.info(f"Filtered predictions for backtest: from {backtest_predictions.index.min()} to {backtest_predictions.index.max()}")
            else:
                backtest_predictions = predictions
                logger.info(f"Using all predictions for backtest: from {predictions.index.min()} to {predictions.index.max()}")
            
            # Run backtest with the filtered predictions
            backtester = SimpleBacktester(backtest_predictions, topk=topk, n_drop=n_drop)
            backtest_results = backtester.run_backtest()
            
            # Save backtest results
            results_file = os.path.join(backtest_dir, "backtest_results.csv")
            detailed_file = os.path.join(backtest_dir, "detailed_results.pkl")
            
            # Save summary results to CSV
            summary_results = {
                "total_return": backtest_results["total_return"],
                "annual_return": backtest_results["annual_return"],
                "sharpe": backtest_results["sharpe"],
                "max_drawdown": backtest_results["max_drawdown"],
                "turnover": backtest_results["turnover"]
            }
            
            pd.DataFrame([summary_results]).to_csv(results_file, index=False)
            
            # Save detailed results to pickle
            with open(detailed_file, "wb") as f:
                pickle.dump(backtest_results, f)
            
            logger.info(f"Backtest results saved to {results_file}")
            
            # Generate portfolio value plot
            if "portfolio_value" in backtest_results:
                plot_file = os.path.join(backtest_dir, "portfolio_value.png")
                backtester.plot_portfolio_value(save_path=plot_file)
                logger.info(f"Portfolio value plot saved to {plot_file}")
                
                if track_mlflow:
                    mlflow.log_metrics(summary_results)
                    mlflow.log_artifact(plot_file)
                    mlflow.log_artifact(results_file)
        else:
            logger.info("Skipping backtest step")
        
        logger.info(f"Workflow completed successfully for model: {model_type}")
        
        return {
            "model_type": model_type,
            "predictions": predictions,
            "backtest_results": backtest_results or {},
            "detailed_results": backtest_results or {},
            "output_dir": output_dir
        }
    
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        traceback.print_exc()
        if track_mlflow:
            mlflow.log_param("workflow_status", "error")
            mlflow.log_param("workflow_error", str(e))
        raise e
    
    finally:
        if track_mlflow:
            close_mlflow()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Scanning Workflow")
    
    # Model group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default="Linear", help="Model type (Linear, LGBM, DEnsemble)")
    model_group.add_argument("--multi", action="store_true", help="Run multiple models and compare")
    model_group.add_argument("--models", nargs="+", default=["Linear", "LGBM", "DEnsemble"], 
                             help="List of models to use with --multi")
    
    # Data generation group
    data_group = parser.add_argument_group("Data Generation")
    data_group.add_argument("--num_stocks", type=int, default=50, help="Number of stocks to generate")
    data_group.add_argument("--price_volatility", type=float, default=0.01, 
                           help="Volatility parameter for price generation")
    
    # Date ranges group
    date_group = parser.add_argument_group("Date Ranges")
    date_group.add_argument("--train_start", type=str, default="2020-01-01", help="Start date for training")
    date_group.add_argument("--train_end", type=str, default="2020-06-30", help="End date for training")
    date_group.add_argument("--valid_start", type=str, default="2020-07-01", help="Start date for validation")
    date_group.add_argument("--valid_end", type=str, default="2020-09-30", help="End date for validation")
    date_group.add_argument("--test_start", type=str, default="2020-10-01", help="Start date for testing")
    date_group.add_argument("--test_end", type=str, default="2020-12-31", help="End date for testing")
    date_group.add_argument("--predict_start", type=str, default=None, 
                           help="Start date for prediction (defaults to test_start)")
    date_group.add_argument("--predict_end", type=str, default=None, 
                           help="End date for prediction (defaults to test_end)")
    date_group.add_argument("--backtest_start", type=str, default=None, 
                           help="Start date for backtesting (defaults to predict_start)")
    date_group.add_argument("--backtest_end", type=str, default=None, 
                           help="End date for backtesting (defaults to predict_end)")
    
    # Strategy group
    strategy_group = parser.add_argument_group("Strategy Configuration")
    strategy_group.add_argument("--topk", type=int, default=10, help="Number of top stocks to select")
    strategy_group.add_argument("--n_drop", type=int, default=2, help="Number of stocks to drop at each rebalance")
    
    # Output group
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output_dir", type=str, default=None, help="Output directory")
    output_group.add_argument("--no_mlflow", action="store_true", help="Disable MLflow tracking")
    
    # Skip steps
    skip_group = parser.add_argument_group("Skip Steps")
    skip_group.add_argument("--skip_data", action="store_true", help="Skip data generation step")
    skip_group.add_argument("--skip_prediction", action="store_true", help="Skip prediction step")
    skip_group.add_argument("--skip_backtest", action="store_true", help="Skip backtesting step")
    
    return parser.parse_args()


def run_multi_model_workflow(args):
    """Run multiple models and compare their results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.output_dir or f"results/multi_model_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    setup_logger(f"{result_dir}/workflow.log")
    logging.info(f"Starting multi-model workflow with models: {args.models}")
    logging.info(f"Results will be saved to: {result_dir}")
    
    model_results = {}
    
    for model_name in args.models:
        logging.info(f"Running workflow for model: {model_name}")
        model_dir = os.path.join(result_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create data directory
        data_dir = os.path.join(model_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate stock data
        data_path = os.path.join(data_dir, "stock_data.pkl")
        logging.info(f"Generating synthetic stock data for {model_name} model")
        data = generate_stock_data(
            start_date=args.train_start, 
            end_date=args.test_end,
            num_stocks=args.num_stocks,
            price_volatility=args.price_volatility
        )
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Saved data to {data_path}")
        
        # Set up model arguments
        model_args = copy.deepcopy(args)
        model_args.model = model_name
        model_args.output_dir = model_dir
        
        # Run model specific workflow
        # Skip data generation as we've already generated it
        try:
            # Run the workflow for this model
            result = run_workflow(
                model_type=model_name,
                dataset_segments={
                    "train": (args.train_start, args.train_end),
                    "valid": (args.valid_start, args.valid_end),
                    "test": (args.test_start, args.test_end)
                },
                topk=args.topk,
                n_drop=args.n_drop,
                output_dir=model_dir,
                track_mlflow=not args.no_mlflow,
                num_stocks=args.num_stocks,
                price_volatility=args.price_volatility,
                predict_start=args.predict_start,
                predict_end=args.predict_end,
                backtest_start=args.backtest_start,
                backtest_end=args.backtest_end,
                skip_data=True,  # We've already generated the data
                skip_prediction=False,
                skip_backtest=args.skip_backtest,
                data=data  # Pass the generated data
            )
            
            model_results[model_name] = result
        except Exception as e:
            logging.error(f"Error running workflow for model {model_name}: {str(e)}")
            traceback.print_exc()
            continue
    
    # Skip comparison if we don't have at least 2 successful models
    if len(model_results) < 1:
        logging.error("No models completed successfully, skipping comparison")
        return result_dir
    
    # Compare models
    logging.info("Comparing model results")
    comparison_dir = os.path.join(result_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create model comparison CSV if we have at least 2 models
    if len(model_results) >= 2:
        comparison_df = pd.DataFrame({
            model: {
                'total_return': results['backtest_results'].get('total_return', 0),
                'annual_return': results['backtest_results'].get('annual_return', 0),
                'sharpe': results['backtest_results'].get('sharpe', 0),
                'max_drawdown': results['backtest_results'].get('max_drawdown', 0),
                'turnover': results['backtest_results'].get('turnover', 0),
            } 
            for model, results in model_results.items()
        }).T
        
        comparison_path = os.path.join(comparison_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path)
        logging.info(f"Saved model comparison to {comparison_path}")
        
        # Create portfolio comparison plot
        plt.figure(figsize=(12, 8))
        for model_name, results in model_results.items():
            if 'detailed_results' in results and 'portfolio_value' in results['detailed_results']:
                portfolio_value = results['detailed_results']['portfolio_value']
                plt.plot(portfolio_value.index, portfolio_value.values, label=model_name)
        
        plt.title("Portfolio Value Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(comparison_dir, "portfolio_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved portfolio comparison plot to {plot_path}")
    else:
        logging.warning("Not enough successful models for comparison")
    
    logging.info(f"Multi-model workflow completed. Results saved to {result_dir}")
    return result_dir


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.multi:
            # Run multi-model workflow
            result_dir = run_multi_model_workflow(args)
        else:
            # Run single-model workflow
            result_dir = run_workflow(
                model_type=args.model,
                dataset_segments={
                    "train": (args.train_start, args.train_end),
                    "valid": (args.valid_start, args.valid_end),
                    "test": (args.test_start, args.test_end)
                },
                topk=args.topk,
                n_drop=args.n_drop,
                output_dir=args.output_dir,
                track_mlflow=not args.no_mlflow,
                num_stocks=args.num_stocks,
                price_volatility=args.price_volatility,
                predict_start=args.predict_start,
                predict_end=args.predict_end,
                backtest_start=args.backtest_start,
                backtest_end=args.backtest_end,
                skip_data=args.skip_data,
                skip_prediction=args.skip_prediction,
                skip_backtest=args.skip_backtest
            )
            result_dir = result_dir["output_dir"]
        
        print(f"Workflow completed successfully. Results saved to {result_dir}")
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 