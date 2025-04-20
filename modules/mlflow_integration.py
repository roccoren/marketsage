import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
# Import qlib conditionally to handle cases where it's not initialized
try:
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    R = None
    SignalRecord = None
from qlib.log import get_module_logger

logger = get_module_logger("MLflowManager")

class MLflowManager:
    """
    Manages MLflow experiment tracking integrated with MarketSage's workflow.
    """
    def __init__(self, experiment_name="marketsage", tracking_uri=None):
        """
        Initialize MLflow tracking.
        
        Args:
            experiment_name (str): Name of the MLflow experiment
            tracking_uri (str, optional): URI for MLflow tracking server. If None, uses local storage.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use /tmp/mlruns to avoid /mnt/d path issues
            if not os.path.exists('mlruns'):
                os.system('mkdir -p /tmp/mlruns && ln -s /tmp/mlruns mlruns')
            
            tracking_uri = "file:mlruns"
            print(f"[DEBUG] Setting MLflow tracking URI to: {tracking_uri}")
            print(f"[DEBUG] MLflow directory symlinked to: {os.path.realpath('mlruns')}")
            mlflow.set_tracking_uri(tracking_uri)
            
        self.experiment_name = experiment_name
        self._setup_experiment()
        
    def _setup_experiment(self):
        """Set up the MLflow experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            
    def start_run(self, run_name=None):
        """
        Start a new MLflow run integrated with MarketSage's workflow if available.
        
        Args:
            run_name (str, optional): Name for the run. If None, uses timestamp.
        """
        try:
            # End any active runs
            active_run = mlflow.active_run()
            if active_run:
                mlflow.end_run()
                logger.info(f"Ended previous MLflow run: {active_run.info.run_id}")

            # End any active Qlib recorder if Qlib is available
            if QLIB_AVAILABLE and R is not None:
                try:
                    recorder = R.get_recorder()
                    if recorder is not None and hasattr(recorder, 'close'):
                        recorder.close()
                    elif recorder is not None and hasattr(recorder, 'stop'):
                        recorder.stop()
                    logger.info("Ended previous Qlib recorder")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to end Qlib recorder: {e}")

            if run_name is None:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Start new Qlib experiment if available
            if QLIB_AVAILABLE and R is not None:
                try:
                    R.start(experiment_name=self.experiment_name)
                    logger.info(f"Started Qlib experiment: {self.experiment_name}")
                except Exception as e:
                    logger.warning(f"Failed to start Qlib experiment: {e}")
                    logger.warning("Continuing with MLflow only")

            # Start MLflow run
            mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            
            logger.info(f"Started MLflow run in experiment {self.experiment_name}")
            logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
            logger.info(f"Artifact URI: {mlflow.active_run().info.artifact_uri}")
            
            return mlflow.active_run().info.run_id
        except Exception as e:
            logger.error(f"Error starting run: {str(e)}")
            # Start MLflow run without Qlib if there was an error
            try:
                mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
                logger.info(f"Started MLflow run without Qlib integration")
                return mlflow.active_run().info.run_id
            except Exception as inner_e:
                logger.error(f"Failed to start MLflow run: {inner_e}")
                raise
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def active_run(self):
        return mlflow.active_run()
    
    def log_parameters(self, params):
        """
        Log parameters to MLflow.
        
        Args:
            params (dict): Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
            
    def log_param(self, key, value):
        """
        Log a single parameter to MLflow.
        
        Args:
            key (str): Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step number for the metrics
        """
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=step)
            except (ValueError, TypeError):
                print(f"Skipping metric {key} with non-numeric value: {value}")
    
    def log_artifact(self, local_path):
        """
        Log an artifact file to MLflow.
        
        Args:
            local_path (str): Path to the local file
        """
        artifact_abs_path = os.path.abspath(local_path)
        print(f"[DEBUG] Logging artifact from {artifact_abs_path}")
        print(f"[DEBUG] Current tracking URI: {mlflow.get_tracking_uri()}")
        print(f"[DEBUG] Current run artifact URI: {mlflow.active_run().info.artifact_uri}")
        mlflow.log_artifact(local_path)
    
    def log_model(self, model, artifact_path="model"):
        """
        Log a model to MLflow.
        
        Args:
            model: The model object to log
            artifact_path (str): Path within the artifact directory to store the model
        """
        mlflow.sklearn.log_model(model, artifact_path)
    
    def log_figure(self, figure, artifact_path):
        """
        Log a matplotlib figure to MLflow.
        
        Args:
            figure (matplotlib.figure.Figure): Figure to log
            artifact_path (str): Path to save the figure
        """
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        figure.savefig(temp_path)
        mlflow.log_artifact(temp_path, artifact_path)
        os.remove(temp_path)

    def _create_prediction_visualization(self, prediction_data, actual_data):
        """
        Create visualization comparing predicted vs actual stock movements.
        
        Args:
            prediction_data (dict): Prediction results
            actual_data (pd.DataFrame): Actual stock data
            
        Returns:
            matplotlib.figure.Figure: Figure with visualization
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical prices
        ax.plot(actual_data.index, actual_data['Close'], label='Historical Price', color='blue')
        
        # Calculate prediction dates and prices
        last_date = actual_data.index[-1]
        last_price = actual_data['Close'].iloc[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
        
        # Calculate forecast change
        forecast_change = prediction_data['forecast_change_percent'] / 100
        
        # Create a simple linear path from last price to predicted price
        predicted_prices = [last_price]
        for i in range(1, 6):
            step_change = last_price * (forecast_change * i / 5)
            predicted_prices.append(last_price + step_change)
        
        # Plot prediction line
        ax.plot([last_date] + forecast_dates, predicted_prices,
                label='Prediction', color='red', linestyle='--')
        
        # Add confidence interval based on volatility
        volatility = prediction_data.get('volatility', 0.5)
        upper_bound = [price * (1 + volatility) for price in predicted_prices]
        lower_bound = [price * (1 - volatility) for price in predicted_prices]
        
        ax.fill_between([last_date] + forecast_dates, lower_bound, upper_bound,
                        color='red', alpha=0.2, label='Confidence Interval')
        
        # Add action markers
        daily_actions = prediction_data.get('daily_actions', [])
        for i, action in enumerate(daily_actions):
            if action.lower() == 'buy':
                ax.scatter(forecast_dates[i], predicted_prices[i+1],
                          marker='^', color='green', s=100, label='Buy Signal')
            elif action.lower() == 'sell':
                ax.scatter(forecast_dates[i], predicted_prices[i+1],
                          marker='v', color='red', s=100, label='Sell Signal')
        
        # Add title and labels
        symbol = actual_data.name if hasattr(actual_data, 'name') else 'Stock'
        ax.set_title(f"{symbol} Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        
        # Add overall action and confidence as text
        action = prediction_data.get('overall_action', 'Unknown')
        confidence = prediction_data.get('confidence', 0)
        change_percent = prediction_data.get('forecast_change_percent', 0)
        
        text = f"Action: {action} (Confidence: {confidence:.2f})\nForecast Change: {change_percent:.2f}%"
        ax.text(0.02, 0.95, text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend (handles duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return fig

    def log_prediction_result(self, symbol, prediction_data, actual_data=None, tags=None):
        """
        Log prediction results to MLflow.
        
        Args:
            symbol (str): Stock symbol
            prediction_data (dict): Prediction results from TradingPredictionIntegrator
            actual_data (pd.DataFrame, optional): Actual stock data for comparison
            tags (dict, optional): Additional tags for the run
        """
        try:
            # Log parameters
            params = {
                "symbol": symbol,
                "prediction_date": datetime.now().strftime('%Y-%m-%d'),
                "prediction_type": prediction_data.get('overall_action', 'Unknown')
            }
            self.log_parameters(params)
            
            # Log metrics
            metrics = {
                "confidence": prediction_data.get('confidence', 0),
                "forecast_change_percent": prediction_data.get('forecast_change_percent', 0),
                "trend": prediction_data.get('trend', 0),
                "volatility": prediction_data.get('volatility', 0)
            }
            self.log_metrics(metrics)
            
            # Create and log visualization
            if actual_data is not None:
                fig = self._create_prediction_visualization(prediction_data, actual_data)
                self.log_figure(fig, "prediction_visualization")
                plt.close(fig)
            
            # Log prediction data as JSON
            prediction_path = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(prediction_path, "w") as f:
                json.dump(prediction_data, f)
            self.log_artifact(prediction_path)
            os.remove(prediction_path)
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            logger.info(f"Successfully logged prediction result for {symbol} to MLflow")
        except Exception as e:
            logger.error(f"Error logging prediction result: {e}")
            raise

    def log_trading_signals(self, signals_df):
        """
        Log trading signals to MLflow.
        
        Args:
            signals_df (pd.DataFrame): DataFrame containing trading signals with columns:
                - Symbol
                - Last Close
                - Signal (BUY/SELL/HOLD)
                - Confidence
                - Predicted Price (optional)
                - Expected Return (optional)
        """
        try:
            # Log signals as a CSV artifact
            signals_path = f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            signals_df.to_csv(signals_path)
            self.log_artifact(signals_path)
            os.remove(signals_path)
            
            # Log aggregate metrics
            metrics = {
                'total_signals': len(signals_df),
                'buy_signals': len(signals_df[signals_df['Signal'] == 'BUY']),
                'sell_signals': len(signals_df[signals_df['Signal'] == 'SELL']),
                'hold_signals': len(signals_df[signals_df['Signal'] == 'HOLD']),
                'avg_confidence': signals_df['Confidence'].mean()
            }
            self.log_metrics(metrics)
            
            # Create and log signal distribution visualization
            plt.figure(figsize=(10, 6))
            signal_counts = signals_df['Signal'].value_counts()
            signal_counts.plot(kind='bar')
            plt.title('Trading Signal Distribution')
            plt.xlabel('Signal Type')
            plt.ylabel('Count')
            plt.tight_layout()
            self.log_figure(plt.gcf(), "signal_distribution")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging trading signals: {e}")
            raise

    def log_stock_analysis(self, analysis_df):
        """
        Log stock analysis results to MLflow.
        
        Args:
            analysis_df (pd.DataFrame): DataFrame containing analysis results with columns:
                - Symbol
                - Base Return
                - Base Sharpe
                - Opt Sharpe
                - Improvement
                - Volatility
                - Win Rate
                - Max Drawdown
        """
        try:
            # Log analysis results as a CSV artifact
            analysis_path = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            analysis_df.to_csv(analysis_path)
            self.log_artifact(analysis_path)
            os.remove(analysis_path)
            
            # Log aggregate metrics
            metrics = {
                'avg_base_return': analysis_df['Base Return'].mean(),
                'avg_base_sharpe': analysis_df['Base Sharpe'].mean(),
                'avg_opt_sharpe': analysis_df['Opt Sharpe'].mean(),
                'avg_improvement': analysis_df['Improvement'].mean(),
                'avg_volatility': analysis_df['Volatility'].mean(),
                'avg_win_rate': analysis_df['Win Rate'].mean(),
                'avg_max_drawdown': analysis_df['Max Drawdown'].mean(),
            }
            self.log_metrics(metrics)
            
            # Create and log performance visualization
            plt.figure(figsize=(12, 6))
            performance_metrics = ['Base Sharpe', 'Opt Sharpe', 'Win Rate']
            analysis_df[performance_metrics].plot(kind='bar')
            plt.title('Performance Metrics by Stock')
            plt.xlabel('Stock')
            plt.ylabel('Value')
            plt.xticks(range(len(analysis_df)), analysis_df['Symbol'], rotation=45)
            plt.tight_layout()
            self.log_figure(plt.gcf(), "performance_metrics")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error logging stock analysis: {e}")
            raise

def log_experiment(data, forecasts, backtest_results):
    """
    Log experiment results to MLflow with Qlib integration if available.
    
    Args:
        data (pd.DataFrame): Historical stock data
        forecasts (dict): Forecasting results
        backtest_results (dict): Backtesting results
    """
    mlflow_manager = MLflowManager()
    
    # Start a new run
    run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Use context manager for Qlib if available, otherwise just start MLflow run directly
    try:
        if QLIB_AVAILABLE and R is not None:
            # Try with Qlib integration
            try:
                with R.start(experiment_name="marketsage"):
                    mlflow_manager.start_run(run_name)
                    _log_experiment_data(mlflow_manager, data, forecasts, backtest_results)
            except Exception as e:
                logger.warning(f"Failed to use Qlib integration: {e}")
                logger.info("Falling back to MLflow-only logging")
                mlflow_manager.start_run(run_name)
                _log_experiment_data(mlflow_manager, data, forecasts, backtest_results)
        else:
            # MLflow only
            mlflow_manager.start_run(run_name)
            _log_experiment_data(mlflow_manager, data, forecasts, backtest_results)
    except Exception as e:
        logger.error(f"Failed to log experiment: {e}")
        # Make sure we end the run if it was started
        if mlflow.active_run():
            mlflow_manager.end_run()

def _log_experiment_data(mlflow_manager, data, forecasts, backtest_results):
    """Helper function to log experiment data to MLflow."""
    try:
        # Log parameters
        params = {
            "symbols": ",".join(data.columns.get_level_values(1).unique()) if isinstance(data, pd.DataFrame) else "unknown",
            "forecast_horizon": len(forecasts) if isinstance(forecasts, list) else "unknown",
            "timestamp": datetime.now().isoformat(),
            "region": "us",  # Default to US market
            "model_type": "ensemble"  # Default model type
        }
        mlflow_manager.log_parameters(params)
        
        # Log metrics
        if isinstance(backtest_results, dict):
            for symbol, results in backtest_results.items():
                metrics = {f"{symbol}_{k}": v for k, v in results.items()}
                mlflow_manager.log_metrics(metrics)
        
        # Log forecasts directly
        if forecasts:
            forecast_df = pd.DataFrame(forecasts, columns=['Forecast'])
            forecast_path = "forecasts.csv"
            forecast_df.to_csv(forecast_path)
            mlflow_manager.log_artifact(forecast_path)
            os.remove(forecast_path)
        
        # Log artifacts
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Save data description with basic statistics
            data_desc_path = "data_description.txt"
            with open(data_desc_path, "w") as f:
                f.write("Data Statistics:\n")
                f.write(str(data.describe()))
            mlflow_manager.log_artifact(data_desc_path)
            os.remove(data_desc_path)
            
            # Create and log visualizations
            if 'Close' in data.columns:
                plt.figure(figsize=(10, 6))
                data['Close'].plot()
                plt.title('Stock Price Analysis')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.tight_layout()
                mlflow_manager.log_figure(plt.gcf(), "charts")
                plt.close()
        
        # Log basic experiment config
        config_path = "experiment_config.json"
        config = {
            "timestamp": datetime.now().isoformat(),
            "region": "us",
            "model_type": "ensemble",
            "forecast_horizon": len(forecasts) if isinstance(forecasts, list) else 0
        }
        with open(config_path, "w") as f:
            json.dump(config, f, default=str)
        mlflow_manager.log_artifact(config_path)
        os.remove(config_path)
        
        logger.info(f"Successfully logged experiment to MLflow. Run ID: {mlflow.active_run().info.run_id}")
    finally:
        # End MLflow run
        mlflow_manager.end_run()