"""
Report generator for backtesting results.
Generates comprehensive performance reports in various formats.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from decimal import Decimal

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates comprehensive performance reports for backtesting results.
    Supports HTML, JSON, and CSV report formats.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for report output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        logger.info(f"ReportGenerator initialized with output directory: {self.output_dir}")
        
    def generate_report(self, 
                        backtest_results: Dict,
                        report_name: str,
                        formats: List[str] = ["html", "json", "csv"]) -> Dict[str, str]:
        """
        Generate a comprehensive performance report in multiple formats.
        
        Args:
            backtest_results: Results from the backtest
            report_name: Name for the report
            formats: List of formats to generate ("html", "json", "csv")
            
        Returns:
            Dict mapping format to report file path
        """
        report_paths = {}
        
        # Create report directory
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)
        
        # Generate reports in requested formats
        if "html" in formats:
            html_path = self.generate_html_report(backtest_results, report_name)
            report_paths["html"] = html_path
            
        if "json" in formats:
            json_path = self.generate_json_report(backtest_results, report_name)
            report_paths["json"] = json_path
            
        if "csv" in formats:
            csv_paths = self.generate_csv_reports(backtest_results, report_name)
            report_paths["csv"] = csv_paths
            
        # Generate charts
        chart_dir = report_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        self.generate_charts(backtest_results, chart_dir)
        
        logger.info(f"Generated reports for {report_name} in formats: {', '.join(formats)}")
        
        return report_paths
        
    def generate_html_report(self, 
                         backtest_results: Dict,
                         report_name: str) -> str:
        """
        Generate an HTML report for the backtest results.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Name for the report file
            
        Returns:
            Path to the generated HTML report
        """
        # Create charts directory
        chart_dir = self.output_dir / "charts" / report_name
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate charts
        self.generate_charts(backtest_results, chart_dir)
        
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report: {report_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart-container {{ margin: 20px 0; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; }}
                .metric-box {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .trades-table {{ max-height: 500px; overflow-y: auto; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report: {report_name}</h1>
            
            <h2>Summary</h2>
            <div class="metrics-container">
        """
        
        # Add summary metrics
        if "performance_metrics" in backtest_results:
            metrics = backtest_results["performance_metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float, Decimal)):
                    color_class = self._get_color_class(float(value)) if key in ["total_return", "annualized_return", "sharpe_ratio"] else ""
                    html_content += f"""
                    <div class="metric-box">
                        <div>{key.replace('_', ' ').title()}</div>
                        <div class="metric-value {color_class}">{value:.2f}%</div>
                    </div>
                    """ if key in ["total_return", "annualized_return"] else f"""
                    <div class="metric-box">
                        <div>{key.replace('_', ' ').title()}</div>
                        <div class="metric-value {color_class}">{value:.2f}</div>
                    </div>
                    """
        
        html_content += """
            </div>
            
            <h2>Equity Curve</h2>
            <div class="chart-container">
                <img src="charts/{}/equity_curve.png" alt="Equity Curve" style="width: 100%;">
            </div>
            
            <h2>Drawdown</h2>
            <div class="chart-container">
                <img src="charts/{}/drawdown.png" alt="Drawdown" style="width: 100%;">
            </div>
            
            <h2>Monthly Returns</h2>
            <div class="chart-container">
                <img src="charts/{}/monthly_returns.png" alt="Monthly Returns" style="width: 100%;">
            </div>
            
            <h2>Trade Distribution</h2>
            <div class="chart-container">
                <img src="charts/{}/trade_distribution.png" alt="Trade Distribution" style="width: 100%;">
            </div>
        """.format(report_name, report_name, report_name, report_name)
        
        # Add trades table with timestamps for verification
        if "portfolio_stats" in backtest_results and "trades_with_timestamps" in backtest_results["portfolio_stats"]:
            trades = backtest_results["portfolio_stats"]["trades_with_timestamps"]
            if trades:
                html_content += """
                <h2>Trades with Timestamps (for Return Verification)</h2>
                <div class="trades-table">
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Action</th>
                            <th>Entry Date</th>
                            <th>Entry Price</th>
                            <th>Quantity</th>
                            <th>Exit Date</th>
                            <th>Exit Price</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Holding Period (Days)</th>
                            <th>Strategy</th>
                        </tr>
                """
                
                for trade in trades:
                    pnl_class = "positive" if trade.get("pnl", 0) > 0 else "negative" if trade.get("pnl", 0) < 0 else ""
                    html_content += f"""
                        <tr>
                            <td>{trade.get("symbol", "")}</td>
                            <td>{trade.get("action", "")}</td>
                            <td>{trade.get("entry_timestamp", "")}</td>
                            <td>{trade.get("entry_price", ""):.4f}</td>
                            <td>{trade.get("quantity", ""):.2f}</td>
                            <td>{trade.get("exit_timestamp", "N/A")}</td>
                            <td>{"{:.4f}".format(float(trade.get("exit_price"))) if trade.get("exit_price") is not None else "N/A"}</td>
                            <td class="{pnl_class}">{"{:.2f}".format(float(trade.get("pnl"))) if trade.get("pnl") is not None else "N/A"}</td>
                            <td class="{pnl_class}">{"{:.2f}%".format(float(trade.get("pnl_percentage"))) if trade.get("pnl_percentage") is not None else "N/A"}</td>
                            <td>{trade.get("holding_period_days", "N/A")}</td>
                            <td>{trade.get("strategy", "")}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        html_file = self.output_dir / f"{report_name}.html"
        with open(html_file, "w") as f:
            f.write(html_content)
            
        return str(html_file)
        
    def generate_json_report(self, 
                             backtest_results: Dict,
                             report_name: str) -> str:
        """
        Generate a JSON performance report.
        
        Args:
            backtest_results: Results from the backtest
            report_name: Name for the report
            
        Returns:
            Path to the generated JSON report
        """
        report_dir = self.output_dir / report_name
        json_path = report_dir / f"{report_name}.json"
        
        # Convert Decimal objects to float for JSON serialization
        serializable_results = self._make_json_serializable(backtest_results)
        
        # Write JSON to file
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        logger.info(f"Generated JSON report: {json_path}")
        
        return str(json_path)
        
    def generate_csv_reports(self, 
                         backtest_results: Dict,
                         report_name: str) -> Dict[str, str]:
        """
        Generate CSV reports for the backtest results.
        
        Args:
            backtest_results: Backtest results dictionary
            report_name: Base name for the report files
            
        Returns:
            Dictionary mapping report type to file path
        """
        csv_files = {}
        
        # Create directory for CSV reports
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        
        # Generate equity curve CSV
        if "equity_curve" in backtest_results:
            equity_df = pd.DataFrame(backtest_results["equity_curve"])
            equity_file = csv_dir / f"{report_name}_equity_curve.csv"
            equity_df.to_csv(equity_file)
            csv_files["equity_curve"] = str(equity_file)
            
        # Generate trades CSV
        if "trades" in backtest_results:
            trades_df = pd.DataFrame(backtest_results["trades"])
            trades_file = csv_dir / f"{report_name}_trades.csv"
            trades_df.to_csv(trades_file)
            csv_files["trades"] = str(trades_file)
            
        # Generate trades with timestamps CSV for verification
        if "portfolio_stats" in backtest_results and "trades_with_timestamps" in backtest_results["portfolio_stats"]:
            trades_with_timestamps = backtest_results["portfolio_stats"]["trades_with_timestamps"]
            if trades_with_timestamps:
                trades_timestamps_df = pd.DataFrame(trades_with_timestamps)
                trades_timestamps_file = csv_dir / f"{report_name}_trades_with_timestamps.csv"
                trades_timestamps_df.to_csv(trades_timestamps_file)
                csv_files["trades_with_timestamps"] = str(trades_timestamps_file)
                
        # Generate monthly returns CSV
        if "monthly_returns" in backtest_results:
            monthly_df = pd.DataFrame(backtest_results["monthly_returns"])
            monthly_file = csv_dir / f"{report_name}_monthly_returns.csv"
            monthly_df.to_csv(monthly_file)
            csv_files["monthly_returns"] = str(monthly_file)
            
        return csv_files
        
    def generate_charts(self, 
                    backtest_results: Dict,
                    chart_dir: Path) -> None:
        """
        Generate charts for the backtest results.
        
        Args:
            backtest_results: Backtest results dictionary
            chart_dir: Directory to save charts
        """
        # Create chart directory
        chart_dir.mkdir(exist_ok=True)
        
        # Generate equity curve chart
        if "equity_curve" in backtest_results:
            equity_df = pd.DataFrame(backtest_results["equity_curve"])
            equity_df = self._prepare_data_for_charts(equity_df)
            
            # Get trades with timestamps for plotting buy/sell points
            trades_with_timestamps = None
            if "portfolio_stats" in backtest_results and "trades_with_timestamps" in backtest_results["portfolio_stats"]:
                trades_with_timestamps = backtest_results["portfolio_stats"]["trades_with_timestamps"]
            
            self._plot_equity_curve(
                equity_df, 
                chart_dir / "equity_curve.png",
                trades_with_timestamps
            )
            
            # Generate drawdown chart
            self._plot_drawdown(equity_df, chart_dir / "drawdown.png")
            
            # Generate monthly returns chart
            self._plot_monthly_returns(equity_df, chart_dir / "monthly_returns.png")
        
        # Generate trade distribution chart
        if "trades" in backtest_results:
            self._plot_trade_distribution(backtest_results["trades"], chart_dir / "trade_distribution.png")
        
        logger.info(f"Generated charts in {chart_dir}")
        
    def _plot_equity_curve(self, 
                       equity_df: pd.DataFrame,
                       output_path: Path,
                       trades_with_timestamps: Optional[List[Dict]] = None) -> None:
        """
        Plot the equity curve with buy/sell points.
        
        Args:
            equity_df: DataFrame with equity curve data
            output_path: Path to save the plot
            trades_with_timestamps: List of trades with timestamps for marking buy/sell points
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert dates to datetime if they're not already
        if not pd.api.types.is_datetime64_any_dtype(equity_df.index):
            equity_df.index = pd.to_datetime(equity_df.index)
            
        # Plot equity curve
        ax.plot(equity_df.index, equity_df['equity'], label='Equity', linewidth=2, color='#1f77b4')
        
        # Add buy/sell markers if trades data is available
        if trades_with_timestamps:
            # Convert trades to DataFrame for easier processing
            trades_df = pd.DataFrame(trades_with_timestamps)
            
            # Process entry points (buys)
            if 'entry_timestamp' in trades_df.columns:
                trades_df['entry_timestamp'] = pd.to_datetime(trades_df['entry_timestamp'])
                
                # Filter to only include timestamps within the equity curve timeframe
                valid_entries = trades_df[trades_df['entry_timestamp'].between(
                    equity_df.index.min(), equity_df.index.max())]
                
                # Plot buy points
                buy_points = valid_entries[valid_entries['action'] == 'BUY']
                if not buy_points.empty:
                    # For each buy point, find the corresponding equity value
                    for _, trade in buy_points.iterrows():
                        # Find closest date in equity curve
                        closest_date = equity_df.index[equity_df.index.get_indexer([trade['entry_timestamp']], method='nearest')[0]]
                        equity_value = equity_df.loc[closest_date, 'equity']
                        ax.scatter(trade['entry_timestamp'], equity_value, color='green', marker='^', s=100, label='_Buy')
                
                # Plot sell points (short entries)
                sell_points = valid_entries[valid_entries['action'] == 'SELL']
                if not sell_points.empty:
                    # For each sell point, find the corresponding equity value
                    for _, trade in sell_points.iterrows():
                        # Find closest date in equity curve
                        closest_date = equity_df.index[equity_df.index.get_indexer([trade['entry_timestamp']], method='nearest')[0]]
                        equity_value = equity_df.loc[closest_date, 'equity']
                        ax.scatter(trade['entry_timestamp'], equity_value, color='red', marker='v', s=100, label='_Sell')
            
            # Process exit points
            if 'exit_timestamp' in trades_df.columns:
                # Filter out NaN exit timestamps (open positions)
                closed_trades = trades_df.dropna(subset=['exit_timestamp'])
                if not closed_trades.empty:
                    closed_trades['exit_timestamp'] = pd.to_datetime(closed_trades['exit_timestamp'])
                    
                    # Filter to only include timestamps within the equity curve timeframe
                    valid_exits = closed_trades[closed_trades['exit_timestamp'].between(
                        equity_df.index.min(), equity_df.index.max())]
                    
                    # Plot exit points
                    for _, trade in valid_exits.iterrows():
                        # Find closest date in equity curve
                        closest_date = equity_df.index[equity_df.index.get_indexer([trade['exit_timestamp']], method='nearest')[0]]
                        equity_value = equity_df.loc[closest_date, 'equity']
                        
                        # Use different markers for buy exits vs sell exits
                        if trade['action'] == 'BUY':
                            ax.scatter(trade['exit_timestamp'], equity_value, color='red', marker='x', s=100, label='_Exit Long')
                        else:
                            ax.scatter(trade['exit_timestamp'], equity_value, color='green', marker='x', s=100, label='_Exit Short')
        
        # Add benchmark if available
        if 'benchmark' in equity_df.columns:
            ax.plot(equity_df.index, equity_df['benchmark'], label='Benchmark', linewidth=2, color='#ff7f0e', alpha=0.7)
            
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.set_title('Equity Curve with Trade Points')
        
        # Add legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close(fig)
        
    def _plot_drawdown(self, 
                       equity_df: pd.DataFrame,
                       output_path: Path) -> None:
        """
        Plot drawdown chart.
        
        Args:
            equity_df: DataFrame with equity history
            output_path: Path to save the chart
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting (convert Decimal to float)
        equity_df = self._prepare_data_for_charts(equity_df)
        
        # Calculate drawdown
        running_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] / running_max - 1.0) * 100
        
        # Plot drawdown
        plt.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='#d62728', linewidth=1)
        
        # Format plot
        plt.title('Drawdown', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
    def _plot_monthly_returns(self, 
                              equity_df: pd.DataFrame,
                              output_path: Path) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_df: DataFrame with equity history
            output_path: Path to save the chart
        """
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting (convert Decimal to float)
        equity_df = self._prepare_data_for_charts(equity_df)
        
        # Calculate returns if not already present
        if 'returns' not in equity_df.columns:
            equity_df['returns'] = equity_df['equity'].pct_change()
            
        # Resample to monthly returns
        monthly_returns = equity_df['returns'].resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns_df = pd.DataFrame(monthly_returns)
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot = monthly_returns_df.pivot_table(
            index='year', 
            columns='month', 
            values='returns'
        )
        
        # Replace month numbers with month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot.columns = [month_names[m] for m in pivot.columns]
        
        # Create a custom colormap (green for positive, red for negative)
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        # Plot heatmap
        sns.heatmap(
            pivot * 100,  # Convert to percentage
            annot=True,
            fmt=".1f",
            cmap=cmap,
            center=0,
            linewidths=1,
            cbar_kws={'label': 'Monthly Return (%)'},
            ax=plt.gca()
        )
        
        # Format plot
        plt.title('Monthly Returns', fontsize=16)
        plt.ylabel('Year', fontsize=12)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
    def _plot_trade_distribution(self, 
                                 trades: List[Dict],
                                 output_path: Path) -> None:
        """
        Plot trade P&L distribution.
        
        Args:
            trades: List of trade dictionaries
            output_path: Path to save the chart
        """
        plt.figure(figsize=(12, 6))
        
        # Extract closed trades with P&L
        closed_trades = [t for t in trades if t.get('type') == 'CLOSE' and 'pnl' in t]
        
        if not closed_trades:
            logger.warning("No closed trades with P&L data available for distribution chart")
            plt.text(0.5, 0.5, "No trade data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
            
        # Extract P&L values and convert to float
        pnl_values = []
        for t in closed_trades:
            try:
                pnl_values.append(float(t['pnl']))
            except (ValueError, TypeError):
                # Skip trades with invalid P&L values
                pass
        
        # Plot histogram
        n, bins, patches = plt.hist(pnl_values, bins=30, alpha=0.7, color='#1f77b4')
        
        # Color negative and positive bars differently
        for i in range(len(patches)):
            if bins[i] < 0:
                patches[i].set_facecolor('#d62728')  # Red for negative
            else:
                patches[i].set_facecolor('#2ca02c')  # Green for positive
                
        # Add a vertical line at zero
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Format plot
        plt.title('Trade P&L Distribution', fontsize=16)
        plt.xlabel('P&L ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis with dollar signs
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'DataProvider':
            return f"DataProvider(base_path={obj.base_path})"
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            # Handle any other non-serializable objects by converting to string
            try:
                return str(obj)
            except:
                return f"<non-serializable: {type(obj).__name__}>"
        else:
            return obj
            
    def _get_color_class(self, value: float) -> str:
        """
        Get CSS color class based on value.
        
        Args:
            value: Value to check
            
        Returns:
            CSS class name
        """
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return ""

    def _prepare_data_for_charts(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for charts by converting Decimal values to float.
        
        Args:
            equity_df: DataFrame with equity history
            
        Returns:
            DataFrame with numeric values as float
        """
        # Make a copy to avoid modifying the original
        df = equity_df.copy()
        
        # Convert all columns to numeric if possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Skip columns that can't be converted
                pass
                
        return df 

    def generate_symbol_report(self, 
                            backtest_results: Dict,
                            symbol: str,
                            output_dir: Path) -> Dict[str, str]:
        """
        Generate a detailed report for a specific symbol.
        Includes all trades, performance metrics, and visualizations.
        
        Args:
            backtest_results: Full backtest results
            symbol: The symbol to generate the report for
            output_dir: Directory to save the report
            
        Returns:
            Dictionary with paths to generated files
        """
        # Create output directory if it doesn't exist
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter trades for this symbol
        all_trades = []
        
        # First check the trades array
        if "trades" in backtest_results:
            all_trades = backtest_results.get("trades", [])
        # Then check portfolio_stats if available
        elif "portfolio_stats" in backtest_results and "trades_with_timestamps" in backtest_results["portfolio_stats"]:
            all_trades = backtest_results["portfolio_stats"]["trades_with_timestamps"]
        
        symbol_trades = [trade for trade in all_trades if trade.get("symbol") == symbol]
        
        # Calculate symbol-specific metrics
        metrics = self._calculate_symbol_metrics(symbol_trades, backtest_results.get("initial_capital", 100000))
        
        # Create symbol performance dataframe
        if not symbol_trades:
            logger.warning(f"No trades found for symbol {symbol}")
            return {}
            
        # Generate trade timeline and standardize field names
        trades_df = pd.DataFrame(symbol_trades)
        
        # Standardize column names - map common field names to our standard field names
        field_mappings = {
            # Entry timestamp field mappings
            "entry_timestamp": ["entry_timestamp", "entry_date", "timestamp"],
            # Exit timestamp field mappings
            "exit_timestamp": ["exit_timestamp", "exit_date"],
            # Direction field mappings
            "direction": ["direction", "action", "side", "trade_type"],
            # Price field mappings
            "entry_price": ["entry_price", "price", "open_price"],
            "exit_price": ["exit_price", "close_price"],
            # PnL field mappings
            "realized_pnl": ["realized_pnl", "pnl", "profit_loss"]
        }
        
        # Standardize column names in the dataframe
        for target_field, source_fields in field_mappings.items():
            if target_field not in trades_df.columns:
                # Try to find one of the source fields
                for source_field in source_fields:
                    if source_field in trades_df.columns:
                        trades_df[target_field] = trades_df[source_field]
                        break
        
        # Ensure datetime columns are correctly formatted
        if "entry_timestamp" in trades_df.columns:
            trades_df["entry_timestamp"] = pd.to_datetime(trades_df["entry_timestamp"])
        if "exit_timestamp" in trades_df.columns:
            trades_df["exit_timestamp"] = pd.to_datetime(trades_df["exit_timestamp"])
        
        # Sort by entry timestamp if available
        if "entry_timestamp" in trades_df.columns:
            trades_df = trades_df.sort_values("entry_timestamp")
        
        # Generate symbol equity curve
        equity_curve = self._reconstruct_symbol_equity_curve(symbol_trades, backtest_results.get("initial_capital", 100000))
        
        # Save results to JSON
        json_path = symbol_dir / f"{symbol}_performance.json"
        with open(json_path, 'w') as f:
            json.dump({
                "symbol": symbol,
                "metrics": metrics,
                "trades": symbol_trades
            }, f, default=self._make_json_serializable, indent=2)
        
        # Generate visualizations
        if len(equity_curve) > 1:
            self._plot_symbol_equity_curve(equity_curve, trades_df, symbol_dir / f"{symbol}_equity_curve.png", symbol)
            self._plot_symbol_drawdown(equity_curve, symbol_dir / f"{symbol}_drawdown.png", symbol)
            self._plot_symbol_trade_distribution(trades_df, symbol_dir / f"{symbol}_trade_distribution.png", symbol)
            self._plot_symbol_win_loss_ratio(trades_df, symbol_dir / f"{symbol}_win_loss.png", symbol)
            
        # Generate HTML report
        html_path = symbol_dir / f"{symbol}_report.html"
        self._generate_symbol_html_report(symbol, metrics, trades_df, html_path)
        
        return {
            "json": str(json_path),
            "html": str(html_path)
        }
        
    def _calculate_symbol_metrics(self, trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate performance metrics for a specific symbol."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_profit": 0,
                "average_loss": 0,
                "max_profit": 0,
                "max_loss": 0,
                "total_profit": 0,
                "total_loss": 0,
                "net_profit": 0
            }
        
        try:
            # 安全地获取PnL值的函数
            def safe_get_pnl(trade):
                try:
                    # 首先尝试获取pnl字段
                    if "pnl" in trade and trade["pnl"] is not None:
                        return float(trade["pnl"])
                    # 然后尝试realized_pnl字段
                    elif "realized_pnl" in trade and trade["realized_pnl"] is not None:
                        return float(trade["realized_pnl"])
                    # 如果都不存在，返回0
                    return 0
                except (ValueError, TypeError):
                    # 如果转换失败，返回0
                    return 0
            
            # 安全地计算指标
            total_trades = len(trades)
            
            # 分类交易
            wins = []
            losses = []
            
            for trade in trades:
                pnl = safe_get_pnl(trade)
                if pnl > 0:
                    wins.append(trade)
                else:
                    losses.append(trade)
            
            win_count = len(wins)
            loss_count = len(losses)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # 计算总利润和总亏损
            total_profit = sum(safe_get_pnl(t) for t in wins)
            total_loss = abs(sum(safe_get_pnl(t) for t in losses))
            
            # 计算平均利润和亏损
            avg_profit = total_profit / win_count if win_count > 0 else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            
            # 计算最大利润和亏损
            max_profit = max([safe_get_pnl(t) for t in wins]) if wins else 0
            max_loss = abs(min([safe_get_pnl(t) for t in losses]) if losses else 0)
            
            # 计算利润因子
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate * 100,
                "profit_factor": profit_factor,
                "average_profit": avg_profit,
                "average_loss": avg_loss,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_profit": total_profit - total_loss
            }
        except Exception as e:
            logger.error(f"Error calculating symbol metrics: {str(e)}")
            # 返回默认值
            return {
                "total_trades": len(trades),
                "win_rate": 0,
                "profit_factor": 0,
                "average_profit": 0,
                "average_loss": 0,
                "max_profit": 0,
                "max_loss": 0,
                "total_profit": 0,
                "total_loss": 0,
                "net_profit": 0
            }
        
    def _reconstruct_symbol_equity_curve(self, trades: List[Dict], initial_equity: float) -> pd.DataFrame:
        """Reconstruct an equity curve for a specific symbol."""
        if not trades:
            return pd.DataFrame(columns=["timestamp", "equity"])
            
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Find the right timestamp column - try exit_timestamp first, then entry_timestamp
            timestamp_col = None
            for col in ["exit_timestamp", "entry_timestamp", "timestamp"]:
                if col in trades_df.columns:
                    timestamp_col = col
                    break
                    
            if not timestamp_col:
                logger.warning("No valid timestamp column found in trades data")
                return pd.DataFrame(columns=["timestamp", "equity"])
                
            # Filter out rows with invalid timestamps
            valid_trades = trades_df.dropna(subset=[timestamp_col])
            
            if valid_trades.empty:
                logger.warning("No valid timestamp data found in trades")
                return pd.DataFrame(columns=["timestamp", "equity"])
                
            # Convert timestamps to datetime safely
            try:
                valid_trades[timestamp_col] = pd.to_datetime(valid_trades[timestamp_col], errors='coerce')
                # Drop any rows that couldn't be converted
                valid_trades = valid_trades.dropna(subset=[timestamp_col])
            except Exception as e:
                logger.warning(f"Error converting timestamps: {str(e)}")
                return pd.DataFrame(columns=["timestamp", "equity"])
            
            if valid_trades.empty:
                logger.warning("No valid datetime data found after conversion")
                return pd.DataFrame(columns=["timestamp", "equity"])
                
            # Sort by timestamp
            valid_trades = valid_trades.sort_values(timestamp_col)
            
            # Create equity curve
            timestamps = []
            equity_values = []
            
            # Start with initial equity
            current_equity = initial_equity
            equity_values.append(current_equity)
            
            # Get starting timestamp (minimum timestamp in the data)
            try:
                start_timestamp = valid_trades[timestamp_col].min()
                timestamps.append(start_timestamp)
            except Exception as e:
                logger.warning(f"Error getting min timestamp: {str(e)}")
                # Use current time as fallback
                timestamps.append(pd.Timestamp.now())
            
            # Find the PnL column
            pnl_col = None
            for col in ["realized_pnl", "pnl", "profit_loss"]:
                if col in valid_trades.columns:
                    pnl_col = col
                    break
                    
            if not pnl_col:
                logger.warning("No valid P&L column found in trades data")
                # Create basic equity curve with just the initial value
                return pd.DataFrame({
                    "timestamp": timestamps,
                    "equity": equity_values
                })
            
            # Add equity after each trade
            for _, trade in valid_trades.iterrows():
                try:
                    pnl = float(trade.get(pnl_col, 0))
                    current_equity += pnl
                    equity_values.append(current_equity)
                    timestamps.append(trade[timestamp_col])
                except (ValueError, TypeError) as e:
                    # Skip trades with invalid P&L
                    logger.warning(f"Error processing trade P&L: {str(e)}")
                    continue
                    
            # Create DataFrame
            equity_df = pd.DataFrame({
                "timestamp": timestamps,
                "equity": equity_values
            })
            
            return equity_df
            
        except Exception as e:
            logger.error(f"Error reconstructing equity curve: {str(e)}")
            # Return empty DataFrame
            return pd.DataFrame(columns=["timestamp", "equity"])
        
    def _plot_symbol_equity_curve(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, 
                              output_path: Path, symbol: str) -> None:
        """Plot equity curve for a specific symbol with trade markers."""
        # Check if the equity dataframe has valid data
        if equity_df.empty or "equity" not in equity_df.columns or "timestamp" not in equity_df.columns:
            # Create an empty chart with a message
            plt.figure(figsize=(12, 6))
            plt.title(f"Equity Curve for {symbol}\n(No equity data available)")
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(equity_df["timestamp"], equity_df["equity"], label="Equity", linewidth=2)
        
        # Add markers for trades if we have the necessary data
        pnl_column = None
        exit_timestamp_column = None
        
        # Check for required columns
        for col in ["realized_pnl", "pnl"]:
            if col in trades_df.columns:
                pnl_column = col
                break
                
        if "exit_timestamp" in trades_df.columns:
            exit_timestamp_column = "exit_timestamp"
            
        if pnl_column and exit_timestamp_column and not trades_df.empty:
            try:
                for _, trade in trades_df.iterrows():
                    if pd.notna(trade[exit_timestamp_column]) and pd.notna(trade[pnl_column]):
                        if trade[pnl_column] > 0:
                            plt.scatter(trade[exit_timestamp_column], 
                                      trade.get("exit_price", 0) * trade.get("quantity", 0), 
                                      color="green", s=50, alpha=0.7, marker="^")
                        else:
                            plt.scatter(trade[exit_timestamp_column], 
                                      trade.get("exit_price", 0) * trade.get("quantity", 0), 
                                      color="red", s=50, alpha=0.7, marker="v")
            except Exception as e:
                logger.warning(f"Error plotting trade markers for {symbol}: {str(e)}")
        
        plt.title(f"Equity Curve for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def _plot_symbol_drawdown(self, equity_df: pd.DataFrame, output_path: Path, symbol: str) -> None:
        """Plot drawdown chart for a specific symbol."""
        # Check if the equity dataframe has valid data
        if equity_df.empty or "equity" not in equity_df.columns or "timestamp" not in equity_df.columns:
            # Create an empty chart with a message
            plt.figure(figsize=(12, 6))
            plt.title(f"Drawdown Chart for {symbol}\n(No equity data available)")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return
            
        try:
            # Calculate running maximum
            equity_df["peak"] = equity_df["equity"].cummax()
            
            # Calculate drawdown
            equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df["timestamp"], equity_df["drawdown"], color="red", linewidth=2)
            plt.fill_between(equity_df["timestamp"], equity_df["drawdown"], 0, color="red", alpha=0.3)
            
            plt.title(f"Drawdown Chart for {symbol}")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.grid(True)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            # If there's an error, create a basic chart
            logger.warning(f"Error creating drawdown chart for {symbol}: {str(e)}")
            plt.figure(figsize=(12, 6))
            plt.title(f"Drawdown Chart for {symbol}\n(Error creating chart)")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
    def _plot_symbol_trade_distribution(self, trades_df: pd.DataFrame, output_path: Path, symbol: str) -> None:
        """Plot trade profit/loss distribution for a specific symbol."""
        # Check for PnL columns in the DataFrame
        pnl_column = None
        for col in ["realized_pnl", "pnl"]:
            if col in trades_df.columns:
                pnl_column = col
                break
                
        if not pnl_column:
            # Create an empty chart with a message
            plt.figure(figsize=(12, 6))
            plt.title(f"Trade Profit/Loss Distribution for {symbol}\n(No profit/loss data available)")
            plt.xlabel("Profit/Loss ($)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot histogram of returns
        trades_df[pnl_column].hist(bins=50, alpha=0.7)
        
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        plt.title(f"Trade Profit/Loss Distribution for {symbol}")
        plt.xlabel("Profit/Loss ($)")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def _plot_symbol_win_loss_ratio(self, trades_df: pd.DataFrame, output_path: Path, symbol: str) -> None:
        """Plot win/loss ratio pie chart for a specific symbol."""
        # Check for PnL columns in the DataFrame
        pnl_column = None
        for col in ["realized_pnl", "pnl"]:
            if col in trades_df.columns:
                pnl_column = col
                break
                
        if not pnl_column:
            # Create an empty chart with a message
            plt.figure(figsize=(8, 8))
            plt.title(f"Win/Loss Ratio for {symbol}\n(No profit/loss data available)")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return
            
        # Count wins and losses
        wins = (trades_df[pnl_column] > 0).sum()
        losses = (trades_df[pnl_column] <= 0).sum()
        
        if wins + losses == 0:
            # Create an empty chart with a message
            plt.figure(figsize=(8, 8))
            plt.title(f"Win/Loss Ratio for {symbol}\n(No trades with profit/loss data)")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return
            
        plt.figure(figsize=(8, 8))
        
        # Create pie chart
        plt.pie([wins, losses], labels=["Wins", "Losses"], autopct="%1.1f%%", 
               colors=["green", "red"], startangle=90, explode=(0.1, 0))
        
        plt.title(f"Win/Loss Ratio for {symbol}")
        plt.axis("equal")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def _generate_symbol_html_report(self, symbol: str, metrics: Dict, 
                                 trades_df: pd.DataFrame, output_path: Path) -> None:
        """Generate HTML report for a specific symbol."""
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Symbol Report: {symbol}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart-container {{ margin: 20px 0; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; }}
                .metric-box {{ flex: 1; min-width: 150px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .trades-table {{ max-height: 500px; overflow-y: auto; }}
                .chart {{ width: 100%; max-width: 800px; margin: 20px auto; }}
            </style>
        </head>
        <body>
            <h1>Trading Report for {symbol}</h1>
            
            <h2>Performance Metrics</h2>
            <div class="metrics-container">
        """
        
        # Add metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, Decimal)):
                if key in ["win_rate", "profit_factor", "net_profit"]:
                    color_class = "positive" if value > 0 else "negative"
                else:
                    color_class = ""
                
                # Format based on metric type
                if key in ["win_rate"]:
                    formatted_value = f"{value:.2f}%"
                elif key in ["profit_factor"]:
                    formatted_value = f"{value:.2f}"
                elif "profit" in key or "loss" in key:
                    formatted_value = f"${value:.2f}"
                else:
                    formatted_value = f"{value}"
                
                html_content += f"""
                <div class="metric-box">
                    <div>{key.replace('_', ' ').title()}</div>
                    <div class="metric-value {color_class}">{formatted_value}</div>
                </div>
                """
        
        html_content += f"""
            </div>
            
            <h2>Visualizations</h2>
            <div class="chart-container">
                <img class="chart" src="{symbol}_equity_curve.png" alt="Equity Curve">
                <img class="chart" src="{symbol}_drawdown.png" alt="Drawdown Chart">
                <img class="chart" src="{symbol}_trade_distribution.png" alt="Trade Distribution">
                <img class="chart" src="{symbol}_win_loss.png" alt="Win/Loss Ratio">
            </div>
            
            <h2>Trades</h2>
            <div class="trades-table">
                <table>
                    <tr>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Direction</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>P&L</th>
                        <th>Return (%)</th>
                    </tr>
        """
        
        # Add trade rows
        if not trades_df.empty:
            # 重新组织交易数据，将开仓和平仓交易配对
            open_trades = {}
            completed_trades = []
            position_ids = {}
            
            # 首先按照时间顺序对交易进行排序
            trades_df = trades_df.sort_values("timestamp") if "timestamp" in trades_df.columns else trades_df
            
            # 识别开仓和平仓交易
            for _, trade in trades_df.iterrows():
                # 获取交易类型
                trade_type = trade.get("type", "").upper()
                position_id = trade.get("position_id", "")
                
                # 如果有position_id，使用它来配对交易
                if position_id:
                    if trade_type == "OPEN":
                        # 存储开仓交易
                        open_trades[position_id] = trade
                    elif trade_type == "CLOSE" and position_id in open_trades:
                        # 找到匹配的开仓交易，创建完整交易记录
                        open_trade = open_trades[position_id]
                        completed_trades.append({
                            "entry_timestamp": open_trade.get("timestamp", ""),
                            "exit_timestamp": trade.get("timestamp", ""),
                            "direction": open_trade.get("action", ""),
                            "entry_price": open_trade.get("price", 0),
                            "exit_price": trade.get("price", 0),
                            "quantity": open_trade.get("quantity", 0),
                            "pnl": trade.get("pnl", 0)
                        })
                        # 从未完成交易中移除
                        del open_trades[position_id]
                else:
                    # 如果没有position_id，使用其他字段来推断开仓/平仓
                    # 可以使用action或其他字段来判断
                    action = trade.get("action", "")
                    symbol = trade.get("symbol", "")
                    timestamp = trade.get("timestamp", "")
                    
                    # 使用symbol+timestamp作为键
                    key = f"{symbol}_{timestamp}"
                    
                    if action in ["BUY", "SELL"]:
                        if key not in position_ids:
                            # 保存这个交易作为开仓交易
                            position_ids[key] = len(position_ids)
                            open_trades[position_ids[key]] = trade
                        else:
                            # 这是一个平仓交易，寻找匹配的开仓交易
                            open_position_id = position_ids[key]
                            if open_position_id in open_trades:
                                open_trade = open_trades[open_position_id]
                                
                                # 创建完整的交易记录
                                pnl = trade.get("pnl", 0)
                                if pd.isna(pnl):
                                    pnl = 0
                                    
                                completed_trades.append({
                                    "entry_timestamp": open_trade.get("timestamp", ""),
                                    "exit_timestamp": trade.get("timestamp", ""),
                                    "direction": open_trade.get("action", ""),
                                    "entry_price": open_trade.get("price", 0),
                                    "exit_price": trade.get("price", 0),
                                    "quantity": open_trade.get("quantity", 0),
                                    "pnl": pnl
                                })
                                
                                # 从未完成交易中移除
                                del open_trades[open_position_id]
            
            # 处理仍然开放的交易（尚未平仓）
            for position_id, open_trade in open_trades.items():
                completed_trades.append({
                    "entry_timestamp": open_trade.get("timestamp", ""),
                    "exit_timestamp": "",  # 尚未平仓
                    "direction": open_trade.get("action", ""),
                    "entry_price": open_trade.get("price", 0),
                    "exit_price": 0,  # 尚未平仓
                    "quantity": open_trade.get("quantity", 0),
                    "pnl": 0  # 尚未实现的盈亏
                })
            
            # 现在使用配对后的交易数据
            for trade in completed_trades:
                try:
                    # 安全地处理日期
                    entry_timestamp = trade.get("entry_timestamp", None)
                    if pd.notna(entry_timestamp) and entry_timestamp != "":
                        try:
                            entry_date = pd.to_datetime(entry_timestamp).strftime("%Y-%m-%d")
                        except:
                            entry_date = "N/A"
                    else:
                        entry_date = "N/A"
                        
                    exit_timestamp = trade.get("exit_timestamp", None)
                    if pd.notna(exit_timestamp) and exit_timestamp != "":
                        try:
                            exit_date = pd.to_datetime(exit_timestamp).strftime("%Y-%m-%d")
                        except:
                            exit_date = "N/A"
                    else:
                        exit_date = "N/A"
                except Exception as e:
                    logger.warning(f"Error formatting dates for {symbol}: {str(e)}")
                    entry_date = "N/A"
                    exit_date = "N/A"
                
                # 获取方向
                direction = trade.get("direction", "N/A")
                
                # 获取价格
                try:
                    entry_price = float(trade.get("entry_price", 0))
                except (ValueError, TypeError):
                    entry_price = 0
                
                try:
                    exit_price = float(trade.get("exit_price", 0))
                except (ValueError, TypeError):
                    exit_price = 0
                
                try:
                    quantity = float(trade.get("quantity", 0))
                except (ValueError, TypeError):
                    quantity = 0
                
                # 获取盈亏
                try:
                    pnl = float(trade.get("pnl", 0))
                except (ValueError, TypeError):
                    pnl = 0
                
                # 计算收益率
                entry_value = entry_price * quantity if entry_price and quantity else 1  # 避免除以零
                return_pct = (pnl / entry_value) * 100 if entry_value != 0 else 0
                
                # 根据盈亏确定颜色
                row_class = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
                
                html_content += f"""
                    <tr>
                        <td>{entry_date}</td>
                        <td>{exit_date}</td>
                        <td>{direction}</td>
                        <td>${entry_price:.2f}</td>
                        <td>${exit_price:.2f}</td>
                        <td>{quantity}</td>
                        <td class="{row_class}">${pnl:.2f}</td>
                        <td class="{row_class}">{return_pct:.2f}%</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path, "w") as f:
            f.write(html_content) 