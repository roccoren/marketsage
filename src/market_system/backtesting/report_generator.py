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