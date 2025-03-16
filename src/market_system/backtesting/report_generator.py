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
        Generate an HTML performance report.
        
        Args:
            backtest_results: Results from the backtest
            report_name: Name for the report
            
        Returns:
            Path to the generated HTML report
        """
        report_dir = self.output_dir / report_name
        html_path = report_dir / f"{report_name}.html"
        
        # Extract data from backtest results
        strategy_name = backtest_results.get("strategy_name", "Unknown Strategy")
        start_date = backtest_results.get("start_date", "Unknown")
        end_date = backtest_results.get("end_date", "Unknown")
        initial_capital = backtest_results.get("initial_capital", 0)
        final_equity = backtest_results.get("final_equity", 0)
        
        # Get performance metrics
        metrics = backtest_results.get("metrics", {})
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report: {strategy_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                }}
                .metric-group {{
                    flex: 1;
                    min-width: 300px;
                    margin: 10px;
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .chart-container {{
                    margin: 20px 0;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Backtest Report: {strategy_name}</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <table>
                        <tr>
                            <td><strong>Strategy:</strong></td>
                            <td>{strategy_name}</td>
                        </tr>
                        <tr>
                            <td><strong>Period:</strong></td>
                            <td>{start_date} to {end_date}</td>
                        </tr>
                        <tr>
                            <td><strong>Initial Capital:</strong></td>
                            <td>${initial_capital:,.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>Final Equity:</strong></td>
                            <td>${final_equity:,.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>Total Return:</strong></td>
                            <td class="{self._get_color_class(metrics.get('total_return', 0))}">{metrics.get('total_return', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td><strong>Annualized Return:</strong></td>
                            <td class="{self._get_color_class(metrics.get('annualized_return', 0))}">{metrics.get('annualized_return', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td><strong>Max Drawdown:</strong></td>
                            <td class="negative">{metrics.get('max_drawdown', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td><strong>Sharpe Ratio:</strong></td>
                            <td class="{self._get_color_class(metrics.get('sharpe_ratio', 0))}">{metrics.get('sharpe_ratio', 0):.2f}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="metrics-container">
                    <div class="metric-group">
                        <h2>Return Metrics</h2>
                        <table>
                            <tr>
                                <td><strong>Total Return:</strong></td>
                                <td class="{self._get_color_class(metrics.get('total_return', 0))}">{metrics.get('total_return', 0):.2%}</td>
                            </tr>
                            <tr>
                                <td><strong>Annualized Return:</strong></td>
                                <td class="{self._get_color_class(metrics.get('annualized_return', 0))}">{metrics.get('annualized_return', 0):.2%}</td>
                            </tr>
                            <tr>
                                <td><strong>Volatility:</strong></td>
                                <td>{metrics.get('volatility', 0):.2%}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="metric-group">
                        <h2>Risk Metrics</h2>
                        <table>
                            <tr>
                                <td><strong>Max Drawdown:</strong></td>
                                <td class="negative">{metrics.get('max_drawdown', 0):.2%}</td>
                            </tr>
                            <tr>
                                <td><strong>Sharpe Ratio:</strong></td>
                                <td class="{self._get_color_class(metrics.get('sharpe_ratio', 0))}">{metrics.get('sharpe_ratio', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>Sortino Ratio:</strong></td>
                                <td class="{self._get_color_class(metrics.get('sortino_ratio', 0))}">{metrics.get('sortino_ratio', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>Calmar Ratio:</strong></td>
                                <td class="{self._get_color_class(metrics.get('calmar_ratio', 0))}">{metrics.get('calmar_ratio', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>VaR (95%):</strong></td>
                                <td>{metrics.get('var_95', 0):.2%}</td>
                            </tr>
                            <tr>
                                <td><strong>CVaR (95%):</strong></td>
                                <td>{metrics.get('cvar_95', 0):.2%}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="metric-group">
                        <h2>Trade Metrics</h2>
                        <table>
                            <tr>
                                <td><strong>Total Trades:</strong></td>
                                <td>{metrics.get('total_trades', 0)}</td>
                            </tr>
                            <tr>
                                <td><strong>Winning Trades:</strong></td>
                                <td>{metrics.get('winning_trades', 0)}</td>
                            </tr>
                            <tr>
                                <td><strong>Losing Trades:</strong></td>
                                <td>{metrics.get('losing_trades', 0)}</td>
                            </tr>
                            <tr>
                                <td><strong>Win Rate:</strong></td>
                                <td>{metrics.get('win_rate', 0):.2%}</td>
                            </tr>
                            <tr>
                                <td><strong>Profit Factor:</strong></td>
                                <td>{metrics.get('profit_factor', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>Average Win:</strong></td>
                                <td class="positive">${metrics.get('average_win', 0):,.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>Average Loss:</strong></td>
                                <td class="negative">${metrics.get('average_loss', 0):,.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>Average Trade:</strong></td>
                                <td class="{self._get_color_class(metrics.get('average_trade', 0))}">${metrics.get('average_trade', 0):,.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>Expectancy:</strong></td>
                                <td class="{self._get_color_class(metrics.get('expectancy', 0))}">${metrics.get('expectancy', 0):,.2f}</td>
                            </tr>
                            <tr>
                                <td><strong>Avg Holding Period:</strong></td>
                                <td>{metrics.get('average_holding_period', 0):.1f} days</td>
                            </tr>
                        </table>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>Equity Curve</h2>
                    <img src="charts/equity_curve.png" alt="Equity Curve" style="width: 100%;">
                </div>
                
                <div class="chart-container">
                    <h2>Drawdown</h2>
                    <img src="charts/drawdown.png" alt="Drawdown" style="width: 100%;">
                </div>
                
                <div class="chart-container">
                    <h2>Monthly Returns</h2>
                    <img src="charts/monthly_returns.png" alt="Monthly Returns" style="width: 100%;">
                </div>
                
                <div class="chart-container">
                    <h2>Trade Distribution</h2>
                    <img src="charts/trade_distribution.png" alt="Trade Distribution" style="width: 100%;">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Generated HTML report: {html_path}")
        
        return str(html_path)
        
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
        Generate CSV performance reports.
        
        Args:
            backtest_results: Results from the backtest
            report_name: Name for the report
            
        Returns:
            Dict mapping report type to file path
        """
        report_dir = self.output_dir / report_name
        csv_paths = {}
        
        # Extract data from backtest results
        equity_curve = backtest_results.get("equity_curve", [])
        trades = backtest_results.get("trades", [])
        metrics = backtest_results.get("metrics", {})
        
        # Generate equity curve CSV
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            equity_path = report_dir / f"{report_name}_equity.csv"
            equity_df.to_csv(equity_path, index=False)
            csv_paths["equity"] = str(equity_path)
            
        # Generate trades CSV
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = report_dir / f"{report_name}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            csv_paths["trades"] = str(trades_path)
            
        # Generate metrics CSV
        if metrics:
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame([metrics])
            metrics_path = report_dir / f"{report_name}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            csv_paths["metrics"] = str(metrics_path)
            
        logger.info(f"Generated CSV reports: {', '.join(csv_paths.keys())}")
        
        return csv_paths
        
    def generate_charts(self, 
                        backtest_results: Dict,
                        chart_dir: Path) -> None:
        """
        Generate charts for the backtest results.
        
        Args:
            backtest_results: Results from the backtest
            chart_dir: Directory to save charts
        """
        # Extract data from backtest results
        equity_curve = backtest_results.get("equity_curve", [])
        trades = backtest_results.get("trades", [])
        
        if not equity_curve:
            logger.warning("No equity curve data available for charts")
            return
            
        # Convert equity curve to DataFrame
        try:
            equity_df = pd.DataFrame(equity_curve)
            
            # Handle different timestamp column names
            timestamp_col = None
            for col in ['timestamp', 'date', 'time']:
                if col in equity_df.columns:
                    timestamp_col = col
                    break
                    
            if timestamp_col:
                equity_df[timestamp_col] = pd.to_datetime(equity_df[timestamp_col])
                equity_df.set_index(timestamp_col, inplace=True)
                equity_df.sort_index(inplace=True)
            
            # Generate equity curve chart
            try:
                self._plot_equity_curve(equity_df, chart_dir / "equity_curve.png")
            except Exception as e:
                logger.error(f"Error plotting equity curve: {str(e)}")
            
            # Generate drawdown chart
            try:
                self._plot_drawdown(equity_df, chart_dir / "drawdown.png")
            except Exception as e:
                logger.error(f"Error plotting drawdown: {str(e)}")
            
            # Generate monthly returns chart
            try:
                self._plot_monthly_returns(equity_df, chart_dir / "monthly_returns.png")
            except Exception as e:
                logger.error(f"Error plotting monthly returns: {str(e)}")
            
            # Generate trade distribution chart
            if trades:
                try:
                    self._plot_trade_distribution(trades, chart_dir / "trade_distribution.png")
                except Exception as e:
                    logger.error(f"Error plotting trade distribution: {str(e)}")
                
            logger.info(f"Generated charts in {chart_dir}")
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
        
    def _plot_equity_curve(self, 
                           equity_df: pd.DataFrame,
                           output_path: Path) -> None:
        """
        Plot equity curve.
        
        Args:
            equity_df: DataFrame with equity history
            output_path: Path to save the chart
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting (convert Decimal to float)
        equity_df = self._prepare_data_for_charts(equity_df)
        
        # Plot equity curve
        plt.plot(equity_df.index, equity_df['equity'], label='Equity', color='#1f77b4', linewidth=2)
        
        # Add initial equity as horizontal line
        initial_equity = equity_df['equity'].iloc[0]
        plt.axhline(y=initial_equity, color='#ff7f0e', linestyle='--', 
                   label=f'Initial Capital (${initial_equity:,.2f})')
        
        # Format plot
        plt.title('Equity Curve', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format y-axis with dollar signs
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
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