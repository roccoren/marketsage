import os
import datetime
import pandas as pd
from pathlib import Path

def test_report_generation():
    """
    Test function to generate a report with the new directory structure and symbol file name.
    """
    # Get current date for the directory name
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    
    # Symbol file name for the report filename
    symbol_file_name = "test-symbols"
    
    # Create report directory with current date
    report_dir = f"reports/backtest/{current_date}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Format dates for filename
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime(2025, 3, 10)
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    # Define HTML path using symbol file name
    html_path = f"{report_dir}/{symbol_file_name}_summary_{start_date_str}_{end_date_str}.html"
    csv_path = f"{report_dir}/{symbol_file_name}_summary_{start_date_str}_{end_date_str}.csv"
    
    # Create a sample DataFrame with non-zero values
    data = {
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'total_return': [0.15, 0.08, 0.12],
        'sharpe_ratio': [1.2, 0.9, 1.1],
        'max_drawdown': [0.05, 0.03, 0.04],
        'win_rate': [0.65, 0.58, 0.62],
        'total_trades': [120, 90, 110],
        'final_equity': [11500, 10800, 11200],
        'report_path': ['path1.html', 'path2.html', 'path3.html']
    }
    df = pd.DataFrame(data)
    
    # Calculate summary metrics
    profitable_symbols = sum(df['total_return'] > 0)
    avg_return = df['total_return'].mean()
    
    # Save results to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved results to CSV: {csv_path}")
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Market Backtest Summary: Test</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Market Backtest Summary: Test</h1>
        <div class="summary">
            <h2>Backtest Information</h2>
            <ul>
                <li>Market: Test</li>
                <li>Period: {start_date} to {end_date}</li>
                <li>Total Symbols: {len(df)}</li>
                <li>Valid Results: {len(df)}</li>
                <li>Profitable Symbols: {profitable_symbols} ({profitable_symbols/len(df)*100:.1f}%)</li>
                <li>Average Return: {avg_return*100:.2f}%</li>
            </ul>
        </div>
        
        <h2>Symbol Results</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Total Trades</th>
                <th>Final Equity</th>
                <th>Report</th>
            </tr>
    """
    
    # Add rows for each symbol
    for _, row in df.iterrows():
        color_class = "positive" if row['total_return'] > 0 else "negative"
        html_content += f"""
        <tr>
            <td>{row['symbol']}</td>
            <td class="{color_class}">{row['total_return']*100:.2f}%</td>
            <td>{row['sharpe_ratio']:.2f}</td>
            <td>{row['max_drawdown']*100:.2f}%</td>
            <td>{row['win_rate']*100:.1f}%</td>
            <td>{row['total_trades']}</td>
            <td>${row['final_equity']:.2f}</td>
            <td><a href="{row['report_path']}">View</a></td>
        </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Summary report generated at {html_path}")

if __name__ == "__main__":
    test_report_generation() 