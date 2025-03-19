import json
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def load_latest_report(output_dir: str) -> Dict[str, Any]:
    """Load the most recent trading analysis report."""
    output_path = Path(output_dir)
    report_files = list(output_path.glob("trading_analysis_*.json"))
    
    if not report_files:
        raise FileNotFoundError("No trading analysis reports found")
    
    # Get the most recent report
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_report, 'r') as f:
        return json.load(f)

def format_signal_summary(signal_summary: Dict[str, Any]) -> str:
    """Format the signal summary section with enhanced display."""
    output = []

    if not signal_summary:
        output.append("No signals found.")
        return "\n".join(output)
    
    # Merge all signals from different strategies
    all_signals = []
    for strategy, data in signal_summary.items():
        if not data.get('signals'):
            continue
        for signal in data['signals']:
            signal['Strategy'] = strategy
            all_signals.append(signal)

    if not all_signals:
        output.append("No signals found.")
        return "\n".join(output)

    # Convert to DataFrame and sort by strength
    signals_df = pd.DataFrame(all_signals)
    if not signals_df.empty:
        # Ensure strength values are numeric
        signals_df['strength'] = signals_df['strength'].apply(
            lambda x: float(x) if isinstance(x, (int, float)) else 
                     1.0 if isinstance(x, str) and x.lower() == 'strong' else 0.0
        )
        signals_df = signals_df.sort_values('strength', ascending=False)

        # Strong signals section
        strong_signals = signals_df[signals_df['strength'] >= 0.7]
        if not strong_signals.empty:
            output.append("\n=== STRONG SIGNALS ===\n")
            output.append(_format_signal_table(strong_signals))

        # Other signals section
        other_signals = signals_df[signals_df['strength'] < 0.7]
        if not other_signals.empty:
            output.append("\n=== OTHER SIGNALS ===\n")
            output.append(_format_signal_table(other_signals))

    return "\n".join(output)

def _format_signal_table(signals_df: pd.DataFrame) -> str:
    """Helper function to format signal table."""
    table_data = []
    for _, signal in signals_df.iterrows():
        table_data.append([
            signal.get('symbol', 'N/A'),
            signal.get('Strategy', 'N/A'),
            signal.get('Summary', 'N/A'),
            f"{signal.get('price', 0):.2f}",
            f"{signal.get('daily_change', 0):.2f}%",
            f"{signal.get('weekly_change', 0):.2f}%",
            signal.get('type', 'N/A'),
            f"{signal.get('strength', 0):.2f}"
        ])

    return tabulate(
        table_data,
        headers=['Symbol', 'Strategy', 'Rating', 'Price', '1D%', '1W%', 'Type', 'Strength'],
        tablefmt='grid',
        floatfmt='.2f'
    )

def format_opportunities(opportunities: list) -> str:
    """Format the high-conviction opportunities section."""
    output = ["\n=== HIGH-CONVICTION OPPORTUNITIES ===\n"]
    
    if not opportunities:
        output.append("No high-conviction opportunities found.")
        return "\n".join(output)
    
    for opp in opportunities:
        output.append(f"\nSymbol: {opp['symbol']}")
        output.append("-" * (len(opp['symbol']) + 8))
        output.append(f"Signal Count: {opp['signal_count']}")
        output.append(f"Strategy Count: {opp['strategy_count']}")
        output.append(f"Confidence Score: {opp['confidence_score']:.2f}")
        
        if 'primary_entry' in opp:
            output.append(f"Primary Entry: {opp['primary_entry']:.2f}")
        if 'primary_exit' in opp:
            output.append(f"Primary Exit: {opp['primary_exit']:.2f}")
            
    return "\n".join(output)

def format_recommendations(recommendations: list) -> str:
    """Format the recommended actions section."""
    output = ["\n=== RECOMMENDED ACTIONS ===\n"]
    
    if not recommendations:
        output.append("No trading recommendations available.")
        return "\n".join(output)
    
    # Main recommendation table
    headers = ['Symbol', 'Date', 'Action', 'Confidence', 'Entry Range', 'Stop Loss', 'Take Profit']
    rows = []
    
    for rec in recommendations:
        entry_range = rec.get('entry_range')
        entry_str = f"{entry_range['low']:.2f}-{entry_range['high']:.2f}" if entry_range else "N/A"
        stop_loss = f"{rec['stop_loss']:.2f}" if rec.get('stop_loss') else "N/A"
        take_profit = "N/A"
        
        # Safely calculate take profit only if entry_range exists and has the required keys
        if entry_range and 'high' in entry_range and 'low' in entry_range:
            if rec['action'] == 'BUY':
                take_profit = f"{float(entry_range['high']) * 1.1:.2f}"
            elif rec['action'] == 'SELL':
                take_profit = f"{float(entry_range['low']) * 0.9:.2f}"
        
        rows.append([
            rec['symbol'],
            rec.get('signal_date', 'N/A'),
            rec['action'],
            f"{rec['confidence_score']:.2f}",
            entry_str,
            stop_loss,
            take_profit
        ])
    
    output.append(tabulate(rows, headers=headers, tablefmt='pipe'))
    # Add detailed trade timing information
    output.append("\nTrade Execution Details:")
    for rec in recommendations:
        if 'trade_timing' in rec:
            output.append(f"\n{rec['symbol']}:")
            output.append(f"- Optimal Entry Time: {rec['trade_timing']['optimal_entry']}")
            output.append(f"- Valid Until: {rec['trade_timing']['valid_until']}")
            output.append(f"- Position Size: {rec['position_size']:.2%}")
    
    return "\n".join(output)

def format_risk_assessment(risk_assessment: Dict[str, Any]) -> str:
    """Format the risk assessment section."""
    output = ["\n=== RISK ASSESSMENT ===\n"]
    
    if not risk_assessment:
        output.append("No risk assessment data available.")
        return "\n".join(output)
    
    # Market Conditions
    mc = risk_assessment['market_conditions']
    output.append("\nMarket Conditions:")
    output.append(f"- Bias: {mc['market_bias']}")
    output.append(f"- Signal Strength: {mc['signal_strength']}")
    output.append(f"- Recent Signal Count: {mc['recent_signal_count']}")
    
    # Volatility Measures
    if risk_assessment['volatility_measures']:
        output.append("\nVolatility Measures:")
        vol_data = []
        for symbol, measures in risk_assessment['volatility_measures'].items():
            vol_data.append([
                symbol,
                f"{measures['daily_volatility']:.3f}",
                f"{measures['avg_true_range']:.3f}"
            ])
        output.append(tabulate(vol_data, 
                             headers=['Symbol', 'Daily Vol', 'ATR'],
                             tablefmt='pipe'))
    
    # Volume Analysis
    if risk_assessment['volume_analysis']:
        output.append("\nVolume Analysis:")
        vol_analysis = []
        for symbol, profile in risk_assessment['volume_analysis'].items():
            vol_analysis.append([
                symbol,
                f"{profile['average_volume']:,.0f}",
                profile['volume_trend'],
                profile['volume_stability']
            ])
        output.append(tabulate(vol_analysis,
                             headers=['Symbol', 'Avg Volume', 'Trend', 'Stability'],
                             tablefmt='pipe'))
    
    return "\n".join(output)

def format_execution_strategy(execution_strategy: Dict[str, Any]) -> str:
    """Format the execution strategy section."""
    output = ["\n=== EXECUTION STRATEGY ===\n"]
    
    if not execution_strategy:
        output.append("No execution strategy data available.")
        return "\n".join(output)
    
    # Order Types
    if execution_strategy['order_types']:
        output.append("\nRecommended Order Types:")
        order_data = []
        for symbol, orders in execution_strategy['order_types'].items():
            order_data.append([
                symbol,
                orders['primary'],
                orders['backup'],
                orders['reasoning']
            ])
        output.append(tabulate(order_data,
                             headers=['Symbol', 'Primary', 'Backup', 'Reasoning'],
                             tablefmt='pipe'))
    
    # Risk Parameters
    if execution_strategy['risk_parameters']:
        output.append("\nRisk Management Parameters:")
        risk_data = []
        for symbol, params in execution_strategy['risk_parameters'].items():
            risk_data.append([
                symbol,
                f"{params['stop_loss_percent']}%",
                f"{params['take_profit_ratio']}:1",
                params['max_loss_per_trade'],
                params['trailing_stop']
            ])
        output.append(tabulate(risk_data,
                             headers=['Symbol', 'Stop Loss', 'Risk:Reward', 'Max Loss', 'Trailing Stop'],
                             tablefmt='pipe'))
    
    # Position Management
    if execution_strategy['position_management']:
        output.append("\nPosition Management Guidelines:")
        for condition, guidelines in execution_strategy['position_management'].items():
            output.append(f"\n{condition.replace('_', ' ').title()}:")
            for guideline in guidelines:
                output.append(f"- {guideline}")
    
    return "\n".join(output)

def display_analysis_report(report: Dict[str, Any]) -> str:
    """Display the full trading analysis report in a formatted way."""
    if not report:
        return "\nNo analysis report data available."
    
    sections = []
    
    # Safely add each section with None checks
    if report.get('signal_summary') is not None:
        sections.append(format_signal_summary(report['signal_summary']))
    if report.get('high_conviction_opportunities') is not None:
        sections.append(format_opportunities(report['high_conviction_opportunities']))
    if report.get('recommended_actions') is not None:
        sections.append(format_recommendations(report['recommended_actions']))
    if report.get('risk_assessment') is not None:
        sections.append(format_risk_assessment(report['risk_assessment']))
    if report.get('execution_strategy') is not None:
        sections.append(format_execution_strategy(report['execution_strategy']))
    
    if not sections:
        return "\nNo valid analysis report sections available."
    
    return "\n\n".join(sections)