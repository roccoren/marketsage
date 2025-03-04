def format_metrics(metrics):
    return (
        f"Total Trades: {metrics['total_trades']}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Total P&L: ${metrics['total_pnl']:.2f}\n"
        f"Average Return: {metrics['avg_return']:.2f}%\n"
        f"Average Win: {metrics['avg_winning_return']:.2f}%\n"
        f"Average Loss: {metrics['avg_losing_return']:.2f}%\n"
        f"Max Drawdown: ${metrics['max_drawdown']:.2f}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}"
    )

def format_trade(trade):
    return (
        f"Trade Type: {trade.signal_type}\n"
        f"Entry Date: {trade.entry_date.date()}\n"
        f"Entry Price: ${trade.entry_price:.2f}\n"
        f"Exit Date: {trade.exit_date.date()}\n"
        f"Exit Price: ${trade.exit_price:.2f}\n"
        f"P&L: ${trade.pnl:.2f} ({trade.return_pct:.2f}%)"
    )

def get_backtest_report(symbol, exchange, result):
    report = f"\nBacktest Results for {symbol} ({exchange})\n"
    report += "=" * 40 + "\n"
    report += format_metrics(result.metrics) + "\n"
    report += "\nTrade History:\n" + "-" * 40 + "\n"
    for trade in result.trades:
         report += format_trade(trade) + "\n\n"
    return report

def display_results(symbol, exchange, result):
    print(get_backtest_report(symbol, exchange, result))