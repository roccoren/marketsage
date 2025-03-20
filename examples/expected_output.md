# Expected Output from Examples

This document shows a sample of what to expect when running the example scripts. The actual numbers will vary based on market conditions and when you run the examples.

## Trio Stocks Prediction Example

When you run `python examples/trio_stocks_prediction.py`, you should see output similar to this:

```
Starting prediction for AAPL, MSFT, NVDA
Setting up environment...
US stock data found.
Creating pipeline for stocks: AAPL, MSFT, NVDA
Initializing Qlib
2023-12-15 14:32:10,123 - INFO - Preparing data from 2020-09-15 to 2023-12-15
2023-12-15 14:32:12,456 - INFO - Loading Alpha158 features
2023-12-15 14:32:15,789 - INFO - Dataset prepared with segments: {'train': ('2020-09-15', '2023-03-15'), 'valid': ('2023-03-15', '2023-09-15'), 'test': ('2023-09-15', '2023-12-15')}
Training model...
2023-12-15 14:32:20,123 - INFO - Training model lgb_model_iteration_0
2023-12-15 14:32:45,678 - INFO - Model lgb_model_iteration_0 trained with scores - Train: 0.6832, Valid: 0.5467
Model trained with scores:
  Train score: 0.6832
  Valid score: 0.5467
Generating predictions...
2023-12-15 14:32:50,123 - INFO - Generating signals with topk_strategy_iteration_0
2023-12-15 14:32:52,456 - INFO - Running backtest with strategy topk_strategy_iteration_0
2023-12-15 14:32:55,789 - INFO - Backtest results: {'annualized_return': 0.2134, 'annualized_volatility': 0.1923, 'max_drawdown': -0.0865, 'total_turnover': 1.2345, 'average_turnover': 0.0789, 'sharpe_ratio': 1.1097}
Backtest results:
  annualized_return: 0.2134
  annualized_volatility: 0.1923
  max_drawdown: -0.0865
  total_turnover: 1.2345
  average_turnover: 0.0789
  sharpe_ratio: 1.1097

=== Latest Predictions ===

AAPL:
  Latest prediction (2023-12-14): 0.008547
  30-day trend: UPWARD (12.45%)
  Signal strength: BUY

MSFT:
  Latest prediction (2023-12-14): 0.011234
  30-day trend: UPWARD (8.76%)
  Signal strength: STRONG BUY

NVDA:
  Latest prediction (2023-12-14): 0.015678
  30-day trend: UPWARD (18.32%)
  Signal strength: STRONG BUY

Prediction trends plot saved to ./output/trio_stocks/plots/prediction_trends.png

=== Stock Ranking (Buy Signal Strength) ===
1. NVDA: 0.015678 (BUY)
2. MSFT: 0.011234 (BUY)
3. AAPL: 0.008547 (BUY)

Recommendation: Consider buying NVDA (strongest positive signal)
```

## Prediction Trends Plot

The script will also generate a plot showing prediction trends over time, which will look something like this:

![Sample Prediction Trends Plot](https://github.com/yourusername/stockscan/raw/main/examples/sample_prediction_trends.png)

The plot shows the prediction scores for each stock over time, with higher values indicating stronger buy signals and lower values indicating sell signals.

## Interpreting the Results

- **Prediction Scores**: The numerical values represent expected returns. Higher positive values indicate stronger buy signals.
- **Trend**: Shows whether the prediction has been improving (UPWARD) or deteriorating (DOWNWARD) over the last 30 days.
- **Signal Strength**: A categorization of the signal into STRONG BUY, BUY, WEAK BUY, WEAK SELL, SELL, or STRONG SELL.
- **Ranking**: Stocks are ranked by their latest prediction score, with the highest scoring stock at the top.
- **Recommendation**: A simple recommendation based on the strongest signal.

Remember that these are not financial recommendations. Always do your own research before making investment decisions. 