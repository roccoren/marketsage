# StockScan - Quantitative Trading Strategy Framework

A flexible and extensible framework for developing and backtesting quantitative trading strategies.

## Features

- Dynamic data loading from multiple sources (yfinance, Alpha Vantage)
- Modular strategy implementation with base class for easy extension
- Comprehensive performance analytics and visualization
- Risk management and position sizing
- Support for multiple assets and portfolio management
- Configurable parameters and settings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stockscan.git
cd stockscan
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
stockscan/
├── src/
│   ├── data/           # Data loading and processing
│   ├── strategies/     # Trading strategy implementations
│   ├── analytics/      # Performance analytics and visualization
│   ├── utils/         # Utility functions
│   └── config/        # Configuration settings
├── scripts/           # Execution scripts
├── data/             # Data storage
│   ├── instruments/  # Stock symbols and metadata
│   ├── calendars/    # Trading calendars
│   └── features/     # Processed stock data
├── outputs/          # Strategy outputs and visualizations
├── tests/           # Unit tests
├── requirements.txt # Project dependencies
└── README.md       # Project documentation
```

## Usage

1. Configure settings in `src/config/settings.py`

2. Run the momentum strategy:
```bash
python scripts/run_strategy.py
```

3. View results in the `outputs` directory:
- `returns.png`: Cumulative returns plot
- `drawdown.png`: Drawdown analysis
- `monthly_returns.png`: Monthly returns heatmap
- `position_weights.png`: Position weights over time
- `metrics.csv`: Performance metrics

## Strategy Implementation

To implement a new strategy:

1. Create a new strategy class in `src/strategies/` that inherits from `BaseStrategy`
2. Implement the required methods:
   - `generate_signals()`
   - `calculate_positions()`
3. Add any additional methods specific to your strategy

Example:
```python
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Implement signal generation logic
        pass
        
    def calculate_positions(self, signals, data):
        # Implement position sizing logic
        pass
```

## Performance Analytics

The framework includes comprehensive performance analytics:

- Returns analysis (total, annual, daily)
- Risk metrics (Sharpe ratio, Sortino ratio, max drawdown)
- Trading metrics (win rate, profit factor)
- Relative performance (alpha, beta, information ratio)
- Visualization tools for returns, drawdown, and position weights

## Configuration

Key configuration settings in `src/config/settings.py`:

- Data sources and parameters
- Strategy parameters
- Portfolio parameters
- Risk management settings
- Performance metrics
- Visualization settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Qlib](https://github.com/microsoft/qlib) from Microsoft Research
- Data provided by [yfinance](https://github.com/ranaroussi/yfinance)
- Visualization powered by [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)