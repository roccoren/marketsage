# StockScan

StockScan is a LEAN-based trading and backtesting platform that provides a flexible framework for developing, testing, and optimizing trading strategies. Built on top of QuantConnect's LEAN engine, it offers a streamlined interface for algorithmic trading development.

## Features

- **LEAN Integration**: Seamless integration with QuantConnect's LEAN engine for robust backtesting
- **Multiple Algorithms**: Pre-built trading algorithms including Trend Following and Mean Reversion strategies
- **Data Management**: Efficient market data handling with support for multiple data providers
- **Parameter Optimization**: Built-in tools for strategy parameter optimization
- **Performance Analytics**: Comprehensive performance metrics and analysis tools
- **Modular Design**: Easy to extend and customize with new algorithms and features

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- quantconnect-lean
- quantconnect-stubs
- pythonnet
- pandas
- numpy
- matplotlib

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stockscan.git
cd stockscan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the example backtest:
```bash
python examples/backtest_example.py
```

## Usage

### Basic Usage

```python
from src.StockScan import create_engine, create_data_handler

# Create a data handler and fetch market data
data_handler = create_data_handler()
data = await data_handler.GetSymbol("AAPL", start_date, end_date)

# Create a backtesting engine
engine = create_engine()

# Run a backtest
results = engine.RunBacktest(
    algorithm_name="TrendFollowingAlgorithm",
    symbols=["AAPL"],
    start_date=start_date,
    end_date=end_date,
    initial_capital=100000
)
```

### Running Different Algorithms

The platform includes several pre-built algorithms:

1. Trend Following Algorithm:
```python
parameters = {
    "FastPeriod": 10,
    "SlowPeriod": 30,
    "VolumeThreshold": 1.2,
    "EnableShort": True
}

results = engine.RunBacktest(
    algorithm_name="TrendFollowingAlgorithm",
    symbols=["AAPL", "MSFT"],
    parameters=parameters
)
```

2. Mean Reversion Algorithm:
```python
parameters = {
    "RsiPeriod": 14,
    "OversoldThreshold": 30,
    "OverboughtThreshold": 70,
    "UseBollingerConfirmation": True
}

results = engine.RunBacktest(
    algorithm_name="MeanReversionAlgorithm",
    symbols=["AAPL"],
    parameters=parameters
)
```

### Parameter Optimization

Optimize strategy parameters:

```python
parameter_space = {
    "FastPeriod": [5, 10, 15, 20],
    "SlowPeriod": [20, 30, 40, 50],
    "EnableShort": [False, True]
}

results = engine.OptimizeParameters(
    algorithm_name="TrendFollowingAlgorithm",
    symbols=["AAPL"],
    parameter_space=parameter_space,
    optimization_metric="SharpeRatio"
)
```

### Creating Custom Algorithms

Create your own trading algorithm by inheriting from the BaseAlgorithm class:

```python
from src.StockScan.Algorithm import BaseAlgorithm

class MyCustomAlgorithm(BaseAlgorithm):
    def Initialize(self):
        # Set up your algorithm parameters
        self.SetCash(100000)
        self.AddEquity("AAPL")
        
    def OnData(self, data):
        # Implement your trading logic
        if self.ShouldBuy(data):
            self.SetHoldings("AAPL", 1.0)
        elif self.ShouldSell(data):
            self.Liquidate()
```

## Directory Structure

```
stockscan/
├── src/
│   └── StockScan/
│       ├── Algorithm/      # Trading algorithms
│       ├── Data/          # Data handling and providers
│       ├── Engine/        # LEAN engine integration
│       ├── Indicators/    # Technical indicators
│       ├── Risk/         # Risk management
│       ├── Reports/      # Performance reporting
│       ├── Utils/        # Utility functions
│       └── Interfaces/   # Abstract interfaces
├── examples/             # Example scripts
├── tests/               # Unit tests
└── requirements.txt     # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- QuantConnect for the LEAN engine
- Contributors to the project
- The algorithmic trading community

## Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue if needed

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.