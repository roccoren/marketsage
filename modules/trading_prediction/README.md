# Trading Prediction Module

This module provides LSTM-based deep learning models for stock trading prediction. It integrates with the main application to provide buy/sell/hold signals based on historical stock data.

## Overview

The trading prediction module uses LSTM (Long Short-Term Memory) neural networks to analyze stock price patterns and generate trading signals. It processes historical OHLCV (Open, High, Low, Close, Volume) data to create features, normalize them, and feed them into an LSTM model that predicts future price movements.

## Features

- Data acquisition from multiple sources
- Advanced feature engineering with technical indicators
- Data normalization and preprocessing
- LSTM model architecture for time series prediction
- Classification output (Buy/Hold/Sell signals)
- Model training, evaluation, and persistence
- Integration with the main application

## Usage

### Running as a Standalone Module

You can run the example script to see the trading prediction in action:

```bash
python -m modules.trading_prediction.example
```

This will:
1. Fetch historical data for AAPL (Apple Inc.)
2. Create technical features and normalize them
3. Train an LSTM model on the data
4. Evaluate the model and generate predictions
5. Save the trained model and visualization plots

### Integration with Main Application

The trading prediction module is fully integrated with the main application. To use it:

```bash
python main.py --symbols-file sample_stocks.txt --prediction-model lstm
```

Command-line options:
- `--symbols-file`: Path to a file containing stock symbols (one per line)
- `--prediction-model`: Set to 'lstm' to use the LSTM-based prediction model

## Model Details

- **Architecture**: LSTM (Long Short-Term Memory)
- **Input**: 60 days of historical data with technical indicators
- **Output**: Classification (0=Sell, 1=Hold, 2=Buy)
- **Features**: Price data, moving averages, momentum indicators, volatility indicators, etc.
- **Training**: Uses early stopping to prevent overfitting

## Files

- `base.py`: Abstract base classes and interfaces
- `data_acquisition.py`: Data fetching functionality
- `feature_engineering.py`: Technical indicator creation and normalization
- `dataset.py`: Dataset preparation for deep learning
- `model_architectures.py`: Neural network architecture definitions
- `model_training.py`: Training and evaluation functionality
- `integration.py`: Integration with the main application
- `example.py`: Example script demonstrating the module

## Model Output

The model generates:
- Trading signals (Buy/Hold/Sell)
- Prediction confidence
- Expected price change
- Trend and volatility metrics

## Visualization

The module can generate:
- Training history plots (loss and accuracy)
- Confusion matrix for prediction evaluation