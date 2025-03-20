"""
Configuration settings for the trading system
"""

from pathlib import Path
from datetime import datetime, timedelta
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Data paths
INSTRUMENTS_DIR = DATA_DIR / "instruments"
CALENDARS_DIR = DATA_DIR / "calendars"
FEATURES_DIR = DATA_DIR / "features" / "day"

# Default symbols and date ranges
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Trading parameters
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# Strategy parameters
STRATEGY_PARAMS = {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "ma_short": 10,
    "ma_medium": 20,
    "ma_long": 50,
    "position_size": 0.1,  # Maximum position size per stock
    "stop_loss": 0.02,     # 2% stop loss
    "take_profit": 0.05    # 5% take profit
}

# Portfolio parameters
DEFAULT_PORTFOLIO_PARAMS = {
    "max_position_size": 0.25,  # Maximum 25% in any single position
    "min_position_size": 0.05,  # Minimum 5% in any single position
    "cash_buffer": 0.1,  # Keep 10% in cash
    "transaction_cost": 0.001,  # 0.1% transaction cost
}

# Risk management parameters
RISK_PARAMS = {
    "max_drawdown": 0.25,  # Maximum 25% drawdown
    "stop_loss": 0.15,  # 15% stop loss
    "take_profit": 0.30,  # 30% take profit
    "position_sizing": "risk_parity",  # equal, risk_parity, or custom
}

# Data sources
DATA_SOURCES = {
    "yfinance": {
        "enabled": True,
        "timeout": 30,
        "retry_count": 3,
    },
    "alpha_vantage": {
        "enabled": False,
        "api_key": None,
        "timeout": 30,
    },
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(OUTPUT_DIR / "trading.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

# Performance metrics
PERFORMANCE_METRICS = [
    "total_return",
    "annual_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "calmar_ratio",
    "information_ratio",
]

# Visualization settings
PLOT_SETTINGS = {
    "style": "ggplot",
    "figsize": (12, 6),
    "dpi": 100,
    "save_format": "png",
    "colors": {
        "buy": "green",
        "sell": "red",
        "hold": "gray",
        "price": "blue",
        "ma_short": "orange",
        "ma_medium": "purple",
        "ma_long": "brown",
        "bb_upper": "gray",
        "bb_lower": "gray",
        "rsi": "purple",
        "macd": "blue",
        "signal": "orange",
        "volume": "gray"
    }
}

# Create necessary directories
for directory in [DATA_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 