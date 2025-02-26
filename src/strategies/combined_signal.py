import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .base import Strategy
from utils.logging_config import configure_logging, load_debug_config

# Load debug configuration
debug_config = load_debug_config()
debug_enabled = debug_config.get('combined_signal', False)

# Configure module logger
logger = configure_logging(
    __name__,
    log_level=debug_config.get('logLevel', 'INFO'),
    log_file=debug_config.get('logFiles', {}).get('combined_signal'),
    debug_mode=debug_enabled
)

class SignalType:
    BUY = "BUY"
    SELL = "SELL"

class CombinedSignalStrategy(Strategy):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name="CombinedSignal")
        self.params = params or {
            # MACD parameters
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            # Bollinger Bands parameters
            'bb_period': 20,
            'bb_std': 2,
            # Momentum parameters
            'momentum_period': 90,  # 3 months
            'volume_ma_period': 20,
            # Signal thresholds
            'momentum_threshold': 0.05,  # 5% minimum momentum
            'volume_threshold': 1.5  # 50% above average volume
        }

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD components."""
        close = pd.to_numeric(data['Close'], errors='coerce')
        exp1 = close.ewm(span=self.params['fast_period'], adjust=False).mean()
        exp2 = close.ewm(span=self.params['slow_period'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.params['signal_period'], adjust=False).mean()
        hist = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': hist
        })

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        close = pd.to_numeric(data['Close'], errors='coerce')
        middle_band = close.rolling(window=self.params['bb_period']).mean()
        std_dev = close.rolling(window=self.params['bb_period']).std()
        upper_band = middle_band + (std_dev * self.params['bb_std'])
        lower_band = middle_band - (std_dev * self.params['bb_std'])

        return pd.DataFrame({
            'MiddleBand': middle_band,
            'UpperBand': upper_band,
            'LowerBand': lower_band
        })

    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price momentum."""
        close = pd.to_numeric(data['Close'], errors='coerce')
        return close.pct_change(periods=self.params['momentum_period'])

    def is_valid_setup(self, row_index: int, data: pd.DataFrame, 
                      macd_data: pd.DataFrame, bb_data: pd.DataFrame, 
                      momentum: pd.Series, signal_type: str) -> bool:
        """Check if all indicators confirm the signal."""
        try:
            price = data['Close'].iloc[row_index]
            volume = data['Volume'].iloc[row_index]
            volume_ma = data['Volume'].rolling(window=self.params['volume_ma_period']).mean().iloc[row_index]
            momentum_value = momentum.iloc[row_index]
            
            # MACD conditions
            macd_line = macd_data['MACD'].iloc[row_index]
            signal_line = macd_data['Signal'].iloc[row_index]
            
            # Bollinger Bands conditions
            middle_band = bb_data['MiddleBand'].iloc[row_index]
            upper_band = bb_data['UpperBand'].iloc[row_index]
            lower_band = bb_data['LowerBand'].iloc[row_index]

            if signal_type == SignalType.BUY:
                macd_condition = macd_line > signal_line and macd_line > 0
                bb_condition = price <= middle_band and price > lower_band
                momentum_condition = momentum_value > self.params['momentum_threshold']
            else:  # SELL
                macd_condition = macd_line < signal_line and macd_line < 0
                bb_condition = price >= middle_band and price < upper_band
                momentum_condition = momentum_value < -self.params['momentum_threshold']

            volume_condition = volume > (volume_ma * self.params['volume_threshold'])

            return all([macd_condition, bb_condition, momentum_condition, volume_condition])

        except Exception as e:
            logger.error(f"Error checking signal setup: {str(e)}")
            return False

    def analyze(self, data: pd.DataFrame) -> List[Dict]:
        """Analyze price data using combined strategy."""
        try:
            if len(data) < max(self.params['slow_period'], self.params['bb_period']):
                logger.warning("Insufficient data for analysis")
                return []

            logger.info(f"Analyzing data from {data.index[0]} to {data.index[-1]}")
            logger.info(f"Total data points before filtering: {len(data)}")

            # Filter data from January 2024
            data = data[data.index >= '2024-01-01']
            logger.info(f"Data points after Jan 2024 filter: {len(data)}")

            if len(data) == 0:
                logger.info("No data available from January 2024")
                return []

            signals = []
            
            # Calculate indicators
            macd_data = self.calculate_macd(data)
            logger.info("MACD calculation completed")
            bb_data = self.calculate_bollinger_bands(data)
            logger.info("Bollinger Bands calculation completed")
            momentum = self.calculate_momentum(data)
            logger.info("Momentum calculation completed")

            # Get last 3 trading days
            recent_data = data.tail(3)
            logger.info(f"Analyzing last {len(recent_data)} trading days")
            
            for i in range(len(recent_data)):
                current_index = recent_data.index[i]
                row_index = data.index.get_loc(current_index)

                logger.debug(f"Checking signals for date: {current_index}")

                # Check for buy signals
                if self.is_valid_setup(row_index, data, macd_data, bb_data, momentum, SignalType.BUY):
                    logger.info(f"Buy signal detected on {current_index}")
                    signal = self._create_signal_dict(
                        data, macd_data, bb_data, momentum, row_index,
                        SignalType.BUY, current_index
                    )
                    signals.append(signal)

                # Check for sell signals
                elif self.is_valid_setup(row_index, data, macd_data, bb_data, momentum, SignalType.SELL):
                    logger.info(f"Sell signal detected on {current_index}")
                    signal = self._create_signal_dict(
                        data, macd_data, bb_data, momentum, row_index,
                        SignalType.SELL, current_index
                    )
                    signals.append(signal)

            logger.info(f"Analysis complete. Found {len(signals)} signals.")
            return signals

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return []

    def _create_signal_dict(self, data: pd.DataFrame, macd_data: pd.DataFrame, 
                          bb_data: pd.DataFrame, momentum: pd.Series, 
                          row_index: int, signal_type: str, 
                          signal_date: pd.Timestamp) -> Dict:
        """Create signal dictionary with all relevant information."""
        return {
            'Date': signal_date.strftime('%Y-%m-%d'),
            'Type': signal_type,
            'Price': float(data['Close'].iloc[row_index]),
            'Volume': float(data['Volume'].iloc[row_index]),
            'MACD': {
                'MACD': float(macd_data['MACD'].iloc[row_index]),
                'Signal': float(macd_data['Signal'].iloc[row_index]),
                'Histogram': float(macd_data['Histogram'].iloc[row_index])
            },
            'BollingerBands': {
                'Upper': float(bb_data['UpperBand'].iloc[row_index]),
                'Middle': float(bb_data['MiddleBand'].iloc[row_index]),
                'Lower': float(bb_data['LowerBand'].iloc[row_index])
            },
            'Momentum': float(momentum.iloc[row_index]),
            'Strength': self.calculate_signal_strength(
                float(macd_data['Histogram'].iloc[row_index]),
                float(data['Volume'].iloc[row_index] / data['Volume'].rolling(window=20).mean().iloc[row_index]),
                float(data['Close'].pct_change().iloc[row_index])
            )
        }