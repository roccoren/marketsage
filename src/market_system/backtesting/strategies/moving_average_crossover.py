"""
Moving Average Crossover Strategy Implementation.

This module implements a strategy that identifies opportunities when a faster moving average 
crosses above a slower moving average, combined with volume and price-to-VWAP filters.
"""

import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from typing import List, Optional, Dict, Tuple, Any
import math

from src.market_system.strategy.base_strategy import BaseStrategy
from src.market_system.domain.models import MarketData, Signal, MarketState
from src.market_system.storage import ParameterStorage

logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates signals based on the crossover of fast and slow moving averages.
    - Buy when fast MA crosses above slow MA (golden cross)
    - Sell when fast MA crosses below slow MA (death cross)
    
    The strategy includes:
    - Adaptive parameters based on volatility
    - Market regime detection
    - RSI, MACD, and volume confirmation
    - Volatility-based position sizing
    - Symbol-specific parameter optimization via ParameterStorage
    """
    
    def __init__(self, 
                fast_period: int = 10, 
                slow_period: int = 30,
                signal_threshold: float = 0.0,
                volume_threshold: float = 1.5,
                priority: int = 1,
                rsi_period: int = 14,
                rsi_overbought: int = 70,
                rsi_oversold: int = 30,
                macd_fast: int = 12,
                macd_slow: int = 26,
                macd_signal: int = 9,
                atr_period: int = 14,
                adaptive_params: bool = True,
                use_market_regime: bool = True,
                use_symbol_specific_params: bool = True,
                param_db_path: str = "data/strategy_params.db"):
        """
        Initialize the strategy with parameters.
        
        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            signal_threshold: Minimum difference between fast and slow MA to generate a signal
            volume_threshold: Minimum volume compared to average to confirm a signal
            priority: Priority level for this strategy (lower number = higher priority)
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            macd_fast: Fast period for MACD
            macd_slow: Slow period for MACD
            macd_signal: Signal period for MACD
            atr_period: Period for ATR calculation
            adaptive_params: Whether to use adaptive parameters based on volatility
            use_market_regime: Whether to use market regime detection
            use_symbol_specific_params: Whether to use symbol-specific parameters from storage
            param_db_path: Path to the parameter database
        """
        super().__init__(f"MAC_{fast_period}_{slow_period}")
        self.base_fast_period = fast_period
        self.base_slow_period = slow_period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_threshold = signal_threshold
        self.volume_threshold = volume_threshold
        self.priority = priority
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.adaptive_params = adaptive_params
        self.use_market_regime = use_market_regime
        self.use_symbol_specific_params = use_symbol_specific_params
        
        # Symbol-specific parameter cache
        self.symbol_params = {}
        
        # Initialize parameter storage if using symbol-specific parameters
        self.param_storage = None
        if self.use_symbol_specific_params:
            try:
                self.param_storage = ParameterStorage(param_db_path)
                logger.info(f"Initialized ParameterStorage with database at {param_db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize ParameterStorage: {str(e)}")
                self.use_symbol_specific_params = False
        
        logger.info(f"Initialized {self.name} with base_fast_period={fast_period}, base_slow_period={slow_period}, " 
                   f"signal_threshold={signal_threshold}, volume_threshold={volume_threshold}, priority={priority}, "
                   f"adaptive_params={adaptive_params}, use_market_regime={use_market_regime}, "
                   f"use_symbol_specific_params={use_symbol_specific_params}")
        
    def analyze(self, 
               market_data: List[MarketData], 
               market_state: MarketState = MarketState.UNDEFINED) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            market_data: List of MarketData objects
            market_state: Current market state
            
        Returns:
            Signal object if a signal is generated, None otherwise
        """
        if not market_data or len(market_data) < max(self.slow_period, self.rsi_period, self.macd_slow) + 15:
            return None
            
        try:
            # Extract data
            symbol = market_data[-1].symbol
            timestamp = market_data[-1].timestamp
            current_price = market_data[-1].close_price
            
            # Load symbol-specific parameters first if enabled
            if self.use_symbol_specific_params and self.param_storage:
                self._load_symbol_parameters(symbol)
            
            # Get closing prices
            closing_prices = [data.close_price for data in market_data]
            
            # Convert to dataframe for indicator calculations
            df = self._convert_to_dataframe(market_data)
            
            # Detect market regime and adapt parameters if enabled
            if self.adaptive_params:
                # Adapt MA periods based on volatility
                self._adapt_parameters(symbol, df)
                
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Detect market regime
            regime = self._detect_market_regime(df, indicators) if self.use_market_regime else None
            
            # Skip if market regime is unfavorable and use_market_regime is enabled
            if self.use_market_regime and regime == "bear":
                logger.info(f"Skipping signal generation for {symbol} due to bearish market regime")
                return None
            
            # Get moving averages
            fast_ma = indicators['fast_ma'][-1]
            slow_ma = indicators['slow_ma'][-1]
            prev_fast_ma = indicators['fast_ma'][-2]
            prev_slow_ma = indicators['slow_ma'][-2]
            
            # Calculate crossover
            current_diff = fast_ma - slow_ma
            previous_diff = prev_fast_ma - prev_slow_ma
            
            # Get volume data
            volumes = [data.volume for data in market_data]
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
            current_volume = volumes[-1]
            
            # Calculate volume confirmation
            volume_confirmation = current_volume > (avg_volume * self.volume_threshold)
            
            # Get RSI and MACD values for confirmation
            current_rsi = indicators['rsi'][-1]
            current_macd = indicators['macd'][-1]
            current_macd_signal = indicators['macd_signal'][-1]
            current_macd_hist = indicators['macd_hist'][-1]
            
            # Check for golden cross (buy signal)
            if (previous_diff <= Decimal(str(self.signal_threshold)) and 
                current_diff > Decimal(str(self.signal_threshold))):
                
                # Enhanced trend confirmation checks
                trend_confirmation = current_price > fast_ma and current_price > slow_ma
                
                # RSI confirmation - not overbought for buys
                rsi_confirmation = current_rsi < self.rsi_overbought
                
                # MACD confirmation - histogram is positive and increasing
                macd_confirmation = (current_macd_hist > 0 and 
                                    current_macd_hist > indicators['macd_hist'][-2])
                
                # Combined confirmation (volume + trend + at least one of RSI/MACD)
                confirmation = (volume_confirmation and trend_confirmation and 
                               (rsi_confirmation or macd_confirmation))
                
                if confirmation:
                    # Calculate confidence score based on multiple factors
                    confidence = self._calculate_confidence(
                        current_diff, 
                        volumes[-5:], 
                        avg_volume,
                        current_rsi, 
                        current_macd_hist,
                        indicators['atr'][-1],
                        current_price
                    )
                    
                    # Calculate optimal stop loss based on volatility
                    atr = indicators['atr'][-1]
                    volatility = atr / current_price
                    
                    # More adaptive stop-loss based on ATR
                    stop_loss = min(Decimal(str(volatility * 2)), Decimal("0.02"))
                    
                    # Take profit at least 2.5:1 risk-reward ratio
                    take_profit = stop_loss * Decimal("2.5")
                    
                    # Set quantity based on available capital (to be calculated by portfolio manager)
                    quantity = None
                    
                    # Create signal with meta information
                    signal = Signal(
                        symbol=symbol,
                        timestamp=timestamp,
                        action="BUY",
                        confidence=confidence,
                        strategy_name=self.name,
                        market_state=market_state,
                        indicators_snapshot={
                            "fast_ma": float(fast_ma),
                            "slow_ma": float(slow_ma),
                            "price": float(current_price),
                            "volume": int(current_volume),
                            "avg_volume": float(avg_volume),
                            "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 0.0,
                            "rsi": float(current_rsi),
                            "macd": float(current_macd),
                            "macd_signal": float(current_macd_signal),
                            "macd_hist": float(current_macd_hist),
                            "atr": float(indicators['atr'][-1]),
                            "market_regime": regime if regime else "unknown"
                        },
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        quantity=quantity,
                        priority=self.priority
                    )
                    
                    logger.info(f"Generated BUY signal for {symbol} at {timestamp} with confidence {confidence}, "
                               f"RSI: {current_rsi:.2f}, MACD Hist: {current_macd_hist:.2f}, "
                               f"Market Regime: {regime if regime else 'unknown'}")
                    return signal
            
            # Check for death cross (sell signal)
            elif (previous_diff >= Decimal(str(self.signal_threshold)) and 
                  current_diff < Decimal(str(self.signal_threshold))):
                
                # Enhanced trend confirmation checks
                trend_confirmation = current_price < fast_ma and current_price < slow_ma
                
                # RSI confirmation - not oversold for sells
                rsi_confirmation = current_rsi > self.rsi_oversold
                
                # MACD confirmation - histogram is negative and decreasing
                macd_confirmation = (current_macd_hist < 0 and 
                                    current_macd_hist < indicators['macd_hist'][-2])
                
                # Combined confirmation (volume + trend + at least one of RSI/MACD)
                confirmation = (volume_confirmation and trend_confirmation and 
                               (rsi_confirmation or macd_confirmation))
                
                if confirmation:
                    # Calculate confidence score based on multiple factors
                    confidence = self._calculate_confidence(
                        abs(current_diff), 
                        volumes[-5:], 
                        avg_volume,
                        current_rsi, 
                        abs(current_macd_hist),
                        indicators['atr'][-1],
                        current_price
                    )
                    
                    # Calculate optimal stop loss based on volatility
                    atr = indicators['atr'][-1]
                    volatility = atr / current_price
                    
                    # More adaptive stop-loss based on ATR
                    stop_loss = min(Decimal(str(volatility * 2)), Decimal("0.02"))
                    
                    # Take profit at least 2.5:1 risk-reward ratio
                    take_profit = stop_loss * Decimal("2.5")
                    
                    # Create signal with meta information
                    signal = Signal(
                        symbol=symbol,
                        timestamp=timestamp,
                        action="SELL",
                        confidence=confidence,
                        strategy_name=self.name,
                        market_state=market_state,
                        indicators_snapshot={
                            "fast_ma": float(fast_ma),
                            "slow_ma": float(slow_ma),
                            "price": float(current_price),
                            "volume": int(current_volume),
                            "avg_volume": float(avg_volume),
                            "volume_ratio": float(current_volume / avg_volume) if avg_volume > 0 else 0.0,
                            "rsi": float(current_rsi),
                            "macd": float(current_macd),
                            "macd_signal": float(current_macd_signal),
                            "macd_hist": float(current_macd_hist),
                            "atr": float(indicators['atr'][-1]),
                            "market_regime": regime if regime else "unknown"
                        },
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        quantity=None,
                        priority=self.priority
                    )
                    
                    logger.info(f"Generated SELL signal for {symbol} at {timestamp} with confidence {confidence}, "
                               f"RSI: {current_rsi:.2f}, MACD Hist: {current_macd_hist:.2f}, "
                               f"Market Regime: {regime if regime else 'unknown'}")
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MovingAverageCrossoverStrategy.analyze: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _load_symbol_parameters(self, symbol: str) -> None:
        """
        Load symbol-specific parameters from the parameter storage.
        
        Args:
            symbol: Symbol to load parameters for
        """
        if not self.param_storage:
            return
            
        try:
            # Check if we have cached parameters for this symbol
            if symbol in self.symbol_params:
                return
                
            # Try to get parameters from storage
            stored_params = self.param_storage.get_parameters(self.name, symbol)
            
            if stored_params:
                # Convert parameters from storage to the right types
                params = {
                    'fast_period': int(stored_params.get('fast_period', self.base_fast_period)),
                    'slow_period': int(stored_params.get('slow_period', self.base_slow_period)),
                    'signal_threshold': float(stored_params.get('signal_threshold', self.signal_threshold)),
                    'volume_threshold': float(stored_params.get('volume_threshold', self.volume_threshold)),
                    'rsi_period': int(stored_params.get('rsi_period', self.rsi_period)),
                    'rsi_overbought': int(stored_params.get('rsi_overbought', self.rsi_overbought)),
                    'rsi_oversold': int(stored_params.get('rsi_oversold', self.rsi_oversold)),
                    'macd_fast': int(stored_params.get('macd_fast', self.macd_fast)),
                    'macd_slow': int(stored_params.get('macd_slow', self.macd_slow)),
                    'macd_signal': int(stored_params.get('macd_signal', self.macd_signal)),
                    'atr_period': int(stored_params.get('atr_period', self.atr_period))
                }
                
                # Update instance variables with symbol-specific parameters
                self.fast_period = params['fast_period']
                self.slow_period = params['slow_period']
                self.signal_threshold = params['signal_threshold']
                self.volume_threshold = params['volume_threshold']
                self.rsi_period = params['rsi_period']
                self.rsi_overbought = params['rsi_overbought']
                self.rsi_oversold = params['rsi_oversold']
                self.macd_fast = params['macd_fast']
                self.macd_slow = params['macd_slow']
                self.macd_signal = params['macd_signal']
                self.atr_period = params['atr_period']
                
                # Cache the parameters
                self.symbol_params[symbol] = params
                
                logger.info(f"Loaded symbol-specific parameters for {symbol} from storage: {params}")
                
        except Exception as e:
            logger.error(f"Error loading symbol-specific parameters for {symbol}: {str(e)}")
    
    def save_symbol_parameters(self, symbol: str, performance_metrics: Dict[str, Any] = None) -> None:
        """
        Save symbol-specific parameters to storage with optional performance metrics.
        
        Args:
            symbol: Symbol to save parameters for
            performance_metrics: Optional performance metrics to record
        """
        if not self.param_storage or symbol not in self.symbol_params:
            return
            
        try:
            # Get parameters from cache
            params = self.symbol_params[symbol]
            
            # Save parameters to storage
            self.param_storage.set_parameters(self.name, symbol, params)
            
            # Record performance if metrics are provided
            if performance_metrics:
                self.param_storage.record_performance(
                    self.name, 
                    symbol, 
                    params, 
                    performance_metrics
                )
                
            logger.info(f"Saved symbol-specific parameters for {symbol} to storage: {params}")
            
        except Exception as e:
            logger.error(f"Error saving symbol-specific parameters for {symbol}: {str(e)}")
    
    def _adapt_parameters(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Adapt strategy parameters based on symbol volatility and price characteristics.
        
        Args:
            symbol: Symbol to adapt parameters for
            df: DataFrame with OHLCV data
        """
        # Check if we have cached parameters for this symbol
        if symbol in self.symbol_params:
            # Use cached parameters
            params = self.symbol_params[symbol]
            self.fast_period = params['fast_period']
            self.slow_period = params['slow_period']
            return
            
        # Calculate volatility using standard deviation of returns
        returns = df['close'].pct_change().dropna()
        daily_volatility = returns.std()
        
        # Calculate average price
        avg_price = df['close'].mean()
        
        # Calculate average volume
        avg_volume = df['volume'].mean()
        
        # Adjust parameters based on volatility
        if daily_volatility > 0.03:  # High volatility
            # Use longer periods for high volatility to filter out noise
            new_fast_period = int(self.base_fast_period * 1.5)
            new_slow_period = int(self.base_slow_period * 1.3)
        elif daily_volatility < 0.01:  # Low volatility
            # Use shorter periods for low volatility to be more responsive
            new_fast_period = max(int(self.base_fast_period * 0.7), 5)
            new_slow_period = max(int(self.base_slow_period * 0.8), 15)
        else:  # Medium volatility
            # Use base periods for medium volatility
            new_fast_period = self.base_fast_period
            new_slow_period = self.base_slow_period
            
        # Additional adjustments for high/low priced stocks
        if avg_price > 500:  # High priced stock
            # More conservative for high-priced stocks
            new_fast_period = min(new_fast_period + 2, 20)
        elif avg_price < 10:  # Low priced stock
            # More responsive for low-priced stocks
            new_fast_period = max(new_fast_period - 2, 5)
            
        # Ensure the slow period is at least twice the fast period
        new_slow_period = max(new_slow_period, new_fast_period * 2)
        
        # Store the adapted parameters in the cache
        self.symbol_params[symbol] = {
            'fast_period': new_fast_period,
            'slow_period': new_slow_period,
            'volatility': daily_volatility,
            'avg_price': avg_price,
            'avg_volume': avg_volume,
            'signal_threshold': self.signal_threshold,
            'volume_threshold': self.volume_threshold,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'atr_period': self.atr_period
        }
        
        # Set the adapted parameters
        self.fast_period = new_fast_period
        self.slow_period = new_slow_period
        
        logger.info(f"Adapted parameters for {symbol}: fast_period={new_fast_period}, "
                   f"slow_period={new_slow_period} (volatility={daily_volatility:.4f})")
        
        # Save the adapted parameters to storage if enabled
        if self.use_symbol_specific_params and self.param_storage:
            self.save_symbol_parameters(symbol)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, List[Decimal]]:
        """
        Calculate technical indicators for analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        result = {}
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan'))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'].fillna(50, inplace=True)  # Fill NaN with neutral value
        
        # Calculate MACD
        df['ema_fast'] = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate ATR for volatility
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Convert to Decimal and add to result
        result['fast_ma'] = [Decimal(str(x)) if not pd.isna(x) else Decimal('0') for x in df['fast_ma']]
        result['slow_ma'] = [Decimal(str(x)) if not pd.isna(x) else Decimal('0') for x in df['slow_ma']]
        result['rsi'] = [Decimal(str(x)) if not pd.isna(x) else Decimal('50') for x in df['rsi']]
        result['macd'] = [Decimal(str(x)) if not pd.isna(x) else Decimal('0') for x in df['macd']]
        result['macd_signal'] = [Decimal(str(x)) if not pd.isna(x) else Decimal('0') for x in df['macd_signal']]
        result['macd_hist'] = [Decimal(str(x)) if not pd.isna(x) else Decimal('0') for x in df['macd_hist']]
        result['atr'] = [Decimal(str(x)) if not pd.isna(x) else Decimal('0') for x in df['atr']]
        
        return result
    
    def _detect_market_regime(self, df: pd.DataFrame, indicators: Dict[str, List[Decimal]]) -> str:
        """
        Detect the current market regime (bull, bear, or neutral).
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Pre-calculated indicators
            
        Returns:
            Market regime as string ("bull", "bear", or "neutral")
        """
        # Use a 50-day and 200-day moving average to detect long-term trend
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # Extract the most recent valid values
        try:
            ma50 = df['ma50'].dropna().iloc[-1]
            ma200 = df['ma200'].dropna().iloc[-1]
        except IndexError:
            # Not enough data for 200-day MA, use shorter trend detection
            try:
                ma20 = df['close'].rolling(window=20).mean().dropna().iloc[-1]
                ma50 = df['ma50'].dropna().iloc[-1]
                
                # Get price data for trend analysis
                close_prices = df['close'].dropna()
                if len(close_prices) < 20:
                    return "neutral"
                    
                current_price = close_prices.iloc[-1]
                
                # Calculate trend based on current price relative to moving averages
                if current_price > ma20 and ma20 > ma50:
                    return "bull"
                elif current_price < ma20 and ma20 < ma50:
                    return "bear"
                else:
                    return "neutral"
            except IndexError:
                return "neutral"
        
        # Get price data for trend analysis
        close_prices = df['close'].dropna()
        if len(close_prices) < 20:
            return "neutral"
            
        current_price = close_prices.iloc[-1]
        
        # Check if the 50-day MA is above the 200-day MA (golden cross)
        if ma50 > ma200 and current_price > ma50:
            return "bull"
        # Check if the 50-day MA is below the 200-day MA (death cross)
        elif ma50 < ma200 and current_price < ma50:
            return "bear"
        # Otherwise, consider it neutral
        else:
            return "neutral"
    
    def _calculate_confidence(self, 
                            diff: Decimal, 
                            recent_volumes: List[int], 
                            avg_volume: float,
                            rsi: Decimal,
                            macd_hist: Decimal,
                            atr: Decimal,
                            price: Decimal) -> Decimal:
        """
        Calculate confidence score based on multiple factors.
        
        Args:
            diff: Difference between fast and slow MA
            recent_volumes: Recent volume data
            avg_volume: Average volume
            rsi: Current RSI value
            macd_hist: Current MACD histogram value
            atr: Current ATR value
            price: Current price
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from MA crossover strength
        base_confidence = min(abs(diff) / Decimal("0.05"), Decimal("1.0"))
        
        # Volume confidence
        volume_ratio = sum(recent_volumes) / (len(recent_volumes) * avg_volume) if avg_volume > 0 else Decimal("0.5")
        volume_confidence = min(volume_ratio / Decimal(str(self.volume_threshold)), Decimal("1.0"))
        
        # RSI confidence - stronger signals at extremes
        rsi_confidence = Decimal("0.5")  # Neutral by default
        if rsi <= Decimal("30"):  # Oversold (good for buy)
            rsi_confidence = Decimal("1.0") - (rsi / Decimal("30"))
        elif rsi >= Decimal("70"):  # Overbought (good for sell)
            rsi_confidence = (rsi - Decimal("70")) / Decimal("30")
        
        # MACD confidence - stronger when histogram is larger
        macd_confidence = min(abs(macd_hist) / Decimal("0.02"), Decimal("1.0"))
        
        # Volatility confidence - more confident with appropriate volatility
        volatility = atr / price if price > 0 else Decimal("0")
        vol_confidence = Decimal("1.0")
        if volatility > Decimal("0.03"):  # High volatility is less confident
            vol_confidence = Decimal("0.7")
        
        # Combine all confidence factors with weights
        weights = {
            'base': Decimal("0.3"),
            'volume': Decimal("0.2"),
            'rsi': Decimal("0.2"),
            'macd': Decimal("0.2"),
            'volatility': Decimal("0.1")
        }
        
        combined_confidence = (
            base_confidence * weights['base'] +
            volume_confidence * weights['volume'] +
            rsi_confidence * weights['rsi'] +
            macd_confidence * weights['macd'] +
            vol_confidence * weights['volatility']
        )
        
        # Ensure the confidence is between 0.1 and 1
        return max(min(combined_confidence, Decimal("1.0")), Decimal("0.1"))
    
    def _calculate_moving_average(self, data: List[Decimal], period: int) -> Decimal:
        """
        Calculate the simple moving average for a given period.
        
        Args:
            data: List of price data
            period: Period for the moving average
            
        Returns:
            Decimal moving average value
        """
        if len(data) < period:
            # Return the average of available data if not enough data
            return sum(data) / Decimal(len(data))
            
        # Calculate SMA for the period
        sma_subset = data[-period:]
        return sum(sma_subset) / Decimal(period)
    
    def get_stop_loss(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate stop loss for a signal based on volatility.
        
        Args:
            data: List of MarketData objects
            signal: Generated signal
            
        Returns:
            Stop loss percentage
        """
        # Extract recent price data
        prices = [float(d.close_price) for d in data[-self.atr_period:]]
        
        # Calculate ATR
        tr_values = []
        for i in range(1, len(prices)):
            high = float(data[-i].high_price)
            low = float(data[-i].low_price)
            prev_close = prices[i-1]
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        
        # Calculate ATR
        atr = sum(tr_values) / len(tr_values) if tr_values else 0
        
        # Current price
        current_price = float(data[-1].close_price)
        
        # Calculate volatility-based stop loss (ATR multiplier)
        if atr > 0:
            # ATR as percentage of price
            atr_pct = atr / current_price
            
            # 2 * ATR for stop loss, but cap at 3% for risk management
            stop_loss_pct = min(atr_pct * 2, 0.03)
            
            # Ensure minimum stop loss
            return max(stop_loss_pct, 0.015)  # At least 1.5% stop loss
        else:
            # Default to 2% if ATR calculation fails
            return 0.02
    
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate take profit for a signal based on volatility and risk:reward ratio.
        
        Args:
            data: List of MarketData objects
            signal: Generated signal
            
        Returns:
            Take profit percentage
        """
        # Get stop loss
        stop_loss_pct = self.get_stop_loss(data, signal)
        
        # Calculate take profit based on risk:reward ratio of 2.5:1
        return stop_loss_pct * 2.5
    
    def _convert_to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """
        Convert list of MarketData objects to pandas DataFrame.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'open': float(d.open_price),
                'high': float(d.high_price),
                'low': float(d.low_price),
                'close': float(d.close_price),
                'volume': d.volume
            }
            for d in data
        ])
        
        return df 