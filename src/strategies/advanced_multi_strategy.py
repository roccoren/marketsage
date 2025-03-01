import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base import Strategy
from ..utils.technical_indicators import TechnicalIndicators
import logging

logger = logging.getLogger(__name__)

class AdvancedMultiStrategy(Strategy):
    """
    Advanced multi-indicator trading strategy that combines several technical indicators
    to generate buy and sell signals with risk management features.
    
    This strategy combines:
    - Trend indicators (SMA, EMA)
    - Momentum indicators (RSI, MACD)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume-based indicators
    
    It also includes risk management features:
    - Position sizing based on ATR
    - Stop-loss and trailing stop mechanisms
    - Profit targets
    - Risk-adjusted entry/exit criteria
    """
    
    def __init__(
        self,
        # Moving Average parameters
        sma_short_period: int = 20,
        sma_long_period: int = 50,
        ema_short_period: int = 12,
        ema_long_period: int = 26,
        
        # RSI parameters
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        
        # Bollinger Bands parameters
        bb_period: int = 20,
        bb_std: float = 2.0,
        
        # MACD parameters
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        
        # ADX parameters
        adx_period: int = 14,
        adx_threshold: int = 25,
        
        # Volume parameters
        volume_ma_period: int = 20,
        volume_threshold: float = 1.5,
        
        # ATR parameters for stop loss and position sizing
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        trailing_stop_factor: float = 3.0,
        
        # Strategy weights
        trend_weight: float = 0.25,
        momentum_weight: float = 0.25,
        volatility_weight: float = 0.25,
        volume_weight: float = 0.25,
        
        # Signal thresholds
        buy_threshold: float = 0.6,
        sell_threshold: float = -0.6,
        
        # Strategy parameters
        min_score_buy: int = 3,
        min_score_sell: int = 3,
        risk_percent: float = 1.0,
        profit_target_factor: float = 2.0
    ):
        """
        Initialize strategy with parameters
        
        Args:
            sma_short_period: Short simple moving average period
            sma_long_period: Long simple moving average period
            ema_short_period: Short exponential moving average period
            ema_long_period: Long exponential moving average period
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            adx_period: ADX period
            adx_threshold: ADX trend strength threshold
            volume_ma_period: Volume moving average period
            volume_threshold: Volume spike threshold ratio
            atr_period: ATR calculation period
            atr_multiplier: Multiplier for ATR stop loss
            trailing_stop_factor: Multiplier for ATR trailing stop
            trend_weight: Weight of trend indicators in overall signal
            momentum_weight: Weight of momentum indicators in overall signal
            volatility_weight: Weight of volatility indicators in overall signal
            volume_weight: Weight of volume indicators in overall signal
            buy_threshold: Combined signal threshold for buy signals
            sell_threshold: Combined signal threshold for sell signals
            min_score_buy: Minimum indicator score for buy signals
            min_score_sell: Minimum indicator score for sell signals
            risk_percent: Risk per trade as percentage of capital
            profit_target_factor: Profit target as multiple of risk
        """
        self.params = {
            'sma_short_period': sma_short_period,
            'sma_long_period': sma_long_period,
            'ema_short_period': ema_short_period,
            'ema_long_period': ema_long_period,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'volume_ma_period': volume_ma_period,
            'atr_period': atr_period
        }
        
        # Strategy weights (must sum to 1.0)
        self.weights = {
            'trend': trend_weight,
            'momentum': momentum_weight,
            'volatility': volatility_weight,
            'volume': volume_weight
        }
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Strategy weights sum to {weight_sum}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        # Signal thresholds
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # Risk management
        self.min_score_buy = min_score_buy
        self.min_score_sell = min_score_sell
        self.volume_threshold = volume_threshold
        self.atr_multiplier = atr_multiplier
        self.trailing_stop_factor = trailing_stop_factor
        self.risk_percent = risk_percent
        self.profit_target_factor = profit_target_factor
        
        # State variables
        self.current_position = None
        self.trailing_stop = None
        self.entry_price = None
        self.stop_loss = None
        self.profit_target = None
        self.last_atr = None
        self.highest_price = None
        
        logger.debug("AdvancedMultiStrategy initialized with parameters: %s", self.params)
        logger.debug("Strategy weights: %s", self.weights)
        logger.debug(f"Signal thresholds - Buy: {buy_threshold}, Sell: {sell_threshold}")

    def analyze(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze price data and generate trading signals based on multiple indicators
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            List of signal dictionaries
        """
        logger.debug(f"Analyzing data frame with {len(data)} bars")
        
        # Check for minimum data requirements
        min_periods = max(
            self.params['sma_long_period'],
            self.params['ema_long_period'],
            self.params['macd_slow'] + self.params['macd_signal'],
            self.params['adx_period']
        )
        
        if len(data) < min_periods + 10:  # Add buffer for calculations
            logger.debug(f"Insufficient data for analysis: {len(data)} < {min_periods + 10}")
            return []
        
        try:
            # Calculate all technical indicators
            indicators = self._calculate_indicators(data)
            
            # Get latest price and volume data
            current_bar = data.iloc[-1]
            price = current_bar['Close']
            volume = current_bar['Volume']
            
            # Calculate average volume
            avg_volume = data['Volume'].rolling(window=self.params['volume_ma_period']).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate component signals (normalized between -1 and 1)
            trend_signal = self._calculate_trend_signal(indicators)
            momentum_signal = self._calculate_momentum_signal(indicators)
            volatility_signal = self._calculate_volatility_signal(indicators, price)
            volume_signal = self._calculate_volume_signal(volume_ratio, indicators)
            
            # Calculate combined weighted signal
            combined_signal = (
                self.weights['trend'] * trend_signal +
                self.weights['momentum'] * momentum_signal +
                self.weights['volatility'] * volatility_signal +
                self.weights['volume'] * volume_signal
            )
            
            # Calculate indicator score (-N to N where N is number of indicators)
            indicator_score = self._calculate_indicator_score(indicators, price, volume_ratio)
            
            # Store ATR for position sizing
            self.last_atr = indicators.get('atr', [0.0])[-1]
            
            # Generate signals based on combined signal and score
            signals = []
            
            # Check for buy signal
            if (combined_signal > self.buy_threshold and 
                indicator_score >= self.min_score_buy):
                
                signal = {
                    'Type': 'BUY',
                    'Date': current_bar.name,
                    'Price': price,
                    'Signal': combined_signal,
                    'Score': indicator_score,
                    'ATR': self.last_atr
                }
                signals.append(signal)
                logger.debug(f"BUY signal generated: price={price}, signal={combined_signal:.2f}, score={indicator_score}")
                
                # Calculate stop loss and profit target
                stop_loss = price - (self.atr_multiplier * self.last_atr)
                profit_target = price + (self.profit_target_factor * self.atr_multiplier * self.last_atr)
                
                signal['StopLoss'] = stop_loss
                signal['ProfitTarget'] = profit_target
                signal['Risk'] = price - stop_loss
                signal['Reward'] = profit_target - price
                signal['RR'] = (profit_target - price) / (price - stop_loss) if price > stop_loss else 0
            
            # Check for sell signal
            elif (combined_signal < self.sell_threshold and 
                  indicator_score <= -self.min_score_sell):
                
                signal = {
                    'Type': 'SELL',
                    'Date': current_bar.name,
                    'Price': price,
                    'Signal': combined_signal,
                    'Score': indicator_score,
                    'ATR': self.last_atr
                }
                signals.append(signal)
                logger.debug(f"SELL signal generated: price={price}, signal={combined_signal:.2f}, score={indicator_score}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in strategy analysis: {str(e)}", exc_info=True)
            return []
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all technical indicators needed for the strategy
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of indicator Series
        """
        indicators = {}
        
        # Moving Averages
        indicators['sma_short'] = self._calculate_sma(data['Close'], self.params['sma_short_period'])
        indicators['sma_long'] = self._calculate_sma(data['Close'], self.params['sma_long_period'])
        indicators['ema_short'] = self._calculate_ema(data['Close'], self.params['ema_short_period'])
        indicators['ema_long'] = self._calculate_ema(data['Close'], self.params['ema_long_period'])
        
        # Crossover signals
        indicators['sma_crossover'] = self._calculate_crossover(
            indicators['sma_short'],
            indicators['sma_long']
        )
        
        indicators['ema_crossover'] = self._calculate_crossover(
            indicators['ema_short'],
            indicators['ema_long']
        )
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(data['Close'], self.params['rsi_period'])
        
        # Bollinger Bands
        bb_result = self._calculate_bollinger_bands(
            data['Close'],
            self.params['bb_period'],
            self.params['bb_std']
        )
        indicators.update(bb_result)
        
        # MACD
        macd_result = self._calculate_macd(
            data['Close'],
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        indicators.update(macd_result)
        
        # ADX
        indicators['adx'] = self._calculate_adx(
            data['High'],
            data['Low'],
            data['Close'],
            self.params['adx_period']
        )
        
        # ATR for position sizing and stops
        indicators['atr'] = self._calculate_atr(
            data['High'],
            data['Low'],
            data['Close'],
            self.params['atr_period']
        )
        
        # Volume indicators
        indicators['volume_ma'] = self._calculate_sma(data['Volume'], self.params['volume_ma_period'])
        indicators['volume_ratio'] = data['Volume'] / indicators['volume_ma']
        
        # On-Balance Volume (OBV)
        indicators['obv'] = self._calculate_obv(data)
        
        return indicators
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_crossover(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Calculate crossover signals: 1 for bullish, -1 for bearish, 0 for no crossover"""
        crossover = pd.Series(0, index=fast_ma.index)
        
        prev_diff = fast_ma.shift(1) - slow_ma.shift(1)
        curr_diff = fast_ma - slow_ma
        
        # Bullish crossover (fast crosses above slow)
        crossover[(prev_diff <= 0) & (curr_diff > 0)] = 1
        
        # Bearish crossover (fast crosses below slow)
        crossover[(prev_diff >= 0) & (curr_diff < 0)] = -1
        
        return crossover
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(
        self,
        series: pd.Series,
        period: int,
        std_dev: float
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return {
            'bb_middle': middle_band,
            'bb_upper': upper_band,
            'bb_lower': lower_band
        }
    
    def _calculate_macd(
        self,
        series: pd.Series,
        fast_period: int,
        slow_period: int,
        signal_period: int
    ) -> Dict[str, pd.Series]:
        """Calculate MACD and signal line"""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Calculate crossover
        prev_diff = macd_line.shift(1) - signal_line.shift(1)
        curr_diff = macd_line - signal_line
        
        crossover = pd.Series(0, index=macd_line.index)
        crossover[(prev_diff <= 0) & (curr_diff > 0)] = 1  # Bullish
        crossover[(prev_diff >= 0) & (curr_diff < 0)] = -1  # Bearish
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram,
            'macd_crossover': crossover
        }
    
    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Plus Directional Movement
        plus_dm = high - high.shift(1)
        plus_dm[plus_dm < 0] = 0
        minus_dm = low.shift(1) - low
        minus_dm[minus_dm < 0] = 0
        
        # When current high - prev high < prev low - current low, +DM = 0
        plus_dm[(high - high.shift(1)) < (low.shift(1) - low)] = 0
        # When prev low - current low < current high - prev high, -DM = 0
        minus_dm[(low.shift(1) - low) < (high - high.shift(1))] = 0
        
        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV)"""
        close = data['Close']
        volume = data['Volume']
        
        obv = pd.Series(0.0, index=close.index)
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def _calculate_trend_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate the trend component signal (-1 to 1)
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            float: Trend signal between -1 (strong downtrend) and 1 (strong uptrend)
        """
        # MA crossover signals (already between -1 and 1)
        sma_signal = indicators['sma_crossover'].iloc[-1]
        ema_signal = indicators['ema_crossover'].iloc[-1]
        
        # Combine the signals, giving more weight to the EMA as it's more responsive
        trend_signal = (0.4 * sma_signal + 0.6 * ema_signal)
        
        # Scale from -1 to 1
        return np.clip(trend_signal, -1, 1)
    
    def _calculate_momentum_signal(self, indicators: Dict[str, pd.Series]) -> float:
        """
        Calculate the momentum component signal (-1 to 1)
        
        Args:
            indicators: Dictionary of calculated indicators
            
        Returns:
            float: Momentum signal between -1 (strong bearish momentum) and 1 (strong bullish momentum)
        """
        # RSI (0-100) to signal (-1 to 1)
        rsi_value = indicators['rsi'].iloc[-1]
        rsi_signal = 0.0
        
        if rsi_value < self.params['rsi_oversold']:
            # Oversold (bullish)
            rsi_signal = -1 + (rsi_value / self.params['rsi_oversold'])
        elif rsi_value > self.params['rsi_overbought']:
            # Overbought (bearish)
            rsi_signal = 1 - ((100 - rsi_value) / (100 - self.params['rsi_overbought']))
        else:
            # Neutral zone - scale to -0.5 to 0.5
            rsi_range = self.params['rsi_overbought'] - self.params['rsi_oversold']
            rsi_signal = (rsi_value - self.params['rsi_oversold']) / rsi_range * 2 - 1
            rsi_signal *= 0.5  # Scale to -0.5 to 0.5
            
        # MACD signal (already between -1 and 1)
        macd_signal = indicators['macd_crossover'].iloc[-1]
        
        # Combine the signals
        momentum_signal = 0.5 * (rsi_signal + macd_signal)
        
        # Scale from -1 to 1
        return np.clip(momentum_signal, -1, 1)
    
    def _calculate_volatility_signal(
        self,
        indicators: Dict[str, pd.Series],
        price: float
    ) -> float:
        """
        Calculate the volatility component signal (-1 to 1)
        
        Args:
            indicators: Dictionary of calculated indicators
            price: Current price
            
        Returns:
            float: Volatility signal between -1 (overbought) and 1 (oversold)
        """
        # Bollinger Band position
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        bb_width = bb_upper - bb_lower
        
        if bb_width == 0:
            return 0
        
        # Calculate where price is within the bands (-1 to 1)
        # Where -1 is at upper band, 0 at middle band, 1 at lower band
        bb_position = 2 * ((bb_upper - price) / bb_width) - 1
        
        # ADX signal (0 for weak trend, up to 1 for strong trend)
        adx_value = indicators['adx'].iloc[-1]
        adx_signal = min(adx_value / 100, 1)  # Normalize to 0-1
        
        # Combine signals: position in BB matters more in strong trends
        # For weak trends, mean reversion is more likely (contrarian)
        # For strong trends, trend continuation is more likely (momentum)
        if adx_value > self.params['adx_threshold']:
            # Strong trend - less contrarian
            volatility_signal = bb_position * 0.5
        else:
            # Weak trend - more contrarian
            volatility_signal = bb_position
        
        # Scale from -1 to 1
        return np.clip(volatility_signal, -1, 1)
    
    def _calculate_volume_signal(
        self,
        volume_ratio: float,
        indicators: Dict[str, pd.Series]
    ) -> float:
        """
        Calculate the volume component signal (-1 to 1)
        
        Args:
            volume_ratio: Current volume / average volume
            indicators: Dictionary of calculated indicators
            
        Returns:
            float: Volume signal between -1 and 1
        """
        # Volume ratio signal (0 to 1)
        volume_strength = min((volume_ratio - 1) / 2, 1.0) if volume_ratio > 1 else 0
        
        # OBV trend (difference from SMA)
        obv = indicators['obv'].iloc[-1]
        obv_sma = indicators['obv'].rolling(window=10).mean().iloc[-1]
        obv_trend = 1 if obv > obv_sma else -1 if obv < obv_sma else 0
        
        # OBV momentum (rate of change)
        obv_change = indicators['obv'].pct_change(periods=5).iloc[-1]
        obv_signal = np.clip(obv_change * 10, -1, 1)  # Scale to -1 to 1
        
        # Combine volume signals
        volume_signal = (
            0.3 * volume_strength * obv_trend +  # Volume strength with direction
            0.7 * obv_signal                     # OBV momentum
        )
        
        # Scale from -1 to 1
        return np.clip(volume_signal, -1, 1)
    
    def _calculate_indicator_score(
        self,
        indicators: Dict[str, pd.Series],
        price: float,
        volume_ratio: float
    ) -> int:
        """
        Calculate a score based on how many indicators are bullish/bearish
        
        Args:
            indicators: Dictionary of calculated indicators
            price: Current price
            volume_ratio: Current volume / average volume
            
        Returns:
            int: Score between -N and N (negative is bearish, positive is bullish)
        """
        score = 0
        
        # Trend indicators
        if price > indicators['sma_short'].iloc[-1]:
            score += 1
        elif price < indicators['sma_short'].iloc[-1]:
            score -= 1
            
        if price > indicators['sma_long'].iloc[-1]:
            score += 1
        elif price < indicators['sma_long'].iloc[-1]:
            score -= 1
            
        if indicators['sma_crossover'].iloc[-1] > 0:
            score += 1
        elif indicators['sma_crossover'].iloc[-1] < 0:
            score -= 1
            
        if indicators['ema_crossover'].iloc[-1] > 0:
            score += 1
        elif indicators['ema_crossover'].iloc[-1] < 0:
            score -= 1
            
        # Momentum indicators
        if indicators['rsi'].iloc[-1] < self.params['rsi_oversold']:
            score += 1
        elif indicators['rsi'].iloc[-1] > self.params['rsi_overbought']:
            score -= 1
            
        if indicators['macd_crossover'].iloc[-1] > 0:
            score += 1
        elif indicators['macd_crossover'].iloc[-1] < 0:
            score -= 1
            
        if indicators['macd_hist'].iloc[-1] > 0:
            score += 1
        elif indicators['macd_hist'].iloc[-1] < 0:
            score -= 1
            
        # Volatility indicators
        if price < indicators['bb_lower'].iloc[-1]:
            score += 1
        elif price > indicators['bb_upper'].iloc[-1]:
            score -= 1
            
        # Volume indicators
        if volume_ratio > self.volume_threshold:
            if price > indicators['ema_short'].iloc[-1]:
                score += 1
            elif price < indicators['ema_short'].iloc[-1]:
                score -= 1
                
        # ADX - trend strength
        if indicators['adx'].iloc[-1] > self.params['adx_threshold']:
            # Strong trend - amplify existing score
            score = int(score * 1.25)
            
        return score