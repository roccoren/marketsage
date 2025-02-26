from strategies.base import Strategy
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class ADXMACDCCI_Strategy(Strategy):
    """Strategy combining ADX, MACD, and CCI indicators."""
    def __init__(self):
        super().__init__("ADX/MACD/CCI Strategy")
        self.logger = logging.getLogger(__name__)

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX indicator with proper DataFrame alignment.
        
        Args:
            data: DataFrame with OHLCV data
            period: ADX calculation period
            
        Returns:
            Series containing ADX values
        """
        # Ensure data is properly aligned and has no gaps
        data = data.copy()  # Create a copy to avoid modifying original
        data = data.sort_index()
        
        # Handle duplicate indices before reindexing
        if data.index.duplicated().any():
            data = data[~data.index.duplicated(keep='last')]
            
        date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(date_range)
        data = data.ffill()  # Forward fill missing values
        data = data.dropna()  # Drop any remaining NaN values
        
        # Calculate +DM and -DM
        high_diff = data['High'].diff()
        low_diff = -data['Low'].diff()
        
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        # Handle Series comparisons properly
        hdm = high_diff > low_diff
        plus_dm.loc[hdm] = high_diff.loc[hdm]
        plus_dm.loc[high_diff <= 0] = 0.0
        
        ldm = low_diff > high_diff
        minus_dm.loc[ldm] = low_diff.loc[ldm]
        minus_dm.loc[low_diff <= 0] = 0.0
        
        # Calculate TR
        tr = pd.DataFrame(index=data.index)
        tr['hl'] = data['High'] - data['Low']
        tr['hc'] = (data['High'] - data['Close'].shift()).abs()
        tr['lc'] = (data['Low'] - data['Close'].shift()).abs()
        tr = tr.max(axis=1)
        
        # Calculate smoothed values with min_periods to handle initial NaN values
        tr_smooth = tr.rolling(window=period, min_periods=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).mean()
        
        # Calculate +DI and -DI with zero division handling
        plus_di = pd.Series(0.0, index=data.index)
        minus_di = pd.Series(0.0, index=data.index)
        
        # Ensure all Series are aligned
        tr_smooth = tr_smooth.reindex(data.index)
        plus_dm_smooth = plus_dm_smooth.reindex(data.index)
        minus_dm_smooth = minus_dm_smooth.reindex(data.index)
        
        valid_tr = tr_smooth > 0
        plus_di[valid_tr] = 100 * plus_dm_smooth[valid_tr] / tr_smooth[valid_tr]
        minus_di[valid_tr] = 100 * minus_dm_smooth[valid_tr] / tr_smooth[valid_tr]
        
        # Calculate DX and ADX with zero division handling
        dx = pd.Series(0.0, index=data.index)
        valid_sum = (plus_di + minus_di) > 0
        dx[valid_sum] = 100 * abs(plus_di - minus_di)[valid_sum] / (plus_di + minus_di)[valid_sum]
        
        adx = dx.rolling(window=period, min_periods=1).mean()
        
        return adx

    def _calculate_macd(self, data: pd.DataFrame) -> pd.Series:
        """Calculate MACD with proper index alignment."""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd

    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate CCI with proper DataFrame alignment."""
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        ma_tp = tp.rolling(window=period).mean()
        mean_dev = abs(tp - ma_tp).rolling(window=period).mean()
        cci = (tp - ma_tp) / (0.015 * mean_dev)
        return cci

    def analyze(self, data: pd.DataFrame) -> Optional[List[Dict]]:
        """Analyze price data to generate trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of signal dictionaries or None if no signals
        """
        # Calculate indicators
        adx = self._calculate_adx(data)
        macd = self._calculate_macd(data)
        cci = self._calculate_cci(data)

        # Filter for last 7 trading days
        last_7_days = datetime.now() - timedelta(days=7)
        data = data[data.index >= last_7_days]
        self.logger.info(f"Analyzing last 7 days of data from {data.index[0]} to {data.index[-1]}")
        
        signals = []
        if not adx.empty and not macd.empty and not cci.empty:
            # Ensure all indicators are properly aligned
            try:
                # Get the last values and convert to float carefully
                latest_adx = adx.iloc[-1]
                latest_macd = macd.iloc[-1]
                latest_cci = cci.iloc[-1]
                
                # Handle Series or scalar values
                if isinstance(latest_adx, pd.Series):
                    latest_adx = float(latest_adx.iloc[0])
                else:
                    latest_adx = float(latest_adx)
                    
                if isinstance(latest_macd, pd.Series):
                    latest_macd = float(latest_macd.iloc[0])
                else:
                    latest_macd = float(latest_macd)
                    
                if isinstance(latest_cci, pd.Series):
                    latest_cci = float(latest_cci.iloc[0])
                else:
                    latest_cci = float(latest_cci)
            except (ValueError, TypeError):
                # If conversion fails, try accessing numpy values
                latest_adx = float(adx.iloc[-1].values[0])
                latest_macd = float(macd.iloc[-1].values[0])
                latest_cci = float(cci.iloc[-1].values[0])
                
            # Calculate entry/exit prices
            current_price = float(data['Close'].iloc[-1])
            atr = data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()
            current_atr = float(atr.iloc[-1])

            # Generate signals based on the indicator values
            if latest_adx > 25:  # Strong trend
                    if latest_macd > 0 and latest_cci < -100:
                        stop_loss = current_price - current_atr
                        take_profit = current_price + (current_atr * 2)
                        signals.append({
                            "Strategy": self.name,
                            "Signal": "BUY",
                            "Timestamp": data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                            "Strength": "Strong",
                            "Reason": f"Strong trend (ADX: {latest_adx:.2f}) with positive MACD ({latest_macd:.2f}) and oversold CCI ({latest_cci:.2f})",
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "trade_timing": {
                                "optimal_entry": "Market open next session",
                                "valid_until": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                            }
                        })
                    elif latest_macd < 0 and latest_cci > 100:
                        stop_loss = current_price + current_atr
                        take_profit = current_price - (current_atr * 2)
                        signals.append({
                            "Strategy": self.name,
                            "Signal": "SELL",
                            "Timestamp": data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                            "Strength": "Strong",
                            "Reason": f"Strong trend (ADX: {latest_adx:.2f}) with negative MACD ({latest_macd:.2f}) and overbought CCI ({latest_cci:.2f})",
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "trade_timing": {
                                "optimal_entry": "Market close current session",
                                "valid_until": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                            }
                        })
                
        return signals if signals else None

    def generate_signal(self, data):
        self.logger.info("Evaluating ADX with MACD and CCI conditions")
        adx = data.get('adx')
        macd = data.get('macd')
        cci = data.get('cci')
        self.logger.debug(f"ADX: {adx}, MACD: {macd}, CCI: {cci}")
        if adx is not None:
            if adx > 25:
                if macd is not None:
                    if macd > 0:
                        self.logger.info("Strong trend with positive MACD: BUY signal")
                        return "buy"
                    elif macd < 0:
                        self.logger.info("Strong trend with negative MACD: SELL signal")
                        return "sell"
                if cci is not None:
                    if cci < -100:
                        self.logger.info("Strong trend with low CCI (oversold): BUY signal")
                        return "buy"
                    elif cci > 100:
                        self.logger.info("Strong trend with high CCI (overbought): SELL signal")
                        return "sell"
            else:
                self.logger.info("Weak trend detected based on ADX")
        self.logger.info("No signal detected: hold position")
        return "hold"
