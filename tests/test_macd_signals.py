import unittest
import pandas as pd
import numpy as np
from strategies.macd import MACDStrategy

class TestMACDSignals(unittest.TestCase):
    def setUp(self):
        self.strategy = MACDStrategy()

    def test_seven_day_signal_detection(self):
        """Test detection of both buy and sell signals within 7 days."""
        # Create sequence with exact crossover points
        base_price = 100
        prices = []
        
        # Initialize with stable prices to establish baseline
        prices.extend([base_price] * 20)
        
        # Create downtrend to setup buy signal
        for i in range(10):
            prices.append(base_price * (1.0 - 0.01 * (i + 1)))  # -1% each day
        
        # Strong recovery to trigger buy signal
        current_price = prices[-1]
        for i in range(5):
            current_price *= 1.03  # +3% each day
            prices.append(current_price)
        
        # Brief consolidation
        prices.extend([current_price] * 3)
        
        # Sharp decline to trigger sell signal
        for i in range(5):
            current_price *= 0.96  # -4% each day
            prices.append(current_price)
        
        # Generate matching volume data
        base_volume = 100000
        volumes = []
        
        # Normal volume for initial period
        volumes.extend([base_volume] * 20)
        
        # Increasing volume during downtrend
        for i in range(10):
            volumes.append(base_volume * (1.2 + i * 0.1))
        
        # High volume during buy signal
        for _ in range(5):
            volumes.append(base_volume * 3.0)
        
        # Normal volume during consolidation
        volumes.extend([base_volume * 1.2] * 3)
        
        # High volume during sell signal
        for _ in range(5):
            volumes.append(base_volume * 2.5)
        
        # Create DataFrame
        dates = pd.date_range(start='2024-01-01', periods=len(prices), freq='D')
        data = pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        # Print recent data for inspection
        print("\nLast 10 days of test data:")
        last_10 = data.tail(10)
        print("\nPrices:")
        print(last_10['Close'].to_frame())
        print("\nVolumes:")
        print(last_10['Volume'].to_frame())
        
        # Calculate and print MACD values
        macd_data = self.strategy.calculate_macd(data)
        print("\nMACD Values for last 10 days:")
        last_10_macd = pd.DataFrame({
            'MACD': macd_data['MACD'].tail(10),
            'Signal': macd_data['Signal'].tail(10),
            'Histogram': macd_data['Histogram'].tail(10)
        })
        print(last_10_macd)
        
        # Print crossover points
        print("\nMACD Crossover Points:")
        for i in range(1, len(macd_data)):
            prev_diff = macd_data['MACD'].iloc[i-1] - macd_data['Signal'].iloc[i-1]
            curr_diff = macd_data['MACD'].iloc[i] - macd_data['Signal'].iloc[i]
            if (prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0):
                print(f"Date: {data.index[i].strftime('%Y-%m-%d')}")
                print(f"MACD/Signal: {macd_data['MACD'].iloc[i]:.4f}/{macd_data['Signal'].iloc[i]:.4f}")
                print(f"Volume Ratio: {data['Volume'].iloc[i]/data['Volume'].iloc[i-5:i].mean():.2f}\n")
        
        # Get signals
        signals = self.strategy.analyze(data)
        
        # Validate signal detection
        self.assertTrue(len(signals) > 0, "Should detect signals in 7-day window")
        
        if signals:
            print("\nDetected Signals:")
            for signal in signals:
                print(f"\n{signal['Type']} Signal on {signal['SignalDate']}:")
                print(f"Price: {signal['Price']:.2f}")
                print(f"Volume Ratio: {signal['VolumeRatio']:.2f}")
                print(f"RSI: {signal['RSI']:.2f}")
                print(f"MACD: {signal['MACD']:.4f}")
                print(f"Signal Line: {signal['Signal']:.4f}")
        
        # Validate signal contents
        self.assertGreater(
            len([s for s in signals if s['Type'] == 'BUY']), 0,
            "Should detect at least one buy signal"
        )
        self.assertGreater(
            len([s for s in signals if s['Type'] == 'SELL']), 0,
            "Should detect at least one sell signal"
        )
    
    def test_golden_cross_detection(self):
        """Test basic golden cross detection."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = []
        prices.extend([100] * 25)
        prices.extend([100 - i for i in range(10)])
        prices.extend([90] * 10)
        prices.extend([90 + i*2 for i in range(5)])
        
        data = pd.DataFrame({
            'Close': prices,
            'Volume': [100000] * len(prices)
        }, index=dates)
        
        self.assertTrue(
            self.strategy.detect_recent_golden_cross(data),
            "Should detect golden cross in recent data"
        )
        
    def test_no_golden_cross(self):
        """Test flat price data generates no golden cross."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'Close': [100] * 50,
            'Volume': [100000] * 50
        }, index=dates)
        
        self.assertFalse(
            self.strategy.detect_recent_golden_cross(data),
            "Should not detect golden cross in flat price data"
        )
        
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Close': [100] * 10,
            'Volume': [100000] * 10
        }, index=dates)
        
        self.assertFalse(
            self.strategy.detect_recent_golden_cross(data),
            "Should return False for insufficient data"
        )
        
if __name__ == '__main__':
    unittest.main()
