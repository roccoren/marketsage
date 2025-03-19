import pandas as pd
from typing import List, Dict, Optional
from .base import Strategy

class PriceCrossoverStrategy(Strategy):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name="PriceCrossover")
        self.params = params or {
            'short_window': 20,
            'long_window': 50,
        }

    def analyze(self, data: pd.DataFrame) -> List[Dict]:
        """
        Analyze price data for simple moving average crossover signals.
        """
        signals = []
        if len(data) < self.params['long_window']:
            return signals

        short_ma = data['Close'].rolling(window=self.params['short_window']).mean()
        long_ma = data['Close'].rolling(window=self.params['long_window']).mean()

        # Check for crossover signal
        if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
            signal = {
                'Price': data['Close'].iloc[-1],
                'ShortMA': short_ma.iloc[-1],
                'LongMA': long_ma.iloc[-1],
                'Type': 'BUY',
            }
            signals.append(signal)

        return signals