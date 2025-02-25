import logging
from strategies.strategy_rsi_stochastic import RSIStochStrategy
from strategies.strategy_atr_moving_average import ATRMovingAverageStrategy
from strategies.strategy_adx_macd_cci import ADXMACDCCI_Strategy
from strategies.strategy_obv import OBVStrategy

# Configure logging to display debug information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_strategies():
    strategies = [
        {
            "name": "RSI/Stochastic Strategy",
            "instance": RSIStochStrategy(),
            "data": {"rsi": 75, "stochastic": 85}  # Example values: overbought condition
        },
        {
            "name": "ATR/Moving Average Strategy",
            "instance": ATRMovingAverageStrategy(),
            "data": {"atr": 1.5, "moving_average": 100, "price": 98}  # Price below MA by ATR margin triggers BUY
        },
        {
            "name": "ADX/MACD/CCI Strategy",
            "instance": ADXMACDCCI_Strategy(),
            "data": {"adx": 30, "macd": 1.2, "cci": -110}  # Strong trend with positive MACD, oversold CCI triggers BUY
        },
        {
            "name": "OBV Strategy",
            "instance": OBVStrategy(),
            "data": {"obv": 120000, "price": 105, "previous_obv": 110000, "previous_price": 100}  # Rising OBV with rising price triggers BUY
        }
    ]
    
    for strat_info in strategies:
        name = strat_info["name"]
        instance = strat_info["instance"]
        data = strat_info["data"]
        logging.info(f"Running {name}")
        signal = instance.generate_signal(data)
        print(f"{name} signal: {signal}")

if __name__ == "__main__":
    run_strategies()
