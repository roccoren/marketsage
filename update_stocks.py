from scanner import StockScanner
import json
from types import SimpleNamespace
import re

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

with open('config.json') as f:
    config_dict = json.load(f)
    
# Convert camelCase keys to snake_case
snake_config = {camel_to_snake(k): v for k, v in config_dict.items()}

# Convert dictionary to object with attributes
config = SimpleNamespace(**snake_config)

scanner = StockScanner(config)
scanner.update_stock_data()