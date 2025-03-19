#!/usr/bin/env python3
import yfinance as yf
import json
import os
from pathlib import Path
from datetime import datetime

# Create directories if they don't exist
os.makedirs("data/NYSE", exist_ok=True)

# Download AAPL data
print("Downloading AAPL data...")
data = yf.download('AAPL', period='1y')

# Convert to the format expected by the exchange data loader
data = data.reset_index()

# Convert DataFrame to serializable format
records = []
for _, row in data.iterrows():
    record = {
        'date': row['Date'].strftime('%Y-%m-%d'),
        'open': float(row['Open']),
        'high': float(row['High']),
        'low': float(row['Low']),
        'close': float(row['Close']),
        'volume': float(row['Volume']),
    }
    records.append(record)

# Save data
with open('data/NYSE/AAPL.json', 'w') as f:
    json.dump({'history': records}, f, indent=2)

print(f"Downloaded AAPL data and saved to data/NYSE/AAPL.json") 