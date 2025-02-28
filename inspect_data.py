import gzip
import json
from pathlib import Path

# Read a sample data file
data_file = Path("test_data/HKEX/9988.HK.json.gz")

with gzip.open(data_file, 'rt') as f:
    data = json.load(f)
    print("Data structure:")
    print(json.dumps(data if isinstance(data, dict) else data[0], indent=2))