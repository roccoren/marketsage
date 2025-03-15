"""Unit tests for optimized market processor module."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import json
import os

from src.market_system.optimized_market_processor import OptimizedMarketProcessor, PerformanceMetrics

class TestOptimizedMarketProcessor(unittest.TestCase):
    """Test cases for OptimizedMarketProcessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data directory and sample market data."""
        # Create temporary directory
        cls.test_dir = tempfile.mkdtemp()
        cls.market_dir = Path(cls.test_dir) / 'test_market'
        cls.symbol_dir = cls.market_dir / 'TEST.SYM'
        cls.symbol_dir.mkdir(parents=True)
        
        # Create sample data for multiple days
        dates = pd.date_range('2025-03-01', '2025-03-15')
        base_price = 100.0
        
        for date in dates:
            data = {
                'date': date.strftime('%Y-%m-%d'),
                'symbol': 'TEST.SYM',
                'market': 'test_market',
                'open': base_price * (1 + np.random.normal(0, 0.01)),
                'high': base_price * (1 + np.random.normal(0, 0.02)),
                'low': base_price * (1 - np.random.normal(0, 0.02)),
                'close': base_price * (1 + np.random.normal(0, 0.01)),
                'volume': int(np.random.randint(1000, 10000)),
                'trading_status': 'active',
                'currency': 'USD',
                'metrics': {
                    'indicators': {
                        'vwap': base_price * (1 + np.random.normal(0, 0.005)),
                        'turnover': int(base_price * np.random.randint(1000, 10000))
                    },
                    'returns': {
                        'daily': np.random.normal(0, 0.01),
                        'intraday_range': np.random.uniform(0.01, 0.03)
                    }
                }
            }
            
            # Write daily data file
            data_file = cls.symbol_dir / f"{date.strftime('%Y-%m-%d')}.json"
            with open(data_file, 'w') as f:
                json.dump(data, f)
    
    def setUp(self):
        """Initialize processor for each test."""
        self.processor = OptimizedMarketProcessor(self.test_dir)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        metrics = PerformanceMetrics()
        
        # Test memory recording
        metrics.record_memory()
        self.assertTrue(len(metrics.metrics['memory_usage']) > 0)
        self.assertGreater(metrics.metrics['memory_usage'][0], 0)
        
        # Test summary generation
        summary = metrics.get_summary()
        self.assertIn('peak_memory_mb', summary)
        self.assertIn('avg_memory_mb', summary)
        self.assertIn('records_per_second', summary)
    
    def test_memory_usage_check(self):
        """Test memory usage monitoring."""
        # Test normal memory usage
        self.assertTrue(self.processor._check_memory_usage())
    
    def test_read_json_chunk(self):
        """Test JSON file reading in chunks."""
        # Get first test file
        test_file = next(self.symbol_dir.glob('*.json'))
        
        # Read and verify chunk
        chunks = list(self.processor._read_json_chunk(test_file, chunk_size=1000))
        self.assertEqual(len(chunks), 1)  # Single chunk for small file
        
        chunk = chunks[0]
        self.assertIn('market_name', chunk)
        self.assertIn('symbol', chunk)
        self.assertIn('data', chunk)
        
        # Verify data structure
        data = chunk['data'][0]
        required_fields = ['timestamp', 'price', 'volume', 'trading_status']
        for field in required_fields:
            self.assertIn(field, data)
    
    def test_process_chunk(self):
        """Test data chunk processing."""
        # Create test chunk
        chunk = {
            'market_name': 'test_market',
            'symbol': 'TEST.SYM',
            'data': [{
                'timestamp': '2025-03-15',
                'price': 100.0,
                'volume': 1000,
                'trading_status': 'active',
                'currency': 'USD',
                'additional_metrics': {
                    'vwap': 99.5,
                    'turnover': 99500,
                    'daily_return': 0.01,
                    'intraday_range': 0.02
                }
            }]
        }
        
        # Process chunk
        result = self.processor._process_chunk(chunk, 'test.json')
        
        # Verify DataFrame structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_dtype(result['timestamp']))
        self.assertTrue(pd.api.types.is_categorical_dtype(result['market_name']))
        self.assertTrue(pd.api.types.is_categorical_dtype(result['trading_status']))
    
    def test_process_market_data(self):
        """Test full market data processing pipeline."""
        # Test with parallel processing
        df_parallel = self.processor.process_market_data('test_market', parallel=True)
        self.assertIsInstance(df_parallel, pd.DataFrame)
        self.assertGreater(len(df_parallel), 0)
        
        # Test with sequential processing
        df_sequential = self.processor.process_market_data('test_market', parallel=False)
        self.assertIsInstance(df_sequential, pd.DataFrame)
        self.assertGreater(len(df_sequential), 0)
        
        # Compare results
        pd.testing.assert_frame_equal(df_parallel.sort_index(), df_sequential.sort_index())
        
        # Test date filtering
        start_date = datetime(2025, 3, 10)
        df_filtered = self.processor.process_market_data('test_market', start_date=start_date)
        self.assertTrue(all(df_filtered['timestamp'] >= pd.Timestamp(start_date)))
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test invalid market directory
        with self.assertRaises(ValueError):
            self.processor.process_market_data('invalid_market')
        
        # Test invalid JSON file
        invalid_file = self.symbol_dir / 'invalid.json'
        with open(invalid_file, 'w') as f:
            f.write('invalid json')
            
        with self.assertRaises(json.JSONDecodeError):
            self.processor.process_file(invalid_file)
    
    def test_performance_monitoring(self):
        """Test performance monitoring during processing."""
        # Process data and check metrics
        _ = self.processor.process_market_data('test_market')
        metrics = self.processor.get_performance_metrics()
        
        # Verify metrics
        self.assertGreater(metrics['file_read_time_seconds'], 0)
        self.assertGreater(metrics['processing_time_seconds'], 0)
        self.assertGreater(metrics['peak_memory_mb'], 0)
        self.assertGreater(metrics['records_per_second'], 0)
    
    def test_data_validation(self):
        """Test data validation and cleaning."""
        # Create test file with missing fields
        invalid_data = {
            'date': '2025-03-16',
            'symbol': 'TEST.SYM',
            'market': 'test_market',
            'close': 100.0,
            # Missing volume
            'trading_status': 'active'
        }
        
        invalid_file = self.symbol_dir / 'invalid_data.json'
        with open(invalid_file, 'w') as f:
            json.dump(invalid_data, f)
        
        # Process should handle missing fields gracefully
        df = self.processor.process_file(invalid_file)
        self.assertTrue('volume' in df.columns)  # Should exist but might be NaN
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()