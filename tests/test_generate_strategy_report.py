"""Unit tests for strategy report generation module."""

import unittest
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime

from src.market_system.generate_strategy_report import format_report, save_report

class TestReportGeneration(unittest.TestCase):
    """Test cases for report generation functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_dir = tempfile.mkdtemp()
        self.sample_report = {
            'analysis_period': {
                'start': '2025-03-01T00:00:00',
                'end': '2025-03-15T00:00:00',
                'trading_days': 10,
                'calendar_days': 15
            },
            'strategies': {
                'moving_average_crossover': {
                    'status': 'Success',
                    'matches': 50,
                    'match_rate': 0.25,
                    'performance': {
                        'success_rate': 0.65,
                        'average_return': 0.02,
                        'vwap_success_rate': 0.7,
                        'winning_trades': 32,
                        'total_trades': 50,
                        'recent_performance': {
                            'win_rate_10d': 0.7,
                            'avg_return_10d': 0.025,
                            'consistency_score': 0.8
                        }
                    }
                }
            },
            'market_detail': {
                'trading_activity': {
                    'activity_rate': 0.95,
                    'active_days': 10,
                    'total_days': 15
                },
                'volume_analysis': {
                    'daily_stats': {
                        'mean': 1000000,
                        'std': 200000,
                        'max': 1500000
                    }
                },
                'price_analysis': {
                    'volatility': 0.015,
                    'daily_stats': {
                        'mean_range': 2.5,
                        'std': 0.5
                    }
                },
                'turnover_analysis': {
                    'daily_stats': {
                        'mean': 10000000,
                        'max': 15000000
                    }
                }
            },
            'report_generated': datetime.now().isoformat()
        }
    
    def test_format_report(self):
        """Test report formatting."""
        formatted_report = format_report(self.sample_report)
        
        # Verify report sections
        self.assertIn('=== Investment Target Screening & Analysis Report ===', formatted_report)
        self.assertIn('A. Basic Screening List', formatted_report)
        self.assertIn('B. Technical Analysis', formatted_report)
        self.assertIn('C. Trading Signal Analysis', formatted_report)
        self.assertIn('D. Comprehensive Evaluation', formatted_report)
        
        # Verify key metrics are included
        self.assertIn('Moving Average Crossover', formatted_report)
        self.assertIn('Success Rate:', formatted_report)
        self.assertIn('Average Return:', formatted_report)
        
        # Check performance metrics
        self.assertIn('65.0%', formatted_report)  # Success rate
        self.assertIn('2.00%', formatted_report)  # Average return
        
        # Verify trading activity metrics
        self.assertIn('95.0%', formatted_report)  # Activity rate
    
    def test_save_report(self):
        """Test report saving functionality."""
        output_dir = Path(self.test_dir) / 'reports'
        market_name = 'test_market'
        
        # Save report
        save_report(self.sample_report, output_dir, market_name)
        
        # Verify files were created
        json_files = list(output_dir.glob('*.json'))
        txt_files = list(output_dir.glob('*.txt'))
        
        self.assertEqual(len(json_files), 1)
        self.assertEqual(len(txt_files), 1)
        
        # Verify JSON content
        with open(json_files[0], 'r') as f:
            saved_report = json.load(f)
        
        self.assertEqual(saved_report['analysis_period'], self.sample_report['analysis_period'])
        self.assertEqual(saved_report['strategies'], self.sample_report['strategies'])
        
        # Verify text content
        with open(txt_files[0], 'r') as f:
            report_text = f.read()
        
        self.assertIn('Investment Target Screening & Analysis Report', report_text)
        self.assertIn('Moving Average Crossover', report_text)
    
    def test_report_validation(self):
        """Test report validation and error handling."""
        # Test with missing required fields
        invalid_report = {
            'strategies': {}  # Missing other required sections
        }
        
        formatted_report = format_report(invalid_report)
        self.assertIn('=== Investment Target Screening & Analysis Report ===', formatted_report)
        
        # Test with empty strategies
        empty_report = {
            'analysis_period': {
                'start': '2025-03-01T00:00:00',
                'end': '2025-03-15T00:00:00'
            },
            'strategies': {},
            'market_detail': {}
        }
        
        formatted_report = format_report(empty_report)
        self.assertNotIn('Strategy-Matched Stocks', formatted_report)
    
    def test_report_file_names(self):
        """Test report file naming convention."""
        output_dir = Path(self.test_dir) / 'reports'
        market_name = 'test_market'
        
        save_report(self.sample_report, output_dir, market_name)
        
        files = list(output_dir.glob(f'{market_name}_strategy_report_*.json'))
        self.assertEqual(len(files), 1)
        
        file_name = files[0].name
        self.assertTrue(file_name.startswith(f'{market_name}_strategy_report_'))
        self.assertTrue(file_name.endswith('.json'))
        
        # Verify timestamp format in filename
        timestamp = file_name.replace(f'{market_name}_strategy_report_', '').replace('.json', '')
        self.assertTrue(len(timestamp) == 15)  # YYYYMMdd_HHMMSS
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()