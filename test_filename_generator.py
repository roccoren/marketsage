"""Test module for filename generator utility."""

import re
import pytest
from datetime import datetime
from utils.filename_generator import FilenameGenerator, InvalidPatternError

def test_default_pattern():
    """Test filename generation with default pattern."""
    generator = FilenameGenerator()
    filename = generator.generate_filename('HKEX', 'sma')
    
    pattern = r'^HKEX-sma-\d{8}-[a-z0-9]{4}\.json$'
    assert re.match(pattern, filename), f"Filename {filename} doesn't match expected pattern"
    
    # Check date part is current date
    today = datetime.now().strftime('%Y%m%d')
    assert today in filename, f"Filename {filename} doesn't contain today's date {today}"

def test_custom_pattern():
    """Test filename generation with custom pattern."""
    generator = FilenameGenerator('{market}_{strategy}_{date}_{random}{extension}')
    filename = generator.generate_filename('NYSE', 'momentum', 'rsi')
    
    pattern = r'^NYSE_momentum-rsi_\d{8}_[a-z0-9]{4}\.json$'
    assert re.match(pattern, filename), f"Filename {filename} doesn't match expected pattern"

def test_invalid_pattern():
    """Test handling of invalid patterns."""
    # Test invalid placeholders
    with pytest.raises(InvalidPatternError) as exc:
        FilenameGenerator('{invalid}_{market}')
    assert 'Invalid placeholders' in str(exc.value)
    
    # Test invalid characters
    with pytest.raises(InvalidPatternError) as exc:
        FilenameGenerator('market<>:"|?*')
    assert 'invalid characters' in str(exc.value)

def test_special_characters():
    """Test handling of special characters in inputs."""
    generator = FilenameGenerator()
    filename = generator.generate_filename('HK<>EX', 'ma|cd', 'r?si')
    
    # Special characters should be removed
    pattern = r'^HKEX-macd-rsi-\d{8}-[a-z0-9]{4}\.json$'
    assert re.match(pattern, filename), f"Filename {filename} doesn't handle special characters correctly"

def test_different_extensions():
    """Test generation with different file extensions."""
    generator = FilenameGenerator()
    
    # Test with .csv extension
    csv_file = generator.generate_filename('HKEX', 'macd', extension='.csv')
    assert csv_file.endswith('.csv'), "File extension not set correctly"
    
    # Test with extension without dot
    txt_file = generator.generate_filename('HKEX', 'macd', extension='txt')
    assert txt_file.endswith('.txt'), "File extension not handled correctly"

def test_random_length():
    """Test custom random string length."""
    generator = FilenameGenerator()
    filename = generator.generate_filename('HKEX', 'macd', random_length=6)
    
    # Extract random part (second to last segment)
    random_part = filename.split('-')[-1].split('.')[0]
    assert len(random_part) == 6, "Random string length not set correctly"

def test_multiple_strategy_names():
    """Test handling of multiple strategy names."""
    generator = FilenameGenerator()
    filename = generator.generate_filename('HKEX', 'macd', 'rsi', 'momentum')
    
    pattern = r'^HKEX-macd-rsi-momentum-\d{8}-[a-z0-9]{4}\.json$'
    assert re.match(pattern, filename), f"Filename {filename} doesn't handle multiple strategies correctly"

if __name__ == '__main__':
    pytest.main([__file__])