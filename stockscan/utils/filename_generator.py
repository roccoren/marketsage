"""Utility module for generating output filenames."""

import datetime
import random
import string
import re
from typing import Dict, Optional

class InvalidPatternError(Exception):
    """Raised when the filename pattern is invalid."""
    pass

class FilenameGenerator:
    """Generator for creating filenames based on custom patterns."""
    
    # Valid placeholders and their descriptions
    VALID_PLACEHOLDERS = {
        '{market}': 'Market name (e.g., HKEX, NYSE)',
        '{strategy}': 'Strategy name(s) joined by hyphens',
        '{date}': 'Current date in YYYYMMDD format',
        '{random}': 'Random alphanumeric string (default 4 chars)',
        '{extension}': 'File extension (default .json)'
    }
    
    def __init__(self, pattern: Optional[str] = None):
        """
        Initialize the filename generator with an optional pattern.
        
        Args:
            pattern: Custom filename pattern. If None, uses default pattern:
                    {market}-{strategy}-{date}-{random}{extension}
        """
        self.set_pattern(pattern or '{market}-{strategy}-{date}-{random}{extension}')
    
    def set_pattern(self, pattern: str) -> None:
        """
        Set and validate the filename pattern.
        
        Args:
            pattern: The filename pattern to use
            
        Raises:
            InvalidPatternError: If the pattern is invalid
        """
        # Check for invalid characters in pattern
        invalid_chars = re.findall(r'[<>:"|?*\x00-\x1f]', pattern)
        if invalid_chars:
            raise InvalidPatternError(
                f"Pattern contains invalid characters: {', '.join(invalid_chars)}"
            )
        
        # Validate placeholders
        placeholders = re.findall(r'\{([^}]+)\}', pattern)
        invalid_placeholders = [
            p for p in placeholders
            if f'{{{p}}}' not in self.VALID_PLACEHOLDERS
        ]
        
        if invalid_placeholders:
            valid_examples = '\n'.join(
                f"  {k}: {v}" for k, v in self.VALID_PLACEHOLDERS.items()
            )
            raise InvalidPatternError(
                f"Invalid placeholders: {', '.join(invalid_placeholders)}\n"
                f"Valid placeholders are:\n{valid_examples}"
            )
        
        self.pattern = pattern
    
    def _generate_random_string(self, length: int = 4) -> str:
        """Generate a random alphanumeric string."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    def generate_filename(self, 
                         market: str,
                         *strategy_names: str,
                         extension: str = '.json',
                         random_length: int = 4) -> str:
        """
        Generate a filename based on the current pattern.
        
        Args:
            market: Market name (e.g., 'HKEX', 'NYSE')
            *strategy_names: One or more strategy names
            extension: File extension (default: .json)
            random_length: Length of random string (default: 4)
            
        Returns:
            str: Generated filename
            
        Example:
            >>> gen = FilenameGenerator()
            >>> gen.generate_filename('HKEX', 'macd', 'rsi')
            'HKEX-macd-rsi-20250226-a7b2.json'
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
            
        # Clean and validate inputs
        market = re.sub(r'[<>:"|?*\x00-\x1f]', '', market)
        strategies = [re.sub(r'[<>:"|?*\x00-\x1f]', '', s) for s in strategy_names]
        
        # Prepare replacement values
        values = {
            '{market}': market,
            '{strategy}': '-'.join(strategies),
            '{date}': datetime.datetime.now().strftime('%Y%m%d'),
            '{random}': self._generate_random_string(random_length),
            '{extension}': extension
        }
        
        # Replace placeholders
        filename = self.pattern
        for placeholder, value in values.items():
            filename = filename.replace(placeholder, value)
            
        return filename

    @classmethod
    def list_placeholders(cls) -> Dict[str, str]:
        """Return dictionary of valid placeholders and their descriptions."""
        return cls.VALID_PLACEHOLDERS

# Example usage
if __name__ == '__main__':
    # Default pattern
    generator = FilenameGenerator()
    print(generator.generate_filename('HKEX', 'macd', 'rsi'))
    
    # Custom pattern
    custom_pattern = '{market}_{strategy}_{date}_{random}{extension}'
    generator.set_pattern(custom_pattern)
    print(generator.generate_filename('NYSE', 'momentum', extension='.csv'))