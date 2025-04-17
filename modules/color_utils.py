#!/usr/bin/env python3
"""
Color utility functions for terminal output.
Provides color coding for buy/sell signals in terminal output.
"""

# Global flag to control color output
disable_color = False

def set_color_enabled(enabled=True):
    """
    Enable or disable color output.
    
    Args:
        enabled: Boolean, if False colors will be disabled
    """
    global disable_color
    disable_color = not enabled

def color_text(text, color_type):
    """
    Color code text based on signal type.
    Tries to use colorama if available, otherwise uses ANSI codes.
    
    Args:
        text: The text to color
        color_type: 'buy', 'sell', or 'hold'
    
    Returns:
        Colored text string
    """
    global disable_color
    
    # If colors are disabled, return plain text
    if disable_color:
        return text
        
    try:
        # Try to use colorama if available
        from colorama import Fore, Style
        if color_type.lower() == 'buy':
            return f"{Fore.GREEN}{text}{Style.RESET_ALL}"
        elif color_type.lower() == 'sell':
            return f"{Fore.RED}{text}{Style.RESET_ALL}"
        else:
            return text
    except ImportError:
        # Fall back to ANSI codes
        if color_type.lower() == 'buy':
            return f"\033[92m{text}\033[0m"  # Green
        elif color_type.lower() == 'sell':
            return f"\033[91m{text}\033[0m"  # Red
        else:
            return text