#!/usr/bin/env python3
"""
Configuration management module for the Qlib-based quantitative analysis project.
This module provides a centralized configuration system that combines configuration
files (YAML/JSON) with environment variables.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


class ConfigManager:
    """
    Configuration manager that combines configuration files with environment variables.
    """
    
    def __init__(self, config_file: Optional[str] = None, env_prefix: str = "QLIB_"):
        """
        Initialize configuration manager.
        
        Args:
            config_file (str, optional): Path to configuration file (YAML or JSON)
            env_prefix (str): Prefix for environment variables
        """
        self.config_file = config_file
        self.env_prefix = env_prefix
        self.config = {}
        
        # Load configuration from file if provided
        if config_file:
            self.load_config_file(config_file)
        
        # Override with environment variables
        self.load_env_vars()
    
    def load_config_file(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file (str): Path to configuration file (YAML or JSON)
        """
        if not os.path.exists(config_file):
            print(f"Configuration file not found: {config_file}")
            return
        
        try:
            file_ext = os.path.splitext(config_file)[1].lower()
            
            with open(config_file, 'r') as f:
                if file_ext == '.yaml' or file_ext == '.yml':
                    self.config = yaml.safe_load(f)
                elif file_ext == '.json':
                    self.config = json.load(f)
                else:
                    print(f"Unsupported configuration file format: {file_ext}")
            
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
    
    def load_env_vars(self) -> None:
        """
        Load configuration from environment variables.
        Environment variables with the specified prefix are converted to nested dictionary keys.
        For example, QLIB_STORAGE_BACKEND=csv becomes {'storage': {'backend': 'csv'}}
        """
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                key = key[len(self.env_prefix):].lower()
                
                # Split by underscore to create nested keys
                parts = key.split('_')
                
                # Convert value to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    value = float(value)
                
                # Build nested dictionary
                current = self.config
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        current[part] = value
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key (dot-separated for nested keys)
            default (Any): Default value if key is not found
            
        Returns:
            Any: Configuration value
        """
        parts = key.split('.')
        current = self.config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key (str): Configuration key (dot-separated for nested keys)
            value (Any): Configuration value
        """
        parts = key.split('.')
        current = self.config
        
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current[part] = value
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
    
    def save(self, config_file: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file (str, optional): Path to configuration file (YAML or JSON)
        """
        config_file = config_file or self.config_file
        
        if not config_file:
            print("No configuration file specified")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            
            file_ext = os.path.splitext(config_file)[1].lower()
            
            with open(config_file, 'w') as f:
                if file_ext == '.yaml' or file_ext == '.yml':
                    yaml.dump(self.config, f, default_flow_style=False)
                elif file_ext == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    print(f"Unsupported configuration file format: {file_ext}")
                    return
            
            print(f"Saved configuration to {config_file}")
        except Exception as e:
            print(f"Error saving configuration file: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dict[str, Any]: All configuration values
        """
        return self.config


# Default configuration manager
_default_config_manager = None


def get_config_manager(config_file: Optional[str] = None, env_prefix: str = "QLIB_") -> ConfigManager:
    """
    Get the default configuration manager.
    
    Args:
        config_file (str, optional): Path to configuration file (YAML or JSON)
        env_prefix (str): Prefix for environment variables
        
    Returns:
        ConfigManager: Default configuration manager
    """
    global _default_config_manager
    
    if _default_config_manager is None:
        _default_config_manager = ConfigManager(config_file, env_prefix)
    
    return _default_config_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value by key from the default configuration manager.
    
    Args:
        key (str): Configuration key (dot-separated for nested keys)
        default (Any): Default value if key is not found
        
    Returns:
        Any: Configuration value
    """
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    Set configuration value by key in the default configuration manager.
    
    Args:
        key (str): Configuration key (dot-separated for nested keys)
        value (Any): Configuration value
    """
    get_config_manager().set(key, value)


def load_config_file(config_file: str) -> None:
    """
    Load configuration from file into the default configuration manager.
    
    Args:
        config_file (str): Path to configuration file (YAML or JSON)
    """
    get_config_manager().load_config_file(config_file)


def save_config(config_file: Optional[str] = None) -> None:
    """
    Save configuration to file from the default configuration manager.
    
    Args:
        config_file (str, optional): Path to configuration file (YAML or JSON)
    """
    get_config_manager().save(config_file)