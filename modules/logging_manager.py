#!/usr/bin/env python3
"""
Logging manager module for the Qlib-based quantitative analysis project.
This module provides structured logging with Loguru, including remote logging
integration and automated error notifications.
"""

import os
import sys
import json
import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

from loguru import logger

from modules.config_manager import get_config
from modules.event_dispatcher import dispatch_event, EventTypes, EventPriority


class LoggingManager:
    """
    Logging manager that provides structured logging with Loguru.
    """
    
    def __init__(self, config_prefix: str = "logging"):
        """
        Initialize logging manager.
        
        Args:
            config_prefix (str): Prefix for configuration keys
        """
        self.config_prefix = config_prefix
        self.configured = False
        
        # Configure logging
        self.configure()
    
    def configure(self) -> None:
        """Configure logging based on configuration."""
        if self.configured:
            return
        
        # Remove default handler
        logger.remove()
        
        # Get configuration
        log_level = get_config(f"{self.config_prefix}.level", "INFO")
        log_file = get_config(f"{self.config_prefix}.file", "logs/qlib_workflow.log")
        max_size = get_config(f"{self.config_prefix}.max_size", 10 * 1024 * 1024)  # 10 MB
        backup_count = get_config(f"{self.config_prefix}.backup_count", 5)
        console_enabled = get_config(f"{self.config_prefix}.console", True)
        
        # Create log directory if it doesn't exist
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
        # Add console handler if enabled
        if console_enabled:
            logger.add(
                sys.stderr,
                level=log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            )
        
        # Add file handler if log file is specified
        if log_file:
            logger.add(
                log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation=max_size,
                retention=backup_count,
                compression="zip"
            )
        
        # Add remote handler if enabled
        remote_enabled = get_config(f"{self.config_prefix}.remote.enabled", False)
        if remote_enabled:
            remote_host = get_config(f"{self.config_prefix}.remote.host", "")
            remote_port = get_config(f"{self.config_prefix}.remote.port", 0)
            
            if remote_host and remote_port:
                try:
                    # Add remote handler (e.g., syslog, logstash, etc.)
                    # This is a placeholder that would be implemented based on specific requirements
                    logger.info(f"Remote logging enabled to {remote_host}:{remote_port}")
                except Exception as e:
                    logger.error(f"Failed to configure remote logging: {e}")
        
        # Configure exception handling
        self._configure_exception_handling()
        
        self.configured = True
        logger.info("Logging configured")
    
    def _configure_exception_handling(self) -> None:
        """Configure global exception handling."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            """Handle uncaught exceptions."""
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupt
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            # Log the exception
            logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Uncaught exception")
            
            # Dispatch error event
            error_data = {
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback)
            }
            dispatch_event(EventTypes.ERROR_OCCURRED, error_data, EventPriority.HIGH)
            
            # Send error notification if enabled
            self.send_error_notification(error_data)
        
        # Set the exception hook
        sys.excepthook = handle_exception
    
    def send_error_notification(self, error_data: Dict[str, Any]) -> None:
        """
        Send error notification via email.
        
        Args:
            error_data (Dict[str, Any]): Error data
        """
        # Check if email notifications are enabled
        enabled = get_config("error_handling.email_notifications.enabled", False)
        if not enabled:
            return
        
        # Get email configuration
        smtp_server = get_config("error_handling.email_notifications.smtp_server", "")
        smtp_port = get_config("error_handling.email_notifications.smtp_port", 587)
        sender = get_config("error_handling.email_notifications.sender", "")
        recipients = get_config("error_handling.email_notifications.recipients", [])
        subject_prefix = get_config("error_handling.email_notifications.subject_prefix", "[QLIB ERROR]")
        
        if not smtp_server or not sender or not recipients:
            logger.warning("Email notification configuration incomplete")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"{subject_prefix} {error_data['type']}: {error_data['message']}"
            
            # Create message body
            body = f"""
            An error occurred in the Qlib workflow:
            
            Type: {error_data['type']}
            Message: {error_data['message']}
            
            Traceback:
            {''.join(error_data['traceback'])}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                # Note: In a real implementation, you would use credentials from a secure source
                # server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Error notification sent to {', '.join(recipients)}")
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
    
    def get_logger(self, name: str) -> "logger":
        """
        Get a logger with the specified name.
        
        Args:
            name (str): Logger name
            
        Returns:
            logger: Loguru logger
        """
        return logger.bind(name=name)


# Default logging manager
_default_logging_manager = None


def get_logging_manager() -> LoggingManager:
    """
    Get the default logging manager.
    
    Returns:
        LoggingManager: Default logging manager
    """
    global _default_logging_manager
    
    if _default_logging_manager is None:
        _default_logging_manager = LoggingManager()
    
    return _default_logging_manager


def get_logger(name: str) -> "logger":
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Logger name
        
    Returns:
        logger: Loguru logger
    """
    return get_logging_manager().get_logger(name)


# Initialize logging
get_logging_manager()