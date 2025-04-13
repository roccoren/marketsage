#!/usr/bin/env python3
"""
Event dispatcher module for the Qlib-based quantitative analysis project.
This module provides a lightweight in-process event dispatcher for inter-module communication
and dynamic, event-triggered orchestration.
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Set
from enum import Enum, auto

from modules.config_manager import get_config


# Configure logging
logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class Event:
    """Event class for inter-module communication."""
    
    def __init__(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                 priority: EventPriority = EventPriority.NORMAL, source: Optional[str] = None):
        """
        Initialize event.
        
        Args:
            event_type (str): Type of the event
            data (Dict[str, Any], optional): Event data
            priority (EventPriority): Event priority
            source (str, optional): Source of the event
        """
        self.event_type = event_type
        self.data = data or {}
        self.priority = priority
        self.source = source
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event(type={self.event_type}, source={self.source}, priority={self.priority.name})"


class EventHandler:
    """Event handler for processing events."""
    
    def __init__(self, callback: Callable[[Event], None], event_types: Optional[List[str]] = None, 
                 priority: EventPriority = EventPriority.NORMAL):
        """
        Initialize event handler.
        
        Args:
            callback (Callable[[Event], None]): Callback function to handle events
            event_types (List[str], optional): List of event types to handle (None for all)
            priority (EventPriority): Handler priority
        """
        self.callback = callback
        self.event_types = event_types
        self.priority = priority
    
    def can_handle(self, event: Event) -> bool:
        """
        Check if the handler can handle the event.
        
        Args:
            event (Event): Event to check
            
        Returns:
            bool: True if the handler can handle the event
        """
        return self.event_types is None or event.event_type in self.event_types
    
    def handle(self, event: Event) -> None:
        """
        Handle the event.
        
        Args:
            event (Event): Event to handle
        """
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Error handling event {event}: {e}")


class EventDispatcher:
    """Event dispatcher for inter-module communication."""
    
    def __init__(self, async_mode: bool = True, max_queue_size: int = 1000):
        """
        Initialize event dispatcher.
        
        Args:
            async_mode (bool): Whether to process events asynchronously
            max_queue_size (int): Maximum size of the event queue
        """
        self.handlers: Dict[int, List[EventHandler]] = {}
        self.async_mode = async_mode
        self.event_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        self.processed_events = 0
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        # Log events if enabled
        if get_config('events.handlers.log_events', True):
            self.add_handler(self._log_event_handler, priority=EventPriority.LOW)
    
    def _log_event_handler(self, event: Event) -> None:
        """
        Default handler for logging events.
        
        Args:
            event (Event): Event to log
        """
        logger.debug(f"Event received: {event}")
    
    def add_handler(self, callback: Callable[[Event], None], event_types: Optional[List[str]] = None, 
                   priority: EventPriority = EventPriority.NORMAL) -> None:
        """
        Add an event handler.
        
        Args:
            callback (Callable[[Event], None]): Callback function to handle events
            event_types (List[str], optional): List of event types to handle (None for all)
            priority (EventPriority): Handler priority
        """
        handler = EventHandler(callback, event_types, priority)
        
        if priority.value not in self.handlers:
            self.handlers[priority.value] = []
        
        self.handlers[priority.value].append(handler)
        logger.debug(f"Added handler for event types: {event_types}")
    
    def remove_handler(self, callback: Callable[[Event], None]) -> None:
        """
        Remove an event handler.
        
        Args:
            callback (Callable[[Event], None]): Callback function to remove
        """
        for priority, handlers in self.handlers.items():
            self.handlers[priority] = [h for h in handlers if h.callback != callback]
    
    def dispatch(self, event: Event) -> None:
        """
        Dispatch an event.
        
        Args:
            event (Event): Event to dispatch
        """
        if self.async_mode:
            try:
                # Put event in queue with priority
                self.event_queue.put((event.priority.value, event))
                logger.debug(f"Event queued: {event}")
            except queue.Full:
                logger.error(f"Event queue is full, dropping event: {event}")
        else:
            self._process_event(event)
    
    def _process_event(self, event: Event) -> None:
        """
        Process an event by calling all matching handlers.
        
        Args:
            event (Event): Event to process
        """
        # Process handlers in priority order
        for priority in sorted(self.handlers.keys()):
            handlers = self.handlers[priority]
            
            for handler in handlers:
                if handler.can_handle(event):
                    handler.handle(event)
        
        self.processed_events += 1
    
    def _event_processor(self) -> None:
        """Event processing thread function."""
        logger.info("Event processor thread started")
        
        while self.running:
            try:
                # Get event with timeout to allow thread to exit
                try:
                    _, event = self.event_queue.get(timeout=1.0)
                    self._process_event(event)
                    self.event_queue.task_done()
                except queue.Empty:
                    continue
            except Exception as e:
                logger.error(f"Error in event processor: {e}")
        
        logger.info("Event processor thread stopped")
    
    def start(self) -> None:
        """Start the event dispatcher."""
        if self.async_mode and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._event_processor, daemon=True)
            self.thread.start()
            logger.info("Event dispatcher started")
    
    def stop(self) -> None:
        """Stop the event dispatcher."""
        if self.async_mode and self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=5.0)
            logger.info(f"Event dispatcher stopped, processed {self.processed_events} events")
    
    def wait_for_queue(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the event queue to be empty.
        
        Args:
            timeout (float, optional): Timeout in seconds
            
        Returns:
            bool: True if the queue is empty, False if timeout occurred
        """
        if not self.async_mode:
            return True
        
        try:
            self.event_queue.join()
            return True
        except Exception:
            return False


# Default event dispatcher
_default_dispatcher = None


def get_event_dispatcher() -> EventDispatcher:
    """
    Get the default event dispatcher.
    
    Returns:
        EventDispatcher: Default event dispatcher
    """
    global _default_dispatcher
    
    if _default_dispatcher is None:
        async_mode = get_config('events.enabled', True)
        _default_dispatcher = EventDispatcher(async_mode=async_mode)
        
        # Start the dispatcher if in async mode
        if async_mode:
            _default_dispatcher.start()
    
    return _default_dispatcher


def dispatch_event(event_type: str, data: Optional[Dict[str, Any]] = None, 
                  priority: EventPriority = EventPriority.NORMAL, source: Optional[str] = None) -> None:
    """
    Dispatch an event using the default event dispatcher.
    
    Args:
        event_type (str): Type of the event
        data (Dict[str, Any], optional): Event data
        priority (EventPriority): Event priority
        source (str, optional): Source of the event
    """
    event = Event(event_type, data, priority, source)
    get_event_dispatcher().dispatch(event)


def add_event_handler(callback: Callable[[Event], None], event_types: Optional[List[str]] = None, 
                     priority: EventPriority = EventPriority.NORMAL) -> None:
    """
    Add an event handler to the default event dispatcher.
    
    Args:
        callback (Callable[[Event], None]): Callback function to handle events
        event_types (List[str], optional): List of event types to handle (None for all)
        priority (EventPriority): Handler priority
    """
    get_event_dispatcher().add_handler(callback, event_types, priority)


def remove_event_handler(callback: Callable[[Event], None]) -> None:
    """
    Remove an event handler from the default event dispatcher.
    
    Args:
        callback (Callable[[Event], None]): Callback function to remove
    """
    get_event_dispatcher().remove_handler(callback)


# Common event types
class EventTypes:
    """Common event types used in the application."""
    # Data events
    DATA_ACQUIRED = "data.acquired"
    DATA_UPDATED = "data.updated"
    
    # Forecasting events
    FORECAST_STARTED = "forecast.started"
    FORECAST_COMPLETED = "forecast.completed"
    
    # Backtesting events
    BACKTEST_STARTED = "backtest.started"
    BACKTEST_COMPLETED = "backtest.completed"
    
    # Model events
    MODEL_TRAINED = "model.trained"
    MODEL_OPTIMIZED = "model.optimized"
    
    # System events
    CONFIG_CHANGED = "system.config_changed"
    SHUTDOWN_REQUESTED = "system.shutdown_requested"
    
    # Error events
    ERROR_OCCURRED = "error.occurred"