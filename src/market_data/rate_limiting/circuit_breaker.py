"""Circuit breaker implementation for fault tolerance."""
import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional, List, Any
from ..exceptions import CircuitBreakerError

T = TypeVar('T')

class CircuitState(Enum):
    """Possible states of the circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # No operations allowed
    HALF_OPEN = "half-open"  # Testing if service is recovered

@dataclass
class FailureRecord:
    """Record of a failure occurrence."""
    timestamp: float
    error: Exception
    context: Optional[dict] = None

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for protecting against API failures.
    
    Features:
    - Configurable failure thresholds and recovery timeouts
    - Detailed failure tracking and error history
    - Support for half-open state testing
    - Async-safe operation
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_timeout: float = 30.0,
        error_types: Optional[List[type]] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_timeout: Seconds to wait in half-open state
            error_types: List of error types to count as failures
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_timeout = half_open_timeout
        self._error_types = error_types or [Exception]
        
        self._state = CircuitState.CLOSED
        self._failures: List[FailureRecord] = []
        self._last_failure_time: float = 0
        self._half_open_start: float = 0
        self._lock = asyncio.Lock()

    def _should_count_failure(self, error: Exception) -> bool:
        """Check if error should count towards failure threshold."""
        return any(isinstance(error, err_type) for err_type in self._error_types)

    def _record_failure(self, error: Exception, context: Optional[dict] = None) -> None:
        """Record a new failure."""
        now = time.time()
        self._failures.append(FailureRecord(now, error, context))
        self._last_failure_time = now
        
        # Remove old failures outside the window
        window_start = now - self._recovery_timeout
        self._failures = [f for f in self._failures if f.timestamp > window_start]

    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened based on failures."""
        if len(self._failures) >= self._failure_threshold:
            recent_failures = [
                f for f in self._failures
                if f.timestamp > time.time() - self._recovery_timeout
            ]
            return len(recent_failures) >= self._failure_threshold
        return False

    async def _try_half_open(self) -> None:
        """Attempt to transition to half-open state."""
        now = time.time()
        if now - self._last_failure_time >= self._recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._half_open_start = now

    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        context: Optional[dict] = None,
        **kwargs: Any
    ) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            context: Optional context for error tracking
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func execution
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by func
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                await self._try_half_open()
                if self._state == CircuitState.OPEN:
                    raise CircuitBreakerError(
                        len(self._failures),
                        "Circuit breaker is open"
                    )
            
            try:
                result = await func(*args, **kwargs)
                
                if self._state == CircuitState.HALF_OPEN:
                    # Success in half-open state - close circuit
                    self._state = CircuitState.CLOSED
                    self._failures.clear()
                
                return result
                
            except Exception as e:
                if self._should_count_failure(e):
                    self._record_failure(e, context)
                    
                    if self._state == CircuitState.HALF_OPEN:
                        self._state = CircuitState.OPEN
                    elif self._should_open_circuit():
                        self._state = CircuitState.OPEN
                
                raise

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get number of tracked failures."""
        return len(self._failures)

    @property
    def last_failure_time(self) -> Optional[float]:
        """Get timestamp of last failure."""
        return self._last_failure_time if self._failures else None

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failures.clear()
        self._last_failure_time = 0
        self._half_open_start = 0