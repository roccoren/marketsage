"""Token bucket implementation for rate limiting."""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from ..exceptions import RateLimitError
from ..models.common import RateLimitConfig

@dataclass
class TokenBucketState:
    """Current state of the token bucket."""
    tokens: float
    last_update: float
    is_suspended: bool = False
    suspend_until: Optional[float] = None

class TokenBucket:
    """
    Token bucket rate limiter for controlling request rates with burst support.
    
    Features:
    - Configurable rates and burst size
    - Support for temporary suspension
    - Async-safe operation
    - Accurate token replenishment
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize token bucket with configuration.
        
        Args:
            config: Rate limiting configuration
        """
        self.rate = 1.0 / config.requests_per_second
        self.max_tokens = config.burst_size
        self.retry_after = config.retry_after
        self._state = TokenBucketState(
            tokens=config.burst_size,
            last_update=time.time()
        )
        self._lock = asyncio.Lock()

    def _replenish_tokens(self) -> None:
        """Replenish tokens based on time passed."""
        now = time.time()
        
        # Check if suspension should be lifted
        if self._state.is_suspended:
            if now >= self._state.suspend_until:
                self._state.is_suspended = False
                self._state.suspend_until = None
                self._state.tokens = self.max_tokens
                self._state.last_update = now
                return
            return

        # Calculate new tokens
        time_passed = now - self._state.last_update
        new_tokens = time_passed / self.rate
        
        if new_tokens >= 1:
            self._state.tokens = min(
                self.max_tokens,
                self._state.tokens + new_tokens
            )
            self._state.last_update = now

    def _suspend(self) -> None:
        """Suspend token bucket temporarily."""
        self._state.is_suspended = True
        self._state.suspend_until = time.time() + self.retry_after
        self._state.tokens = 0

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Acquire tokens, blocking if necessary.
        
        Args:
            tokens: Number of tokens to acquire (default: 1.0)
            
        Raises:
            RateLimitError: If bucket is suspended
        """
        async with self._lock:
            while True:
                self._replenish_tokens()
                
                if self._state.is_suspended:
                    wait_time = self._state.suspend_until - time.time()
                    if wait_time > 0:
                        raise RateLimitError(
                            f"Rate limit exceeded. Try again in {wait_time:.1f} seconds"
                        )
                
                if self._state.tokens >= tokens:
                    self._state.tokens -= tokens
                    return
                
                # If we're out of tokens, calculate wait time
                wait_time = (tokens - self._state.tokens) * self.rate
                if wait_time > self.retry_after:
                    self._suspend()
                    raise RateLimitError(
                        f"Would exceed rate limit. Try again in {self.retry_after} seconds"
                    )
                    
                await asyncio.sleep(wait_time)

    def get_tokens(self) -> float:
        """Get current number of available tokens."""
        self._replenish_tokens()
        return self._state.tokens

    def is_suspended(self) -> bool:
        """Check if bucket is currently suspended."""
        if self._state.is_suspended:
            self._replenish_tokens()  # May lift suspension
        return self._state.is_suspended