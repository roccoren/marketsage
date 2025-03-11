"""Tests for rate limiting components."""
import pytest
import asyncio
import time
from datetime import datetime

from src.market_data.rate_limiting.token_bucket import TokenBucket
from src.market_data.rate_limiting.circuit_breaker import CircuitBreaker, CircuitState
from src.market_data.models.common import RateLimitConfig
from src.market_data.exceptions import RateLimitError, CircuitBreakerError

@pytest.fixture
def rate_limit_config():
    return RateLimitConfig(
        requests_per_second=10.0,
        burst_size=5,
        retry_after=1
    )

@pytest.fixture
def token_bucket(rate_limit_config):
    return TokenBucket(rate_limit_config)

@pytest.mark.asyncio
async def test_token_bucket_basic_rate_limiting(token_bucket):
    """Test basic rate limiting functionality."""
    # Should be able to make burst_size requests immediately
    for _ in range(5):
        await token_bucket.acquire()
        
    # Next request should be rate limited
    with pytest.raises(RateLimitError):
        await token_bucket.acquire()
        
    # Wait for token replenishment
    await asyncio.sleep(0.2)  # Should have 2 new tokens (10 per second)
    
    # Should be able to make 2 requests
    await token_bucket.acquire()
    await token_bucket.acquire()
    
    # Third request should be rate limited
    with pytest.raises(RateLimitError):
        await token_bucket.acquire()

@pytest.mark.asyncio
async def test_token_bucket_suspension(token_bucket):
    """Test bucket suspension after too many requests."""
    # Exhaust tokens
    for _ in range(5):
        await token_bucket.acquire()
        
    # Make requests that trigger suspension
    for _ in range(3):
        with pytest.raises(RateLimitError):
            await token_bucket.acquire()
            
    assert token_bucket.is_suspended()
    
    # Wait for suspension to lift
    await asyncio.sleep(1.1)
    
    # Should be able to make requests again
    await token_bucket.acquire()
    assert not token_bucket.is_suspended()

@pytest.fixture
def circuit_breaker():
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
        half_open_timeout=0.5
    )

@pytest.mark.asyncio
async def test_circuit_breaker_basic_operation(circuit_breaker):
    """Test basic circuit breaker state transitions."""
    # Start in closed state
    assert circuit_breaker.state == CircuitState.CLOSED
    
    # Successful calls should work
    async def success():
        return "ok"
        
    result = await circuit_breaker.call(success)
    assert result == "ok"
    assert circuit_breaker.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures(circuit_breaker):
    """Test circuit opens after failure threshold."""
    failure_count = 0
    
    async def fail():
        nonlocal failure_count
        failure_count += 1
        raise ValueError("test failure")
        
    # Generate failures
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(fail)
            
    assert circuit_breaker.state == CircuitState.OPEN
    assert failure_count == 3
    
    # Should reject requests when open
    with pytest.raises(CircuitBreakerError):
        await circuit_breaker.call(fail)
        
    # Failure count shouldn't increase when circuit is open
    assert failure_count == 3

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery(circuit_breaker):
    """Test recovery through half-open state."""
    # Open the circuit
    async def fail():
        raise ValueError("test failure")
        
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(fail)
            
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    await asyncio.sleep(1.1)
    
    # Next request should transition to half-open
    async def success():
        return "ok"
        
    result = await circuit_breaker.call(success)
    assert result == "ok"
    assert circuit_breaker.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_circuit_breaker_remains_open_on_half_open_failure(circuit_breaker):
    """Test circuit reopens on failure in half-open state."""
    # Open the circuit
    async def fail():
        raise ValueError("test failure")
        
    for _ in range(3):
        with pytest.raises(ValueError):
            await circuit_breaker.call(fail)
            
    # Wait for recovery timeout
    await asyncio.sleep(1.1)
    
    # Failure in half-open should reopen circuit
    with pytest.raises(ValueError):
        await circuit_breaker.call(fail)
        
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Should reject requests immediately
    with pytest.raises(CircuitBreakerError):
        await circuit_breaker.call(fail)

@pytest.mark.asyncio
async def test_concurrent_rate_limiting(token_bucket):
    """Test rate limiting under concurrent load."""
    async def make_request(i: int):
        try:
            await token_bucket.acquire()
            return i
        except RateLimitError:
            return None
            
    # Launch 10 concurrent requests
    tasks = [make_request(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Should get 5 successful requests (burst_size) and 5 None
    assert len([r for r in results if r is not None]) == 5
    assert len([r for r in results if r is None]) == 5

@pytest.mark.asyncio
async def test_concurrent_circuit_breaking(circuit_breaker):
    """Test circuit breaker under concurrent load."""
    success_count = 0
    failure_count = 0
    
    async def flaky_operation(i: int):
        if i % 2 == 0:
            return "ok"
        raise ValueError("test failure")
        
    async def tracked_operation(i: int):
        nonlocal success_count, failure_count
        try:
            result = await circuit_breaker.call(flaky_operation, i)
            success_count += 1
            return result
        except Exception:
            failure_count += 1
            return None
            
    # Launch 10 concurrent operations
    tasks = [tracked_operation(i) for i in range(10)]
    await asyncio.gather(*tasks)
    
    # Circuit should be open after 3 failures
    assert circuit_breaker.state == CircuitState.OPEN
    assert 2 <= success_count <= 3  # May vary based on timing
    assert failure_count >= 3