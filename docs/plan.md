# Performance Optimization Plan for Historical Data Fetching

## Current Issues
1. Synchronous yfinance calls within async context
2. Small batch sizes (25 symbols) with artificial delays
3. Redundant Ticker object creation
4. Underutilization of existing async infrastructure

## Optimization Strategy

### 1. Leverage Existing Async Infrastructure
- Utilize the `YahooFinanceClient` class in async_utils.py which already has:
  - Proper rate limiting with TokenBucket
  - Circuit breaker protection
  - Async session management
  - Request batching capabilities

### 2. Implement Concurrent Processing
- Increase batch size for parallel processing (from 25 to 50-100)
- Remove artificial delays between chunks
- Use proper async HTTP requests instead of synchronous yfinance calls
- Implement connection pooling for better resource utilization

### 3. Caching Improvements
- Enhance the existing caching mechanism in yfinance_utils.py
- Implement multi-level caching:
  - Memory cache for frequently accessed data
  - SQLite cache for persistence
  - Smart cache invalidation based on market hours

### 4. Code Changes Required
1. Update YahooFinanceClient:
   - Add optimized batch processing for historical data
   - Implement connection pooling
   - Add memory caching layer

2. Modify update_stocks.py:
   - Replace direct yfinance calls with YahooFinanceClient
   - Optimize batch processing
   - Remove synchronous code blocks

3. Update caching strategy:
   - Enhance CachedSession configuration
   - Add memory cache layer
   - Implement smart cache invalidation

## Expected Improvements
- Reduce fetch time per stock from 10 seconds to 2-3 seconds
- Better resource utilization through proper async implementation
- Improved reliability with circuit breaker protection
- More efficient caching reducing redundant API calls

## Migration Strategy
1. Implement changes in a new branch
2. Add comprehensive tests for new functionality
3. Benchmark performance improvements
4. Gradually roll out changes to minimize disruption