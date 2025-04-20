#!/usr/bin/env python3
"""
StockScan API - Main Application Entry Point
"""

import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Import routers
# These will be uncommented as we implement them
from api.routers import auth
from api.routers import stocks
from api.routers import predictions
from api.routers import schedules
# from api.routers import backtest
# from api.routers import portfolios
# from api.routers import alerts
# from api.routers import users

# Import configuration
from api.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# Include routers
app.include_router(auth.router, prefix=f"{settings.API_PREFIX}/auth", tags=["Authentication"])
app.include_router(stocks.router, prefix=f"{settings.API_PREFIX}/stocks", tags=["Stocks"])
app.include_router(predictions.router, prefix=f"{settings.API_PREFIX}/predictions", tags=["Predictions"])
app.include_router(schedules.router, prefix=f"{settings.API_PREFIX}/schedules", tags=["Scheduling"])
# app.include_router(backtest.router, prefix=f"{settings.API_PREFIX}/backtest", tags=["Backtesting"])
# app.include_router(portfolios.router, prefix=f"{settings.API_PREFIX}/portfolios", tags=["Portfolio Management"])
# app.include_router(alerts.router, prefix=f"{settings.API_PREFIX}/alerts", tags=["Alerts"])
# app.include_router(users.router, prefix=f"{settings.API_PREFIX}/users", tags=["User Management"])


# Health check endpoint
@app.get(f"{settings.API_PREFIX}/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": settings.PROJECT_VERSION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )