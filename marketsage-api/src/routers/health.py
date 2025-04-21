#!/usr/bin/env python3
"""
Health check router for the MarketSage API.

This module provides endpoints for checking the health and status of the API
and its connections to other components.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from marketsage_api.core.config import settings
from marketsage_api.core.dependencies import (
    DataHubClient,
    MarketSageCoreClient,
    TSDSClient,
    DbSession,
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Dict[str, Any]: Health status information
    """
    return {
        "status": "ok",
        "version": settings.PROJECT_VERSION,
        "service": settings.PROJECT_NAME,
    }


@router.get("/health/detailed", tags=["Health"])
async def detailed_health_check(
    db: DbSession,
    datahub_client: DataHubClient,
    marketsage_core_client: MarketSageCoreClient,
    tsds_client: TSDSClient,
) -> Dict[str, Any]:
    """
    Detailed health check endpoint that tests connections to all dependencies.
    
    Args:
        db: Database session
        datahub_client: DataHub client
        marketsage_core_client: MarketSage Core client
        tsds_client: Time Series Data Service client
        
    Returns:
        Dict[str, Any]: Detailed health status information
    """
    health_status = {
        "status": "ok",
        "version": settings.PROJECT_VERSION,
        "service": settings.PROJECT_NAME,
        "components": {},
    }
    
    # Check database connection
    try:
        db.execute("SELECT 1")
        health_status["components"]["database"] = {"status": "ok"}
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        health_status["components"]["database"] = {
            "status": "error",
            "message": str(e),
        }
        health_status["status"] = "degraded"
    
    # Check DataHub connection
    try:
        datahub_status = datahub_client.check_health()
        health_status["components"]["datahub"] = {
            "status": "ok",
            "version": datahub_status.get("version", "unknown"),
        }
    except Exception as e:
        logger.error(f"DataHub connection error: {e}")
        health_status["components"]["datahub"] = {
            "status": "error",
            "message": str(e),
        }
        health_status["status"] = "degraded"
    
    # Check MarketSage Core connection
    try:
        core_status = marketsage_core_client.check_health()
        health_status["components"]["marketsage_core"] = {
            "status": "ok",
            "version": core_status.get("version", "unknown"),
        }
    except Exception as e:
        logger.error(f"MarketSage Core connection error: {e}")
        health_status["components"]["marketsage_core"] = {
            "status": "error",
            "message": str(e),
        }
        health_status["status"] = "degraded"
    
    # Check Time Series Data Service connection
    try:
        tsds_status = tsds_client.check_health()
        health_status["components"]["tsds"] = {
            "status": "ok",
            "version": tsds_status.get("version", "unknown"),
        }
    except Exception as e:
        logger.error(f"TSDS connection error: {e}")
        health_status["components"]["tsds"] = {
            "status": "error",
            "message": str(e),
        }
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/readiness", tags=["Health"])
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check endpoint for Kubernetes.
    
    Returns:
        Dict[str, str]: Readiness status
    """
    return {"status": "ready"}


@router.get("/liveness", tags=["Health"])
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check endpoint for Kubernetes.
    
    Returns:
        Dict[str, str]: Liveness status
    """
    return {"status": "alive"}