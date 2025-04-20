#!/usr/bin/env python3
"""
Schedules router for the StockScan API.
"""

import uuid
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.core.dependencies import (
    AnalystUser,
    CurrentUser,
    DbSession,
    get_current_active_user,
)
from api.schemas.schedule import (
    Schedule,
    ScheduleCreate,
    ScheduleExecution,
    ScheduleExecutionList,
    ScheduleList,
    ScheduleUpdate,
)

# Import existing modules for scheduling
from modules.scheduler import Scheduler

router = APIRouter()


@router.post("/", response_model=Schedule)
async def create_schedule(
    schedule_data: ScheduleCreate,
    current_user: dict = Depends(get_current_active_user),
    db: Session = Depends(DbSession),
) -> Any:
    """
    Create a new schedule.
    
    Args:
        schedule_data: Schedule data
        current_user: Current user
        db: Database session
        
    Returns:
        Created schedule
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Generate schedule ID
    schedule_id = str(uuid.uuid4())
    
    # Create schedule
    schedule = {
        "id": schedule_id,
        "name": schedule_data.name,
        "task_type": schedule_data.task_type,
        "cron_expression": schedule_data.cron_expression,
        "interval_seconds": schedule_data.interval_seconds,
        "params": schedule_data.params.dict(),
        "user_id": current_user["user_id"],
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    # Calculate next run time
    next_run = scheduler.calculate_next_run(
        cron_expression=schedule_data.cron_expression,
        interval_seconds=schedule_data.interval_seconds,
    )
    schedule["next_run"] = next_run
    
    # Add schedule to scheduler
    try:
        scheduler.add_schedule(schedule)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error adding schedule: {str(e)}",
        )
    
    return schedule


@router.get("/", response_model=ScheduleList)
async def list_schedules(
    task_type: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    List schedules.
    
    Args:
        task_type: Filter by task type
        status: Filter by status
        page: Page number
        page_size: Page size
        current_user: Current user
        
    Returns:
        List of schedules
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Get schedules
    schedules = scheduler.get_schedules(
        user_id=current_user["user_id"],
        task_type=task_type,
        status=status,
    )
    
    # Apply pagination
    total = len(schedules)
    pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paginated_schedules = schedules[start:end]
    
    return {
        "schedules": paginated_schedules,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


@router.get("/{schedule_id}", response_model=Schedule)
async def get_schedule(
    schedule_id: str,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Get schedule details.
    
    Args:
        schedule_id: Schedule ID
        current_user: Current user
        
    Returns:
        Schedule details
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Get schedule
    schedule = scheduler.get_schedule(schedule_id)
    
    # Check if schedule exists
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule with ID {schedule_id} not found",
        )
    
    # Check if user has access to schedule
    if schedule.get("user_id") != current_user["user_id"]:
        # In a real implementation, we would check if the user has access to the schedule
        # For now, we'll just allow access if the user is an admin or analyst
        is_admin_or_analyst = False  # This would be a database lookup
        if not is_admin_or_analyst:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this schedule",
            )
    
    return schedule


@router.put("/{schedule_id}", response_model=Schedule)
async def update_schedule(
    schedule_id: str,
    schedule_data: ScheduleUpdate,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Update schedule.
    
    Args:
        schedule_id: Schedule ID
        schedule_data: Schedule data
        current_user: Current user
        
    Returns:
        Updated schedule
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Get schedule
    schedule = scheduler.get_schedule(schedule_id)
    
    # Check if schedule exists
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule with ID {schedule_id} not found",
        )
    
    # Check if user has access to schedule
    if schedule.get("user_id") != current_user["user_id"]:
        # In a real implementation, we would check if the user has access to the schedule
        # For now, we'll just allow access if the user is an admin or analyst
        is_admin_or_analyst = False  # This would be a database lookup
        if not is_admin_or_analyst:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this schedule",
            )
    
    # Update schedule
    update_data = schedule_data.dict(exclude_unset=True)
    
    if "params" in update_data and update_data["params"]:
        update_data["params"] = update_data["params"].dict()
    
    # Update next run time if cron_expression or interval_seconds changed
    if "cron_expression" in update_data or "interval_seconds" in update_data:
        cron_expression = update_data.get("cron_expression", schedule.get("cron_expression"))
        interval_seconds = update_data.get("interval_seconds", schedule.get("interval_seconds"))
        next_run = scheduler.calculate_next_run(
            cron_expression=cron_expression,
            interval_seconds=interval_seconds,
        )
        update_data["next_run"] = next_run
    
    # Update updated_at timestamp
    update_data["updated_at"] = datetime.utcnow()
    
    # Update schedule
    try:
        updated_schedule = scheduler.update_schedule(schedule_id, update_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error updating schedule: {str(e)}",
        )
    
    return updated_schedule


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    schedule_id: str,
    current_user: dict = Depends(get_current_active_user),
) -> None:
    """
    Delete schedule.
    
    Args:
        schedule_id: Schedule ID
        current_user: Current user
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Get schedule
    schedule = scheduler.get_schedule(schedule_id)
    
    # Check if schedule exists
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule with ID {schedule_id} not found",
        )
    
    # Check if user has access to schedule
    if schedule.get("user_id") != current_user["user_id"]:
        # In a real implementation, we would check if the user has access to the schedule
        # For now, we'll just allow access if the user is an admin or analyst
        is_admin_or_analyst = False  # This would be a database lookup
        if not is_admin_or_analyst:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this schedule",
            )
    
    # Delete schedule
    try:
        scheduler.delete_schedule(schedule_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error deleting schedule: {str(e)}",
        )


@router.post("/{schedule_id}/execute", response_model=ScheduleExecution)
async def execute_schedule(
    schedule_id: str,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Execute schedule immediately.
    
    Args:
        schedule_id: Schedule ID
        current_user: Current user
        
    Returns:
        Schedule execution
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Get schedule
    schedule = scheduler.get_schedule(schedule_id)
    
    # Check if schedule exists
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule with ID {schedule_id} not found",
        )
    
    # Check if user has access to schedule
    if schedule.get("user_id") != current_user["user_id"]:
        # In a real implementation, we would check if the user has access to the schedule
        # For now, we'll just allow access if the user is an admin or analyst
        is_admin_or_analyst = False  # This would be a database lookup
        if not is_admin_or_analyst:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to execute this schedule",
            )
    
    # Execute schedule
    try:
        execution_id = str(uuid.uuid4())
        execution = {
            "id": execution_id,
            "schedule_id": schedule_id,
            "start_time": datetime.utcnow(),
            "status": "running",
        }
        
        # Execute task
        result = scheduler.execute_task(
            task_type=schedule["task_type"],
            params=schedule["params"],
        )
        
        # Update execution
        execution["end_time"] = datetime.utcnow()
        execution["status"] = "completed"
        execution["result"] = result
        
        # Store execution
        scheduler.store_execution(execution)
        
        return execution
    except Exception as e:
        # Update execution with error
        execution = {
            "id": execution_id if 'execution_id' in locals() else str(uuid.uuid4()),
            "schedule_id": schedule_id,
            "start_time": datetime.utcnow() if 'execution' not in locals() else execution["start_time"],
            "end_time": datetime.utcnow(),
            "status": "failed",
            "error": str(e),
        }
        
        # Store execution
        scheduler.store_execution(execution)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing schedule: {str(e)}",
        )


@router.get("/{schedule_id}/executions", response_model=ScheduleExecutionList)
async def list_schedule_executions(
    schedule_id: str,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    List schedule executions.
    
    Args:
        schedule_id: Schedule ID
        status: Filter by status
        page: Page number
        page_size: Page size
        current_user: Current user
        
    Returns:
        List of schedule executions
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Get schedule
    schedule = scheduler.get_schedule(schedule_id)
    
    # Check if schedule exists
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule with ID {schedule_id} not found",
        )
    
    # Check if user has access to schedule
    if schedule.get("user_id") != current_user["user_id"]:
        # In a real implementation, we would check if the user has access to the schedule
        # For now, we'll just allow access if the user is an admin or analyst
        is_admin_or_analyst = False  # This would be a database lookup
        if not is_admin_or_analyst:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this schedule",
            )
    
    # Get executions
    executions = scheduler.get_executions(
        schedule_id=schedule_id,
        status=status,
    )
    
    # Apply pagination
    total = len(executions)
    pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paginated_executions = executions[start:end]
    
    return {
        "executions": paginated_executions,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


@router.get("/executions/{execution_id}", response_model=ScheduleExecution)
async def get_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Get execution details.
    
    Args:
        execution_id: Execution ID
        current_user: Current user
        
    Returns:
        Execution details
    """
    # Initialize scheduler
    scheduler = Scheduler()
    
    # Get execution
    execution = scheduler.get_execution(execution_id)
    
    # Check if execution exists
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution with ID {execution_id} not found",
        )
    
    # Get schedule
    schedule = scheduler.get_schedule(execution["schedule_id"])
    
    # Check if user has access to schedule
    if schedule.get("user_id") != current_user["user_id"]:
        # In a real implementation, we would check if the user has access to the schedule
        # For now, we'll just allow access if the user is an admin or analyst
        is_admin_or_analyst = False  # This would be a database lookup
        if not is_admin_or_analyst:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this execution",
            )
    
    return execution