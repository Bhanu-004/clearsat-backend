# backend/app/routes/analysis.py
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, Query
from datetime import datetime, timedelta
from bson import ObjectId
import asyncio
import logging
from typing import List, Optional

from app.models.analysis import AnalysisCreate, AnalysisResponse, AnalysisStatus, AnalysisType, SatelliteSource
from app.auth.auth import get_current_active_user
from app.database import get_analysis_collection, get_user_collection
from backend.app.services.satellite_service import ee_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["analysis"])

async def check_guest_limits(user: dict):
    """Enhanced guest user limits with better messaging"""
    if user.get("role") == "guest":
        analysis_count = user.get("analysis_count", 0)
        if analysis_count >= settings.GUEST_ANALYSIS_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "message": f"Guest users are limited to {settings.GUEST_ANALYSIS_LIMIT} analysis",
                    "action": "Please register for full access",
                    "limit": settings.GUEST_ANALYSIS_LIMIT,
                    "used": analysis_count
                }
            )

async def update_user_analysis_count(user_id: str, increment: bool = True):
    """Update user's analysis count with error handling"""
    user_collection = get_user_collection()
    change = 1 if increment else -1
    
    try:
        await user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$inc": {"analysis_count": change}}
        )
    except Exception as e:
        logger.error(f"Failed to update user analysis count: {e}")

async def cleanup_failed_analyses():
    """Background task to clean up stuck analyses"""
    analysis_collection = get_analysis_collection()
    
    try:
        # Find analyses stuck in processing for more than 1 hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        result = await analysis_collection.update_many(
            {
                "status": AnalysisStatus.PROCESSING,
                "started_at": {"$lt": cutoff_time}
            },
            {
                "$set": {
                    "status": AnalysisStatus.FAILED,
                    "error_message": "Analysis timed out",
                    "completed_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"🧹 Cleaned up {result.modified_count} stuck analyses")
            
    except Exception as e:
        logger.error(f"Failed to clean up stuck analyses: {e}")

async def validate_analysis_parameters(analysis_data: AnalysisCreate):
    """Validate analysis parameters before processing"""
    try:
        # Date validation
        start_date = datetime.strptime(analysis_data.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(analysis_data.end_date, '%Y-%m-%d')
        
        if end_date <= start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="End date must be after start date"
            )
        
        if (end_date - start_date).days > settings.MAX_ANALYSIS_DAYS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Analysis period cannot exceed {settings.MAX_ANALYSIS_DAYS} days"
            )
        
        # Validate buffer size
        if analysis_data.buffer_km > settings.MAX_ANALYSIS_BUFFER_KM:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Buffer size cannot exceed {settings.MAX_ANALYSIS_BUFFER_KM} km"
            )
        
        # Validate coordinates
        lat, lng = analysis_data.location.latitude, analysis_data.location.longitude
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid coordinates provided"
            )
            
        return True
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

async def perform_analysis_task(analysis_id: str, analysis_data: dict):
    """Enhanced background analysis task with timeout and progress tracking"""
    analysis_collection = get_analysis_collection()
    
    try:
        # Update status to processing
        await analysis_collection.update_one(
            {"_id": ObjectId(analysis_id)},
            {"$set": {
                "status": AnalysisStatus.PROCESSING,
                "started_at": datetime.utcnow()
            }}
        )
        
        # Set timeout for analysis (30 minutes)
        try:
            results = await asyncio.wait_for(
                ee_service.perform_analysis(analysis_data),
                timeout=1800  # 30 minutes
            )
        except asyncio.TimeoutError:
            raise TimeoutError("Analysis timed out after 30 minutes")
        
        # Update with results
        await analysis_collection.update_one(
            {"_id": ObjectId(analysis_id)},
            {"$set": {
                "status": AnalysisStatus.COMPLETED,
                "results": results,
                "completed_at": datetime.utcnow()
            }}
        )
        
        logger.info(f"✅ Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Analysis {analysis_id} failed: {error_message}")
        
        # Update with error
        await analysis_collection.update_one(
            {"_id": ObjectId(analysis_id)},
            {"$set": {
                "status": AnalysisStatus.FAILED,
                "error_message": error_message,
                "completed_at": datetime.utcnow()
            }}
        )

@router.post("/", response_model=AnalysisResponse)
async def create_analysis(
    analysis_data: AnalysisCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user)
):
    """Create analysis with enhanced validation and error handling"""
    # Check guest limits
    await check_guest_limits(current_user)
    
    analysis_collection = get_analysis_collection()
    
    try:
        # Validate analysis parameters
        await validate_analysis_parameters(analysis_data)
        
        # Enhance location data
        location_info = ee_service.get_location_name(
            analysis_data.location.latitude,
            analysis_data.location.longitude
        )
        
        # Create analysis document
        analysis_dict = analysis_data.dict()
        analysis_dict["location"].update(location_info)
        analysis_dict["user_id"] = str(current_user["_id"])
        analysis_dict["status"] = AnalysisStatus.PENDING
        analysis_dict["created_at"] = datetime.utcnow()
        
        # Insert analysis record
        result = await analysis_collection.insert_one(analysis_dict)
        analysis_id = str(result.inserted_id)
        
        # Update user analysis count
        await update_user_analysis_count(current_user["_id"])
        
        # Start background analysis task
        background_tasks.add_task(perform_analysis_task, analysis_id, analysis_dict)
        
        # Clean up stuck analyses in background
        background_tasks.add_task(cleanup_failed_analyses)
        
        # Return initial response
        analysis_dict["id"] = analysis_id
        return AnalysisResponse(**analysis_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create analysis"
        )

@router.get("/", response_model=List[AnalysisResponse])
async def get_user_analyses(
    current_user: dict = Depends(get_current_active_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[AnalysisStatus] = None
):
    """Get user analyses with pagination and filtering"""
    analysis_collection = get_analysis_collection()
    
    try:
        # Build query
        query = {"user_id": str(current_user["_id"])}
        if status:
            query["status"] = status
        
        # Get analyses with pagination
        analyses = await analysis_collection.find(query) \
            .sort("created_at", -1) \
            .skip(skip) \
            .limit(limit) \
            .to_list(length=limit)
        
        response_analyses = []
        for analysis in analyses:
            analysis["id"] = str(analysis["_id"])
            response_analyses.append(AnalysisResponse(**analysis))
        
        return response_analyses
        
    except Exception as e:
        logger.error(f"Failed to fetch analyses: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analyses"
        )

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get specific analysis with enhanced error handling"""
    analysis_collection = get_analysis_collection()
    
    try:
        # Validate analysis ID format
        if not ObjectId.is_valid(analysis_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid analysis ID format"
            )
        
        analysis = await analysis_collection.find_one({
            "_id": ObjectId(analysis_id),
            "user_id": str(current_user["_id"])
        })
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        analysis["id"] = str(analysis["_id"])
        return AnalysisResponse(**analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analysis"
        )

@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Delete analysis with comprehensive error handling"""
    analysis_collection = get_analysis_collection()
    
    try:
        # Validate analysis ID format
        if not ObjectId.is_valid(analysis_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid analysis ID format"
            )
        
        result = await analysis_collection.delete_one({
            "_id": ObjectId(analysis_id),
            "user_id": str(current_user["_id"])
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Update user analysis count
        await update_user_analysis_count(current_user["_id"], increment=False)
        
        return {"message": "Analysis deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete analysis"
        )

@router.get("/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get analysis status for progress tracking"""
    analysis_collection = get_analysis_collection()
    
    try:
        if not ObjectId.is_valid(analysis_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid analysis ID format"
            )
        
        analysis = await analysis_collection.find_one(
            {"_id": ObjectId(analysis_id), "user_id": str(current_user["_id"])},
            {"status": 1, "started_at": 1, "completed_at": 1, "error_message": 1}
        )
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        return {
            "analysis_id": analysis_id,
            "status": analysis.get("status"),
            "started_at": analysis.get("started_at"),
            "completed_at": analysis.get("completed_at"),
            "error_message": analysis.get("error_message")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis status {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analysis status"
        )

@router.get("/stats/summary")
async def get_analysis_summary(current_user: dict = Depends(get_current_active_user)):
    """Get analysis statistics for the current user"""
    analysis_collection = get_analysis_collection()
    
    try:
        pipeline = [
            {"$match": {"user_id": str(current_user["_id"])}},
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1},
                "latest": {"$max": "$created_at"}
            }}
        ]
        
        stats = await analysis_collection.aggregate(pipeline).to_list(length=10)
        
        summary = {
            "total": 0,
            "by_status": {},
            "recent_activity": None
        }
        
        for stat in stats:
            status_name = stat["_id"]
            summary["by_status"][status_name] = stat["count"]
            summary["total"] += stat["count"]
            
            # Track most recent activity
            if not summary["recent_activity"] or stat["latest"] > summary["recent_activity"]:
                summary["recent_activity"] = stat["latest"]
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get analysis summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analysis statistics"
        )