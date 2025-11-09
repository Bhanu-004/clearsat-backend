# backend/app/routes/profile.py
from fastapi import APIRouter, HTTPException, status, Depends
from bson import ObjectId
from datetime import datetime

from app.auth.auth import get_current_active_user
from app.database import get_user_collection, get_analysis_collection
from app.models.user import UserResponse

router = APIRouter(prefix="/profile", tags=["profile"])

@router.get("/overview", response_model=dict)
async def get_profile_overview(current_user: dict = Depends(get_current_active_user)):
    """Get comprehensive profile overview"""
    user_collection = get_user_collection()
    analysis_collection = get_analysis_collection()
    
    # Get user with full details
    user = await user_collection.find_one({"_id": current_user["_id"]})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get analysis statistics
    analysis_pipeline = [
        {"$match": {"user_id": str(current_user["_id"])}},
        {"$group": {
            "_id": "$status",
            "count": {"$sum": 1}
        }}
    ]
    
    analysis_stats = await analysis_collection.aggregate(analysis_pipeline).to_list(length=10)
    
    # Get recent analyses
    recent_analyses = await analysis_collection.find(
        {"user_id": str(current_user["_id"])}
    ).sort("created_at", -1).limit(5).to_list(length=5)
    
    # Format recent analyses
    formatted_analyses = []
    for analysis in recent_analyses:
        formatted_analyses.append({
            "id": str(analysis["_id"]),
            "analysis_type": analysis["analysis_type"],
            "location_name": analysis["location"]["name"],
            "status": analysis["status"],
            "created_at": analysis["created_at"],
            "satellite_source": analysis["satellite_source"]
        })
    
    # Calculate account metrics
    account_age_days = (datetime.utcnow() - user["created_at"]).days
    days_since_last_login = (datetime.utcnow() - user.get("last_login", user["created_at"])).days
    
    return {
        "user": UserResponse(
            id=str(user["_id"]),
            email=user["email"],
            full_name=user["full_name"],
            role=user["role"],
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            analysis_count=user.get("analysis_count", 0),
            preferences=user.get("preferences", {})
        ),
        "statistics": {
            "total_analyses": user.get("analysis_count", 0),
            "analysis_status": {stat["_id"]: stat["count"] for stat in analysis_stats},
            "account_age_days": account_age_days,
            "days_since_last_login": days_since_last_login,
            "completion_rate": len([a for a in recent_analyses if a.get("status") == "completed"]) / max(len(recent_analyses), 1)
        },
        "recent_analyses": formatted_analyses,
        "account_health": {
            "email_verified": user.get("email_verified", False),
            "is_active": user.get("is_active", True),
            "has_strong_password": True,  # Would need password strength check
            "recent_activity": days_since_last_login <= 30
        }
    }