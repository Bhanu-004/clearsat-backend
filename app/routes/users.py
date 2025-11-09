# backend/app/routes/users.py
from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from datetime import datetime, timedelta
from bson import ObjectId
import secrets
import re

from app.config import settings
from app.models.user import (
    UserCreate, UserResponse, Token, UserUpdate, UserRole, 
    UserProfileUpdate, PasswordChange, UserPreferencesUpdate, UserPreferences
)
from app.auth.auth import (
    get_password_hash, 
    verify_password, 
    create_access_token, 
    get_current_active_user,
    validate_password_strength,
    get_current_user
)
from app.database import get_user_collection

router = APIRouter(prefix="/users", tags=["users"])

async def cleanup_expired_guests():
    """Background task to clean up expired guest users"""
    user_collection = get_user_collection()
    
    try:
        result = await user_collection.delete_many({
            "role": UserRole.GUEST,
            "guest_expires_at": {"$lt": datetime.utcnow()}
        })
        
        if result.deleted_count > 0:
            print(f"🧹 Cleaned up {result.deleted_count} expired guest users")
    except Exception as e:
        print(f"Guest cleanup error: {e}")

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, background_tasks: BackgroundTasks):
    """Register a new user with enhanced validation"""
    user_collection = get_user_collection()
    
    # Validate email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Validate password strength
    if not validate_password_strength(user_data.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters with uppercase, lowercase, number and special character"
        )
    
    # Check if user already exists
    existing_user = await user_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Create new user with default preferences
    user_dict = user_data.dict()
    user_dict["password_hash"] = get_password_hash(user_data.password)
    user_dict["created_at"] = datetime.utcnow()
    user_dict["last_login"] = datetime.utcnow()
    user_dict["is_active"] = True
    user_dict["analysis_count"] = 0
    user_dict["email_verified"] = False
    
    # Set default preferences
    user_dict["preferences"] = UserPreferences().dict()
    
    # Remove plain password
    del user_dict["password"]
    
    result = await user_collection.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)
    
    # Clean up expired guests in background
    background_tasks.add_task(cleanup_expired_guests)
    
    return UserResponse(**user_dict)

@router.post("/login", response_model=Token)
async def login_user(
    email: str, 
    password: str,
    background_tasks: BackgroundTasks
):
    """User login with enhanced security"""
    user_collection = get_user_collection()
    
    # Basic input validation
    if not email or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email and password are required"
        )
    
    user = await user_collection.find_one({"email": email})
    if not user or not verify_password(password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Update last login
    await user_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]}
    )
    
    user_response = UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        created_at=user["created_at"],
        last_login=user["last_login"],
        analysis_count=user.get("analysis_count", 0),
        preferences=UserPreferences(**user.get("preferences", {}))
    )
    
    # Clean up expired guests in background
    background_tasks.add_task(cleanup_expired_guests)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )

@router.post("/guest-login", response_model=Token)
async def guest_login(background_tasks: BackgroundTasks):
    """Create a temporary guest account with cleanup"""
    user_collection = get_user_collection()
    
    # Generate unique guest credentials
    guest_email = f"guest_{secrets.token_hex(8)}@clearsat.com"
    guest_name = f"Guest_{secrets.token_hex(4)}"
    
    guest_expires_at = datetime.utcnow() + timedelta(hours=settings.GUEST_SESSION_HOURS)
    
    guest_user = {
        "email": guest_email,
        "full_name": guest_name,
        "role": UserRole.GUEST,
        "password_hash": get_password_hash(secrets.token_urlsafe(32)),
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
        "is_active": True,
        "analysis_count": 0,
        "is_guest": True,
        "guest_expires_at": guest_expires_at,
        "email_verified": False,
        "preferences": UserPreferences().dict()
    }
    
    result = await user_collection.insert_one(guest_user)
    guest_user["id"] = str(result.inserted_id)
    
    # Create access token
    access_token = create_access_token(
        data={"sub": guest_user["email"], "role": guest_user["role"]}
    )
    
    user_response = UserResponse(
        id=guest_user["id"],
        email=guest_user["email"],
        full_name=guest_user["full_name"],
        role=guest_user["role"],
        created_at=guest_user["created_at"],
        last_login=guest_user["last_login"],
        analysis_count=0,
        preferences=UserPreferences(**guest_user["preferences"])
    )
    
    # Clean up expired guests in background
    background_tasks.add_task(cleanup_expired_guests)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=str(current_user["_id"]),
        email=current_user["email"],
        full_name=current_user["full_name"],
        role=current_user["role"],
        created_at=current_user["created_at"],
        last_login=current_user.get("last_login"),
        analysis_count=current_user.get("analysis_count", 0),
        preferences=UserPreferences(**current_user.get("preferences", {}))
    )

@router.put("/me", response_model=UserResponse)
async def update_user_info(
    update_data: UserUpdate,
    current_user: dict = Depends(get_current_active_user)
):
    """Update user information with validation"""
    user_collection = get_user_collection()
    
    update_dict = update_data.dict(exclude_unset=True, exclude_none=True)
    
    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for update"
        )
    
    # Validate email if provided
    if "email" in update_dict:
        existing_user = await user_collection.find_one({
            "email": update_dict["email"],
            "_id": {"$ne": current_user["_id"]}
        })
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
    
    await user_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": update_dict}
    )
    
    # Get updated user
    updated_user = await user_collection.find_one({"_id": current_user["_id"]})
    
    return UserResponse(
        id=str(updated_user["_id"]),
        email=updated_user["email"],
        full_name=updated_user["full_name"],
        role=updated_user["role"],
        created_at=updated_user["created_at"],
        last_login=updated_user.get("last_login"),
        analysis_count=updated_user.get("analysis_count", 0),
        preferences=UserPreferences(**updated_user.get("preferences", {}))
    )

@router.put("/me/profile", response_model=UserResponse)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user: dict = Depends(get_current_active_user)
):
    """Update user profile information"""
    user_collection = get_user_collection()
    
    update_dict = profile_data.dict(exclude_unset=True, exclude_none=True)
    
    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for update"
        )
    
    # Validate email if provided
    if "email" in update_dict:
        existing_user = await user_collection.find_one({
            "email": update_dict["email"],
            "_id": {"$ne": current_user["_id"]}
        })
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
    
    await user_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": update_dict}
    )
    
    # Get updated user
    updated_user = await user_collection.find_one({"_id": current_user["_id"]})
    
    return UserResponse(
        id=str(updated_user["_id"]),
        email=updated_user["email"],
        full_name=updated_user["full_name"],
        role=updated_user["role"],
        created_at=updated_user["created_at"],
        last_login=updated_user.get("last_login"),
        analysis_count=updated_user.get("analysis_count", 0),
        preferences=UserPreferences(**updated_user.get("preferences", {}))
    )

@router.post("/me/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: dict = Depends(get_current_active_user)
):
    """Change user password with current password verification"""
    user_collection = get_user_collection()
    
    # Verify current password
    if not verify_password(password_data.current_password, current_user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password strength
    if not validate_password_strength(password_data.new_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters with uppercase, lowercase, number and special character"
        )
    
    # Update password
    new_password_hash = get_password_hash(password_data.new_password)
    await user_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"password_hash": new_password_hash}}
    )
    
    return {"message": "Password updated successfully"}

@router.get("/me/preferences", response_model=UserPreferences)
async def get_user_preferences(current_user: dict = Depends(get_current_active_user)):
    """Get user preferences"""
    preferences = current_user.get("preferences", {})
    return UserPreferences(**preferences)

@router.put("/me/preferences", response_model=UserPreferences)
async def update_user_preferences(
    preferences_data: UserPreferencesUpdate,
    current_user: dict = Depends(get_current_active_user)
):
    """Update user preferences"""
    user_collection = get_user_collection()
    
    update_dict = preferences_data.dict(exclude_unset=True, exclude_none=True)
    
    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for update"
        )
    
    # Get current preferences and merge with updates
    current_preferences = current_user.get("preferences", {})
    updated_preferences = {**current_preferences, **update_dict}
    
    await user_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"preferences": updated_preferences}}
    )
    
    return UserPreferences(**updated_preferences)

@router.get("/stats")
async def get_user_stats(current_user: dict = Depends(get_current_active_user)):
    """Get comprehensive user statistics"""
    user_collection = get_user_collection()
    
    # Get user with analysis count
    user = await user_collection.find_one(
        {"_id": current_user["_id"]},
        {"analysis_count": 1, "created_at": 1, "last_login": 1}
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Calculate account age
    account_age_days = (datetime.utcnow() - user["created_at"]).days
    
    return {
        "analysis_count": user.get("analysis_count", 0),
        "account_age_days": account_age_days,
        "last_login": user.get("last_login"),
        "member_since": user["created_at"]
    }

# Admin-only endpoints
@router.get("/admin/users")
async def get_all_users(
    current_user: dict = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 50
):
    """Get all users (admin only)"""
    if current_user.get("role") != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user_collection = get_user_collection()
    
    users = await user_collection.find({}) \
        .sort("created_at", -1) \
        .skip(skip) \
        .limit(limit) \
        .to_list(length=limit)
    
    user_list = []
    for user in users:
        user_list.append({
            "id": str(user["_id"]),
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "created_at": user["created_at"],
            "last_login": user.get("last_login"),
            "analysis_count": user.get("analysis_count", 0),
            "is_active": user.get("is_active", True)
        })
    
    return user_list

@router.get("/admin/stats")
async def get_admin_stats(current_user: dict = Depends(get_current_active_user)):
    """Get admin statistics (admin only)"""
    if current_user.get("role") != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user_collection = get_user_collection()
    
    # Get total user counts by role
    pipeline = [
        {
            "$group": {
                "_id": "$role",
                "count": {"$sum": 1},
                "total_analyses": {"$sum": "$analysis_count"}
            }
        }
    ]
    
    role_stats = await user_collection.aggregate(pipeline).to_list(length=10)
    
    # Get recent signups (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_signups = await user_collection.count_documents({
        "created_at": {"$gte": week_ago}
    })
    
    # Get total analyses
    total_analyses_cursor = user_collection.aggregate([
        {"$group": {"_id": None, "total": {"$sum": "$analysis_count"}}}
    ])
    total_analyses_result = await total_analyses_cursor.to_list(length=1)
    total_analyses = total_analyses_result[0]["total"] if total_analyses_result else 0
    
    return {
        "role_stats": role_stats,
        "recent_signups": recent_signups,
        "total_analyses": total_analyses,
        "total_users": await user_collection.count_documents({})
    }