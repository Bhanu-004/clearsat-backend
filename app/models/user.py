# backend/app/models/user.py
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, validator, Field
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class UserPreferences(BaseModel):
    default_satellite_source: str = "landsat"
    default_analysis_type: str = "NDVI"
    default_buffer_km: int = 10
    email_notifications: bool = True
    newsletter_subscription: bool = False
    map_auto_zoom: bool = True

class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = UserRole.USER

    @validator('full_name')
    def validate_full_name(cls, v):
        if not v.strip():
            raise ValueError('Full name cannot be empty')
        return v.strip()

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None

    @validator('full_name')
    def validate_full_name(cls, v):
        if v is not None and not v.strip():
            raise ValueError('Full name cannot be empty')
        return v.strip() if v else v

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

class UserPreferencesUpdate(BaseModel):
    default_satellite_source: Optional[str] = None
    default_analysis_type: Optional[str] = None
    default_buffer_km: Optional[int] = None
    email_notifications: Optional[bool] = None
    newsletter_subscription: Optional[bool] = None
    map_auto_zoom: Optional[bool] = None

class UserInDB(UserBase):
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    analysis_count: int = 0
    email_verified: bool = False
    preferences: Optional[Dict[str, Any]] = None

class UserResponse(UserBase):
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None
    analysis_count: int = 0
    preferences: Optional[UserPreferences] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[UserRole] = None