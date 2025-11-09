# backend/app/auth/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import re

from app.config import settings
from app.models.user import UserRole, TokenData
from app.database import get_user_collection

# Enhanced password hashing
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__rounds=12
)

security = HTTPBearer()

# Password validation
def validate_password_strength(password: str) -> bool:
    """Validate password meets security requirements"""
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash securely"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password securely"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token securely"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> dict:
    """Retrieve current user from JWT with enhanced security"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        email: str = payload.get("sub")
        role: str = payload.get("role")
        token_type: str = payload.get("type")
        
        if email is None or role is None or token_type != "access":
            raise credentials_exception
            
        token_data = TokenData(email=email, role=role)
    except JWTError as e:
        print(f"JWT decode error: {e}")
        raise credentials_exception

    user_collection = get_user_collection()
    user = await user_collection.find_one({"email": token_data.email})
    
    if user is None:
        raise credentials_exception
        
    # Check if guest user has expired
    if user.get("role") == UserRole.GUEST and user.get("guest_expires_at"):
        if datetime.utcnow() > user.get("guest_expires_at"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Guest session expired"
            )
    
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Ensure current user is active"""
    if not current_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    return current_user

def require_role(required_role: UserRole):
    """Role-based access control with admin override"""
    def role_checker(current_user: dict = Depends(get_current_active_user)):
        user_role = current_user.get("role")
        
        # Admin has access to everything
        if user_role == UserRole.ADMIN:
            return current_user
            
        # Check if user has required role
        if user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker