# backend/app/main.py
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from datetime import datetime
import time
import logging
import traceback
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import uvicorn

from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection, get_user_collection
from app.routes import users, analysis, reports, profile
from app.auth.auth import get_password_hash
from app.services.satellite_service import satellite_service as ee_service
from app.models.user import UserRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('clearsat.log') if settings.is_production else logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

async def create_default_admin():
    """Create default admin user with enhanced security"""
    user_collection = get_user_collection()
    
    try:
        # Check if admin already exists
        admin_user = await user_collection.find_one({"email": "admin@clearsat.com"})
        if not admin_user:
            admin_data = {
                "email": "admin@clearsat.com",
                "full_name": "System Administrator",
                "role": UserRole.ADMIN,
                "password_hash": get_password_hash("admin123"),
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "is_active": True,
                "analysis_count": 0,
                "email_verified": True,
                "preferences": {
                    "default_satellite_source": "landsat",
                    "default_analysis_type": "NDVI",
                    "default_buffer_km": 10,
                    "email_notifications": True,
                    "newsletter_subscription": False,
                    "map_auto_zoom": True
                }
            }
            
            await user_collection.insert_one(admin_data)
            logger.info("🎉 Default admin user created successfully!")
            logger.info("📧 Email: admin@clearsat.com") 
            logger.info("🔑 Password: Admin123!")
            logger.warning("⚠️  IMPORTANT: Change these credentials in production!")
        else:
            logger.info("✅ Admin user already exists")
            
    except Exception as e:
        logger.error(f"❌ Failed to create admin user: {e}")

async def initialize_services():
    """Initialize all external services with error handling"""
    services_initialized = {}
    
    try:
        # Initialize MongoDB
        await connect_to_mongo()
        services_initialized['mongodb'] = True
        logger.info("✅ MongoDB initialized")
    except Exception as e:
        logger.error(f"❌ MongoDB initialization failed: {e}")
        services_initialized['mongodb'] = False
    
    # Earth Engine initialization with better error handling
    earth_engine_initialized = False
    try:
        if ee_service:
            ee_service.initialize_ee()
            earth_engine_initialized = ee_service.initialized
        else:
            logger.error("Earth Engine service not available")
    except Exception as e:
        logger.error(f"Earth Engine initialization failed: {e}")
        earth_engine_initialized = False
    
    services_initialized['earth_engine'] = earth_engine_initialized
    
    if earth_engine_initialized:
        logger.info("✅ Earth Engine initialized")
    else:
        logger.warning("⚠️ Earth Engine not initialized - satellite features disabled")
    
    try:
        # Create default admin
        await create_default_admin()
        services_initialized['admin_user'] = True
    except Exception as e:
        logger.error(f"❌ Admin user creation failed: {e}")
        services_initialized['admin_user'] = False
    
    return services_initialized

async def cleanup_services():
    """Cleanup services gracefully"""
    try:
        await close_mongo_connection()
        logger.info("✅ Services cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ Service cleanup failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting ClearSat API...")
    startup_time = time.time()
    
    services_initialized = await initialize_services()
    
    # Check critical services
    if not services_initialized.get('mongodb'):
        logger.critical("❌ CRITICAL: MongoDB failed to initialize")
    if not services_initialized.get('earth_engine'):
        logger.warning("⚠️ WARNING: Earth Engine failed to initialize - satellite features disabled")
    
    startup_duration = time.time() - startup_time
    logger.info(f"✅ ClearSat API started in {startup_duration:.2f}s")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ClearSat API...")
    await cleanup_services()
    logger.info("✅ ClearSat API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="""
    ClearSat - Satellite Imagery Analysis Platform
    
    Provides environmental monitoring and analysis using satellite data.
    Features include vegetation indices, land cover classification, and water body detection.
    """,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# UPDATED CORS for production - REPLACE with your actual Netlify URL
# Replace your current CORS middleware with this:
# IMPROVE YOUR CORS:
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://clearsat.netlify.app", 
        "https://clearsat-frontend.netlify.app",
        "https://*.netlify.app",  # ADD THIS WILDCARD
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Enhanced validation error handler with proper error serialization"""
    logger.warning(f"Validation error for {request.url}: {exc.errors()}")
    
    # Extract clean error messages without ValueError objects
    clean_errors = []
    for error in exc.errors():
        clean_error = {
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        }
        # Extract the error message from ctx if it contains a ValueError
        if "ctx" in error and "error" in error["ctx"]:
            error_ctx = error["ctx"]
            if isinstance(error_ctx["error"], ValueError):
                clean_error["msg"] = str(error_ctx["error"])
        clean_errors.append(clean_error)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request data",
            "errors": clean_errors,
            "path": request.url.path
        },
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging"""
    logger.error(f"Unhandled exception for {request.method} {request.url}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_id": str(hash(time.time())),
            "path": request.url.path
        }
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "detail": "Endpoint not found",
            "path": request.url.path,
            "available_endpoints": ["/docs", "/health", "/users/", "/analysis/", "/reports/", "/profile/"]
        }
    )

# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Skip logging for health checks
    if request.url.path == "/health":
        response = await call_next(request)
        return response
    
    logger.info(f"📥 {request.method} {request.url.path} - Client: {request.client.host}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"📤 {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            f"💥 {request.method} {request.url.path} - "
            f"Error: {str(exc)} - "
            f"Time: {process_time:.3f}s"
        )
        raise

# Include routers with prefixes
app.include_router(users.router)
app.include_router(analysis.router)
app.include_router(reports.router)
app.include_router(profile.router)

# Enhanced health check endpoint
@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "ClearSat API",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.PROJECT_VERSION,
        "environment": settings.ENVIRONMENT
    }
    
    # Check MongoDB
    try:
        user_collection = get_user_collection()
        await user_collection.find_one({})
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = "disconnected"
        health_status["status"] = "degraded"
        logger.error(f"Health check - Database error: {e}")
    
    # Check Earth Engine
    try:
        if ee_service and ee_service.initialized:
            health_status["earth_engine"] = "connected"
        else:
            health_status["earth_engine"] = "disconnected"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["earth_engine"] = "error"
        health_status["status"] = "degraded"
        logger.error(f"Health check - Earth Engine error: {e}")
    
    return health_status

@app.get("/")
async def root():
    """Enhanced root endpoint with API information"""
    return {
        "message": "Welcome to ClearSat API",
        "version": settings.PROJECT_VERSION,
        "description": "Satellite Imagery Analysis Platform",
        "environment": settings.ENVIRONMENT,
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "users": "/users/",
            "analysis": "/analysis/",
            "reports": "/reports/",
            "profile": "/profile/"
        },
        "features": [
            "Vegetation Health Analysis (NDVI, EVI)",
            "Water Body Detection (NDWI)",
            "Urban Development Monitoring (BUI, NDBI)",
            "Land Cover Classification",
            "PDF Report Generation",
            "User Profiles & Settings"
        ]
    }

@app.get("/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "description": "Environmental monitoring using satellite imagery",
        "status": "operational",
        "maintainer": "ClearSat Team",
        "documentation": "/docs",
        "support": "support@clearsat.com"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info",
        access_log=True,
        timeout_keep_alive=5,
        limit_max_requests=1000
    )