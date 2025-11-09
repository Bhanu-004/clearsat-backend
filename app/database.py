# backend/app/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
from app.config import settings
import asyncio

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Connect to MongoDB with connection pooling"""
    try:
        mongodb.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            maxPoolSize=settings.MONGODB_MAX_POOL_SIZE,
            minPoolSize=settings.MONGODB_MIN_POOL_SIZE,
            retryWrites=True,
            socketTimeoutMS=30000,
            connectTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        
        # Test connection
        await mongodb.client.admin.command('ping')
        mongodb.database = mongodb.client[settings.DATABASE_NAME]
        
        # Create indexes
        await create_indexes()
        
        print("✅ Connected to MongoDB with connection pooling")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise

async def create_indexes():
    """Create database indexes for better performance"""
    try:
        database = get_database()  # Get database instance
        
        # User collection indexes
        user_collection = database["users"]
        await user_collection.create_index([("email", ASCENDING)], unique=True)
        await user_collection.create_index([("role", ASCENDING)])
        await user_collection.create_index([("created_at", ASCENDING)])
        
        # Analysis collection indexes
        analysis_collection = database["analyses"]
        await analysis_collection.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
        await analysis_collection.create_index([("status", ASCENDING)])
        await analysis_collection.create_index([("location.latitude", ASCENDING), ("location.longitude", ASCENDING)])
        
        # Report collection indexes
        report_collection = database["reports"]
        await report_collection.create_index([("analysis_id", ASCENDING)])
        await report_collection.create_index([("created_at", ASCENDING)])
        
        print("✅ Database indexes created successfully")
    except Exception as e:
        print(f"⚠️ Index creation warning: {e}")

async def close_mongo_connection():
    """Close MongoDB connection gracefully"""
    if mongodb.client:
        mongodb.client.close()
        print("❌ Disconnected from MongoDB")

def get_database():
    return mongodb.database

def get_user_collection():
    return mongodb.database["users"]

def get_analysis_collection():
    return mongodb.database["analyses"]

def get_report_collection():
    return mongodb.database["reports"]