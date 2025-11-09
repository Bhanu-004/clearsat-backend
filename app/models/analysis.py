# backend/app/models/analysis.py
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum

class AnalysisType(str, Enum):
    NDVI = "NDVI"
    NDWI = "NDWI"
    EVI = "EVI"
    BUI = "BUI"
    NDBI = "NDBI"
    LST = "LST"
    LAND_COVER = "LAND_COVER"
    ESA_LAND_COVER = "ESA_LAND_COVER"

class SatelliteSource(str, Enum):
    LANDSAT = "landsat"
    SENTINEL = "sentinel"
    MODIS = "modis" 

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Location(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    name: str = Field(..., min_length=1)
    state: Optional[str] = None
    district: Optional[str] = None

class AnalysisCreate(BaseModel):
    location: Location
    start_date: str
    end_date: str
    analysis_type: AnalysisType
    satellite_source: SatelliteSource = SatelliteSource.LANDSAT
    buffer_km: int = Field(default=10, ge=1, le=50)
    cloud_cover: int = Field(default=20, ge=0, le=100)

    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v

    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values:
            start = datetime.strptime(values['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if end <= start:
                raise ValueError('End date must be after start date')
            if (end - start).days > 365:
                raise ValueError('Analysis period cannot exceed 1 year')
        return v

class AnalysisUpdate(BaseModel):
    status: Optional[AnalysisStatus] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class AnalysisResponse(BaseModel):
    id: str
    user_id: str
    location: Location
    start_date: str
    end_date: str
    analysis_type: AnalysisType
    satellite_source: SatelliteSource
    status: AnalysisStatus
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class TimeSeriesData(BaseModel):
    date: str
    value: float

class AnalysisResults(BaseModel):
    time_series: List[TimeSeriesData]
    statistics: Dict[str, float]
    insights: List[str]
    images: Dict[str, str]
    report_data: Dict[str, Any]