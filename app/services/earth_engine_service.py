# backend/app/services/earth_engine_service.py
import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import urllib.request
from io import BytesIO
from PIL import Image, ImageDraw
import base64
import json
from typing import Dict, List, Optional, Tuple
import os
import requests
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import io

logger = logging.getLogger(__name__)

# Add MODIS dataset configurations at the top level
MODIS_DATASETS = {
    'modis_ndvi': {
        'collection': 'MODIS/006/MOD13Q1',
        'bands': ['NDVI', 'EVI'],
        'scale': 250,
        'description': 'MODIS Vegetation Indices (16-day, 250m)'
    },
    'modis_lst': {
        'collection': 'MODIS/006/MOD11A2', 
        'bands': ['LST_Day_1km', 'LST_Night_1km'],
        'scale': 1000,
        'description': 'MODIS Land Surface Temperature (8-day, 1km)'
    },
    'modis_surface_reflectance': {
        'collection': 'MODIS/006/MOD09GA',
        'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'],
        'scale': 500,
        'description': 'MODIS Surface Reflectance (Daily, 500m)'
    }
}

class EarthEngineService:
    def __init__(self):
        self.initialized = False
        self.ee_retry_count = 3
        
    def _force_ee_auth_path(self):
        """Force Earth Engine to use specific credentials path"""
        try:
            # Set the custom path BEFORE any EE operations
            custom_path = r'C:\Users\rama\AppData\Local\earthengine'
            os.environ['EE_CONFIG'] = custom_path
            
            # Monkey patch the credentials path function
            import ee.oauth
            
            original_get_credentials_path = ee.oauth.get_credentials_path
            
            def fixed_get_credentials_path():
                custom_path = os.environ.get('EE_CONFIG')
                if custom_path:
                    return os.path.join(custom_path, 'credentials')
                return original_get_credentials_path()
            
            ee.oauth.get_credentials_path = fixed_get_credentials_path
            logger.info("✅ Earth Engine auth path forced successfully")
            
        except Exception as e:
            logger.warning(f"Could not force EE auth path: {e}")
        
    def initialize_ee(self):
        """Initialize Earth Engine with robust error handling"""
        try:
            if self.initialized:
                return
                
            # Force authentication path first
            self._force_ee_auth_path()
                
            # Use service account credentials if provided
            service_account = os.getenv('EARTH_ENGINE_SERVICE_ACCOUNT')
            private_key = os.getenv('EARTH_ENGINE_PRIVATE_KEY')
            
            if service_account and private_key:
                logger.info("Using service account credentials")
                credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
                ee.Initialize(credentials)
            else:
                # Use default initialization with retry logic
                logger.info("Using personal account credentials")
                for attempt in range(self.ee_retry_count):
                    try:
                        ee.Initialize()
                        break
                    except ee.EEException as e:
                        if attempt == self.ee_retry_count - 1:
                            # Last attempt - provide helpful error message
                            error_msg = str(e)
                            if "Please authorize access" in error_msg:
                                logger.error("""
❌ Earth Engine authentication required!
Run this command in your terminal:
earthengine authenticate

Or create a service account and set EARTH_ENGINE_SERVICE_ACCOUNT and EARTH_ENGINE_PRIVATE_KEY environment variables.
                                """)
                            raise
                        logger.warning(f"Earth Engine initialization attempt {attempt + 1} failed: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            self.initialized = True
            logger.info("✅ Earth Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Earth Engine initialization failed: {e}")
            self.initialized = False
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ee.EEException, requests.RequestException))
    )
    def get_location_name(self, lat: float, lng: float) -> Dict[str, str]:
        """Get detailed location name with robust error handling"""
        try:
            # Validate coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError(f"Invalid coordinates: {lat}, {lng}")
            
            # Try OpenStreetMap Nominatim API
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json&addressdetails=1"
            headers = {
                'User-Agent': 'ClearSat-App/1.0 (https://github.com/clearsat-app)',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            address = data.get('address', {})
            
            # Extract location details with fallbacks
            name = (
                address.get('city') or 
                address.get('town') or 
                address.get('village') or 
                address.get('municipality') or
                address.get('suburb') or
                address.get('county') or
                f"Location ({lat:.4f}, {lng:.4f})"
            )
            
            state = address.get('state', 'Unknown')
            district = (
                address.get('county') or 
                address.get('state_district') or 
                address.get('region') or 
                'Unknown'
            )
            
            logger.info(f"📍 Geocoded: {name}, {district}, {state}")
            return {
                "name": name,
                "state": state,
                "district": district
            }
                
        except requests.RequestException as e:
            logger.warning(f"Geocoding API failed: {e}, using fallback")
            return self.get_location_name_fallback(lat, lng)
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return self.get_location_name_fallback(lat, lng)

    def get_location_name_fallback(self, lat: float, lng: float) -> Dict[str, str]:
        """Enhanced fallback location mapping"""
        try:
            # Extended Indian cities mapping
            indian_cities = {
                (28.6139, 77.2090): {"name": "New Delhi", "state": "Delhi", "district": "Central Delhi"},
                (19.0760, 72.8777): {"name": "Mumbai", "state": "Maharashtra", "district": "Mumbai City"},
                (12.9716, 77.5946): {"name": "Bengaluru", "state": "Karnataka", "district": "Bangalore Urban"},
                (13.0827, 80.2707): {"name": "Chennai", "state": "Tamil Nadu", "district": "Chennai"},
                (22.5726, 88.3639): {"name": "Kolkata", "state": "West Bengal", "district": "Kolkata"},
                (17.3850, 78.4867): {"name": "Hyderabad", "state": "Telangana", "district": "Hyderabad"},
                (23.0225, 72.5714): {"name": "Ahmedabad", "state": "Gujarat", "district": "Ahmedabad"},
                (18.5204, 73.8567): {"name": "Pune", "state": "Maharashtra", "district": "Pune"},
                (26.9124, 75.7873): {"name": "Jaipur", "state": "Rajasthan", "district": "Jaipur"},
                (26.8467, 80.9462): {"name": "Lucknow", "state": "Uttar Pradesh", "district": "Lucknow"},
                (21.1702, 72.8311): {"name": "Surat", "state": "Gujarat", "district": "Surat"},
                (15.2993, 74.1240): {"name": "Goa", "state": "Goa", "district": "North Goa"},
                (30.7333, 76.7794): {"name": "Chandigarh", "state": "Chandigarh", "district": "Chandigarh"},
                (11.0168, 76.9558): {"name": "Coimbatore", "state": "Tamil Nadu", "district": "Coimbatore"},
                (25.5941, 85.1376): {"name": "Patna", "state": "Bihar", "district": "Patna"},
                (34.0837, 74.7973): {"name": "Srinagar", "state": "Jammu & Kashmir", "district": "Srinagar"},
                (30.3165, 78.0322): {"name": "Dehradun", "state": "Uttarakhand", "district": "Dehradun"},
                (26.4499, 74.6399): {"name": "Ajmer", "state": "Rajasthan", "district": "Ajmer"},
                (9.9312, 76.2673): {"name": "Kochi", "state": "Kerala", "district": "Ernakulam"},
                (21.1458, 79.0882): {"name": "Nagpur", "state": "Maharashtra", "district": "Nagpur"}
            }
            
            # Find nearest city with improved distance calculation
            nearest_city = None
            min_distance = float('inf')
            
            for city_coords, city_info in indian_cities.items():
                city_lat, city_lng = city_coords
                # Haversine distance approximation
                distance = ((lat - city_lat)**2 + (lng - city_lng)**2)**0.5
                if distance < min_distance and distance < 0.5:  # Within 0.5 degrees (~55km)
                    min_distance = distance
                    nearest_city = city_info
            
            if nearest_city:
                logger.info(f"📍 Fallback geocoding: {nearest_city['name']}")
                return nearest_city
            
        except Exception as e:
            logger.error(f"Fallback geocoding failed: {e}")
        
        # Final fallback with formatted coordinates
        return {
            "name": f"Location ({lat:.4f}, {lng:.4f})",
            "state": "Unknown",
            "district": "Unknown"
        }

    def create_region(self, lat: float, lng: float, buffer_km: int = 10):
        """Create analysis region with validation"""
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise ValueError(f"Invalid coordinates: {lat}, {lng}")
        if buffer_km < 1 or buffer_km > 50:
            raise ValueError(f"Buffer must be between 1 and 50 km, got {buffer_km}")
            
        point = ee.Geometry.Point([lng, lat])
        return point.buffer(buffer_km * 1000)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_landsat_collection(self, region, start_date: str, end_date: str, cloud_cover: int = 20):
        """Get Landsat collection with retry logic"""
        return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_sentinel_collection(self, region, start_date: str, end_date: str, cloud_cover: int = 20):
        """Get Sentinel collection with retry logic"""
        return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_modis_collection(self, region, start_date: str, end_date: str, dataset_name: str = 'modis_ndvi'):
        """Get MODIS collection with retry logic"""
        if dataset_name not in MODIS_DATASETS:
            raise ValueError(f"Unknown MODIS dataset: {dataset_name}")
        
        dataset_config = MODIS_DATASETS[dataset_name]
        collection = ee.ImageCollection(dataset_config['collection']) \
            .filterBounds(region) \
            .filterDate(start_date, end_date)
        
        return collection, dataset_config

    def calculate_ndvi(self, image):
        """Calculate NDVI with band validation"""
        try:
            if 'SR_B5' in image.bandNames().getInfo():  # Landsat
                nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
                red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
            elif 'B8' in image.bandNames().getInfo():  # Sentinel-2
                nir = image.select('B8')
                red = image.select('B4')
            else:
                raise ValueError("Unsupported satellite type for NDVI")
            
            ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
            return ndvi.clamp(-1, 1)  # Ensure valid range
            
        except Exception as e:
            logger.error(f"NDVI calculation error: {e}")
            raise

    def calculate_modis_ndvi(self, image):
        """Calculate NDVI from MODIS"""
        try:
            ndvi = image.select('NDVI').multiply(0.0001).rename('NDVI')  # MODIS scaling factor
            return ndvi.clamp(-1, 1)
        except Exception as e:
            logger.error(f"MODIS NDVI calculation error: {e}")
            raise

    def calculate_ndwi(self, image):
        """Calculate NDWI with band validation"""
        try:
            if 'SR_B3' in image.bandNames().getInfo():  # Landsat
                green = image.select('SR_B3').multiply(0.0000275).add(-0.2)
                nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
            elif 'B3' in image.bandNames().getInfo():  # Sentinel-2
                green = image.select('B3')
                nir = image.select('B8')
            else:
                raise ValueError("Unsupported satellite type for NDWI")
            
            ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
            return ndwi.clamp(-1, 1)  # Ensure valid range
            
        except Exception as e:
            logger.error(f"NDWI calculation error: {e}")
            raise

    def calculate_evi(self, image):
        """Calculate EVI with band validation"""
        try:
            if 'SR_B5' in image.bandNames().getInfo():  # Landsat
                nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
                red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
                blue = image.select('SR_B2').multiply(0.0000275).add(-0.2)
            elif 'B8' in image.bandNames().getInfo():  # Sentinel-2
                nir = image.select('B8')
                red = image.select('B4')
                blue = image.select('B2')
            else:
                raise ValueError("Unsupported satellite type for EVI")
            
            # EVI formula: 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
            evi = image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': nir,
                    'RED': red,
                    'BLUE': blue
                }
            ).rename('EVI')
            return evi.clamp(-1, 2)  # EVI can exceed 1 in some cases
            
        except Exception as e:
            logger.error(f"EVI calculation error: {e}")
            raise

    def calculate_modis_evi(self, image):
        """Calculate EVI from MODIS"""
        try:
            evi = image.select('EVI').multiply(0.0001).rename('EVI')  # MODIS scaling factor
            return evi.clamp(-1, 2)
        except Exception as e:
            logger.error(f"MODIS EVI calculation error: {e}")
            raise

    def calculate_bui(self, image):
        """Calculate BUI with band validation"""
        try:
            if 'SR_B6' in image.bandNames().getInfo():  # Landsat
                swir1 = image.select('SR_B6').multiply(0.0000275).add(-0.2)
                nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
                green = image.select('SR_B3').multiply(0.0000275).add(-0.2)
            elif 'B11' in image.bandNames().getInfo():  # Sentinel-2
                swir1 = image.select('B11')
                nir = image.select('B8')
                green = image.select('B3')
            else:
                raise ValueError("Unsupported satellite type for BUI")
            
            bui = image.expression(
                '((SWIR1 - NIR) / (SWIR1 + NIR)) - ((NIR - GREEN) / (NIR + GREEN))',
                {
                    'SWIR1': swir1,
                    'NIR': nir,
                    'GREEN': green
                }
            ).rename('BUI')
            return bui.clamp(-2, 2)
            
        except Exception as e:
            logger.error(f"BUI calculation error: {e}")
            raise

    def calculate_ndbi(self, image):
        """Calculate NDBI with band validation"""
        try:
            if 'SR_B6' in image.bandNames().getInfo():  # Landsat
                swir1 = image.select('SR_B6').multiply(0.0000275).add(-0.2)
                nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
            elif 'B11' in image.bandNames().getInfo():  # Sentinel-2
                swir1 = image.select('B11')
                nir = image.select('B8')
            else:
                raise ValueError("Unsupported satellite type for NDBI")
            
            ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI')
            return ndbi.clamp(-1, 1)
            
        except Exception as e:
            logger.error(f"NDBI calculation error: {e}")
            raise

    def calculate_lst(self, image):
        """Calculate LST with band validation"""
        try:
            if 'ST_B10' in image.bandNames().getInfo():  # Landsat
                # Convert to Celsius from Kelvin
                lst_kelvin = image.select('ST_B10').multiply(0.00341802).add(149.0)
                lst_celsius = lst_kelvin.subtract(273.15)
                return lst_celsius.rename('LST')
            else:
                raise ValueError("LST calculation currently supported only for Landsat")
                
        except Exception as e:
            logger.error(f"LST calculation error: {e}")
            raise

    def calculate_modis_lst(self, image):
        """Calculate LST from MODIS"""
        try:
            # MODIS LST is in Kelvin, convert to Celsius
            lst_day = image.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST')
            return lst_day.clamp(-50, 60)  # Reasonable Earth temperature range
        except Exception as e:
            logger.error(f"MODIS LST calculation error: {e}")
            raise

    def calculate_land_cover(self, image):
        """Calculate basic land cover classification"""
        try:
            ndvi = self.calculate_ndvi(image)
            ndbi = self.calculate_ndbi(image)
            
            # Enhanced classification
            vegetation = ndvi.gt(0.3)
            builtup = ndbi.gt(0.1)
            water = ndvi.lt(0).And(ndbi.lt(0))
            
            # Create land cover classes: 1=Vegetation, 2=Built-up, 3=Water, 4=Other
            land_cover = vegetation.multiply(1) \
                .add(builtup.multiply(2)) \
                .add(water.multiply(3))
            land_cover = land_cover.where(land_cover.eq(0), 4)  # Other where 0
            
            return land_cover.rename('LAND_COVER')
            
        except Exception as e:
            logger.error(f"Land cover calculation error: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def calculate_esa_land_cover(self, region, start_date: str, end_date: str):
        """Calculate ESA WorldCover land cover with retry logic"""
        try:
            # ESA WorldCover is a static dataset for 2020/2021
            # Use the most recent version available
            esa_collection = ee.ImageCollection("ESA/WorldCover/v200")
            
            # For static datasets, we don't filter by date, just get the first image
            esa_image = esa_collection \
                .filterBounds(region) \
                .first()
            
            if esa_image is None:
                raise ValueError("No ESA land cover data available for the given region")
            
            return esa_image.clip(region).select('Map').rename('ESA_LAND_COVER')
            
        except Exception as e:
            logger.error(f"ESA land cover error: {e}")
            raise

    def get_esa_land_cover_stats(self, esa_image, region):
        """Get statistics for ESA land cover classes with error handling"""
        try:
            esa_classes = {
                10: "Trees", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
                50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
                80: "Permanent water bodies", 90: "Herbaceous wetland",
                95: "Mangroves", 100: "Moss and lichen"
            }
            
            area_image = ee.Image.pixelArea().addBands(esa_image)
            
            stats = area_image.reduceRegion(
                reducer=ee.Reducer.sum().group(1, 'class'),
                geometry=region,
                scale=10,
                maxPixels=1e9,
                bestEffort=True
            ).getInfo()
            
            total_area = 0
            class_areas = {}
            class_percentages = {}
            
            for group in stats.get('groups', []):
                class_id = group['class']
                area_sq_m = group['sum']
                area_hectares = area_sq_m / 10000
                class_name = esa_classes.get(class_id, f"Class {class_id}")
                class_areas[class_name] = area_hectares
                total_area += area_hectares
            
            # Calculate percentages
            for class_name, area in class_areas.items():
                if total_area > 0:
                    percentage = (area / total_area) * 100
                    class_percentages[class_name] = round(percentage, 2)
            
            dominant_cover = max(class_percentages, key=class_percentages.get) if class_percentages else "Unknown"
            
            return {
                'total_area_hectares': round(total_area, 2),
                'class_areas': {k: round(v, 2) for k, v in class_areas.items()},
                'class_percentages': class_percentages,
                'dominant_land_cover': dominant_cover
            }
            
        except Exception as e:
            logger.error(f"ESA stats calculation error: {e}")
            raise

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def get_time_series_data(self, collection, region, index_function):
        """Extract time series data with robust error handling"""
        dates_list = []
        values_list = []
        
        try:
            collection_list = collection.toList(collection.size())
            n_images = collection.size().getInfo()
            
            if n_images == 0:
                return pd.DataFrame({'date': dates_list, 'value': values_list})
            
            for i in range(n_images):
                try:
                    image = ee.Image(collection_list.get(i))
                    date = image.date().format('YYYY-MM-dd').getInfo()
                    
                    index_image = index_function(image)
                    mean_dict = index_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=region,
                        scale=100,
                        bestEffort=True,
                        maxPixels=1e8
                    ).getInfo()
                    
                    value = list(mean_dict.values())[0] if mean_dict else None
                    
                    if value is not None and not np.isnan(value):
                        dates_list.append(date)
                        values_list.append(float(value))
                        
                except Exception as e:
                    logger.warning(f"Skipping image {i}: {e}")
                    continue
            
            return pd.DataFrame({'date': dates_list, 'value': values_list})
            
        except Exception as e:
            logger.error(f"Time series extraction error: {e}")
            return pd.DataFrame({'date': dates_list, 'value': values_list})

    def generate_insights(self, analysis_type: str, statistics: Dict, time_series_data: List) -> List[str]:
        """Generate enhanced human-readable insights"""
        insights = []
        mean_val = statistics.get('mean', 0)
        count = statistics.get('count', 0)
        
        # Add data quality insight
        if count < 5:
            insights.append("📊 Limited data points available for analysis")
        elif count > 20:
            insights.append("📈 Robust dataset with good temporal coverage")
        
        if analysis_type == "NDVI":
            if mean_val > 0.6:
                insights.extend([
                    "🌿 Excellent vegetation health with dense canopy cover",
                    "💪 Ideal for agricultural monitoring and forest management",
                    "🌳 High biomass accumulation detected"
                ])
            elif mean_val > 0.3:
                insights.extend([
                    "🌾 Moderate vegetation with healthy crop/plant growth",
                    "📈 Good conditions for agricultural productivity",
                    "💧 Adequate moisture levels supporting vegetation"
                ])
            elif mean_val > 0.1:
                insights.extend([
                    "🌱 Sparse vegetation, may indicate early growth or stress",
                    "⚠️ Monitor for potential irrigation needs",
                    "🔍 Consider soil health assessment"
                ])
            else:
                insights.extend([
                    "🏢 Low vegetation density, typical of urban or barren areas",
                    "🔍 Consider urban planning or land use analysis",
                    "🌡️ Potential for urban heat island effect"
                ])
                
        elif analysis_type == "NDWI":
            if mean_val > 0.2:
                insights.extend([
                    "💧 Significant water presence detected",
                    "🌊 Suitable for water resource management and flood monitoring",
                    "💦 High moisture content in vegetation and soil"
                ])
            elif mean_val > 0:
                insights.extend([
                    "💦 Moderate moisture levels in vegetation and soil",
                    "🌧️ Good conditions for crop health",
                    "⚖️ Balanced water content for ecosystem"
                ])
            else:
                insights.extend([
                    "🏜️ Dry conditions with limited water content",
                    "🚰 Consider irrigation assessment",
                    "🌵 Potential drought stress indicators"
                ])
                
        elif analysis_type == "EVI":
            if mean_val > 0.5:
                insights.extend([
                    "🌳 Excellent vegetation vigor with minimal atmospheric influence",
                    "🌱 Ideal for dense vegetation monitoring in humid regions",
                    "📊 Superior canopy structure metrics"
                ])
            elif mean_val > 0.2:
                insights.extend([
                    "🌿 Good vegetation condition with reliable canopy metrics",
                    "📊 Suitable for precision agriculture applications",
                    "🌤️ Reduced atmospheric distortion in measurements"
                ])
            else:
                insights.extend([
                    "🏙️ Low to moderate vegetation cover",
                    "🔬 Consider soil quality and land management practices",
                    "🏗️ Potential urban or mixed-use landscape"
                ])
        
        # Add seasonal context for India
        insights.extend([
            "🇮🇳 Analysis focused on Indian subcontinent conditions",
            "📅 Consider seasonal variations in monsoon climate",
            "🌍 Regional environmental factors incorporated"
        ])
        
        return insights[:8]  # Limit to top 8 insights

    def generate_fallback_image(self, region, title: str) -> str:
        """Generate a fallback image when Earth Engine fails"""
        try:
            # Create a simple colored image as fallback
            width, height = 512, 512
            image = Image.new('RGB', (width, height), color='lightblue')
            draw = ImageDraw.Draw(image)
            
            # Draw a simple representation
            draw.ellipse([50, 50, 462, 462], outline='green', width=5)
            draw.text((width//2, height//2), "Satellite Image\nNot Available", 
                     fill='black', anchor='mm', align='center')
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            image_data = buffer.getvalue()
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"Fallback image generation also failed: {e}")
            # Return a tiny transparent pixel as last resort
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def generate_satellite_image(self, composite, region, title: str) -> Optional[str]:
        """Generate base64 encoded satellite image with robust error handling"""
        try:
            region_bounds = region.bounds().getInfo()['coordinates'][0]
            band_names = composite.bandNames().getInfo()
            
            # Enhanced visualization parameters with better defaults
            if 'SR_B4' in band_names:  # Landsat
                vis_params = {
                    'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # Red, Green, Blue for true color
                    'min': 0.0,
                    'max': 0.3,
                    'gamma': 1.2
                }
            elif 'B4' in band_names:  # Sentinel-2
                vis_params = {
                    'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.2
                }
            elif 'NDVI' in band_names:  # For index images
                vis_params = {
                    'bands': ['NDVI'],
                    'min': -1,
                    'max': 1,
                    'palette': ['blue', 'white', 'green']
                }
            else:
                # Use first available bands as fallback
                available_bands = band_names[:3] if len(band_names) >= 3 else band_names
                vis_params = {
                    'bands': available_bands,
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.2
                }
            
            # Generate thumbnail with better parameters
            thumbnail_url = composite.visualize(**vis_params).getThumbURL({
                'region': region_bounds,
                'dimensions': 512,
                'format': 'png',
                'crs': 'EPSG:3857'
            })
            
            # Download with timeout and retry
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/png,image/*;q=0.8,*/*;q=0.5'
            }
            
            req = urllib.request.Request(thumbnail_url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=60) as response:
                image_data = response.read()
                
            # Validate image size - should be reasonable for a 512x512 PNG
            if len(image_data) < 5000:  # Increased minimum size
                logger.warning(f"Generated image too small ({len(image_data)} bytes), likely error")
                return self.generate_fallback_image(region, title)
                
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"Satellite image generation error: {e}")
            return self.generate_fallback_image(region, title)

    async def perform_analysis(self, analysis_data: Dict) -> Dict:
        """Main analysis function with comprehensive error handling"""
        if not self.initialized:
            self.initialize_ee()
            
        try:
            # Validate input data
            location = analysis_data['location']
            lat, lng = location['latitude'], location['longitude']
            start_date = analysis_data['start_date']
            end_date = analysis_data['end_date']
            
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError(f"Invalid coordinates: {lat}, {lng}")
            
            # Create analysis region
            region = self.create_region(lat, lng, analysis_data.get('buffer_km', 10))
            
            # Handle ESA Land Cover separately
            if analysis_data['analysis_type'] == 'ESA_LAND_COVER':
                esa_image = self.calculate_esa_land_cover(region, start_date, end_date)
                esa_stats = self.get_esa_land_cover_stats(esa_image, region)
                insights = self.generate_insights(analysis_data['analysis_type'], esa_stats, [])
                land_cover_image = self.generate_satellite_image(esa_image, region, "ESA Land Cover")
                
                results = {
                    'time_series': [],
                    'statistics': esa_stats,
                    'insights': insights,
                    'images': {'satellite': land_cover_image},
                    'report_data': {
                        'location_name': location['name'],
                        'analysis_period': f"{start_date} to {end_date}",
                        'satellite_used': 'ESA_WORLDCOVER',
                        'area_covered': f"{analysis_data.get('buffer_km', 10)}km radius"
                    }
                }
                return results

            # Handle MODIS datasets
            satellite_source = analysis_data.get('satellite_source', 'landsat')
            if satellite_source == 'modis':
                # Determine which MODIS dataset to use based on analysis type
                modis_dataset_map = {
                    'NDVI': 'modis_ndvi',
                    'EVI': 'modis_ndvi',
                    'LST': 'modis_lst'
                }
                
                dataset_name = modis_dataset_map.get(analysis_data['analysis_type'], 'modis_ndvi')
                collection, dataset_config = self.get_modis_collection(region, start_date, end_date, dataset_name)
                
                # Check collection size
                collection_size = collection.size().getInfo()
                if collection_size == 0:
                    raise ValueError(f"No MODIS images found for the given parameters")
                
                logger.info(f"📡 Processing {collection_size} MODIS images from {dataset_config['collection']}")
                
                # Get composite
                composite = collection.median().clip(region)
                
                # MODIS analysis functions
                modis_functions = {
                    'NDVI': self.calculate_modis_ndvi,
                    'EVI': self.calculate_modis_evi,
                    'LST': self.calculate_modis_lst
                }
                
                index_func = modis_functions.get(analysis_data['analysis_type'])
                if not index_func:
                    # For unsupported MODIS analysis types, fall back to standard calculation
                    logger.warning(f"Analysis type {analysis_data['analysis_type']} not directly supported for MODIS, using standard calculation")
                    index_func = self.calculate_ndvi  # fallback
                
                # Get time series data
                df = self.get_time_series_data(collection, region, index_func)
                
                if df.empty:
                    raise ValueError("No valid data extracted from MODIS images")
                
                # Calculate enhanced statistics
                statistics = {
                    'mean': float(df['value'].mean()),
                    'median': float(df['value'].median()),
                    'std': float(df['value'].std()),
                    'min': float(df['value'].min()),
                    'max': float(df['value'].max()),
                    'count': len(df),
                    'range': float(df['value'].max() - df['value'].min()),
                    'q1': float(df['value'].quantile(0.25)),
                    'q3': float(df['value'].quantile(0.75))
                }
                
                # Generate insights and images
                insights = self.generate_insights(analysis_data['analysis_type'], statistics, df.to_dict('records'))
                satellite_image = self.generate_satellite_image(composite, region, f"MODIS {analysis_data['analysis_type']} Analysis")
                
                # Prepare comprehensive results
                results = {
                    'time_series': df.to_dict('records'),
                    'statistics': statistics,
                    'insights': insights,
                    'images': {'satellite': satellite_image},
                    'report_data': {
                        'location_name': location['name'],
                        'analysis_period': f"{start_date} to {end_date}",
                        'satellite_used': f"MODIS ({dataset_config['description']})",
                        'area_covered': f"{analysis_data.get('buffer_km', 10)}km radius",
                        'images_processed': collection_size,
                        'data_points': len(df)
                    }
                }
                
                logger.info(f"✅ MODIS analysis completed successfully with {len(df)} data points")
                return results
            
            # For other analysis types (Landsat/Sentinel)
            if satellite_source == 'landsat':
                collection = self.get_landsat_collection(region, start_date, end_date, analysis_data.get('cloud_cover', 20))
            elif satellite_source == 'sentinel':
                collection = self.get_sentinel_collection(region, start_date, end_date, analysis_data.get('cloud_cover', 20))
            else:
                collection = self.get_landsat_collection(region, start_date, end_date, analysis_data.get('cloud_cover', 20))
            
            # Check collection size
            collection_size = collection.size().getInfo()
            if collection_size == 0:
                # Try with extended date range for static datasets
                if analysis_data['analysis_type'] in ['LAND_COVER', 'ESA_LAND_COVER']:
                    # Use a wider date range for land cover
                    extended_start = '2020-01-01'
                    extended_end = '2023-12-31'
                    if satellite_source == 'landsat':
                        collection = self.get_landsat_collection(region, extended_start, extended_end, analysis_data.get('cloud_cover', 50))
                    elif satellite_source == 'sentinel':
                        collection = self.get_sentinel_collection(region, extended_start, extended_end, analysis_data.get('cloud_cover', 50))
                    
                    collection_size = collection.size().getInfo()
                    if collection_size == 0:
                        raise ValueError(f"No {satellite_source} images found even with extended date range")
                else:
                    raise ValueError(f"No {satellite_source} images found for the given parameters")
            
            logger.info(f"📡 Processing {collection_size} satellite images")
            
            # Get composite and calculate index
            composite = collection.median().clip(region)
            
            index_functions = {
                'NDVI': self.calculate_ndvi,
                'NDWI': self.calculate_ndwi,
                'EVI': self.calculate_evi,
                'BUI': self.calculate_bui,
                'NDBI': self.calculate_ndbi,
                'LST': self.calculate_lst,
                'LAND_COVER': self.calculate_land_cover
            }
            
            index_func = index_functions.get(analysis_data['analysis_type'])
            if not index_func:
                raise ValueError(f"Unsupported analysis type: {analysis_data['analysis_type']}")
            
            # Get time series data
            df = self.get_time_series_data(collection, region, index_func)
            
            if df.empty:
                # For land cover, create dummy time series
                if analysis_data['analysis_type'] == 'LAND_COVER':
                    df = pd.DataFrame({
                        'date': [start_date],
                        'value': [0.5]  # Default value for land cover
                    })
                else:
                    raise ValueError("No valid data extracted from satellite images")
            
            # Calculate enhanced statistics
            statistics = {
                'mean': float(df['value'].mean()),
                'median': float(df['value'].median()),
                'std': float(df['value'].std()),
                'min': float(df['value'].min()),
                'max': float(df['value'].max()),
                'count': len(df),
                'range': float(df['value'].max() - df['value'].min()),
                'q1': float(df['value'].quantile(0.25)),
                'q3': float(df['value'].quantile(0.75))
            }
            
            # Generate insights and images
            insights = self.generate_insights(analysis_data['analysis_type'], statistics, df.to_dict('records'))
            satellite_image = self.generate_satellite_image(composite, region, f"{analysis_data['analysis_type']} Analysis")
            
            # Prepare comprehensive results
            results = {
                'time_series': df.to_dict('records'),
                'statistics': statistics,
                'insights': insights,
                'images': {'satellite': satellite_image},
                'report_data': {
                    'location_name': location['name'],
                    'analysis_period': f"{start_date} to {end_date}",
                    'satellite_used': satellite_source.upper(),
                    'area_covered': f"{analysis_data.get('buffer_km', 10)}km radius",
                    'images_processed': collection_size,
                    'data_points': len(df)
                }
            }
            
            logger.info(f"✅ Analysis completed successfully with {len(df)} data points")
            return results
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise

# Global instance with error handling
try:
    ee_service = EarthEngineService()
except Exception as e:
    logger.error(f"Failed to initialize Earth Engine service: {e}")
    ee_service = None