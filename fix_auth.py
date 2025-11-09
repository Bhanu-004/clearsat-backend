# backend/force_ee_auth.py
import os
import ee
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def force_earth_engine_auth():
    """Force Earth Engine authentication with custom path"""
    try:
        # Force the path BEFORE importing ee
        custom_path = r'C:\Users\rama\AppData\Local\earthengine'
        os.environ['EE_CONFIG'] = custom_path
        
        # Ensure the directory exists
        os.makedirs(custom_path, exist_ok=True)
        
        # Monkey patch the problematic function
        import ee.oauth

        original_get_credentials_path = ee.oauth.get_credentials_path

        def fixed_get_credentials_path():
            custom_path = os.environ.get('EE_CONFIG')
            if custom_path:
                return os.path.join(custom_path, 'credentials')
            return original_get_credentials_path()

        ee.oauth.get_credentials_path = fixed_get_credentials_path

        print("🔄 Starting Earth Engine authentication...")
        print("📁 Using credentials path:", custom_path)
        
        # Authenticate
        ee.Authenticate(auth_mode='notebook')
        
        print("✅ Authentication successful!")
        
        # Test initialization
        ee.Initialize()
        print("✅ Earth Engine initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    success = force_earth_engine_auth()
    if success:
        print("\n🎉 Earth Engine is now ready to use!")
        print("You can now run: uvicorn app.main:app --reload")
    else:
        print("\n💥 Authentication failed. Please check the error above.")
        sys.exit(1)