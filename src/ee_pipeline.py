#### `src/ee_pipeline.py`
import ee
import numpy as np
import streamlit as st
from PIL import Image
from geopy.geocoders import Nominatim

# Use caching so that initialization happens only once
@st.cache_resource
def get_ee_credentials():
    """Authenticates and initializes Earth Engine with project ID."""
    try:
        # First try to initialize with your project ID
        ee.Initialize(project='ee-deepeshy')
        st.success("Google Earth Engine initialized successfully ✅")
        return True
    except Exception as e:
        try:
            # Fallback: try without project ID (for users with different setup)
            ee.Initialize()
            st.success("Google Earth Engine initialized successfully ✅")
            return True
        except Exception as e2:
            st.error(f"Could not initialize Earth Engine: {e}")
            st.error("Please run `earthengine authenticate` in your terminal first.")
            st.info("Steps to authenticate:")
            st.write("1. Open terminal/command prompt")
            st.write("2. Run: `earthengine authenticate`")
            st.write("3. Follow the authentication process")
            st.write("4. Restart this Streamlit app")
            st.stop()
            return False

def get_s2_images(roi, start_date, end_date):
    """
    Fetches two least cloudy Sentinel-2 images for a given ROI and date range.
    """
    try:
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(roi)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Increased threshold
                         .sort('CLOUDY_PIXEL_PERCENTAGE'))

        count = s2_collection.size().getInfo()
        if count < 2:
            # Try with higher cloud threshold
            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(roi)
                             .filterDate(start_date, end_date)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                             .sort('CLOUDY_PIXEL_PERCENTAGE'))
            count = s2_collection.size().getInfo()
            
        if count < 2:
            return None, None, f"Could not find two images for the selected period. Found {count} images. Please try a wider date range or different location."

        # Get the two least cloudy images
        img1 = ee.Image(s2_collection.toList(2).get(0))
        img2 = ee.Image(s2_collection.toList(2).get(1))

        vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}

        # Fetch image data as NumPy arrays
        # We fetch a thumbnail, which is efficient for visualization
        thumb_url1 = img1.visualize(**vis_params).getThumbURL({
            'dimensions': '64x64', 
            'region': roi, 
            'format': 'png'
        })
        thumb_url2 = img2.visualize(**vis_params).getThumbURL({
            'dimensions': '64x64', 
            'region': roi, 
            'format': 'png'
        })
        
        return thumb_url1, thumb_url2, f"Successfully found {count} images. Using the two clearest."
        
    except Exception as e:
        return None, None, f"Error fetching satellite images: {str(e)}"

def get_map_tile_url(roi):
    """Gets a map tile layer URL for the ROI."""
    try:
        image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate('2023-01-01', '2023-12-31')
                 .median())  # Use a median composite for a clear view
        
        vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
        map_id = image.getMapId(vis_params)
        return map_id['tile_fetcher'].url_format
    except Exception as e:
        st.warning(f"Could not get map tiles: {str(e)}")
        return None