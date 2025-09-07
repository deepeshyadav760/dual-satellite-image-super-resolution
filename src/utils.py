from PIL import Image
import numpy as np
from io import BytesIO
import requests

def load_image_from_url(url):
    """Loads an image from a URL and converts to a NumPy array."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Ensure the image is 64x64 as expected by the model
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
        return np.array(img)
    except Exception as e:
        raise Exception(f"Failed to load image from URL: {str(e)}")

def convert_array_to_bytes(image_array):
    """Converts a NumPy array to bytes for download."""
    try:
        # Ensure values are in [0, 1] range
        image_array = np.clip(image_array, 0, 1)
        
        # Convert to 8-bit image
        img_8bit = (image_array * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(img_8bit)
        
        # Convert to bytes
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        raise Exception(f"Failed to convert array to bytes: {str(e)}")

def calculate_area_hectares(bounds):
    """A simplified area calculation. Not perfectly accurate but good for validation."""
    try:
        lat1, lon1 = bounds[0]
        lat2, lon2 = bounds[1]
        
        # Rough approximation using haversine-like calculation
        lat_diff = abs(lat2 - lat1)
        lon_diff = abs(lon2 - lon1)
        avg_lat = (lat1 + lat2) / 2
        
        # Convert degrees to approximate km
        lat_km = lat_diff * 111.32  # 1 degree latitude ≈ 111.32 km
        lon_km = lon_diff * 111.32 * np.cos(np.radians(avg_lat))  # adjust for longitude
        
        area_sq_km = lat_km * lon_km
        area_hectares = area_sq_km * 100  # 1 sq km = 100 hectares
        
        return area_hectares
    except Exception as e:
        return 0.0  # Return 0 if calculation fails
