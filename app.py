# app.py

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import tensorflow as tf
import numpy as np
from datetime import date, timedelta
import ee
import os

from src.ee_pipeline import get_s2_images, get_map_tile_url, get_ee_credentials
from src.utils import convert_array_to_bytes, load_image_from_url, calculate_area_hectares

# --- Page Configuration ---
st.set_page_config(
    page_title="Satellite Super-Resolution",
    page_icon="🛰️",
    layout="wide"
)

# --- Initialize Earth Engine first ---
get_ee_credentials()

# --- Model Loading ---
@st.cache_resource
def load_sr_model(model_path):
    try:
        if not os.path.exists(model_path):
            return None, f"Model file not found at '{model_path}'"
        return tf.keras.models.load_model(model_path, compile=False), "Model loaded successfully"
        # return tf.keras.models.load_model(model_path), "Model loaded successfully"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Use forward slashes for cross-platform compatibility
MODEL_PATH = "models/sr_generator_compatible.h5"
# Try compatible model first if it exists
COMPATIBLE_MODEL_PATH = "models/sr_generator_compatible.h5"

if os.path.exists(COMPATIBLE_MODEL_PATH):
    model, model_status = load_sr_model(COMPATIBLE_MODEL_PATH)
    st.info("Using compatible model version")
else:
    model, model_status = load_sr_model(MODEL_PATH)

if model is None:
    st.error(f"❌ {model_status}")
    st.info("Please ensure:")
    st.write("1. Create a `models/` folder in your project directory")
    st.write("2. Place your `sr_generator_final.h5` file inside the `models/` folder")
    st.write("3. Restart the application")
    st.stop()
else:
    st.success(f"✅ {model_status}")

# --- Main App ---
st.title("🛰️ Interactive Satellite Super-Resolution")
st.markdown("Draw a region of interest (ROI) on the map, select a date, and generate a high-resolution image from two low-resolution Sentinel-2 captures.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    
    # Date Input
    selected_date = st.date_input(
        "Select a target date",
        value=date(2023, 7, 1),
        min_value=date(2017, 3, 23),
        max_value=date.today()
    )
    time_window = st.slider("Time window (days)", 15, 180, 60)
    
    start_date = selected_date - timedelta(days=time_window // 2)
    end_date = selected_date + timedelta(days=time_window // 2)
    
    st.info(f"Searching for images between:\n**{start_date}** and **{end_date}**")

    st.markdown("---")
    st.markdown("Made with ❤️ for a clearer world.")


# --- Map and ROI Drawing ---
if 'center' not in st.session_state:
    st.session_state.center = [20.5937, 78.9629]  # Default to India
    st.session_state.zoom = 5

m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom, tiles="CartoDB positron")
Draw(export=True, draw_options={'rectangle': {'shapeOptions': {'color': '#00A36C'}}}).add_to(m)

st.write("### 1. Draw your Region of Interest (ROI) on the map")
map_data = st_folium(m, width='100%', height=500)

# --- Processing Pipeline ---
if map_data and map_data.get("last_active_drawing"):
    roi_geom = map_data["last_active_drawing"]["geometry"]
    if roi_geom and roi_geom.get("type") == "Polygon":
        coords = roi_geom['coordinates'][0]
        bounds = [[coords[0][1], coords[0][0]], [coords[2][1], coords[2][0]]]
        area = calculate_area_hectares(bounds)

        st.info(f"ROI Area: **{area:.2f} hectares**")
        
        # Area Validation
        if area > 10000:
            st.error("ROI is too large! Please select an area smaller than 100 hectares.")
        else:
            if st.button("✨ Generate Super-Resolved Image", type="primary"):
                # **MODIFICATION**: Clear previous results from session state
                for key in ['lr1_url', 'lr2_url', 'sr_image', 'ee_roi', 'bounds']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                with st.spinner("Step 1/3: Fetching low-resolution satellite images..."):
                    try:
                        ee_coords = [coords[0] + coords[2]]
                        ee_roi = ee.Geometry.Rectangle(ee_coords[0])
                        
                        lr1_url, lr2_url, message = get_s2_images(ee_roi, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                        st.info(message)

                        # **MODIFICATION**: Store results in session_state on success
                        if lr1_url and lr2_url:
                            st.session_state.lr1_url = lr1_url
                            st.session_state.lr2_url = lr2_url
                            st.session_state.ee_roi = ee_roi
                            st.session_state.bounds = bounds

                    except Exception as e:
                        st.error(f"Error fetching satellite images: {str(e)}")

# **MODIFICATION**: This entire block is new. It runs on every script re-run to display results if they exist in the session state.
if 'lr1_url' in st.session_state and 'lr2_url' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        # **MODIFICATION**: Fixed deprecation warning
        st.image(st.session_state.lr1_url, caption="Low-Res Image 1 (Input)", use_container_width=True)
    with col2:
        # **MODIFICATION**: Fixed deprecation warning
        st.image(st.session_state.lr2_url, caption="Low-Res Image 2 (Input)", use_container_width=True)
    
    # Only run the model if the super-resolved image isn't already generated
    if 'sr_image' not in st.session_state:
        with st.spinner("Step 2/3: Running super-resolution model..."):
            try:
                lr1 = load_image_from_url(st.session_state.lr1_url) / 255.0
                lr2 = load_image_from_url(st.session_state.lr2_url) / 255.0
                
                if lr1.shape != (64, 64, 3) or lr2.shape != (64, 64, 3):
                    st.warning(f"Input images have unexpected shape: {lr1.shape}, {lr2.shape}")
                
                # **MODIFICATION**: Store the generated image in session_state
                st.session_state.sr_image = model.predict([np.expand_dims(lr1, axis=0), np.expand_dims(lr2, axis=0)])[0]

            except Exception as e:
                st.error(f"Error during super-resolution: {str(e)}")

    # Display the final output if the super-resolved image exists in the state
    if 'sr_image' in st.session_state and st.session_state.sr_image is not None:
        with st.spinner("Step 3/3: Preparing final output..."):
            st.success("Super-Resolution Complete!")
            # **MODIFICATION**: Fixed deprecation warning
            st.image(st.session_state.sr_image, caption="Generated High-Resolution Image", use_container_width=True)

            st.download_button(
                label="📥 Download High-Resolution Image",
                data=convert_array_to_bytes(st.session_state.sr_image),
                file_name=f"super_resolved_{selected_date}.png",
                mime="image/png"
            )
            
            # --- Display result on a new map ---
            st.write("### Final Result on Map")
            result_map = folium.Map(location=[(st.session_state.bounds[0][0]+st.session_state.bounds[1][0])/2, (st.session_state.bounds[0][1]+st.session_state.bounds[1][1])/2], zoom_start=14)
            
            try:
                tiles = get_map_tile_url(st.session_state.ee_roi)
                folium.TileLayer(tiles=tiles, attr='Google Earth Engine', name='Satellite View').add_to(result_map)
                
                img_overlay = folium.raster_layers.ImageOverlay(
                    image=st.session_state.sr_image,
                    bounds=st.session_state.bounds,
                    opacity=0.8,
                    name="Super-Resolved Overlay"
                )
                img_overlay.add_to(result_map)
                folium.LayerControl().add_to(result_map)
                
                st_folium(result_map, width='100%', height=500)
            except Exception as e:
                st.warning(f"Could not display result on map: {str(e)}")

