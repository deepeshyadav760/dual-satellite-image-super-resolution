# 🛰️ Interactive Satellite Super-Resolution

A web application built with Streamlit and TensorFlow that generates high-resolution satellite imagery from two low-resolution Sentinel-2 captures using a deep learning model. The app leverages Google Earth Engine to fetch satellite data in real-time.

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 📖 About The Project

This project addresses the challenge of obtaining high-resolution satellite imagery, which is often commercially expensive or unavailable for specific timeframes. By using a deep learning super-resolution model, this application can synthesize a clearer, more detailed image from two readily available, lower-resolution images from the Copernicus Sentinel-2 satellite mission.

Users can interactively select any region of interest (ROI) on a global map, choose a target date, and the application's backend pipeline handles the rest:
1.  It queries the Google Earth Engine catalog for two suitable, low-cloud-cover images within the specified timeframe.
2.  It feeds these two 64x64 pixel images into a pre-trained TensorFlow/Keras model.
3.  The model generates a new, higher-resolution image that reconstructs details and sharpens features.
4.  The result is displayed to the user, who can download it or view it overlaid on the map.

---

## ✨ Key Features

- **Interactive Map Interface:** Draw a rectangular Region of Interest (ROI) directly on a Folium map.
- **Flexible Date Selection:** Choose a target date and a time window to search for satellite imagery.
- **Real-Time Data Fetching:** Integrates with Google Earth Engine to find and retrieve the latest available Sentinel-2 images.
- **Deep Learning Powered:** Utilizes a TensorFlow/Keras super-resolution model to enhance image quality.
- **Comparative View:** Displays both low-resolution input images alongside the final high-resolution output.
- **Downloadable Results:** Save the generated high-resolution image as a PNG file.
- **Map Overlay:** View the final super-resolved image overlaid on its geographical location.

---

## 💻 Technology Stack

- **Backend & Machine Learning:**
  - [Python](https://www.python.org/)
  - [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- **Web Framework:**
  - [Streamlit](https://streamlit.io/)
- **Geospatial:**
  - [Google Earth Engine (ee-api)](https://earthengine.google.com/)
  - [Folium](https://python-visualization.github.io/folium/) & [streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
- **Core Libraries:**
  - [NumPy](https://numpy.org/)
  - [Pillow](https://python-pillow.org/)
  - [Geopy](https://geopy.readthedocs.io/en/stable/)

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8 - 3.11
- An active Google account with Google Earth Engine access enabled.

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/SuperResolutionApp.git
    cd SuperResolutionApp
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all the required Python libraries using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Authenticate with Google Earth Engine**
    This is a **critical one-time step**. You must authenticate your machine with GEE. Run the following command and follow the prompts in your browser to log in with your Google account.
    ```bash
    earthengine authenticate
    ```

5.  **Set Up the Model**
    - Create a directory named `models` in the root of the project.
    - Download the pre-trained model file (`sr_generator_compatible.h5` or `sr_generator_final.h5`).
    - Place the `.h5` model file inside the `models/` directory.

6.  **Run the Streamlit App**
    Once the setup is complete, run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    The application should now be open and running in your web browser!

---

## ⚙️ How It Works

The application workflow is as follows:

1.  **User Input:** The user draws a rectangle on the map and selects a date.
2.  **Data Fetching (`ee_pipeline.py`):**
    - The geographic coordinates are sent to Google Earth Engine.
    - A search is performed for Sentinel-2 images that match the location and date range.
    - The two least cloudy images are identified, and thumbnail URLs (for 64x64 images) are generated.
3.  **Model Inference (`app.py`):**
    - The two low-resolution images are loaded from their URLs and preprocessed into NumPy arrays.
    - These arrays are fed as a two-channel input to the loaded TensorFlow super-resolution model (`model.predict()`).
4.  **Displaying Results:**
    - The model's output (a high-resolution image array) is displayed in the Streamlit interface.
    - The user is given an option to download the image.
    - The output image is also overlaid on a new Folium map at its exact coordinates for geographical context.

---

## 📂 Project Structure

SuperResolutionApp/
│
├── models/
│   └── sr_generator_compatible.h5    # Pre-trained Keras model
│
├── src/
│   ├── __init__.py
│   ├── ee_pipeline.py                # Functions for Google Earth Engine API
│   └── utils.py                      # Helper functions (image conversion, etc.)
│
├── app.py                            # Main Streamlit application script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
