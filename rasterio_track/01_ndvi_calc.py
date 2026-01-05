import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

# 1. DOWNLOAD REAL DATA 
# downloading a GeoTIFF from Rasterio's GitHub repo.
# It's an RGB image (Red, Green, Blue).
image_url = "https://github.com/rasterio/rasterio/raw/master/tests/data/RGB.byte.tif"
filename = "real_satellite_sample.tif"

if not os.path.exists(filename):
    print(f"Downloading real satellite data from {image_url}...")
    response = requests.get(image_url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print("Download Complete!")
else:
    print("File already exists. Skipping download.")

# 2. OPEN AND ANALYZE
with rasterio.open(filename) as src:
    print("\n--- Metadata ---")
    print(f"Dimensions: {src.width} x {src.height} pixels")
    print(f"Coordinate System: {src.crs}")  # Real EPSG coordinates!
    print(f"Bands: {src.count}") # Should be 3 (R, G, B)
    # Note: Rasterio reads as (Bands, Height, Width) -> (3, 700, 800)
    image_data = src.read()
    
    # --- 3. VISUALIZATION ---
    # We use Rasterio's built-in 'show' function which handles coordinates nicely
    print("Displaying Image...")
    plt.figure(figsize=(8, 8))
    show(src, title="Real Satellite Imagery (RGB)")

    # --- 4. OPTIONAL: FAKE NDVI (Proof of Concept) ---
    # Since this specific sample image doesn't have an Infrared band (it's only RGB),
    # we can't calculate REAL NDVI. But here is how you access a specific band:
    
    red_band = src.read(1).astype(float)   # Band 1 is usually Red
    green_band = src.read(2).astype(float) # Band 2 is Green
    
    # Just to show we can do math on real pixels:
    # Let's highlight very bright areas
    bright_spots = (red_band > 100) & (green_band > 100)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(bright_spots, cmap='Greys_r')
    plt.title("Bright Spots Mask (Processed from Real Data)")
    plt.show()