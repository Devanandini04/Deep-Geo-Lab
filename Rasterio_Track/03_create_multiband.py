import rasterio
from rasterio.transform import from_origin
import numpy as np

# Dimensions (100x100 pixels)
height, width = 100, 100

# --- LAYER 1: RED BAND (Visible Light) ---
# Mitti (Soil) Red light ko reflect karti hai (Brightish)
red_band = np.ones((height, width), dtype=rasterio.float32) * 150 

# --- LAYER 2: NIR BAND (Near Infrared) ---
# Mitti NIR ko bhi reflect karti hai
nir_band = np.ones((height, width), dtype=rasterio.float32) * 160

# --- ADD FEATURES ---

# 1. RIVER 🌊 (Water absorbs everything -> Dark in both bands)
for i in range(100):
    red_band[i, i] = 10   # Dark
    nir_band[i, i] = 10   # Dark
    # River ko thoda wide banate hain
    if i+1 < 100:
        red_band[i, i+1] = 10
        nir_band[i, i+1] = 10

# 2. PARK / JUNGLE 🌳 (The Vegetation Patch)
# Logic: Plants 'Red' light ko kha jate hain (Low Value)
#        Lekin 'NIR' light ko wapas phek dete hain (High Value)

# Hum (20,60) se (40,80) coordinates par ek park banate hain
park_slice = (slice(20, 40), slice(60, 80))

red_band[park_slice] = 20   # Absorbed (Dark)
nir_band[park_slice] = 250  # Reflected (Very Bright!)

# --- SAVE AS GEO-TIFF ---
filename = "Rasterio_Track/fake_lucknow_multiband.tif"
transform = from_origin(80.9462, 26.8467, 0.0001, 0.0001)

print(f"Generating Multi-Band Map at {filename}...")

with rasterio.open(
    filename,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=2,               # Important: Ab humare paas 2 Bands hain!
    dtype=rasterio.float32,
    crs='+proj=latlong',
    transform=transform
) as dst:
    dst.write(red_band, 1)  # Band 1 = Red
    dst.write(nir_band, 2)  # Band 2 = NIR

print("✅ Success! Created a map with Hidden Vegetation Data.")