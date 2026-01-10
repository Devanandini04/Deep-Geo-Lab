import rasterio
import numpy as np
import matplotlib.pyplot as plt

# File path
filename = "Rasterio_Track/fake_lucknow_multiband.tif"

print(f"🔄 Processing {filename} for Vegetation Detection...")

with rasterio.open(filename) as src:
    # Band 1 is Red, Band 2 is NIR (Jo humne pichle code mein banaya tha)
    red = src.read(1).astype(float)
    nir = src.read(2).astype(float)
    
    # Metadata save kar lete hain taaki result save kar sakein
    profile = src.profile

# --- THE MATH MAGIC (NDVI FORMULA) ---
# NDVI = (NIR - Red) / (NIR + Red)
# Zero se divide hone se bachne ke liye hum denominator mein chhota number add nahi kar rahe
# kyunki humara dummy data saaf hai. Real data mein error handling hoti hai.

numerator = nir - red
denominator = nir + red

# Jahan denominator 0 ho, wahan division mat karo (Avoid Error)
ndvi = np.zeros(red.shape)
mask = denominator > 0
ndvi[mask] = numerator[mask] / denominator[mask]

print("✅ NDVI Calculation Complete!")

# --- VISUALIZATION ---
plt.figure(figsize=(10, 5))

# Plot 1: Original Red Band (Dikhta kaisa hai)
plt.subplot(1, 2, 1)
plt.imshow(red, cmap='gray')
plt.title("Band 1: Red Light (Normal Vision)")
plt.colorbar()

# Plot 2: NDVI Result (Computer Vision)
plt.subplot(1, 2, 2)
# 'RdYlGn' colormap use karte hain: Red=Bad, Green=Vegetation 🌳
plt.imshow(ndvi, cmap='RdYlGn') 
plt.title("NDVI: Detected Vegetation")
plt.colorbar()

plt.show()

# Optional: Print coordinate of the detected jungle
# Hum check karte hain ki max value kahan hai
max_ind = np.unravel_index(np.argmax(ndvi, axis=None), ndvi.shape)
print(f"📍 Strongest Vegetation found at Pixel Coordinates: {max_ind}")