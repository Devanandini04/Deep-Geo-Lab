import rasterio
import numpy as np
import matplotlib.pyplot as plt

# File path (Wahi purana map use karenge jisme River hai)
filename = "Rasterio_Track/fake_lucknow_multiband.tif"

print(f"🌊 Analyzing {filename} for Water Bodies...")

with rasterio.open(filename) as src:
    # Band 2 (NIR) padhte hain
    # Logic: Paani NIR light ko pee jata hai (Absorb), isliye wo Dark (Low Value) hoga.
    nir_band = src.read(2)

    # --- STEP 1: THRESHOLDING (Water Mask) ---
    # Hum computer se keh rahe hain: "Jahan bhi NIR value 50 se kam hai, wo Paani hai"
    # (Humne map banate waqt River ki value 10 rakhi thi)
    water_mask = nir_band < 50

    print("✅ Water Detected!")

    # --- STEP 2: VISUALIZATION & COASTLINE ---
    plt.figure(figsize=(10, 5))

    # Original NIR View
    plt.subplot(1, 2, 1)
    plt.imshow(nir_band, cmap='gray')
    plt.title("Satellite View (NIR Band)")
    plt.colorbar()

    # Water Mask + Coastline
    plt.subplot(1, 2, 2)
    plt.imshow(water_mask, cmap='Blues') # Blue color for water
    plt.title("Extracted Water & Coastline")
    
    # Coastline Draw karna (Contour Method)
    # Ye function 'water_mask' ke kinaro par line kheench dega
    plt.contour(water_mask, levels=[0.5], colors='red', linewidths=2)
    
    plt.show()

print("🚀 Coastline Extraction Logic Complete!")