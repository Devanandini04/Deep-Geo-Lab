import rasterio
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt

# 1. Create Dummy Data (100x100 pixels)
# Ek black box banate hain (uint8 type taaki memory kam le)
data = np.zeros((100, 100), dtype=rasterio.uint8)

# 2. Draw a "River" 🌊 (Diagonal white line)
# Pixel value 255 ka matlab hai 'Bright White'
for i in range(100):
    data[i, i] = 255
    if i+1 < 100: data[i, i+1] = 255
    if i-1 >= 0:  data[i, i-1] = 255

# 3. Define Geography (Magic Part 🌍)
# Hum bata rahe hain ki ye image 'Lucknow' ke coordinates par hai
# (West_Longitude, North_Latitude, Pixel_Size_X, Pixel_Size_Y)
transform = from_origin(80.9462, 26.8467, 0.0001, 0.0001)

# 4. Save as GeoTIFF
filename = "Rasterio_Track/fake_lucknow.tif"

print("Generating Map...")

# Metadata set kar rahe hain (Driver, Size, Count, CRS)
with rasterio.open(
    filename,
    'w',
    driver='GTiff',
    height=data.shape[0],
    width=data.shape[1],
    count=1,              # 1 Band (Black & White)
    dtype=data.dtype,
    crs='+proj=latlong',  # Coordinate System (Lat/Lon)
    transform=transform
) as dst:
    dst.write(data, 1)

print(f"Success! 🗺️ Map saved at {filename}")

# 5. Visualization Check (Bas dekhne ke liye)
plt.imshow(data, cmap='gray')
plt.title("Fake Satellite Image (River)")
plt.show()