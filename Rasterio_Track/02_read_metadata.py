import rasterio

# Humari banayi hui nakli satellite image
filename = "Rasterio_Track/fake_lucknow.tif"

print(f"🕵️‍♂️ Inspecting: {filename}...\n")

with rasterio.open(filename) as dataset:
    
    # 1. Image Size
    print(f"📏 Size: {dataset.width}x{dataset.height} pixels")
    print(f"🔢 Bands: {dataset.count} (Layers)")
    
    # 2. Location (Bounding Box) - The Magic Part 🌍
    print(f"\n🌍 Geographical Bounds (Lat/Lon):")
    print(f"   West:  {dataset.bounds.left}")
    print(f"   East:  {dataset.bounds.right}")
    print(f"   North: {dataset.bounds.top}")
    print(f"   South: {dataset.bounds.bottom}")
    
    # 3. Coordinate System
    print(f"\n🗺️  CRS (Coordinate System): {dataset.crs}")
    
    # Check
    if dataset.bounds.left == 80.9462:
        print("\n✅ PASS: Coordinates match Lucknow!")
    else:
        print("\n❌ FAIL: Coordinates are wrong.")