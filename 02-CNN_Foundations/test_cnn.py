import numpy as np
from convolution import conv_forward

np.random.seed(1)

# --- 1. DUMMY DATA BANANA ---
# Imagine humare paas 10 images hain (Batch size = 10)
# Har image 4x4 pixels ki hai aur usme 3 channels hain (RGB)
A_prev = np.random.randn(10, 4, 4, 3)

# --- 2. FILTERS BANANA ---
# Hum 2 filters use kar rahe hain
# Har filter 2x2 size ka hai
W = np.random.randn(2, 2, 3, 8) # (f, f, n_C_prev, n_C)

# Bias for 8 filters
b = np.random.randn(1, 1, 1, 8)

# --- 3. HYPERPARAMETERS ---
# Stride = 2 (2 steps jump karega)
# Pad = 2 (Border pe 2 lines zeros ki)
hparameters = {"pad" : 2, "stride": 2}

print("Testing Convolution Function...")

# --- 4. RUNNING YOUR CODE ---
try:
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("\nSUCCESS! 🎉 Convolution chal gaya.")
    
    # Check Dimensions
    print("Z (Output) shape:", Z.shape)
    
    # Expected Check
    print("Mean value of Z:", np.mean(Z))
    
    # Verification
    if Z.shape == (10, 4, 4, 8):
        print("\nPASS: Output dimensions ekdum sahi hain!")
    else:
        print("\nFAIL: Dimensions galat hain. Expected (10, 4, 4, 8)")

except Exception as e:
    print("\nERROR: Kuch gadbad hai code mein!")
    print(e)