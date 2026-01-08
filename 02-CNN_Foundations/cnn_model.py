import numpy as np
from convolution import conv_forward
from pooling import pool_forward
from flatten import flatten_forward

def relu(Z):
    """
    ReLU Activation: Negative values ko 0 kar deta hai.
    Ye CNN ka sabse common activation function hai.
    """
    return np.maximum(0, Z)

class CNN_Network:
    def __init__(self):
        print("🛠️ Initializing CNN Model...")
        np.random.seed(1)
        
        # --- LAYER 1: CONVOLUTION FILTERS ---
        # 8 Filters, size 3x3, expecting 3 input channels (RGB)
        # Shape: (f, f, n_C_prev, n_C)
        self.W = np.random.randn(3, 3, 3, 8) 
        self.b = np.random.randn(1, 1, 1, 8)
        
        # Hyperparameters
        self.conv_params = {"pad": 1, "stride": 1} # Pad=1 taaki size kam na ho
        self.pool_params = {"stride": 2, "f": 2}   # Size half karne ke liye

    def forward(self, X):
        """
        Pura Forward Pass: Image -> Conv -> ReLU -> Pool -> Flatten
        """
        print(f"\n🚀 Input Image Shape: {X.shape}")

        # 1. Convolution (Features dhundo)
        Z1, _ = conv_forward(X, self.W, self.b, self.conv_params)
        print(f"   Step 1 (Conv):   {Z1.shape}")

        # 2. ReLU (Non-linearity)
        A1 = relu(Z1)
        print(f"   Step 2 (ReLU):   {A1.shape}")

        # 3. Pooling (Compress)
        P1, _ = pool_forward(A1, self.pool_params, mode="max")
        print(f"   Step 3 (Pool):   {P1.shape}")

        # 4. Flatten (Ready for ANN)
        F = flatten_forward(P1)
        print(f"   Step 4 (Flat):   {F.shape}")
        
        return F

# --- TESTING BLOCK ---
if __name__ == "__main__":
    # Create Dummy Data: 5 images, 64x64 pixels, 3 Channels (RGB)
    fake_images = np.random.randn(5, 64, 64, 3)
    
    # Initialize Model
    model = CNN_Network()
    
    # Pass Data
    output = model.forward(fake_images)
    
    print("\n✅ MISSION SUCCESS!")
    print(f"Final Vector Size: {output.shape} (Ready for Classification)")