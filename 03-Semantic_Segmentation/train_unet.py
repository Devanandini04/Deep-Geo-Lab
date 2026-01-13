import torch
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
import torchvision

# 1. Setup Device (GPU agar hai, nahi toh CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {device}")

# 2. Initialize Model
model = UNet().to(device)

# 3. Loss Function & Optimizer
# BCEWithLogitsLoss best hai Binary Segmentation (Water vs Land) ke liye
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🧠 Model Initialized. Starting Training Loop (with Dummy Data)...")

# 4. Training Loop (Trial Run)
# Hum 10 baar (Epochs) model ko sikhayenge
for epoch in range(1, 11):
    
    # --- Generate Fake Data (Batch Size=2) ---
    # Input: 2 Images, 3 Channels, 160x160
    inputs = torch.randn(2, 3, 160, 160).to(device)
    # Target: 2 Masks, 1 Channel, 160x160 (Values 0 ya 1)
    targets = torch.randint(0, 2, (2, 1, 160, 160)).float().to(device)
    
    # --- Forward Pass ---
    optimizer.zero_grad()       # Purane gradients saaf karo
    outputs = model(inputs)     # Model se prediction lo
    
    # --- Calculate Loss (Galti kitni hai?) ---
    loss = criterion(outputs, targets)
    
    # --- Backward Pass (Seekho) ---
    loss.backward()             # Galti se seekho (Backprop)
    optimizer.step()            # Weights update karo
    
    print(f"Epoch {epoch}/10 | Loss: {loss.item():.4f}")

print("✅ Training Loop Verified! Model is capable of learning.")