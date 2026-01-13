import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # --- ENCODER (Left Side: Image chhota hota jayega) ---
        # Logic: Conv -> ReLU -> Conv -> ReLU -> MaxPool
        
        self.enc1 = self.conv_block(3, 64)   # Input: 3 Channels (RGB)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # --- BOTTLENECK (Bottom: Sabse chhota size, sabse zyada features) ---
        self.bottleneck = self.double_conv(512, 1024)
        
        # --- DECODER (Right Side: Image wapas bada hoga) ---
        # Logic: UpSample -> Concat (Jodna) -> Conv -> ReLU
        
        self.up4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512) # 1024 kyun? 512 (Up) + 512 (Copy from Enc4)
        
        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        # Final Output Layer (1 Channel: Water or Land)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 1. Downsampling (Encoder)
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # 2. Bottleneck
        b = self.bottleneck(p4)
        
        # 3. Upsampling (Decoder + Skip Connections)
        d4 = self.up4(b)
        d4 = torch.cat((d4, self.crop(e4, d4)), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((d3, self.crop(e3, d3)), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, self.crop(e2, d2)), dim=1)
        d2 = self.dec2(d2)
        
        # --- FIX IS HERE ---
        d1 = self.up1(d2)  # Corrected: d2 input jayega
        d1 = torch.cat((d1, self.crop(e1, d1)), dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))

    # --- HELPER FUNCTIONS ---
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def double_conv(self, in_c, out_c):
        return self.conv_block(in_c, out_c)

    def up_conv(self, in_c, out_c):
        return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
    
    def pool(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)
        
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

# Test Block
if __name__ == "__main__":
    import torchvision
    model = UNet()
    # Dummy Image: Batch=1, Channels=3, Size=160x160
    x = torch.randn(1, 3, 160, 160)
    print("Testing U-Net Shape...")
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}") # Should be (1, 1, 160, 160)
    print("✅ U-Net Ready!")