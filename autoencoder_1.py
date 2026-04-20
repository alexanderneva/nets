import torch
import torch.nn as nn

#### random image shaped like CIFAR

x0 = torch.randn(1,3,32,32)
print(f"Image shape {x0.shape}")
# increasing the channels
# stride samples every 2 pixel
# padding pad image to 34 x 34 to pass the whole 3x3 kernel to every pix
enc1 = nn.Conv2d(in_channels=3, out_channels=16,kernel_size=3,stride=2,padding=1)
x0 = enc1(x0)
print(f"Shape Encoder1: {x0.shape}")
## second encoding
enc2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1)
x0 = enc2(x0)
print(f"Shape Encoder2: {x0.shape}")


#### decoder
dec1 = nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=2)
x0 = dec1(x0)
print(f"Shape Decoder1: {x0.shape}")
### decrease back to 3 channels
dec2 = nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=2,stride=2)
x0 = dec2(x0)
print(f"Shape Decoder2: {x0.shape}")

### making torch module

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        ### Encoder block
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        ### Decoder block
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,3,kernel_size=2,stride=2),
            nn.Sigmoid()
        )
    def forward(self, x):
        # compress the image
        bottleneck = self.encoder(x)
        # reconstruct
        reconstruct = self.decoder(bottleneck)
        return reconstruct
model = Autoencoder()
print(model.parameters)

class UNet_1(nn.Module):
    def __init__(self):
        super().__init__()
        ### Downsample block (Conv + ReLU)
        self.down_block_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        self.down_block_2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.up_block_1 = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=2,stride=2),
            nn.ReLU()
        )
        ### upsample

        self.up_block_2 = nn.Sequential(
            nn.ConvTranspose2d(16,3,kernel_size=2,stride=2),
            nn.Sigmoid()
        )
        self.blender = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        # skip connection features
        skip_1 = self.down_block_1(x)
        bottleneck = self.down_block_2(skip_1)
        reconstruct_1 = self.up_block_1(bottleneck)
        concat = torch.cat((skip_1,reconstruct_1),dim=1)
        blended = self.blender(concat)
        reconstruct = self.up_block_2(blended)
        return reconstruct


