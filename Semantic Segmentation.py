import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Standard Double Convolution Block used throughout U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 2. The U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):  # out_channels=1 for binary mask
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ENCODER (Downsampling)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        # BOTTLENECK
        self.bottleneck = DoubleConv(512, 1024)

        # DECODER (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512)  # 1024 due to skip connection concatenation

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)

        # FINAL OUTPUT
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Pass
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        # Bottleneck
        b = self.bottleneck(self.pool(d4))

        # Decoder Pass with Skip Connections
        u4 = self.upconv4(b)
        u4 = torch.cat([d4, u4], dim=1)  # Concatenating the skip connection
        u4 = self.up4(u4)

        u3 = self.upconv3(u4)
        u3 = torch.cat([d3, u3], dim=1)
        u3 = self.up3(u3)

        u2 = self.upconv2(u3)
        u2 = torch.cat([d2, u2], dim=1)
        u2 = self.up2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat([d1, u1], dim=1)
        u1 = self.up1(u1)

        return self.out_conv(u1)


# 3. Dice Loss Implementation (For precise boundary delineation)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten predictions and targets
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice_score