# U-Net model

import torch
import torch.nn as nn
from torchvision import transforms


class UNet_ISBI(nn.Module):
    def __init__(self):
        super(UNet_ISBI, self).__init__()

        def DoubleConv(in_channels, out_channels, kernel_size=3):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )

            return layer

        # Contracting part
        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = DoubleConv(512, 1024)
        self.drop5 = nn.Dropout(0.5)

        # Expansive part
        self.up6 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.fc = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        layer1 = self.conv1(x)
        out = self.pool1(layer1)
        layer2 = self.conv2(out)
        out = self.pool2(layer2)
        layer3 = self.conv3(out)
        out = self.pool3(layer3)
        layer4 = self.conv4(out)
        drop = self.drop4(layer4)
        out = self.pool4(drop)
        layer5 = self.conv5(out)
        drop = self.drop5(layer5)

        up = self.up6(drop)
        cat1 = torch.cat((transforms.CenterCrop((up.shape[2], up.shape[3]))(layer4), up), dim=1)
        layer6 = self.conv6(cat1)
        up = self.up7(layer6)
        cat2 = torch.cat((transforms.CenterCrop((up.shape[2], up.shape[3]))(layer3), up), dim=1)
        layer7 = self.conv7(cat2)
        up = self.up8(layer7)
        cat3 = torch.cat((transforms.CenterCrop((up.shape[2], up.shape[3]))(layer2), up), dim=1)
        layer8 = self.conv8(cat3)
        up = self.up9(layer8)
        cat4 = torch.cat((transforms.CenterCrop((up.shape[2], up.shape[3]))(layer1), up), dim=1)
        layer9 = self.conv9(cat4)

        out = self.fc(layer9)
        return out
