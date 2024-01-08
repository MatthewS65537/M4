import torch
import torch.nn as nn
import torch.nn.functional as F


class UNET(nn.Module):
    def __init__(self, device=None, dtype=torch.float32):
        super(UNET, self).__init__()
        self.conv1 = nn.Conv1d(768, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(32, 1, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose1d(1, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose1d(64, 128, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose1d(128, 256, kernel_size=3, padding=1)
        self.upconv5 = nn.ConvTranspose1d(256, 512, kernel_size=3, padding=1)
        self.upconv6 = nn.ConvTranspose1d(512, 768, kernel_size=3, padding=1)
        self.device = device
        if not device == None:
            self.to(device)
        self.dtype = dtype
        self.to(dtype=dtype)

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = F.relu(self.upconv5(x))
        x = self.upconv6(x)
        return x