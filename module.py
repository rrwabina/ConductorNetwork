import torch
import torch.nn as nn

class CnvMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CnvMod, self).__init__()
        self.block  = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size = (5, 5), stride = 1, padding = 1),
            nn.BatchNorm2d(output_channel, eps = 0.001, momentum = 0.9),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class EncMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EncMod, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size = (3, 3), stride = 1, padding = 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DecMod, self).__init__()
        self.block = nn.Sequential(
            #Double check why kernel size is even
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size = (3, 3), stride = 1, padding = 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            CnvMod(output_channel, output_channel)
        )

    def forward(self, x):
        return self.block(x)

class Map(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Map, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size = (5, 5), stride = 1, padding = 1)
        )

    def forward(self, x):
        return self.block(x)