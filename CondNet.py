import torch
import torch.nn as nn
from typing import List, Dict

class CnvMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(CnvMod, self).__init__()
        self.block  = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size = 1, stride = 1, padding = 1),
            nn.BatchNorm2d(output_channel, eps = 0.001, momentum = 0.9),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class EncMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(EncMod, self).__init__()
        self.block = nn.Sequential(
            CnvMod(input_channel, output_channel),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecMod(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DecMod, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size = 1, stride = 1, padding = 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class Map(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Map, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size = 1, stride = 1, padding = 1),
            nn.LogSigmoid()
        )

    def forward(self, x):
        return self.block(x)         

from torchsummary import summary

class EncoderTrack(nn.Module): 
    def __init__(self):
        super(EncoderTrack, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 5) 
        ])
        self.decodermodule = DecMod(64, 64)
        self.encodermodule = EncMod(64, 128)

    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        x = self.decodermodule(x)
        x = self.encodermodule(x)
        return x

    def forward(self, x):
        return self.encoder_track(x) 

class EncoderSubTrackA(nn.Module):
    def __init__(self):
        super(EncoderSubTrackA, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 1) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, x):
        return self.encoder_track(x)

class EncoderSubTrackB(nn.Module):
    def __init__(self):
        super(EncoderSubTrackB, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 2) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, x):
        return self.encoder_track(x)

class EncoderSubTrackC(nn.Module):
    def __init__(self):
        super(EncoderSubTrackC, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 3) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, x):
        return self.encoder_track(x)

class EncoderSubTrackD(nn.Module):
    def __init__(self):
        super(EncoderSubTrackD, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 4) 
        ])
    def encoder_track(self, x):
        for encoder in self.encodermodules:
            x = encoder(x)
        return x
    def forward(self, x):
        return self.encoder_track(x)

class DecoderTrackA(nn.Module):
    def __init__(self):
        super(DecoderTrackA, self).__init__()
        self.convmodules = CnvMod(256, 128)
        self.decodermodules = DecMod(128, 64)
        self.map = Map(64, 32)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackB(nn.Module):
    def __init__(self):
        super(DecoderTrackB, self).__init__()
        self.convmodules = CnvMod(32, 64)
        self.decodermodules = DecMod(64, 32)
        self.map = Map(32, 16)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackC(nn.Module):
    def __init__(self):
        super(DecoderTrackC, self).__init__()
        self.convmodules = CnvMod(16, 32)
        self.decodermodules = DecMod(32, 16)
        self.map = Map(16, 8)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackD(nn.Module):
    def __init__(self):
        super(DecoderTrackD, self).__init__()
        self.convmodules = CnvMod(8, 16)
        self.decodermodules = DecMod(16, 8)
        self.map = Map(8, 4)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class DecoderTrackE(nn.Module):
    def __init__(self):
        super(DecoderTrackE, self).__init__()
        self.convmodules = CnvMod(4, 8)
        self.decodermodules = DecMod(8, 4)
        self.map = Map(4, 2)

    def forward(self, x):
        return self.map(self.decodermodules(self.convmodules(x)))

class Cond(nn.Module):
    def __init__(self):
        super(Cond, self).__init__()
        self.encoder = EncoderTrack()

        self.decoderA = DecoderTrackA()
        self.decoderB = DecoderTrackB()
        self.decoderC = DecoderTrackC()
        self.decoderD = DecoderTrackD()
        self.decoderE = DecoderTrackE()

        self.sub_encoderD = EncoderSubTrackD()
        self.sub_encoderC = EncoderSubTrackC()
        self.sub_encoderB = EncoderSubTrackB()
        self.sub_encoderA = EncoderSubTrackA()

    def forward(self, a):
        x = torch.concat((self.encoder(a), self.encoder(a)), dim = 1)
        x = self.decoderA(x)                                                    #(32, 17, 17)

        skipA = torch.concat((self.sub_encoderD(a), self.sub_encoderD(a)))      #(32, 17, 17)
        skipB = torch.concat((self.sub_encoderC(a), self.sub_encoderC(a)))      #(16, 33, 33)
        skipC = torch.concat((self.sub_encoderB(a), self.sub_encoderB(a)))      #(08, 65, 65)
        skipD = torch.concat((self.sub_encoderA(a), self.sub_encoderA(a)))      #(04, 129, 129)  

        x = torch.concat((x[:, :32, :6, :6], skipA[:, :32, :6, :6]))            #(32, 17, 17)
        x = self.decoderB(x)                                                    #(16, 08, 08)
        x = torch.concat((x[:, :32, :8, :8], skipB[:, :32, :8, :8]))            #(16, 08, 08)
        x = self.decoderC(x)                                                    #(08, 10, 10)
        x = torch.concat((x[:, :, :10, :10], skipC[:, :, :10, :10]))            #(08, 10, 10)
        x = self.decoderD(x)                                                    #(04, 12, 12)
        x = torch.concat((x[:, :, :12, :12], skipD[:, :, :12, :12]))            #(04, 12, 12) 
        x = self.decoderE(x)                                                    #(02, 14, 14)
 
        return x
        
# summary(Cond(), (1, 256, 256))