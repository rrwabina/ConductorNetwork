from typing import List, Dict

import torch
import torch.nn as nn
from segmentation_models.condnet.module import EncMod, DecMod, CnvMod, Map 

from typing import List, Dict

class EncoderTrack(nn.Module): 
    ''' U = 2 inputs, V = 1 output, I = 6 depth '''
    def __init__(self):
        super(EncoderTrack, self).__init__()
        self.encodermodules = nn.ModuleList([
            EncMod(1 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)) for i in range(0, 6) 
        ])
        self.decodermodule = DecMod(128, 256)
        self.encodermodule = EncMod(128, 256)

    def encoder_track(self, x: List[torch.Tensor]) -> torch.Tensor:
        out = []
        for encoder in self.encodermodules:
            x = encoder(x)
            out.append(x)
        return out

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.encoder_track(x) #Check if there exists a mapping

class DecoderTrack(nn.Module):
    ''' U = 2 inputs, V = 1 output, I = 6 depth '''
    def __init__(self):
        super(DecoderTrack, self).__init__()
        self.convmodules = nn.ModuleList([
            CnvMod(64 if i == 2 else 2 ** (i + 3), 2 ** (i + 2)) for i in range(2, 0, -1)
        ])
        self.decodermodules = nn.ModuleList([
            DecMod(2 ** (j + 2), 1 if j == 1 else 2 ** (j + 1)) for j in range(2, 1, -1)
        ])
        self.map = Map(input_channel = 8, output_channel = 1)

    def decoder_track(self, x: List[torch.Tensor]) -> torch.Tensor:
        # inp = x.pop()
        for conv, decoder in zip(self.convmodules, self.decodermodules):
            inp = decoder(conv(x))
        return inp

    def forward(self, x):
        for _ in range(0, 4):
            output = self.map(self.decoder_track(x))
        return output

class CondNet(nn.Module):
    def __init__(self):
        super(CondNet, self).__init__()
        self.encoders = nn.ModuleList([
            EncoderTrack() for _ in range(0, 2)
        ])
        self.decoders = nn.ModuleList([
            DecoderTrack() for _ in range(0, 5)
        ]) 
        self.base_decoder = DecMod(256, 256)

    def encoder_track(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for encoder in self.encoders:
            x = encoder(x)
            out.append(x)
        out.append(self.base_decoder(out.pop()))
        return out

    def forward(self, x: torch.Tensor):
        encoder_outputs = self.encoder_track(x)
        return self.decoders(encoder_outputs)

class CondNetSimulation(nn.Module):
    def __init__(self):
        super(CondNetSimulation, self).__init__()
        self.encodertrack = EncoderTrack()
        self.decodertrack = DecoderTrack()
        
        self.decoders = nn.ModuleList([DecoderTrack() for _ in range(0, 6)])

    def forward(self, input1, input2):
        input1 = self.encodertrack(input1) 
        input2 = self.encodertrack(input2) 
        
        concat = torch.cat((input1[3], input2[3]), dim = 1)
        return self.decodertrack(concat) 
