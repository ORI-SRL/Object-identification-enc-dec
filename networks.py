import torch
import torch.nn as nn

class TwoLayerConv(nn.Module):
    def __init__(self):
        super(TwoLayerConv, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, (1, 19), padding=0, dtype=float),
            nn.ReLU(True),
            nn.Conv2d(16, 4, (10, 1), padding=0, dtype=float),
            nn.ReLU(True),
            )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, (10, 1), dtype=float),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 1, (1, 19), dtype=float),
            nn.LeakyReLU(True)
            )
        # self.pad = nn.functional.pad((0, 0, 1, 0))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class TwoLayerWDropout(nn.Module):
    def __init__(self):
        super(TwoLayerWDropout, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, (1, 19), padding=0, dtype=float),
            nn.LeakyReLU(True),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 4, (10, 1), padding=0, dtype=float),
            nn.LeakyReLU(True),
            nn.Dropout(p=0.25)
            )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, (10, 1), dtype=float),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, (1, 19), dtype=float),
            nn.ReLU(True)
            )
        # self.pad = nn.functional.pad((0, 0, 1, 0))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class TwoLayerWBatchNorm(nn.Module):
    def __init__(self):
        super(TwoLayerWBatchNorm, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, (1, 19), padding=0, dtype=float),
            nn.BatchNorm2d(16, dtype=float),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 4, (10, 1), padding=0, dtype=float),
            nn.BatchNorm2d(4, dtype=float),
            nn.LeakyReLU(True),
            )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, (10, 1), dtype=float),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, (1, 19), dtype=float),
            nn.ReLU(True)
            )
        # self.pad = nn.functional.pad((0, 0, 1, 0))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x