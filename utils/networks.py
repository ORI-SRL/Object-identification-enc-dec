import torch
import torch.nn as nn


class TwoLayerConv(nn.Module):

    def __init__(self):
        super(TwoLayerConv, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 19), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, (5, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Embedding
        x = self.encoder_conv(torch.zeros((1, 1, 10, 19)))
        self.encoder_ff = nn.Sequential(nn.Linear(x.shape[-1], 100),
                                        nn.Tanh(),
                                        nn.Linear(100, 3))  # luca: squeeze the bottleneck to two/three neurons

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (10, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (1, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (1, 10)),
        )

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        x_enc = self.encoder_ff(x_enc)  # here x_enc is the encoded two-dimensional vector and can be investigated

        x_dec = self.decoder(x_enc.unsqueeze(1).unsqueeze(1))

        return x_dec, x_enc


class TwoLayerWDropout(nn.Module):

    def __init__(self):
        super(TwoLayerWDropout, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 19), padding=0),
            nn.Dropout2d(.15),
            nn.ReLU(),
            nn.Conv2d(32, 16, (5, 1), padding=0),
            nn.Dropout2d(.15),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Embedding
        x = self.encoder_conv(torch.zeros((1, 1, 10, 19)))
        self.encoder_ff = nn.Sequential(nn.Linear(x.shape[-1], 100),
                                        nn.Tanh(),
                                        nn.Linear(100, 3))  # luca: squeeze the bottleneck to two neurons

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (10, 2)),
            nn.Dropout2d(.2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (1, 7)),
            nn.Dropout2d(.2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (1, 10)),
        )

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        x_enc = self.encoder_ff(x_enc)  # here x_enc is the encoded two-dimensional vector and can be investigated

        x_dec = self.decoder(x_enc.unsqueeze(1).unsqueeze(1))

        return x_dec, x_enc


class TwoLayerWBatchNorm(nn.Module):

    def __init__(self):
        super(TwoLayerWBatchNorm, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 19), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, (5, 1), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 1), padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten()
        )
        # Embedding
        x = self.encoder_conv(torch.zeros((1, 1, 10, 19)))
        self.encoder_ff = nn.Sequential(nn.Linear(x.shape[-1], 100),
                                        nn.Tanh(),
                                        nn.Linear(100, 3))  # luca: squeeze the bottleneck to two neurons

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (10, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (1, 7)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (1, 10)),
        )

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        x_enc = self.encoder_ff(x_enc)  # here x_enc is the encoded two-dimensional vector and can be investigated

        x_dec = self.decoder(x_enc.unsqueeze(1).unsqueeze(1))

        return x_dec, x_enc


class TwoLayerConv7Grasp(nn.Module):

    def __init__(self):
        super(TwoLayerConv7Grasp, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 19), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, (4, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Embedding
        x = self.encoder_conv(torch.zeros((1, 1, 7, 19)))
        self.encoder_ff = nn.Sequential(nn.Linear(x.shape[-1], 100),
                                        nn.Tanh(),
                                        nn.Linear(100, 3))  # luca: squeeze the bottleneck to two/three neurons

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (7, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (1, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (1, 10)),
        )

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        x_enc = self.encoder_ff(x_enc)  # here x_enc is the encoded three-dimensional vector and can be investigated

        x_dec = self.decoder(x_enc.unsqueeze(1).unsqueeze(1))

        return x_dec, x_enc


class TwoLayerConv5Grasp(nn.Module):

    def __init__(self):
        super(TwoLayerConv5Grasp, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 19), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 8, (2, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Embedding
        x = self.encoder_conv(torch.zeros((1, 1, 5, 19)))
        self.encoder_ff = nn.Sequential(nn.Linear(x.shape[-1], 100),
                                        nn.Tanh(),
                                        nn.Linear(100, 3))  # luca: squeeze the bottleneck to two/three neurons

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (5, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (1, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (1, 10)),
        )

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        x_enc = self.encoder_ff(x_enc)  # here x_enc is the encoded two-dimensional vector and can be investigated

        x_dec = self.decoder(x_enc.unsqueeze(1).unsqueeze(1))

        return x_dec, x_enc


class TwoLayerConv3Grasp(nn.Module):

    def __init__(self):
        super(TwoLayerConv3Grasp, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 19), padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 16, (5, 1), padding=0),
            # nn.ReLU(),
            nn.Conv2d(32, 8, (2, 1), padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        # Embedding
        x = self.encoder_conv(torch.zeros((1, 1, 3, 19)))
        self.encoder_ff = nn.Sequential(nn.Linear(x.shape[-1], 100),
                                        nn.Tanh(),
                                        nn.Linear(100, 3))  # luca: squeeze the bottleneck to two/three neurons

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (3, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (1, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (1, 10)),
        )

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        x_enc = self.encoder_ff(x_enc)  # here x_enc is the encoded two-dimensional vector and can be investigated

        x_dec = self.decoder(x_enc.unsqueeze(1).unsqueeze(1))

        return x_dec, x_enc


