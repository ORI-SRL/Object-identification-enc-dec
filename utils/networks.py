import torch
import torch.nn as nn


class IterativeRNN(nn.Module):

    def __init__(self):
        super(IterativeRNN, self).__init__()

        self.lin1 = nn.Linear(83, 128)
        self.lin2 = nn.Linear(128, 200)
        self.lin3 = nn.Linear(200, 64)
        self.linOut = nn.Linear(64, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.15)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), -1)
        combined = self.drop(combined)
        h1 = self.lin1(combined)
        h1 = self.relu(h1)
        h2 = self.lin2(h1)
        h2 = self.tanh(h2)
        h3 = self.lin3(h2)
        h3 = self.tanh(h3)
        output = self.linOut(h3)
        output = self.tanh(output)
        output = self.softmax(output)
        return output, h3


class IterativeFFNN(nn.Module):

    def __init__(self):
        super(IterativeFFNN, self).__init__()

        self.fc1 = nn.Linear(26, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 7)
        self.drop = nn.Dropout(.15)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pred_in):

        next_pred = pred_in
        output = torch.empty((x.size(0), 0, 7)).to('cuda:0')
        final = torch.zeros(x.size(0), 1, 7)
        # for j in range(x.size(-2)):
        row = x.reshape(x.size(0), 1, x.size(-1))  # [:, j, :]

        #    if torch.sum(row) != 0:
        #        if j == 0:
        row = torch.cat((row, pred_in.reshape(x.size(0), 1, 7)), dim=-1)  # concatenate the data with the predictions
        #        else:
        #            row = torch.cat((row, next_pred), dim=-1)
        h1 = self.fc1(row)
        rel1 = self.relu(h1)
        h2 = self.fc2(rel1)
        rel2 = self.relu(h2)
        h3 = self.fc3(rel2)
        rel3 = self.relu(h3)
        h_out = self.fc4(rel3)
        drop_out = self.drop(h_out)
        final = self.tanh(drop_out)
        next_pred = self.softmax(final)
        output = torch.cat((output, final.reshape(x.size(0), 1, 7)), dim=1)
        return final.reshape(x.size(0), 7), output


class IterativeCNN(nn.Module):

    def __init__(self):
        super(IterativeCNN, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv1d(1, 32, 19, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 7),
            nn.Dropout1d(0.15),
            nn.Tanh(),
        )
        self.conv1 = nn.Conv1d(1, 32, 19, padding=0)
        self.conv2 = nn.Conv1d(32, 16, 9, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 7)
        self.drop = nn.Dropout1d(0.15)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pred_in):
        next_pred = pred_in
        output = torch.empty((x.size(0), 0, 7)).to('cuda:0')
        final = torch.zeros(x.size(0), 1, 7)

        # for j in range(x.size(-2)):
        row = x.reshape(x.size(0), 1, x.size(-1))
        # block = torch.cat([row, torch.zeros(x.size(0), 1).to('cuda:0')], dim=-1)
        # block.resize(4, 5)
        # block = nn.functional.pad(block, (1, 1))

        # if torch.sum(row) != 0:
        #     if j == 0:
        #         arr = torch.cat((row, pred_in), dim=-1).reshape(x.size(0), 1, 26)  # concatenate the data with the predictions
        #     else:
        arr = torch.cat((row, next_pred.reshape(x.size(0), 1, 7)), dim=-1)
        final = self.fwd(arr)  # h1 = self.conv1(arr)
        # rel1 = self.relu(h1)
            # h2 = self.conv2(rel1)
            # rel2 = self.relu(h2)
            # flat = self.flatten(rel1)
            # lin = self.fc(flat)
            # drop_out = self.drop(lin)
            # final = self.tanh(drop_out)
        next_pred = self.softmax(final)
        output = torch.cat((output, final.reshape(x.size(0), 1, 7)), dim=1)
        return final, output


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


class TwoLayerWDropoutWBatchNorm(nn.Module):

    def __init__(self):
        super(TwoLayerWDropoutWBatchNorm, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 19), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(.15),
            nn.Conv2d(32, 16, (5, 1), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(.15),
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
