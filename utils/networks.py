import torch
import torch.nn as nn


class LSTM(nn.Module):  # this takes in the previous prediction to inform the next layer

    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=19, hidden_size=64, num_layers=1, batch_first=True)  #, dropout=0.15)
        self.linOut = nn.Linear(64, 7)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        # output = self.relu(output)
        output = self.linOut(output)
        return output, hidden


class SilhouetteRNN(nn.Module):

    def __init__(self):
        super(SilhouetteRNN, self).__init__()

        self.lin1 = nn.Linear(83, 64)
        self.linOut = nn.Linear(64, 7)
        self.silOut = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.15)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), -1)
        combined = self.drop(combined)
        h1 = self.lin1(combined)
        h1 = self.relu(h1)
        output = self.linOut(h1)
        sil_out = self.silOut(h1)
        return output, h1, sil_out


class IterativeRCNN(nn.Module):

    def __init__(self):
        super(IterativeRCNN, self).__init__()

        self.conv = nn.Conv1d(1, 32, 19, padding=0)
        self.batch = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        nn.Conv1d(32, 16, 5, padding=0)
        nn.Flatten()
        nn.Linear(216, 32)

        self.pred_out = nn.Linear(32, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, prev_hidden):
        combined = torch.cat((x, prev_hidden), -1)
        c1 = self.conv(combined)
        h_out = self.relu(c1)
        output = self.pred_out(c1)
        output = self.relu(output)
        output = self.softmax(output)
        return output, h_out


class IterativeRNN(nn.Module):  # this takes in the previous hidden layer output to inform the next layer

    def __init__(self):
        super(IterativeRNN, self).__init__()

        self.lin1 = nn.Linear(83, 64)  # best params found at 128, 200, 64 neurons
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
        output = self.linOut(h1)
        return output, h1


class IterativeRNN2(nn.Module):  # this takes in the previous prediction to inform the next layer

    def __init__(self):
        super(LSTM, self).__init__()

        self.lin1 = nn.Linear(26, 64)  # best params found at 128, 200, 64 neurons
        self.lin2 = nn.Linear(128, 200)
        self.lin3 = nn.Linear(200, 64)
        self.linOut = nn.Linear(64, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.15)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pred_in):
        combined = torch.cat((x, pred_in), -1)
        combined = self.drop(combined)
        h1 = self.lin1(combined)
        h1 = self.relu(h1)
        output = self.linOut(h1)
        return output, output


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

        row = x.reshape(x.size(0), 1, x.size(-1))
        arr = torch.cat((row, next_pred.reshape(x.size(0), 1, 7)), dim=-1)
        final = self.fwd(arr)  # h1 = self.conv1(arr)
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
