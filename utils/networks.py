import torch
import torch.nn as nn


class LSTM(nn.Module):  # this takes in the previous prediction to inform the next layer

    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=19, hidden_size=64, num_layers=1, batch_first=True)  # , dropout=0.15)
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
        self.drop = nn.Dropout(0.15)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.hid1 = nn.Linear(256, 32)
        self.pred_out = nn.Linear(32, 7)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, prev_hidden):
        combined = torch.cat((x, prev_hidden), -1)
        c1 = self.conv(combined)
        c1 = self.drop(c1)
        h1 = self.relu(c1)
        flat = self.flat(h1)
        h_out = self.hid1(flat)
        output = self.pred_out(h_out)
        pred_back = self.softmax(self.relu(output))
        # output = self.softmax(output)
        return output, pred_back.unsqueeze(1)  # h_out


class IterativeRNN(nn.Module):  # this takes in the previous hidden layer output to inform the next layer

    def __init__(self):
        super(IterativeRNN, self).__init__()

        self.lin1 = nn.Linear(83, 64)
        self.linOut = nn.Linear(64, 7)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.15)
        self.norm = nn.BatchNorm1d(64)
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
        super(IterativeRNN2, self).__init__()

        self.lin1 = nn.Linear(26, 64)  # best params found at 128, 200, 64 neurons
        self.linOut = nn.Linear(64, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.15)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pred_in):
        combined = torch.cat((x, pred_in), -1)
        drop = self.drop(combined)
        h1 = self.lin1(drop)
        h1 = self.relu(h1)
        output = self.linOut(h1)
        # pred_back = output  # self.softmax(output)
        # if not self.training:
        # LUCA NOTE: it's better to do all this work in the main, outside the network
        #     # retain to allow for taking backward on each loop
        #     pred_back.backward(torch.ones_like(pred_back), retain_graph=True)
        #     saliency = torch.mean(combined.grad.data.abs(), dim=0)

        return output  # , pred_back, saliency


class IterativeRNN3(nn.Module):  # this takes in the previous prediction to inform the next layer

    def __init__(self):
        super(IterativeRNN3, self).__init__()

        self.lin1 = nn.Linear(26, 64)  # best params found at 128, 200, 64 neurons
        self.lin2 = nn.Linear(64, 64)
        self.linOut = nn.Linear(64, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.15)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pred_in):
        x = torch.cat((x, pred_in), -1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.drop(x)
        output = self.linOut(x)

        return output  # , pred_back, saliency


class IterativeRNN4(nn.Module):  # this takes in the previous prediction to inform the next layer

    embed_states = []

    def __init__(self):
        super(IterativeRNN4, self).__init__()

        self.lin1 = nn.Linear(19, 64)  # best params found at 128, 200, 64 neurons
        # self.lin2 = nn.Linear(263, 7)  # best params found at 128, 200, 64 neurons
        self.cnn1_block = nn.Sequential(
                              nn.Conv2d(1, 32, (3, 3)),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Dropout(p=.1),
                              nn.Conv2d(32, 16, (3, 3)),
                              nn.BatchNorm2d(16),
                              nn.Dropout(p=.1),
                              nn.ReLU())
        self.embed_layers = [self.lin1, self.cnn1_block]

        self.lin2 = nn.Linear(263, 32)

        self.linOut = nn.Linear(32, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.15)

        self.embed = None

        self.softmax = nn.Softmax(dim=-1)

    def save_embed_states(self):
        self.embed_states = []
        for layer in self.embed_layers:
            self.embed_states.append(layer.state_dict())

    def freeze_embed(self):
        for layer in self.embed_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def _reset_embed_state(self):
        assert len(self.embed_states) > 0, "please save the states before resetting them"
        for layer, layer_state in zip(self.embed_layers, self.embed_states):
            layer.load_state_dict(layer_state)
        self.freeze_embed()

    def get_embed(self):
        return self.embed

    def forward(self, x, pred_in):
        # x = self.sensdrop(x)
        x = self.lin1(x)
        embed_layer = x.reshape(-1, 1, 8, 8)

        self.embed = embed_layer.detach()

        x_aug = self.cnn1_block(embed_layer)
        x = torch.cat((x_aug.view(x.shape[0], -1), pred_in), -1)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.drop(x)
        output = self.linOut(x)

        return output  # , pred_back, saliency


class IterativeRNN4_embed(nn.Module):  # this takes in the previous prediction to inform the next layer

    def __init__(self):
        super(IterativeRNN4_embed, self).__init__()

        self.lin1 = nn.Linear(19, 64)  # best params found at 128, 200, 64 neurons
        # self.lin2 = nn.Linear(263, 7)  # best params found at 128, 200, 64 neurons
        self.cnn1_block = nn.Sequential(
                              nn.Conv2d(1, 32, (3, 3)),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Dropout(p=.1),
                              nn.Conv2d(32, 16, (3, 3)),
                              nn.BatchNorm2d(16),
                              nn.Dropout(p=.1),
                              nn.ReLU())
        self.lin2 = nn.Linear(263, 32)
        self.linOut = nn.Linear(32, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.15)

        self.embed = None

        self.softmax = nn.Softmax(dim=-1)

    def get_embed(self):
        return self.embed

    def forward(self, embed, pred_in):
        # x = self.sensdrop(x)
        # x = self.lin1(x)
        embed_layer = embed.reshape(-1, 1, 8, 8)

        self.embed = embed_layer.detach()

        x_aug = self.cnn1_block(embed_layer)
        x = torch.cat((x_aug.view(embed.shape[0], -1), pred_in), -1)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.drop(x)
        output = self.linOut(x)

        return output  # , pred_back, saliency

