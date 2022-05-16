import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class ObjectGraspsDataset(Dataset):
    """Object Grasps Dataset"""

    def __init__(self, data_array_file, labels_file, root_dir, transform=None, train: bool = True):
        """
        Args:
            array_file (string): Path to the saved ndarray file with grasp raw data.
            labels_file (string): Path to the saved ndarray file with labels.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        input_data = np.load(data_array_file)
        input_targets = np.load(labels_file)
        norm_data = input_data/np.linalg.norm(input_data, axis=1)[:, None]
        # reshape data to align sensors
        X = np.reshape(norm_data, (-1, 1, 10, 19))
        y = input_targets
        self.root_dir = root_dir
        self.transform = transform
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=1
        )

        if train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        obj_name = self.targets[idx]
        data_line = self.data[idx, :]
        sample = {'image': obj_name, 'data': data_line}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample

#Converting data to torch.FloatTensor
transform = transforms.ToTensor()

# load grasp dataset into train and test
train_data = ObjectGraspsDataset('shuffled_data_11_03_22.npy', 'labels_data_11_03_22.npy', './', train=True, transform=transform)
test_data = ObjectGraspsDataset('shuffled_data_11_03_22.npy', 'labels_data_11_03_22.npy', './', train=False, transform=transform)
# ignore negative values or ones that only vary with bit noise
train_data.data[train_data.data < 0] = 0
test_data.data[test_data.data < 0] = 0



# Define the image classes
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']

# Download the training and test datasets
# train_data = np.reshape(X_train, (-1, 1, 10, 19))
# test_data = np.reshape(X_test, (-1, 1, 10, 19))

# Prepare data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)   # torch.from_numpy(train_data)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)   # torch.from_numpy(test_data)


# Utility functions to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))


# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 5, padding=1, dtype=float)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1, dtype=float)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 3, padding=1, dtype=float)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 5, padding=1, dtype=float)
        self.unpool = nn.MaxUnpool2d(2, 2)
        # self.pad = nn.functional.pad((0, 0, 1, 0))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, p1 = self.pool(x)
        x = F.relu(self.conv2(x))
        x, p2 = self.pool(x)
        x = self.unpool(x, p2)
        x = F.relu(self.t_conv1(x))
        x = self.unpool(x, p1)
        x = F.pad(x, (0, 1), 'constant', 0)
        x = torch.softmax(self.t_conv2(x), dim=0)

        return x


# Instantiate the model
model = ConvAutoencoder()
print(model)

# Loss function - for multiclass classification this should be Cross Entropy after a softmax activation
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


device = get_device()
print(device)
model.to(device)

# Epochs
n_epochs = 100
loss_out = []

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    for data in train_loader:
        frame = data["data"]
        frame = frame.to(device)
        optimizer.zero_grad()
        outputs = model(frame)
        loss = criterion(outputs, frame)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * frame.size(0)

    train_loss = train_loss / len(train_loader)
    loss_out.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

# not sure where to take this next to evaluate the loss and evaluate the outputs
torch.save(model.state_dict(), 'final_model_state.pt')

model = ConvAutoencoder(*args, **kwargs)
model.load_state_dict(torch.load('final_model_state.pt'))
model.eval()