import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
from utils.pytorch_helpers import learn_model, test_model, seed_experiment
from utils.data_handlers import ObjectGraspsDataset
from torch.utils.data import DataLoader
from utils.networks import TwoLayerConv
from utils.networks import TwoLayerWDropout
from utils.networks import TwoLayerWBatchNorm

DATA_PATH = os.path.abspath(os.getcwd())
DATA_FOLDER = './data/'
MODEL_SAVE_FOLDER = './saved_model_states/'

# luca: seeding the experiment is useful to get reproduceable results
seed_experiment(123)

# load grasp dataset into train and test
train_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_data_11_03_22.npy',
                                 f'{DATA_FOLDER}labels_data_11_03_22.npy',
                                 train=True)
test_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_data_11_03_22.npy',
                                f'{DATA_FOLDER}labels_data_11_03_22.npy',
                                train=False)

# Define the image classes
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']

# Prepare data loaders
batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)   # torch.from_numpy(train_data)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)   # torch.from_numpy(test_data)

models = [TwoLayerConv, TwoLayerWBatchNorm, TwoLayerWDropout]
loss_comparison_dict = {}

TRAIN_MODEL = False
if TRAIN_MODEL:
    for ModelArchitecture in models:
        model = ModelArchitecture()
        print(f'Total params: {(sum(p.numel() for p in model.parameters()) / 1000000.0):.2f}M')

        # Loss function - for multiclass classification this should be Cross Entropy after a softmax activation
        criterion = nn.MSELoss()
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print(model)

        batch_params, batch_losses = learn_model(model, train_loader, test_loader, optimizer, criterion,
                                                 n_epochs=150,
                                                 max_patience=20,
                                                 save_folder=MODEL_SAVE_FOLDER,
                                                 save=True,
                                                 show=True)

        loss_comparison_dict[model.__class__.__name__] = batch_losses
else:
    for ModelStates in os.listdir('./saved_model_states'):
        model = torch.load(DATA_PATH)
        encoded_data = test_model(model, train_loader, test_loader)


# plot stuff to choose model

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Model Comparison')

for key in loss_comparison_dict.keys():
    plt.plot(loss_comparison_dict[key], label=key)

plt.xlabel('epochs')
plt.ylabel('test loss')
plt.legend()
plt.show()

#model.load_state_dict(torch.load('final_model_state.pt'))
#model.eval()

print('finished')
