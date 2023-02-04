# import torch.nn as nn
# import matplotlib.pyplot as plt
import os
# from os.path import exists
# import csv
import random

import pandas as pd
import torch

from utils.pytorch_helpers import *
from utils.data_handlers import *
from torch.utils.data import DataLoader
from utils.networks import *
# from utils.ml_classifiers import *
from utils.plot_helpers import *
from utils.experiment_runs import *

plt.rcParams.update({'font.size': 15})

DATA_PATH = os.path.abspath(os.getcwd())
DATA_FOLDER = "./data/validation_data_523/"
MODEL_SAVE_FOLDER = './saved_model_states/iterative/shifted/'
FILE_PREFIX = "shifted_"
n_grasps = [10]  # , 7, 5, 3, 1]
loss_comparison_dict = {}
sil_comparison_dict = {}
ml_dict = {}
train_ratio, valid_ratio = .6, .2  # test will be the remaining .2

SEED = 1234
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']
batch_size = 32
n_epochs = 1500
patience = 100
noise_level = .00

TRAIN_MODEL = True
TEST_MODEL = False
USE_PREVIOUS = True
JOINT_DATA = False
COMPARE_LOSSES = True
ITERATIVE = True
RNN = True
ONLINE_VALIDATION = False
TUNING = True


'''This is where the model can either be tuned and updated, or learnt from scratch with the combined data'''

print("Creating 'old' dataset splits...")
old_data = GraspDataset(x_filename="data/raw_data/base_unshuffled_original_data.npy",
                        y_filename="data/raw_data/base_unshuffled_original_labels.npy")
old_train_data, old_valid_data, old_test_data = old_data.get_splits(train_ratio=train_ratio,
                                                                    valid_ratio=valid_ratio)

print("Creating 'new' dataset splits...")
new_data = GraspDataset(x_filename="data/raw_data/base_unshuffled_tuning_data.npy",
                        y_filename="data/raw_data/base_unshuffled_tuning_labels.npy",
                        normalize=True)
new_train_data, new_valid_data, new_test_data = new_data.get_splits(train_ratio=train_ratio,
                                                                    valid_ratio=valid_ratio)
print("Datasets ready!")

dropout_sensors = range(10)


drops = []
valid_losses = []
valid_accs = []
for dp in dropout_sensors:
    seed_experiment(SEED)
    model = IterativeRNN4()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model, batch_params, batch_kpis = train_rcnn_network(model, optimizer, criterion, batch_size,
                                                           dp=dp,
                                                           n_epochs=n_epochs,
                                                           old_data=(old_train_data, old_valid_data, old_test_data),
                                                           new_data=(new_train_data, new_valid_data, new_test_data),
                                                           max_patience=patience,
                                                           save_folder=MODEL_SAVE_FOLDER,
                                                           oldnew=JOINT_DATA,
                                                           noise_level=noise_level,
                                                           save=True,
                                                           show=True)

    drops.append(dp)
    valid_losses.append(batch_kpis['valid_loss'].item())
    valid_accs.append(batch_kpis['valid_acc'])

print(drops)
print(valid_losses)
print(valid_accs)
with open(f'{MODEL_SAVE_FOLDER}results.npy', 'wb') as f:
    np.save(f, np.array(drops))
    np.save(f, np.array(valid_losses))
    np.save(f, np.array(valid_accs))



print('finished')
