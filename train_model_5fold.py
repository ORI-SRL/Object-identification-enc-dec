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
FOLDS = 5
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']
batch_size = 32
n_epochs = 1500
patience = 30
noise_level = .00
drops = 0

SHOW = True
SAVE = True
JOINT_DATA = False


'''This is where the model can either be tuned and updated, or learnt from scratch with the combined data'''

print("Creating 'old' dataset splits...")
old_data = GraspDataset(x_filename="data/raw_data/base_unshuffled_original_data.npy",
                        y_filename="data/raw_data/base_unshuffled_original_labels.npy")
old_data_folds = old_data.get_fold_splits(k=5)

print("Creating 'new' dataset splits...")
new_data = GraspDataset(x_filename="data/raw_data/base_unshuffled_tuning_data.npy",
                        y_filename="data/raw_data/base_unshuffled_tuning_labels.npy",
                        normalize=True)
new_data_folds = new_data.get_fold_splits(k=5)
print("Datasets ready!")

valid_accs = {i:[] for i in range(10)}
predictions = {i:[] for i in range(10)}
true_labels = {i:[] for i in range(10)}

for k in range(FOLDS):
    seed_experiment(SEED)
    model = IterativeRNN4()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model, batch_params, batch_kpis = train_rcnn_network(model, optimizer, criterion, batch_size,
                                                         dp=drops,
                                                         n_epochs=n_epochs,
                                                         old_data=(old_data_folds[k]['train'], old_data_folds[k]['valid'], None),
                                                         new_data=(new_data_folds[k]['train'], new_data_folds[k]['valid'], None),
                                                         max_patience=patience,
                                                         save_folder=MODEL_SAVE_FOLDER,
                                                         oldnew=JOINT_DATA,
                                                         noise_level=noise_level,
                                                         save=SAVE,
                                                         show=SHOW)

    for grasp in range(10):
        valid_accs[grasp].append(batch_kpis['grasp_accuracies'][grasp])
        predictions[grasp].extend(batch_kpis['predictions'][grasp])
        true_labels[grasp].extend(batch_kpis['true_labels'][grasp])


print("5-fold valid_accuracy")
for grasp in range(10):
    print(f"{grasp+1} grasp(s): {np.array(valid_accs[grasp]).mean()*100:.5f}%")

    plot_confusion(predictions=predictions[grasp],
                   true_labels=true_labels[grasp],
                   unique_labels=old_data_folds[0]['train'].labels,
                   title=f'{grasp + 1} grasps - {model.__class__.__name__}',
                   save_folder='./figures/',
                   show=SHOW,
                   save=SAVE)

print('finished')
