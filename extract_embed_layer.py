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
from utils.online_validation import *


DATA_PATH = os.path.abspath(os.getcwd())
DATA_FOLDER = "./data/validation_data_523/"
MODEL_SAVE_FOLDER = './saved_model_states/iterative/shifted/'
FILE_PREFIX = "shifted_"
n_grasps = [10]  # , 7, 5, 3, 1]
loss_comparison_dict = {}
sil_comparison_dict = {}
ml_dict = {}
train_ratio, valid_ratio = .6, .2  # test will be the remaining .2


SEED = 123
# luca: seeding the experiment is useful to get reproducible results
seed_experiment(SEED)
# Define the object classes
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']
# Prepare data loaders
batch_size = 32
n_epochs = 1000
plt.rcParams.update({'font.size': 15})

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
if TUNING:
    PREP_TUNE = True

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

    # model = IterativeRNN4()

    model = IterativeRNN4_embed()
    model_name = model.__class__.__name__.split('_')[0]

    if USE_PREVIOUS:
        '''either use the previous model and update, or train from scratch'''
        model_state = f'{MODEL_SAVE_FOLDER}{model_name}_dropout_model_state.pt'
        if exists(model_state):
            model.load_state_dict(torch.load(model_state))
            print("Model loaded!")
            # plot_embed(model, old_train_data, batch_size, device=get_device(), show=True, save=False)
            # plot_embed_optimize(model, data=old_data, device=get_device(), show=True, save=False)
            plot_embed_optimize_direct(model, data=old_data, device=get_device(), show=True, save=True)
            exit()
        else:
            print("Could not find model to load!")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model, batch_params, batch_losses = tune_RNN_network(model, optimizer, criterion, batch_size,
                                                         n_epochs=n_epochs,
                                                         old_data=(old_train_data, old_valid_data, old_test_data),
                                                         new_data=(new_train_data, new_valid_data, new_test_data),
                                                         max_patience=30,
                                                         save_folder=MODEL_SAVE_FOLDER,
                                                         oldnew=JOINT_DATA,
                                                         save=True,
                                                         show=True)

    """test_tuned_model will return the predicted vs true labels for use in confusion matrix plotting"""
    grasp_pred_labels = test_tuned_model(model, n_epochs, batch_size, criterion,
                                         old_data=(old_train_data, old_valid_data, old_test_data),
                                         new_data=(new_train_data, new_valid_data, new_test_data),
                                         oldnew=JOINT_DATA, show_confusion=False)
    model_file = f'{MODEL_SAVE_FOLDER}{model_name}_labels'
    print('finished')
