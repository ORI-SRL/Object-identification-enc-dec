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
MODEL_SAVE_FOLDER_FINETUNE = './saved_model_states/iterative/finetuned/'
FILE_PREFIX = "shifted_"
n_grasps = [10]  # , 7, 5, 3, 1]
loss_comparison_dict = {}
sil_comparison_dict = {}
ml_dict = {}
train_ratio, valid_ratio = .8, .2  # test will be the remaining .2

SEED = 1234
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']
batch_size = 32
n_epochs = 1500
patience = 50
noise_level = .00
drops = 0

TRAIN_MODEL = True
TEST_MODEL = False
USE_PREVIOUS = True
JOINT_DATA = True
COMPARE_LOSSES = True
ITERATIVE = True
RNN = True
ONLINE_VALIDATION = False
TUNING = True


def get_model(Model, use_previous=False, save_folder=''):
    model = Model()
    model_name = model.__class__.__name__.split('_')[0]
    exist = False

    model_state = None
    if use_previous:
        '''either use the previous model and update, or train from scratch'''
        model_state = f'{save_folder}{model_name}_dropout_model_state.pt'
        if exists(model_state):
            model.load_state_dict(torch.load(model_state))
            print("Model loaded!")
            exist = True
        else:
            print("Could not find model to load!")

    return model, model_state, model_name, exist


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
seed_experiment(SEED)
model = IterativeRNN4()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
model, batch_params, batch_kpis_train = train_rcnn_network(model, optimizer, criterion, batch_size,
                                                           dp=drops,
                                                           n_epochs=n_epochs,
                                                           old_data=(old_train_data, old_valid_data, old_test_data),
                                                           new_data=(new_train_data, new_valid_data, new_test_data),
                                                           max_patience=patience,
                                                           save_folder=MODEL_SAVE_FOLDER_FINETUNE,
                                                           oldnew=JOINT_DATA,
                                                           noise_level=noise_level,
                                                           save=True,
                                                           show=True)


seed_experiment(SEED)
model, _, _, exist = get_model(IterativeRNN4, use_previous=USE_PREVIOUS, save_folder=MODEL_SAVE_FOLDER)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
if exist:
    model, batch_params, batch_kpis_finetune = train_rcnn_network(model, optimizer, criterion, batch_size,
                                                               dp=drops,
                                                               n_epochs=n_epochs,
                                                               old_data=(old_train_data, old_valid_data, old_test_data),
                                                               new_data=(new_train_data, new_valid_data, new_test_data),
                                                               max_patience=patience,
                                                               save_folder=MODEL_SAVE_FOLDER_FINETUNE,
                                                               oldnew=JOINT_DATA,
                                                               noise_level=noise_level,
                                                               save=True,
                                                               show=True)

seed_experiment(SEED)
model, _, _, exist = get_model(IterativeRNN4, use_previous=USE_PREVIOUS, save_folder=MODEL_SAVE_FOLDER)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
if exist:
    model.save_embed_states()
    model.freeze_embed()
    model, batch_params, batch_kpis_finetunehead = finetune_rcnn_network(model, optimizer, criterion, batch_size,
                                                                   dp=drops,
                                                                   n_epochs=n_epochs,
                                                                   old_data=(old_train_data, old_valid_data, old_test_data),
                                                                   new_data=(new_train_data, new_valid_data, new_test_data),
                                                                   max_patience=patience,
                                                                   save_folder=MODEL_SAVE_FOLDER_FINETUNE,
                                                                   oldnew=JOINT_DATA,
                                                                   noise_level=noise_level,
                                                                   save=True,
                                                                   show=True)


with open(f'{MODEL_SAVE_FOLDER_FINETUNE}finetune_results.npy', 'wb') as f:
    np.save(f, np.array(batch_kpis_train['valid_losses']))
    np.save(f, np.array(batch_kpis_finetune['valid_losses']))
    np.save(f, np.array(batch_kpis_finetunehead['valid_losses']))
    np.save(f, np.array(batch_kpis_train['valid_accs']))
    np.save(f, np.array(batch_kpis_finetune['valid_accs']))
    np.save(f, np.array(batch_kpis_finetunehead['valid_accs']))

plot_finetune_comparison([batch_kpis_train['valid_losses'], batch_kpis_finetune['valid_losses'], batch_kpis_finetunehead['valid_losses']],
                         [batch_kpis_train['valid_accs'], batch_kpis_finetune['valid_accs'], batch_kpis_finetunehead['valid_accs']],
                         ['re-training', 'model fine-tune', 'embed-transfer'],
                         save_folder='./figures/',
                         show=True,
                         save=True)

print('finished')
