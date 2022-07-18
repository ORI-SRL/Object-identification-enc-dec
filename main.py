import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
from os.path import exists
import csv
import pickle
from utils.pytorch_helpers import learn_model, test_model, seed_experiment
from utils.data_handlers import ObjectGraspsDataset
from torch.utils.data import DataLoader
from utils.networks import *
from utils.ml_classifiers import svm_classifier, knn_classifier, tree_searches, compare_classifiers
from utils.loss_plotting import *
import numpy as np

DATA_PATH = os.path.abspath(os.getcwd())
DATA_FOLDER = './data/'
MODEL_SAVE_FOLDER = './saved_model_states/'
n_grasps = [10, 7, 5, 3, 1]
models = [TwoLayerWBatchNorm]  # TwoLayerConv, , TwoLayerWDropout
loss_comparison_dict = {}
sil_comparison_dict = {}
ml_dict = {}

# luca: seeding the experiment is useful to get reproduceable results
seed_experiment(123)
# Define the object classes
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']
# Prepare data loaders
batch_size = 32

TRAIN_MODEL = False
TEST_MODEL = True
USE_PREVIOUS = True
COMPARE_LOSSES = False

for ModelArchitecture in models:
    for num_grasps in n_grasps:

        # load grasp dataset into train and test
        train_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_train_data.npy',  # 'shuffled_data_11_03_22.npy'
                                         f'{DATA_FOLDER}shuffled_train_labels.npy',  # 'labels_data_11_03_22.npy'
                                         num_grasps, train=True, pre_sort=True)
        test_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_test_data.npy',
                                        f'{DATA_FOLDER}shuffled_test_labels.npy',
                                        num_grasps, train_data.max_vals, train_data.min_vals,
                                        train=False, pre_sort=True)

        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                  shuffle=True)  # torch.from_numpy(train_data)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0,
                                 shuffle=True)  # torch.from_numpy(test_data)
        model = ModelArchitecture()

        if COMPARE_LOSSES:
            print(f'{model.__class__.__name__}_{num_grasps}_grasps')
            loss_file = f'{MODEL_SAVE_FOLDER}losses/{model.__class__.__name__}_{num_grasps}_losses.csv'
            plot_silhouette(loss_file, model, num_grasps)
            model_state = f'./saved_model_states/{model.__class__.__name__}_{num_grasps}grasps_model_state.pt'
            model.load_state_dict(torch.load(model_state))
            model.eval()
            _, _, _, _, silhouette_score = test_model(
                model, train_loader, test_loader, classes, compare=False)
            plt.close('all')
            sil_comparison_dict[
                f'{model.__class__.__name__}_{num_grasps}'] = silhouette_score.cpu().detach().numpy().item()

        if TRAIN_MODEL:

            print(f'Total params: {(sum(p.numel() for p in model.parameters()) / 1000000.0):.2f}M')

            if USE_PREVIOUS:
                model_state = f'./saved_model_states/{model.__class__.__name__}_{num_grasps}grasps_model_state.pt'
                if exists(model_state):
                    model.load_state_dict(torch.load(model_state))
            # Loss function - for multiclass classification this should be Cross Entropy after a softmax activation
            criterion = nn.MSELoss()
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

            print(model)

            batch_params, batch_losses = learn_model(model, train_loader, test_loader, optimizer, criterion, num_grasps,
                                                     n_epochs=1500,
                                                     max_patience=100,
                                                     save_folder=MODEL_SAVE_FOLDER,
                                                     save=True,
                                                     show=True)

            loss_comparison_dict[model.__class__.__name__] = batch_losses

            # plot stuff to choose model

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # plt.title('Model Comparison')

            # for key in loss_comparison_dict.keys():
            #     plt.plot(loss_comparison_dict[key], label=key)

            # plt.xlabel('epochs')
            # plt.ylabel('test loss')
            # plt.legend()
            # plt.show()

        elif TEST_MODEL:

            # os.listdir('./saved_model_states'):
            print(f'{model.__class__.__name__}_{num_grasps}_grasps')
            model_state = f'./saved_model_states/{model.__class__.__name__}_{num_grasps}grasps_model_state.pt'
            model.load_state_dict(torch.load(model_state))
            model.eval()
            train_data, train_labels, test_data, test_labels, silhouette_score = test_model(
                model, train_loader, test_loader, classes, num_grasps, compare=False)

            svm_params, svm_acc = svm_classifier(train_data.detach().numpy(), train_labels,
                                                 test_data.detach().numpy(), test_labels, num_grasps, learn=False)
            knn_params, knn_acc = knn_classifier(train_data.detach().numpy(), train_labels,
                                                 test_data.detach().numpy(), test_labels, num_grasps, learn=False)
            tree_params, tree_acc = tree_searches(train_data.detach().numpy(), train_labels,
                                                  test_data.detach().numpy(), test_labels, num_grasps, learn=False)
            print('svm accuracy: {:.4f}\t knn accuracy: {:.4f} \t tree accuracy: {:.4f}'.format(svm_acc, knn_acc, tree_acc))
            ml_dict[f'{model.__class__.__name__}_{num_grasps}_svm'] = svm_params
            ml_dict[f'{model.__class__.__name__}_{num_grasps}_knn'] = knn_params
            ml_dict[f'{model.__class__.__name__}_{num_grasps}_tree'] = tree_params
            with open('./saved_model_states/classifier_comparison.pkl', 'wb') as f:
                pickle.dump(ml_dict, f)
            plt.close('all')
if COMPARE_LOSSES:
    with open('./saved_model_states/silhouette_comparison.pkl', 'wb') as f:
        pickle.dump(sil_comparison_dict, f)
print('finished')
