# import torch.nn as nn
# import matplotlib.pyplot as plt
import os
# from os.path import exists
# import csv
import pandas as pd
import torch

from utils.pytorch_helpers import *
from utils.data_handlers import *
from torch.utils.data import DataLoader
from utils.networks import *
# from utils.ml_classifiers import *
from utils.loss_plotting import *
from utils.online_validation import *
import numpy as np

DATA_PATH = os.path.abspath(os.getcwd())
DATA_FOLDER = "./data/combined_tune_data/"
MODEL_SAVE_FOLDER = './saved_model_states/iterative/shifted/'
FILE_PREFIX = "combined_"
n_grasps = [10]  # , 7, 5, 3, 1]
models = [IterativeRNN2]  # TwoLayerConv, , TwoLayerWDropout IterativeRNN2
loss_comparison_dict = {}
sil_comparison_dict = {}
ml_dict = {}

# luca: seeding the experiment is useful to get reproducible results
seed_experiment(123)
# Define the object classes
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']
# Prepare data loaders
batch_size = 32

TRAIN_MODEL = True
TEST_MODEL = True
USE_PREVIOUS = False
COMPARE_LOSSES = False
ITERATIVE = True
RNN = True
ONLINE_VALIDATION = False
TUNING = True

# load grasp datasets
train_data = ObjectGraspsDataset(f'{DATA_FOLDER}{FILE_PREFIX}shuffled_train_data.npy',
                                 f'{DATA_FOLDER}{FILE_PREFIX}shuffled_train_labels.npy', 10, train=True,
                                 pre_sort=True, random_pad=False)
test_data = ObjectGraspsDataset(f'{DATA_FOLDER}{FILE_PREFIX}shuffled_test_data.npy',
                                f'{DATA_FOLDER}{FILE_PREFIX}shuffled_test_labels.npy', 10, train_data.max_vals,
                                train_data.min_vals, train=False, pre_sort=True, random_pad=False)
validation_data = ObjectGraspsDataset(f'{DATA_FOLDER}{FILE_PREFIX}shuffled_val_data.npy',
                                      f'{DATA_FOLDER}{FILE_PREFIX}shuffled_val_labels.npy', 10,
                                      train_data.max_vals,
                                      train_data.min_vals, train=False, pre_sort=True, random_pad=False)
online_data = ObjectGraspsDataset(f'{DATA_FOLDER}../shifted_data/shuffled_online_data.npy',
                                  f'{DATA_FOLDER}../shifted_data/shuffled_online_labels.npy', 10, train_data.max_vals,
                                  train_data.min_vals, train=False,
                                  pre_sort=True, random_pad=False)

if ITERATIVE or RNN:
    train_data.data = train_data.data.reshape(train_data.data.size(0), train_data.data.size(2),
                                              train_data.data.size(3))
    test_data.data = test_data.data.reshape(test_data.data.size(0), test_data.data.size(2),
                                            test_data.data.size(3))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0,
                             shuffle=True)
    norm_vals = torch.stack([train_data.max_vals, train_data.min_vals])
    model = models[0]()
    test_new_grasps(model, DATA_FOLDER, MODEL_SAVE_FOLDER, classes, norm_vals)

    shuffle_online_data(DATA_FOLDER, 50, classes)
    online_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_online_data.npy',
                                      f'{DATA_FOLDER}shuffled_online_labels.npy', 10, train_data.max_vals,
                                      train_data.min_vals, train=False,
                                      pre_sort=True, random_pad=False)
    online_data.data = online_data.data.reshape(online_data.data.size(0), online_data.data.size(2),
                                                online_data.data.size(3))
    online_loader = DataLoader(online_data, batch_size=batch_size, num_workers=0,
                               shuffle=True)
    model_state = f'{MODEL_SAVE_FOLDER}{model.__class__.__name__}_dropout_model_state.pt'
    if exists(model_state):
        model.load_state_dict(torch.load(model_state))
    # Loss function - for multiclass classification this should be Cross Entropy after a softmax activation
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    batch_params, batch_losses = learn_iter_model(model, online_loader, test_loader, optimizer, criterion,
                                                  classes,
                                                  n_epochs=1500,
                                                  max_patience=75,
                                                  save_folder=MODEL_SAVE_FOLDER,
                                                  save=False,
                                                  show=True)

    # gather_grasps(DATA_FOLDER, classes, norm_vals)
    # online_loop(model, MODEL_SAVE_FOLDER, norm_vals, classes)

for ModelArchitecture in models:
    for num_grasps in n_grasps:

        model = ModelArchitecture()

        # load grasp dataset into train and test
        train_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_train_data.npy',
                                         f'{DATA_FOLDER}shuffled_train_labels.npy', num_grasps, train=True,
                                         pre_sort=True, random_pad=False)
        test_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_test_data.npy',
                                        f'{DATA_FOLDER}shuffled_test_labels.npy', num_grasps, train_data.max_vals,
                                        train_data.min_vals, train=False, pre_sort=True, random_pad=False)
        validation_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_val_data.npy',
                                              f'{DATA_FOLDER}shuffled_val_labels.npy', num_grasps, train_data.max_vals,
                                              train_data.min_vals, train=False, pre_sort=True, random_pad=False)

        if ITERATIVE or RNN:
            train_data.data = train_data.data.reshape(train_data.data.size(0), train_data.data.size(2),
                                                      train_data.data.size(3))
            test_data.data = test_data.data.reshape(test_data.data.size(0), test_data.data.size(2),
                                                    test_data.data.size(3))
            validation_data.data = validation_data.data.reshape(validation_data.data.size(0),
                                                                validation_data.data.size(2),
                                                                validation_data.data.size(3))

        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0,
                                  shuffle=True)  # torch.from_numpy(train_data)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0,
                                 shuffle=True)  # torch.from_numpy(test_data)
        val_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0,
                                shuffle=True)

        if TRAIN_MODEL:

            print(f'Total params: {(sum(p.numel() for p in model.parameters()) / 1000000.0):.2f}M')

            if USE_PREVIOUS:
                model_state = f'{MODEL_SAVE_FOLDER}{model.__class__.__name__}_dropout_model_state.pt'
                if exists(model_state):
                    model.load_state_dict(torch.load(model_state))
            # Loss function - for multiclass classification this should be Cross Entropy after a softmax activation
            criterion = nn.CrossEntropyLoss()
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            print(model)
            if ITERATIVE and not RNN:
                batch_params, batch_losses = learn_iter_model(model, train_loader, test_loader, optimizer, criterion,
                                                              classes,
                                                              n_epochs=1500,
                                                              max_patience=75,
                                                              save_folder=MODEL_SAVE_FOLDER,
                                                              save=True,
                                                              show=True)
            elif RNN:
                batch_params, batch_losses = train_RNN(model, train_loader, test_loader, optimizer, criterion,
                                                       classes, batch_size,
                                                       n_epochs=1000,
                                                       max_patience=75,
                                                       save_folder=MODEL_SAVE_FOLDER,
                                                       save=True,
                                                       show=True)
            else:
                batch_params, batch_losses = learn_model(model, train_loader, test_loader, optimizer, criterion,
                                                         num_grasps,
                                                         n_epochs=1500,
                                                         max_patience=50,
                                                         save_folder=MODEL_SAVE_FOLDER,
                                                         save=True,
                                                         show=True)

            loss_comparison_dict[model.__class__.__name__] = batch_losses

        if TEST_MODEL:
            # os.listdir('./saved_model_states'):
            model_name = model.__class__.__name__
            print(f'{model_name}_{num_grasps}_grasps')
            model_state = f'{MODEL_SAVE_FOLDER}{model_name}_dropout_model_state.pt'
            model.load_state_dict(torch.load(model_state))
            model.eval()
            criterion = nn.CrossEntropyLoss()
            if ITERATIVE:
                true_labels, pred_labels, grasp_true, grasp_pred = test_iter_model(model, val_loader, classes,
                                                                                   criterion)
                model_file = f'{MODEL_SAVE_FOLDER}{model_name}_labels'
                # save_params(model_file, true_labels, pred_labels)

                df = pd.DataFrame(columns=["True Values", "Pred Values"])
                df["True Values"], df["Pred Values"] = true_labels, pred_labels
                plot_confusion(pred_labels, true_labels, model_name, n_grasps, iter=True)

                for grasp in grasp_true:
                    plot_confusion(grasp_pred[grasp], grasp_true[grasp], model_name, int(grasp), iter=True)

                with open(f'{model_file}.csv', 'w') as f:
                    w = csv.writer(f)
                    w.writerow(true_labels)
                    w.writerow(pred_labels)
                    w.writerow(grasp_true.values())
                    w.writerow(grasp_pred.values())

            else:
                train_data, train_labels, test_data, test_labels, silhouette_score = test_model(
                    model, train_loader, test_loader, classes, num_grasps, compare=False)

                # svm_params, svm_acc = svm_classifier(train_data.detach().numpy(), train_labels,
                #                                     test_data.detach().numpy(), test_labels, num_grasps, learn=False)
                knn_params, knn_acc = knn_classifier(train_data.detach().numpy(), train_labels,
                                                     test_data.detach().numpy(), test_labels, num_grasps, learn=False)
                tree_params, tree_acc = tree_searches(train_data.detach().numpy(), train_labels,
                                                      test_data.detach().numpy(), test_labels, num_grasps, learn=False)
                print('knn accuracy: {:.4f} \t tree accuracy: {:.4f}'.format(
                    knn_acc, tree_acc))  # 'svm accuracy: {:.4f}\t,
                # ml_dict[f'{model.__class__.__name__}_{num_grasps}_svm'] = svm_params
                ml_dict[f'{model.__class__.__name__}_{num_grasps}_knn'] = knn_params
                ml_dict[f'{model.__class__.__name__}_{num_grasps}_tree'] = tree_params
                with open(f'./{MODEL_SAVE_FOLDER}classifier_comparison.pkl', 'wb') as f:
                    pickle.dump(ml_dict, f)
                plt.close('all')
if COMPARE_LOSSES:
    for ModelArchitecture in models:
        model = ModelArchitecture()
        # print(f'{model.__class__.__name__}_{n_grasps}_grasps')
        loss_file = f'{MODEL_SAVE_FOLDER}{model.__class__.__name__}'  # _{num_grasps}_losses.csv'
        if ITERATIVE:
            plot_losses(loss_file, model)
        else:
            plot_silhouette(loss_file, model, n_grasps)
        plt.show()
        '''for num_grasps in n_grasps:
            model_state = f'{MODEL_SAVE_FOLDER}{model.__class__.__name__}_{num_grasps}grasps_model_state.pt'
            model.load_state_dict(torch.load(model_state))
            model.eval()
            _, _, _, _, silhouette_score = test_model(
                model, train_loader, test_loader, classes, num_grasps, compare=False)
            plt.close('all')
            sil_comparison_dict[
                f'{model.__class__.__name__}_{num_grasps}'] = silhouette_score.cpu().detach().numpy().item()
    print(sil_comparison_dict)'''
    with open(f'{MODEL_SAVE_FOLDER}silhouette_comparison.pkl', 'wb') as f:
        pickle.dump(sil_comparison_dict, f)
plt.show()
print('finished')
