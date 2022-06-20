import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
import pickle
from utils.pytorch_helpers import learn_model, test_model, seed_experiment
from utils.data_handlers import ObjectGraspsDataset
from torch.utils.data import DataLoader
from utils.networks import *
from utils.ml_classifiers import svm_classifier, knn_classifier, tree_searches, compare_classifiers

DATA_PATH = os.path.abspath(os.getcwd())
DATA_FOLDER = './data/'
MODEL_SAVE_FOLDER = './saved_model_states/'
n_grasps = 5

# luca: seeding the experiment is useful to get reproduceable results
seed_experiment(123)

# load grasp dataset into train and test
train_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_data_11_03_22.npy',
                                 f'{DATA_FOLDER}labels_data_11_03_22.npy',
                                 n_grasps, train=True)
test_data = ObjectGraspsDataset(f'{DATA_FOLDER}shuffled_data_11_03_22.npy',
                                f'{DATA_FOLDER}labels_data_11_03_22.npy',
                                n_grasps, train=False)

# Define the object classes
classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']

# Prepare data loaders
batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)  # torch.from_numpy(train_data)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)  # torch.from_numpy(test_data)

if n_grasps == 10:
    models = [TwoLayerConv, TwoLayerWBatchNorm, TwoLayerWDropout]  #
elif n_grasps == 5:
    models = [TwoLayerConv5Grasp]
loss_comparison_dict = {}

TRAIN_MODEL = False
if TRAIN_MODEL:
    for ModelArchitecture in models:
        model = ModelArchitecture()
        print(f'Total params: {(sum(p.numel() for p in model.parameters()) / 1000000.0):.2f}M')

        # Loss function - for multiclass classification this should be Cross Entropy after a softmax activation
        criterion = nn.MSELoss()
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=7.5e-4)

        print(model)

        batch_params, batch_losses = learn_model(model, train_loader, test_loader, optimizer, criterion, n_grasps,
                                                 n_epochs=500,
                                                 max_patience=50,
                                                 save_folder=MODEL_SAVE_FOLDER,
                                                 save=True,
                                                 show=True)

        loss_comparison_dict[model.__class__.__name__] = batch_losses

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

else:
    # for ModelArchitecture in models:
    # os.listdir('./saved_model_states'):
    model = TwoLayerConv5Grasp()  # ModelArchitecture()
    model_state = f'./saved_model_states/{model.__class__.__name__}_5grasps_model_state.pt'
    model.load_state_dict(torch.load(model_state))
    model.eval()
    train_data, train_labels, test_data, test_labels = test_model(
        model, train_loader, test_loader, classes, compare=False)

    # svm_params = svm_classifier(train_data.detach().numpy(), train_labels,
    #                             test_data.detach().numpy(), test_labels, learn=True)
    knn_params = knn_classifier(train_data.detach().numpy(), train_labels,
                                test_data.detach().numpy(), test_labels, learn=True)
    tree_params = tree_searches(train_data.detach().numpy(), train_labels,
                                test_data.detach().numpy(), test_labels, learn=True)

    save_folder = './saved_model_states/ml_states/'
    pickle.dump(knn_params, open(f'{save_folder}{model.__class__.__name__}_knn_params', 'wb'))
    # pickle.dump(svm_params, open(f'{save_folder}{model.__class__.__name__}_svm_params', 'wb'))
    pickle.dump(tree_params, open(f'{save_folder}{model.__class__.__name__}_tree_params', 'wb'))

# model.load_state_dict(torch.load('final_model_state.pt'))
# model.eval()

print('finished')
