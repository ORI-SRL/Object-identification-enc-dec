import copy
import numpy as np
import pandas as pd
from scipy import stats
from os.path import exists
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import time
from utils.ml_classifiers import *
from utils.pytorch_helpers import *
from utils.plot_helpers import *
import pickle
import math
from skimage.filters.rank import entropy
from skimage.morphology import disk, square


# from utils.networks import IterativeRNN2

def train_rcnn_network(model, optimizer, criterion, batch_size, old_data=None, new_data=None, n_epochs=50, dp=0,
                       max_patience=25, save_folder='./figures/', oldnew=True, noise_level=.05, save=True, show=True):

    model_name = model.__class__.__name__

    device = get_device()
    print(device)
    model.to(device)

    # Epochs
    train_loss_out = []
    valid_loss_out = []
    train_acc_out = []
    valid_acc_out = []

    patience = 0
    best_loss_dict = {'train_loss': None, 'valid_loss': None, 'train_acc': None,
                      'valid_acc': None, 'epoch': None}
    best_params = None

    hidden_size = 7

    """Convert data into tensors"""
    old_train_data, old_valid_data, _ = old_data
    new_train_data, new_valid_data, _ = new_data

    """Calculate how many batches there are"""

    batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size  # enforce even batch sizes
    half_batch = int(batch_size / 2)
    if oldnew:
        train_batch_reminder = len(new_train_data) % half_batch
        valid_batch_reminder = len(new_train_data) % half_batch

        n_train_batches = int(len(new_train_data) / half_batch) if train_batch_reminder == 0 else int(
            len(new_train_data) / half_batch) + 1
        n_valid_batches = int(len(new_valid_data) / half_batch) if valid_batch_reminder == 0 else int(
            len(new_valid_data) / half_batch) + 1
    else:
        train_batch_reminder = len(old_train_data) % batch_size
        valid_batch_reminder = len(old_valid_data) % batch_size

        n_train_batches = int(len(old_train_data) / batch_size) if train_batch_reminder == 0 else int(
            len(old_train_data) / batch_size) + 1
        n_valid_batches = int(len(old_valid_data) / batch_size) if valid_batch_reminder == 0 else int(
            len(old_valid_data) / batch_size) + 1

    old_train_indices = list(range(len(old_train_data)))
    old_valid_indices = list(range(len(old_valid_data)))

    new_train_indices = list(range(len(new_train_data)))
    new_valid_indices = list(range(len(new_valid_data)))

    entropy_out = {label: [] for label in old_valid_data.labels} # entropy(image, disk(5))

    n_grasps = 10
    grasp_accuracy = np.zeros((10, 2))  # setup for accuracy at each grasp number
    grasp_accuracy[:, 0] = np.linspace(1, 10, 10)

    for epoch in range(n_epochs):
        train_loss, valid_loss, train_accuracy, valid_accuracy = 0.0, 0.0, 0.0, 0.0
        cycle = 0
        # confusion_ints = torch.zeros((7, 7)).to(device)

        model.train()

        random.shuffle(old_train_indices)
        random.shuffle(old_valid_indices)
        random.shuffle(new_train_indices)
        random.shuffle(new_valid_indices)

        for i in range(n_train_batches):

            # Take each training batch and process
            if oldnew:
                batch_start = i * half_batch
                batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_train_data) \
                    else len(new_train_data)
            else:
                batch_start = i * batch_size
                batch_end = i * batch_size + batch_size if i * batch_size + batch_size < len(old_train_data) \
                    else len(old_train_data)

            X_old, y_old, _ = old_train_data[old_train_indices[batch_start:batch_end]]
            X_new, y_new, _ = new_train_data[new_train_indices[batch_start:batch_end]]

            # concatenate the new and old data then add noise to prevent overfitting
            X_cat = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) if oldnew else \
                X_old.reshape(-1, 10, 19).to(device)

            # noise = torch.abs(torch.normal(0.0, noise_level, X_cat.shape)).to(device)
            # X_cat += noise

            y_cat = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_old.to(device)
            batch_ints = list(range(len(y_cat)))
            random.shuffle(batch_ints)
            X = X_cat[batch_ints, :, :]
            y = y_cat[batch_ints]

            # randomly pick a number of grasps to train
            all_no_of_grasps = list(range(n_grasps))
            random.shuffle(all_no_of_grasps)

            # randomly pick a a grasp order
            grasps_order = list(range(n_grasps))
            random.shuffle(grasps_order)

            # randomly drop sensors
            if dp > 0:
                drop = np.random.randint(dp)
                X[:, :, random.sample(range(X.shape[-1]), k=drop)] = 0.

            model.train()
            for k in range(n_grasps):

                # randomly switch in zero rows to vary the number of grasps being identified
                no_of_grasps = all_no_of_grasps[k]  # np.random.randint(1, 11)
                X_pad = X[:, grasps_order[:no_of_grasps + 1], :]

                # set hidden layer
                hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

                optimizer.zero_grad()
                """ iterate through each grasp and run the model """
                for j in range(0, no_of_grasps + 1):
                    output = model(X_pad[:, j, :], hidden)
                    hidden = nn.functional.softmax(output, dim=-1)

                frame_loss = criterion(output, y.squeeze())
                frame_loss.backward()
                optimizer.step()
                # output = nn.functional.softmax(output, dim=-1)

                _, preds = output.detach().max(dim=1)
                frame_accuracy = torch.sum(preds == y.flatten()).cpu().numpy() / len(preds)
                train_accuracy += frame_accuracy
                train_loss += frame_loss
                cycle += 1
        train_loss = train_loss.detach().cpu() / (n_train_batches * n_grasps)
        train_accuracy = train_accuracy / (n_train_batches * n_grasps)
        train_loss_out.append(train_loss)
        train_acc_out.append(train_accuracy)

        grasp_accuracies = {i: 0. for i in range(n_grasps)}  # setup for accuracy at each grasp number
        grasp_predictions = {i: [] for i in range(n_grasps)}
        grasp_true_labels = {i: [] for i in range(n_grasps)}

        epoch_entropy = {label: [] for label in old_valid_data.labels}

        model.eval()
        for i in range(n_valid_batches):

            # Take each validation batch and process
            if oldnew:
                batch_start = i * half_batch
                batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_valid_data) \
                    else len(new_valid_data)
            else:
                batch_start = i * batch_size
                batch_end = i * batch_size + batch_size if i * batch_size + batch_size < len(old_valid_data) \
                    else len(old_valid_data)

            X_old, y_old, y_labels_old = old_valid_data[old_valid_indices[batch_start:batch_end]]
            X_new, y_new, y_labels_new = new_valid_data[new_valid_indices[batch_start:batch_end]]

            X_cat = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) if oldnew else \
                X_old.reshape(-1, 10, 19).to(device)

            # noise = torch.abs(torch.normal(0.0, noise_level, X_cat.shape)).to(device)
            # X_cat += noise

            y_cat = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_old.to(device)
            y_labels = np.concatenate([y_labels_old, y_labels_new]) if oldnew else y_labels_old

            batch_ints = list(range(len(y_cat)))
            random.shuffle(batch_ints)
            X = X_cat[batch_ints, :, :]
            y = y_cat[batch_ints]
            y_labels = y_labels[batch_ints]

            all_no_of_grasps = list(range(n_grasps))
            random.shuffle(all_no_of_grasps)

            # randomly pick a a grasp order
            grasps_order = list(range(n_grasps))
            random.shuffle(grasps_order)

            # randomly drop sensors
            if dp > 0:
                drop = np.random.randint(dp)
                X[:, :, random.sample(range(X.shape[-1]), k=drop)] = 0.

            for k in range(n_grasps):

                # randomly switch in zero rows to vary the number of grasps being identified
                no_of_grasps = all_no_of_grasps[k]  # np.random.randint(1, 11)
                X_pad = X[:, grasps_order[:no_of_grasps + 1], :]

                # set the first hidden layer as a vanilla prediction or zeros
                hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

                """ Run the model through each grasp """
                for j in range(0, no_of_grasps + 1):
                    output = model(X_pad[:, j, :].float(), hidden)
                    hidden = nn.functional.softmax(output, dim=-1)

                valid_loss += criterion(output, y.squeeze())

                _, preds = output.detach().max(dim=1)
                frame_accuracy = torch.sum(preds == y.flatten()).cpu().numpy() / len(preds)
                valid_accuracy += frame_accuracy

                grasp_accuracies[no_of_grasps] += frame_accuracy
                grasp_predictions[no_of_grasps].extend(old_train_data.get_labels(preds.cpu().numpy()))
                grasp_true_labels[no_of_grasps].extend(y_labels.flatten().tolist())

                if no_of_grasps == 9:
                    for cls, labels in enumerate(old_valid_data.labels):
                        embeds = model.get_embed()[y == cls].cpu().detach().numpy()
                        ent = np.sum([entropy(nn.Tanh()(torch.Tensor(embeds[i])), square(3)) for i in range(len(embeds))])/len(embeds)
                        epoch_entropy[labels].append(ent)

        # calculate the testing accuracy and losses and divide by the number of batches
        valid_accuracy = valid_accuracy / (n_valid_batches * n_grasps)

        for i in range(n_grasps): grasp_accuracies[i] /= n_valid_batches

        valid_loss = valid_loss.detach().cpu() / (n_valid_batches * n_grasps)
        valid_loss_out.append(valid_loss)
        valid_acc_out.append(valid_accuracy)

        for key in entropy_out.keys():
            entropy_out[key].append(np.mean(epoch_entropy[key]))

        # confusion_perc = confusion_ints / torch.sum(confusion_ints, dim=1)

        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation loss: {:.4f} \t Train accuracy {:.2f} '
              '\t Validation accuracy {:.2f} \t Patience {:}'
              .format(epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, patience))
        # empty cache to prevent overusing the memory
        torch.cuda.empty_cache()
        # Early Stopping Logic

        if best_loss_dict['valid_loss'] is None or valid_loss < best_loss_dict['valid_loss']:
            best_params = copy.copy(model.state_dict())
            best_loss_dict = {'train_loss': train_loss,
                              'valid_loss': valid_loss,
                              'train_acc': train_accuracy,
                              'valid_acc': valid_accuracy,
                              'grasp_accuracies': grasp_accuracies,
                              'predictions': grasp_predictions,
                              'true_labels': grasp_true_labels,
                              'epoch': epoch, }
            patience = 0

            for i in range(n_grasps):
                grasp_accuracy[i, 1] = grasp_accuracies[i]
        else:
            patience += 1

        if patience >= max_patience:
            print(f'Early stopping: training terminated at epoch {epoch} due to es, '
                  f'patience exceeded at {epoch - max_patience}')
            print(f'Best accuracies: Training: {best_loss_dict["train_acc"]} \t Testing: {best_loss_dict["valid_acc"]}')
            break

    if save and best_params is not None:
        model_file = f'{save_folder}{model_name}_dropout'
        save_params(model_file, best_loss_dict, best_params)

    if show:
        # plot model losses
        plot_model(best_loss_dict, train_loss_out, valid_loss_out, train_acc_out, valid_acc_out, type="accuracy", save_folder='./figures/', save=save)
        plot_entropies(entropy_out, labels=old_valid_data.labels, save_folder='./figures/', save=save)

    print(f'Grasp accuracy: \n{grasp_accuracy}')
    return model, best_params, best_loss_dict


def test_tuned_model(model, n_epochs, batch_size, criterion, old_data=None, new_data=None, oldnew=False,
                     noise_level=0.05, save_folder='./', show=True, save=False):


    device = get_device()
    print(device)
    model.to(device)

    # Epochs
    test_loss_out = []
    test_acc_out = []

    # set zero values for all initial parameters
    test_loss, test_accuracy = 0.0, 0.0
    grasp_pred_labels = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
    true_labels = []
    pred_labels = []
    hidden_size = 7
    n_grasps = 10

    grasp_accuracy = np.zeros((10, 2)).astype(float)  # setup for accuracy at each grasp number
    grasp_accuracy[:, 0] = np.linspace(1, 10, 10)

    """Extract data"""
    _, _, old_test_data = old_data
    _, _, new_test_data = new_data
    """Calculate how many batches there are"""

    batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size  # enforce even batch sizes
    half_batch = int(batch_size / 2)

    if oldnew:
        test_batch_reminder = len(new_test_data) % half_batch

        n_test_batches = int(len(new_test_data) / half_batch) if test_batch_reminder == 0 else int(
            len(new_test_data) / half_batch) + 1
    else:
        test_batch_reminder = len(old_test_data) % batch_size
        n_test_batches = int(len(old_test_data) / batch_size) if test_batch_reminder == 0\
            else int(len(old_test_data) / batch_size) + 1

    old_test_indeces = list(range(len(old_test_data)))
    new_test_indeces = list(range(len(new_test_data)))

    model.eval()
    random.shuffle(old_test_indeces)
    random.shuffle(new_test_indeces)

    for i in range(n_test_batches):

        # Take each testing batch and process
        if oldnew:
            batch_start = i * half_batch
            batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_test_data) \
                else len(new_test_data)
        else:
            batch_start = i * batch_size
            batch_end = i * batch_size + batch_size if i * batch_size + batch_size < len(old_test_data) \
                else len(old_test_data)

        X_old, y_old, y_labels_old = old_test_data[old_test_indeces[batch_start:batch_end]]
        X_new, y_new, y_labels_new = new_test_data[new_test_indeces[batch_start:batch_end]]

        X = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) \
            if oldnew else X_old.reshape(-1, 10, 19).to(device)

        # noise = torch.abs(torch.normal(0.0, noise_level, X.shape)).to(device)
        # X += noise

        y = torch.cat([y_old, y_new], dim=0).to(device)if oldnew else y_old.to(device)
        y_labels = np.concatenate([y_labels_old, y_labels_new]) if oldnew else y_labels_old

        true_labels.extend(y_labels.squeeze().tolist())

        all_no_of_grasps = list(range(n_grasps))
        random.shuffle(all_no_of_grasps)

        # randomly pick a a grasp order
        grasps_order = list(range(n_grasps))
        random.shuffle(grasps_order)

        # randomly drop sensors

        for k in range(n_grasps):
            # randomly switch in zero rows to vary the number of grasps being identified
            no_of_grasps = all_no_of_grasps[k]  # np.random.randint(1, 11)
            X_pad = X[:, grasps_order[:no_of_grasps + 1], :]

            # set hidden layer
            hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

            """ iterate through each grasp and run the model """
            for j in range(0, no_of_grasps + 1):
                output = model(X_pad[:, j, :], hidden)
                hidden = nn.functional.softmax(output, dim=-1)

            loss2 = criterion(output, y.squeeze())
            test_loss += loss2.item()

            # calculate accuracy of classification
            _, preds = output.detach().max(dim=1)
            frame_accuracy = torch.sum(preds == y.flatten()).cpu().numpy() / len(preds)

            test_accuracy += frame_accuracy
            grasp_accuracy[no_of_grasps, 1] += frame_accuracy

            # add the prediction and true to the grasp_labels dict
            pred_labels_tmp = old_test_data.get_labels(preds.cpu().numpy())
            pred_labels.extend(pred_labels_tmp)
            grasp_pred_labels[str(no_of_grasps+1)].extend(pred_labels_tmp)

    test_accuracy = test_accuracy / (n_epochs * n_grasps)
    test_loss = test_loss / (n_epochs * n_grasps)
    test_loss_out.append(test_loss)
    test_acc_out.append(test_accuracy)

    grasp_accuracy[:, 1] = grasp_accuracy[:, 1] / n_test_batches
    print(f'Grasp accuracy: \n{grasp_accuracy}')

    if show or save:
        for grasp_no in range(n_grasps):
            plot_confusion(predictions=grasp_pred_labels[str(grasp_no + 1)],
                           true_labels=true_labels,
                           unique_labels=new_test_data.labels,
                           title=f'{grasp_no + 1} grasps - {model.__class__.__name__}',
                           save_folder='./figures/',
                           show=show,
                           save=save)


    return grasp_pred_labels
