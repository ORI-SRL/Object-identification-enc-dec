import copy
import numpy as np
# import os
from os.path import exists
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib import cm
# import serial
import time
from utils.ml_classifiers import *
from utils.pytorch_helpers import *
import pickle
import math


# from utils.networks import IterativeRNN2



def tune_RNN_network(model, optimizer, criterion, batch_size, old_data=None, new_data=None, n_epochs=50,
                     max_patience=25, save_folder='./', oldnew=True, save=True, show=True):
    model_name, device, train_loss_out, valid_loss_out, train_acc_out, valid_acc_out, patience, best_loss_dict, \
        best_params = model_init(model)
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
        train_batch_reminder = len(old_train_data) % half_batch
        valid_batch_reminder = len(old_valid_data) % half_batch

        n_train_batches = int(len(old_train_data) / half_batch) if train_batch_reminder == 0 else int(
            len(old_train_data) / half_batch) + 1
        n_valid_batches = int(len(old_valid_data) / half_batch) if valid_batch_reminder == 0 else int(
            len(old_valid_data) / half_batch) + 1

    old_train_indices = list(range(len(old_train_data)))
    old_valid_indices = list(range(len(old_valid_data)))

    new_train_indices = list(range(len(new_train_data)))
    new_valid_indices = list(range(len(new_valid_data)))

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
            batch_start = i * half_batch
            if oldnew:
                batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_train_data) \
                    else len(new_train_data)
            else:
                batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_train_data) \
                    else len(old_train_data)

            X_old, y_old, _ = old_train_data[old_train_indices[batch_start:batch_end]]
            X_new, y_new, _ = new_train_data[new_train_indices[batch_start:batch_end]]

            # concatenate the new and old data then add noise to prevent overfitting
            X_cat = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) if oldnew else \
                X_old.reshape(-1, 10, 19).to(device)
            # X_cat[X_cat < 1] = 0

            y_cat = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_old.to(device)
            batch_ints = list(range(len(y_cat)))
            random.shuffle(batch_ints)
            X = X_cat[batch_ints, :, :]
            y = y_cat[batch_ints]

            padded_ints = list(range(n_grasps))
            random.shuffle(padded_ints)

            model.train()
            for k in range(n_grasps):
                frame_loss = 0

                # randomly switch in zero rows to vary the number of grasps being identified
                padded_start = padded_ints[k]  # np.random.randint(1, 11)
                X_pad = X[:, :padded_start + 1, :]

                # set hidden layer
                hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

                optimizer.zero_grad()
                """ iterate through each grasp and run the model """
                output = model(X_pad[:, 0, :], hidden)
                hidden = nn.functional.softmax(output, dim=-1)
                for j in range(1, padded_start + 1):
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

        accuracies = np.zeros(n_grasps).astype(float)  # setup for accuracy at each grasp number

        model.eval()
        for i in range(n_valid_batches):

            # Take each validation batch and process
            batch_start = i * half_batch
            if oldnew:
                batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_valid_data) \
                    else len(new_valid_data)
            else:
                batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_valid_data) \
                    else len(old_train_data)

            X_old, y_old, _ = old_valid_data[old_valid_indices[batch_start:batch_end]]
            X_new, y_new, _ = new_valid_data[new_valid_indices[batch_start:batch_end]]

            X_cat = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) if oldnew else \
                X_old.reshape(-1, 10, 19).to(device)
            # noise = torch.normal(0, 0.2, X_cat.shape).to(device)
            # X_cat += noise
            # X_cat[X_cat < 1] = 0
            y_cat = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_old.to(device)
            batch_ints = list(range(len(y_cat)))
            random.shuffle(batch_ints)
            X = X_cat[batch_ints, :, :]
            y = y_cat[batch_ints]

            padded_ints = list(range(n_grasps))
            random.shuffle(padded_ints)

            for k in range(n_grasps):

                # randomly switch in zero rows to vary the number of grasps being identified
                padded_start = padded_ints[k]  # np.random.randint(1, 11)
                X_pad = X[:, :padded_start + 1, :]

                # set the first hidden layer as a vanilla prediction or zeros
                hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

                output = model(X_pad[:, 0, :].float(), hidden)
                hidden = nn.functional.softmax(output, dim=-1)
                """ Run the model through each grasp """
                for j in range(1, padded_start + 1):
                    output = model(X_pad[:, j, :].float(), hidden)
                    hidden = nn.functional.softmax(output, dim=-1)
                valid_loss += criterion(output, y.squeeze())

                _, preds = output.detach().max(dim=1)
                frame_accuracy = torch.sum(preds == y.flatten()).cpu().numpy() / len(preds)
                valid_accuracy += frame_accuracy
                accuracies[padded_start] += frame_accuracy

        # calculate the testing accuracy and losses and divide by the number of batches
        valid_accuracy = valid_accuracy / (n_valid_batches * n_grasps)
        accuracies = accuracies / n_valid_batches
        valid_loss = valid_loss.detach().cpu() / (n_valid_batches * n_grasps)
        valid_loss_out.append(valid_loss)
        valid_acc_out.append(valid_accuracy)

        # confusion_perc = confusion_ints / torch.sum(confusion_ints, dim=1)

        print('Epoch: {} \tTraining Loss: {:.4f} \tValidation loss: {:.4f} \t Train accuracy {:.2f} '
              '\t Validation accuracy {:.2f} \t Patience {:}'
              .format(epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, patience))
        # empty cache to prevent overusing the memory
        torch.cuda.empty_cache()
        # Early Stopping Logic

        if best_loss_dict['valid_loss'] is None or valid_loss < best_loss_dict['valid_loss']:
            best_params = copy.copy(model.state_dict())
            best_loss_dict = {'train_loss': train_loss, 'valid_loss': valid_loss, 'train_acc': train_accuracy,
                              'valid_acc': valid_accuracy, 'epoch': epoch}
            patience = 0

            grasp_accuracy[:, 1] = accuracies
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
        plot_model(best_loss_dict, train_loss_out, valid_loss_out, train_acc_out, valid_acc_out, type="accuracy")

    print(f'Grasp accuracy: {grasp_accuracy}')
    return model, best_params, best_loss_dict


def test_tuned_model(model, n_epochs, batch_size, criterion, old_data=None, new_data=None, oldnew=False,
                     show_confusion=True):

    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
        best_params = model_init(model)

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
        test_batch_reminder = len(old_test_data) % half_batch

        n_test_batches = int(len(old_test_data) / half_batch) if test_batch_reminder == 0 else int(
            len(old_test_data) / half_batch) + 1

    old_test_indeces = list(range(len(old_test_data)))
    new_test_indeces = list(range(len(new_test_data)))

    model.eval()
    random.shuffle(old_test_indeces)
    random.shuffle(new_test_indeces)

    for i in range(n_test_batches):

        # Take each testing batch and process
        batch_start = i * half_batch
        batch_end = i * half_batch + half_batch \
            if i * half_batch + half_batch < len(new_test_data) \
            else len(new_test_data)

        X_old, y_old, y_labels_old = old_test_data[old_test_indeces[batch_start:batch_end]]
        X_new, y_new, y_labels_new = new_test_data[new_test_indeces[batch_start:batch_end]]

        X = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) \
            if oldnew else X_old.reshape(-1, 10, 19).to(device)
        y = torch.cat([y_old, y_new], dim=0).to(device)if oldnew else y_old.to(device)
        y_labels = np.concatenate([y_labels_old, y_labels_new]) if oldnew else y_old

        true_labels.extend(y_labels.squeeze().tolist())

        padded_ints = list(range(n_grasps))
        random.shuffle(padded_ints)

        for k in range(n_grasps):
            # randomly switch in zero rows to vary the number of grasps being identified
            padded_start = padded_ints[k]  # np.random.randint(1, 11)
            X_pad = X[:, :padded_start + 1, :]

            # set hidden layer
            hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

            """ iterate through each grasp and run the model """
            output = model(X_pad[:, 0, :], hidden)
            hidden = output
            for j in range(1, padded_start + 1):
                output = model(X_pad[:, j, :], hidden)
                hidden = output

            loss2 = criterion(output, y.squeeze())
            test_loss += loss2.item()

            # calculate accuracy of classification
            _, preds = output.detach().max(dim=1)
            frame_accuracy = torch.sum(preds == y.flatten()).cpu().numpy() / len(preds)

            test_accuracy += frame_accuracy
            grasp_accuracy[padded_start, 1] += frame_accuracy

            # add the prediction and true to the grasp_labels dict
            pred_labels_tmp = old_test_data.get_labels(preds.cpu().numpy())
            pred_labels.extend(pred_labels_tmp)
            grasp_pred_labels[str(padded_start+1)].extend(pred_labels_tmp)

    test_accuracy = test_accuracy / (n_epochs * n_grasps)
    test_loss = test_loss / (n_epochs * n_grasps)
    test_loss_out.append(test_loss)
    test_acc_out.append(test_accuracy)

    grasp_accuracy[:, 1] = grasp_accuracy[:, 1] / n_test_batches
    print(f'Grasp accuracy: {grasp_accuracy}')

    if show_confusion:
        for grasp_no in range(n_grasps):
            unique_labels = new_test_data.labels
            cm = confusion_matrix(true_labels, grasp_pred_labels[str(grasp_no+1)], labels=unique_labels)
            # cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels).plot()
            cm = cm.astype('float64')
            for row in range(len(unique_labels)):
                cm[row, :] = cm[row, :] / cm[row, :].sum()
            fig = plt.figure()
            plt.title(f'{grasp_no+1} grasps - {model.__class__.__name__}')
            fig.set_size_inches(8, 5)
            sns.set(font_scale=1.2)
            cm_display_percentages = sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues',
                                                 xticklabels=unique_labels,
                                                 yticklabels=unique_labels, vmin=0, vmax=1).plot()

            plt.show()

    return grasp_pred_labels

