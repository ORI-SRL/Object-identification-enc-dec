import pickle
import csv
import matplotlib.pyplot as plt
import os
from os.path import exists
import numpy as np


def plot_silhouette(file_path, model, n_grasps):
    model_name = model.__class__.__name__
    loss_dict = dict()
    f_size = 16
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.set_title('Losses', fontsize=f_size + 4)
    ax1.set_ylabel('Loss', fontsize=f_size)
    ax1.set_xlabel('Epoch', fontsize=f_size)
    ax2.set_title('Silhouette score', fontsize=f_size + 4)
    ax2.set_xlabel('Epoch', fontsize=f_size)
    ax2.set_ylabel('Silhouette score', fontsize=f_size)
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    cIdx = 0

    for num_grasps in n_grasps:
        file_name = f'{file_path}_{num_grasps}_losses.csv'
        if exists(file_name):
            with open(file_name, mode='r') as infile:
                reader = csv.reader(infile)
                for row in reader:
                    if len(row) > 0:
                        if row[0]:
                            data_str_long = row[1][1:-1]
                            data_str = data_str_long.split(', ')
                            if "tensor" in data_str_long:
                                data_str = data_str[0::2]
                                data_float = [-float(x[7:14]) for x in data_str]
                            else:
                                data_float = [float(x) + 0.015 for x in data_str]
                            # print(row)
                            loss_dict[row[0]] = data_float

            # 'training': train_loss_out, 'testing': test_loss_out, 'training_silhouette': train_sil_out
            train_loss = loss_dict['training']
            test_loss = loss_dict['testing']
            train_sil_score = loss_dict['training_silhouette']
            test_sil_score = loss_dict['testing_silhouette']
            epochs = np.linspace(1, len(train_loss), len(train_loss))
            max_silhouette = max(test_sil_score)
            max_sil_idx = np.argmax(test_sil_score)
            plt.rcParams.update({'font.size': 12})

            ax1.plot(epochs, train_loss, '--', color=colours[cIdx], label=f'{num_grasps} Grasps Training')
            ax1.plot(epochs, test_loss, '-', color=colours[cIdx], label=f'{num_grasps} Grasps Validation')
            ax1.legend(prop={"size": 12})
            ax2.plot(epochs, train_sil_score, '--', color=colours[cIdx], label=f'{num_grasps} Grasps Training')
            ax2.plot(epochs, test_sil_score, '-', color=colours[cIdx], label=f'{num_grasps} Grasps Testing')
            ax2.scatter(max_sil_idx, max_silhouette, color=colours[cIdx])
            ax2.scatter(max_sil_idx, max_silhouette, color=colours[cIdx])
            cIdx += 1
            # ax2.legend()
            # fig.suptitle('Three Layer Convolution with Batch Norm, Losses and Silhouette Score')
            # plt.show()
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(14)
    plt.show()


def plot_losses(file, model):
    file_name = f'{file}.csv'
    losses_data = []
    if exists(file_name):
        with open(file_name, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if len(row) > 0:
                    data_str_long = row[1][1:-1]
                    data_str = data_str_long.split(', ')
                    for idx, data_entry in enumerate(data_str):
                        if "tensor" in data_entry:
                            t_start = data_entry.find('(')
                            t_end = data_entry.find(')')
                            data_str[idx] = float(data_entry[t_start + 1:t_end])
                        else:
                            data_str = [float(x) for x in data_str]
                    # print(row)
                    losses_data.append(data_str)

                # loss_dict[row[0]] = data_float
    train_loss = losses_data[0]
    test_loss = losses_data[1]
    type_train = losses_data[2]
    type_test = losses_data[3]
    fig, [ax1, ax2] = plt.subplots(1, 2)
    x = list(range(1, len(test_loss) + 1))
    ax1.plot(x, train_loss, label="Training loss")
    ax1.plot(x, test_loss, label="Testing loss")
    # ax1.plot(best_loss['epoch'], best_loss['train_loss'])
    # ax1.plot(best_loss['epoch'], best_loss['test_loss'])
    ax1.set_xlabel('epoch #')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(type_train, label=f"Training Accuracy")
    ax2.plot(type_test, label=f"Testing Accuracy")
    # ax2.plot(best_loss['epoch'], best_loss['train_acc'])
    # ax2.plot(best_loss['epoch'], best_loss['test_acc'])
    ax2.set_xlabel('epoch #')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
