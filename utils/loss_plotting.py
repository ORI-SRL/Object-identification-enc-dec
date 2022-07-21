import pickle
import csv
import matplotlib.pyplot as plt
import os
from os.path import exists
import numpy as np


def plot_silhouette(file_path, model, n_grasps):
    model_name = model.__class__.__name__
    loss_dict = dict()
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.set_title('Losses')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2.set_title('Silhouette score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Silhouette score')

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
            silhouette_score = loss_dict['training_silhouette']
            max_silhouette = max(silhouette_score)
            max_sil_idx = np.argmax(silhouette_score)

            ax1.plot(train_loss, label=f'{num_grasps} Grasps Training Losses')
            ax1.plot(test_loss, label=f'{num_grasps} Grasps Validation Losses')
            ax1.legend()
            ax2.plot(silhouette_score, label=f'{num_grasps} Grasps Silhouette Score')
            ax2.scatter(max_sil_idx, max_silhouette)
            ax2.legend()
            # fig.suptitle('Three Layer Convolution with Batch Norm, Losses and Silhouette Score')
            # plt.show()
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
