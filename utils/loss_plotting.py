import pickle
import csv
import matplotlib.pyplot as plt
import os
from os.path import exists
import numpy as np


def plot_silhouette(file_path, model, n_grasps):
    model_name = model.__class__.__name__
    loss_dict = dict()
    if exists(file_path):
        with open(file_path, mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if len(row)>0:
                    if row[0]:
                        data_str_long = row[1][1:-1]
                        data_str = data_str_long.split(', ')
                        if "tensor" in data_str_long:
                            data_str = data_str[0::2]
                            data_float = [-float(x[7:14]) for x in data_str]
                        else:
                            data_float = [float(x) for x in data_str]
                        # print(row)
                        loss_dict[row[0]] = data_float

        # 'training': train_loss_out, 'testing': test_loss_out, 'training_silhouette': train_sil_out
        train_loss = loss_dict['training']
        test_loss = loss_dict['testing']
        silhouette_score = loss_dict['training_silhouette']
        max_silhouette = max(silhouette_score)
        max_sil_idx = np.argmax(silhouette_score)

        fig, [ax1, ax2] = plt.subplots(1, 2)
        ax1.plot(train_loss, label='Training losses')
        ax1.plot(test_loss, label='Testing losses')
        ax1.legend()
        ax2.plot(silhouette_score, label='Silhouette Score')
        ax2.scatter(max_sil_idx, max_silhouette)
        ax2.legend()
        fig.suptitle(f'{model_name} {n_grasps} grasps')
        ax1.set_title('Losses')
        ax2.set_title(f'Silhouette score. max: {max_silhouette}')
        # plt.show()
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
