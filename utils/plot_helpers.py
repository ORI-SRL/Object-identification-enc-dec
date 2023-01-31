# import pickle
import csv
import matplotlib.pyplot as plt
# import os
from os.path import exists
import numpy as np
from utils.simple_io import *
from matplotlib.gridspec import GridSpec
import torch


def plot_silhouette(file_path, model, n_grasps):
    # model_name = model.__class__.__name__
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


def plot_saliencies(frame_sal, hidden_sal, frame_std, hidden_std, classes):
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    f_size = 14

    # normalise the salience
    # frame_normed = frame_sal/torch.max(frame_sal)
    # hidden_normed = hidden_sal/torch.max(hidden_sal)
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    fig, axs = plt.subplot_mosaic([['1', '2a', '3'],
                                   ['1', '2b', '3']])
    axs['1'].set_title('Salience vs grasps', fontsize=f_size + 4)
    axs['1'].set_ylabel('Salience', fontsize=f_size)
    axs['1'].set_xlabel('Number of grasps', fontsize=f_size)
    # ax2.set_title('Scaled Salience vs grasps', fontsize=f_size + 4)
    axs['2a'].set_xlabel('Grasps', fontsize=f_size)
    axs['2a'].set_ylabel('Frame Salience', fontsize=f_size)
    axs['2b'].set_xlabel('Grasps', fontsize=f_size)
    axs['2b'].set_ylabel('Hidden Salience', fontsize=f_size)
    axs['3'].set_title('Salience trends', fontsize=f_size + 4)
    axs['3'].set_ylabel('Salience', fontsize=f_size)
    axs['3'].set_xlabel('Number of grasps', fontsize=f_size)
    for row in range(len(frame_sal)):
        axs['1'].plot(frame_sal[row], '-', label=f'{classes[row]} Frame Salience', color=colours[row])
        axs['1'].plot(hidden_sal[row], '--', label=f'{classes[row]} Hidden Salience', color=colours[row])
        axs['2a'].plot(frame_sal[row], '-', label=f'{classes[row]}', color=colours[row])
        axs['2b'].plot(hidden_sal[row], '--', label=f'{classes[row]}', color=colours[row])
    frm_arr = np.zeros((len(frame_sal), len(frame_sal[0])))
    hid_arr = np.zeros((len(hidden_sal), len(hidden_sal[0])))
    for labels, vals in frame_sal.items():
        frm_arr[labels, :] = vals
    for labels, vals in hidden_sal.items():
        hid_arr[labels, :] = vals
    hid_arr_norm = (hid_arr - np.mean(hid_arr)) / np.std(hid_arr)
    frm_arr_norm = (frm_arr - np.mean(frm_arr)) / np.std(frm_arr)
    hid_means = np.mean(hid_arr_norm, axis=0)
    frm_means = np.mean(frm_arr_norm, axis=0)
    hid_std = np.std(hid_arr_norm, axis=0)
    frm_std = np.std(frm_arr_norm, axis=0)
    axs['3'].plot(frm_means, 'tab:orange', linewidth=3, label='Hidden Trend')
    axs['3'].plot(hid_means, 'b--', linewidth=3, label='Frame Trend')
    # axs['3'].errorbar(range(10), frm_means, yerr=frm_std/2, fmt='m', linewidth=3, label='Hidden Trend')
    # axs['3'].errorbar(range(10), hid_means, yerr=hid_std/2, fmt='b--', linewidth=3, label='Frame Trend')

    fig2, ax2 = plt.subplots()
    ax2.plot(frm_means, 'tab:orange', linewidth=3, label='Hidden Trend')
    ax2.plot(hid_means, 'b--', linewidth=3, label='Frame Trend')
    # normalise each row rather than whole matrix
    '''frame_obj_norm = torch.empty((7, 10))
    hidden_obj_norm = torch.empty((7, 10))
    frame_std_norm = {}
    hid_std_norm = {}
    for row in range(frame_sal.size(0)):
        frame_obj_norm[row, :] = frame_sal[row, :] / torch.max(frame_sal, dim=1).values[row]
        frame_std_norm[row] = [x / torch.max(frame_sal, dim=1).values[row] for x in frame_std[row]]
        hidden_obj_norm[row, :] = hidden_sal[row, :] / torch.max(hidden_sal, dim=1).values[row]
        hid_std_norm[row] = [x / torch.max(hidden_sal, dim=1).values[row] for x in hidden_std[row]]'''

    x = range(len(frame_sal[0]))
    fig_objs, axs_objs = plt.subplot_mosaic([[1, 2, 3],
                                             [4, 5, 6],
                                             ['.', 7, '.']])
    for labels, ax in axs_objs.items():
        frame_sal[labels - 1] = (frame_sal[labels - 1] - np.mean(frm_arr)) / np.std(frm_arr)
        hidden_sal[labels - 1] = (hidden_sal[labels - 1] - np.mean(hid_arr)) / np.std(hid_arr)
        frame_std[labels - 1] = np.std(frame_sal[labels - 1])
        hidden_std[labels - 1] = np.std(hidden_sal[labels - 1])
        ax.errorbar(x, frame_sal[labels - 1], yerr=np.array(frame_std[labels - 1]) / 2, fmt='tab:orange', label='Frame')
        ax.errorbar(x, hidden_sal[labels - 1], yerr=np.array(hidden_std[labels - 1]) / 2, fmt='b--', label='Hidden')
        ax.set_title(classes[labels - 1])
    axs_objs[1].legend()
    axs['1'].legend(labels=['Frame Salience', 'Hidden Salience'])
    axs['2a'].legend()
    axs['3'].legend()


def plot_entropies(entropies, labels):
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.grid(True)

    for label in labels:
        plt.plot(entropies[label], label=label, alpha=.8, marker='o')

    ax.set_xlabel('epoch #', fontsize=25)
    ax.set_ylabel('Embedded Layer Entropy', fontsize=25)
    ax.legend(loc='upper right', ncol=2, fontsize=18)
    plt.show()


def plot_embeddings(outputs_trained, outputs_rnd, true_labels, all_embeds_trained, all_embeds_rnd, lbl_to_cls_dict, save_folder, show, save):

    objects = sorted(list(lbl_to_cls_dict.keys()))
    hidden_size = len(objects)
    fig = plt.figure(figsize=(hidden_size * 5, 10))
    gs = GridSpec(2, hidden_size * 5, figure=fig)
    gs.tight_layout(fig, pad=.4, w_pad=0.5, h_pad=1.0)

    # ----------------------------------------------------------------------------------#

    for i, cls in enumerate(objects):
        indices = true_labels == cls

        idx_trained = outputs_trained[indices][:, lbl_to_cls_dict[cls]].argmax()
        img_trained = all_embeds_trained[indices][idx_trained]
        ax1 = fig.add_subplot(gs[0, i * 5:(i + 1) * 5])
        ax1.imshow(img_trained, interpolation='bilinear', cmap='Blues')
        ax1.set_xlabel(cls, fontsize=40, weight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])

        idx_rnd = outputs_rnd[indices][:, lbl_to_cls_dict[cls]].argmax()
        img_rnd = all_embeds_rnd[indices][idx_rnd]
        ax2 = fig.add_subplot(gs[1, i * 5:(i + 1) * 5])
        ax2.imshow(img_rnd, interpolation='bilinear', cmap='Blues')
        ax2.set_xlabel(cls, fontsize=40, weight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])

    # fig.suptitle("Embedding Activation", fontsize=46)
    if save:
        if not folder_exists(save_folder):
            folder_create(save_folder)
        fig_name = f"{save_folder}embedding_trained.png"
        fig.savefig(fig_name, dpi=300)
        print(f"saved figure: '{fig_name}'")
    if show:
        plt.show()

    plt.close('all')


def plot_attention_loadings(input_attentions, rec_attentions, save_folder='./figures/', show=True, save=False):

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 5))

    input_min = []
    input_max = []
    rec_min = []
    rec_max = []
    for obj in input_attentions.keys():
        data = np.array([input_attentions[obj][key] for key in input_attentions[obj].keys()])
        input_min.append(data.min())
        input_max.append(data.max())

        data = np.array([rec_attentions[obj][key] for key in rec_attentions[obj].keys()])
        rec_min.append(np.array([lst for lst in rec_attentions[obj]]).min())
        rec_max.append(np.array([lst for lst in rec_attentions[obj]]).max())

    input_min = min(input_min)
    input_max = max(input_max)
    rec_min = min(rec_min)
    rec_max = max(rec_max)


    for obj in input_attentions.keys():
        obj_input_means = []
        obj_input_stds = []
        obj_rec_means = []
        obj_rec_stds = []
        for i in range(len(list(input_attentions[obj].keys()))):
            norm_data = (input_attentions[obj][i] - input_min) / (input_max - input_min)
            obj_input_means.append(np.mean(norm_data))
            obj_input_stds.append(np.std(norm_data)/np.sqrt(len(norm_data)))

            norm_data = (rec_attentions[obj][i] - rec_min) / (rec_max - rec_min)
            obj_rec_means.append(np.mean(norm_data))
            obj_rec_stds.append(np.std(norm_data)/np.sqrt(len(norm_data)))

        x = list(range(1, len(obj_input_means)+1))

        p = ax1.errorbar(x, obj_input_means, yerr=obj_input_stds, label=f"obj", marker='o', ls='-')
        ax2.errorbar(x, obj_rec_means, yerr=obj_rec_stds, label=f"obj", marker='d', ls='--', color=p[0].get_color())

    ax1.set_ylabel("Scaled grasp input attention", fontsize=20)
    ax1.set_xlabel("Grasp #", fontsize=24)
    ax2.set_ylabel("Scaled recurrent input attention", fontsize=20)
    ax2.set_xlabel("Grasp #", fontsize=24)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)

    # fig.suptitle("Embedding Activation", fontsize=46)
    if save:
        if not folder_exists(save_folder):
            folder_create(save_folder)
        fig_name = f"{save_folder}attentions.png"
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"saved figure: '{fig_name}'")
    if show:
        plt.show()

    plt.close('all')
