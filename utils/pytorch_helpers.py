import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
import copy
import random
# from sklearn.metrics import silhouette_score, silhouette_samples
from utils.simple_io import *
from sklearn import preprocessing
from utils import silhouette
import scipy.io
from utils.plot_helpers import plot_saliencies, plot_embeddings


def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def save_params(filename, loss, params):
    model_file = f'{filename}_model_state.pt'
    torch.save(params, model_file)
    # open file for writing, "w" is writing
    w = csv.writer(open(f'{filename}_losses.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in loss.items():
        # write every key and value to file
        if torch.is_tensor(val):
            val = val.numpy()
        w.writerow([key, val])


def plot_model(best_loss, train_loss, valid_loss, train_acc, train_val, type):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    x = list(range(1, len(valid_loss) + 1))
    smoothing_level = 5.

    sm_train_loss = pd.DataFrame(train_loss).ewm(com=smoothing_level).mean()
    p = ax1.plot(x, train_loss, alpha=.2)
    ax1.plot(x, sm_train_loss.squeeze(), label="Training loss", alpha=.8, color=p[0].get_color())

    sm_valid_loss = pd.DataFrame(valid_loss).ewm(com=smoothing_level).mean()
    p = ax1.plot(x, valid_loss, alpha=.2)
    ax1.plot(x, sm_valid_loss.squeeze(), label="Validation loss", alpha=.8, color=p[0].get_color())

    ax1.set_xlabel('epoch #')
    ax1.set_ylabel('Loss')
    ax1.legend()

    train_acc = np.array(train_acc) * 100
    train_val = np.array(train_val) * 100

    sm_train_acc = pd.DataFrame(train_acc).ewm(com=smoothing_level).mean()
    p = ax2.plot(train_acc, alpha=.2)
    ax2.plot(sm_train_acc, label=f"Training {type}", alpha=.8, color=p[0].get_color())

    sm_valid_acc = pd.DataFrame(train_val).ewm(com=smoothing_level).mean()
    p = ax2.plot(train_val, alpha=0.2)
    ax2.plot(sm_valid_acc, label=f"Validation {type}", alpha=.8, color=p[0].get_color())

    ax2.set_xlabel('epoch #')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    fig.set_size_inches(12, 4.8)  # size in pixels
    plt.show()

    return fig


def plot_entropies(entropies, labels):
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.grid(True)

    for label in labels:
        plt.plot(entropies[label], label=label, alpha=.8, marker='o')

    ax.set_xlabel('epoch #', fontsize=25)
    ax.set_ylabel('Embedded Layer Entropy', fontsize=25)
    ax.legend(loc='upper right', ncol=2, fontsize=18)
    plt.show()


def plot_embed(trained_model, data, batch_size, device='cpu', save_folder='./figures/', show=True, save=False):
    """Convert data into tensors"""

    rnd_model = copy.deepcopy(trained_model)
    rnd_model.__init__()

    trained_model = trained_model.to(device)
    rnd_model = rnd_model.to(device)

    trained_model.eval()
    rnd_model.eval()

    # _, valid_data, _ = data

    batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size  # enforce even batch sizes
    half_batch = int(batch_size / 2)
    valid_batch_reminder = len(data) % half_batch
    n_valid_batches = int(len(data) / half_batch) if valid_batch_reminder == 0 else int(
        len(data) / half_batch) + 1

    hidden_size = 7
    n_grasps = 10

    valid_indices = list(range(len(data)))
    random.shuffle(valid_indices)

    true_labels = []
    raw_outputs_trained = []
    raw_outputs_rnd = []
    embeddings_trained = []
    embeddings_rnd = []

    rnd_acc = 0
    trained_acc = 0
    for i in range(n_valid_batches):

        # Take each testing batch and process
        batch_start = i * half_batch
        batch_end = i * half_batch + half_batch \
            if i * half_batch + half_batch < len(data) \
            else len(data)

        X, y, y_labels = data[valid_indices[batch_start:batch_end]]

        X = X.reshape(-1, 10, 19).to(device)
        y = y.to(device)
        y_labels = y_labels.squeeze()

        true_labels.extend(y_labels.squeeze().tolist())

        padded_ints = list(range(n_grasps))
        random.shuffle(padded_ints)

        # randomly switch in zero rows to vary the number of grasps being identified
        padded_start = 9  # np.random.randint(1, 11)
        X_pad = X[:, :padded_start + 1, :]

        # set hidden layer
        hidden_trained = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)
        hidden_rnd = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

        """ iterate through each grasp and run the model """
        output_trained = trained_model(X_pad[:, 0, :], hidden_trained)
        hidden_trained = output_trained

        output_rnd = rnd_model(X_pad[:, 0, :], hidden_rnd)
        hidden_rnd = output_rnd

        for j in range(1, padded_start + 1):
            output_trained = trained_model(X_pad[:, j, :], hidden_trained)
            hidden_trained = nn.functional.softmax(output_trained, dim=-1)

            output_rnd = rnd_model(X_pad[:, j, :], hidden_rnd)
            hidden_rnd = nn.functional.softmax(output_rnd, dim=-1)

            # calculate accuracy of classification
            _, preds_trained = output_trained.detach().max(dim=1)
            _, preds_rnd = output_rnd.detach().max(dim=1)


        trained_acc += torch.sum(preds_trained == y.flatten()).cpu().numpy() / len(preds_trained)
        rnd_acc += torch.sum(preds_rnd == y.flatten()).cpu().numpy() / len(preds_rnd)

        embedding_trained = trained_model.get_embed().cpu().numpy()
        embedding_rnd = rnd_model.get_embed().cpu().numpy()

        raw_outputs_trained.extend(output_trained.detach().squeeze().cpu().numpy())
        raw_outputs_rnd.extend(output_trained.detach().squeeze().cpu().numpy())

        embeddings_trained.extend(embedding_trained.squeeze())
        embeddings_rnd.extend(embedding_rnd.squeeze())

    rnd_acc /= n_valid_batches
    trained_acc /= n_valid_batches

    print(f"trained_acc: {trained_acc}, rnd_acc: {rnd_acc}")
    true_labels = np.array(true_labels)
    all_embeds_trained = np.stack(embeddings_trained)
    all_embeds_rnd = np.stack(embeddings_rnd)
    all_outputs_trained = np.stack(raw_outputs_trained)
    all_outputs_rnd = np.stack(raw_outputs_rnd)

    lbl_to_cls_dict = data.label_to_cls
    plot_embeddings(outputs_trained=all_outputs_trained,
                    outputs_rnd=all_outputs_rnd,
                    true_labels=true_labels,
                    all_embeds_trained=all_embeds_trained,
                    all_embeds_rnd=all_embeds_rnd,
                    lbl_to_cls_dict=lbl_to_cls_dict,
                    save_folder=save_folder,
                    show=show,
                    save=save)


def plot_embed_optimize(model_trained, model_state_trained, data, device='cpu', save_folder='./figures/', show=True, save=False):
    """Convert data into tensors"""

    cls_dict = data.label_to_cls

    model_rnd = copy.deepcopy(model_trained)
    model_rnd.__init__()
    model_state_rnd = model_rnd.state_dict()

    model_trained = model_trained.to(device)
    model_rnd = model_rnd.to(device)

    model_trained.eval()
    model_rnd.eval()

    search_space = 100
    objects = sorted(list(cls_dict.keys()))
    hidden_size = len(objects)

    input_codes = []
    target_classes = []
    for cls in cls_dict.keys():
        input_codes += [torch.rand(search_space, 10, 19, device=device)]
        target_classes += [cls_dict[cls]]*search_space

    input_codes_trained = torch.cat(input_codes, dim=0).to(device=device)
    input_codes_rnd = copy.deepcopy(input_codes_trained)
    input_codes_trained.requires_grad = True
    input_codes_rnd.requires_grad = True

    target_classes = torch.Tensor(target_classes).long().to(device)

    optim_rnd = torch.optim.SGD([input_codes_trained], lr=1e-5)
    optim_trained = torch.optim.SGD([input_codes_rnd], lr=1e-5)

    embeddings_trained = []
    embeddings_rnd = []
    raw_outputs_trained = []
    raw_outputs_rnd = []

    n_epochs = 10000

    for i in range(n_epochs):
        model_trained.load_state_dict(torch.load(model_state_trained))
        model_rnd.load_state_dict(model_state_rnd)

        # Take each testing batch and process
        clipped_code_trained = torch.nn.Sigmoid()(input_codes_trained)
        clipped_code_rnd = torch.nn.Sigmoid()(input_codes_rnd)

        # randomly switch in zero rows to vary the number of grasps being identified
        padded_start = 1  # np.random.randint(1, 11)

        # set hidden layer
        hidden_trained = torch.full((search_space*hidden_size, hidden_size), 1 / hidden_size).to(device)
        hidden_rnd = torch.full((search_space*hidden_size, hidden_size), 1 / hidden_size).to(device)

        """ iterate through each grasp and run the model """
        output_trained = model_trained(clipped_code_trained[:, 0, :], hidden_trained)
        hidden_trained = output_trained

        output_rnd = model_rnd(clipped_code_rnd[:, 0, :], hidden_rnd)
        hidden_rnd = output_rnd

        for j in range(1, padded_start + 1):
            output_trained = model_trained(clipped_code_trained[:, j, :], hidden_trained)
            hidden_trained = nn.functional.softmax(output_trained, dim=-1)

            output_rnd = model_rnd(clipped_code_rnd[:, j, :], hidden_rnd)
            hidden_rnd = nn.functional.softmax(output_rnd, dim=-1)

        loss_rnd = nn.CrossEntropyLoss()(output_rnd, target_classes).to(device)
        loss_trained = nn.CrossEntropyLoss()(output_trained, target_classes)

        loss_rnd.backward()
        optim_rnd.step()

        loss_trained.backward()
        optim_trained.step()

        cls_max_trained = output_trained.detach().max(dim=0).values
        cls_max_rnd = output_rnd.detach().max(dim=0).values

        if np.all(cls_max_trained.cpu().numpy() > 50.) or i >= n_epochs-1:
            embedding_trained = model_trained.get_embed().cpu().numpy()
            embedding_rnd = model_rnd.get_embed().cpu().numpy()

            embeddings_trained.extend(embedding_trained.squeeze())
            embeddings_rnd.extend(embedding_rnd.squeeze())
            raw_outputs_trained = output_trained.detach().squeeze().cpu().numpy()
            raw_outputs_rnd = output_rnd.detach().squeeze().cpu().numpy()
            break

        print(f"epoch: {i} --- trained: {cls_max_trained},    rnd: {cls_max_rnd}")
        print()

    embeddings_rnd = np.stack(embeddings_rnd)
    embeddings_trained = np.stack(embeddings_trained)
    plot_embeddings(outputs_trained=raw_outputs_trained,
                    outputs_rnd=raw_outputs_rnd,
                    true_labels=np.array(data.get_labels(target_classes)),
                    all_embeds_trained=embeddings_trained,
                    all_embeds_rnd=embeddings_rnd,
                    lbl_to_cls_dict=cls_dict,
                    save_folder=save_folder,
                    show=show,
                    save=save)


def plot_embed_optimize_direct(model_trained, model_state_trained, data, device='cpu', save_folder='./figures/', show=True, save=True):
    """Convert data into tensors"""

    cls_dict = data.label_to_cls

    model_rnd = copy.deepcopy(model_trained)
    model_rnd.__init__()
    model_state_rnd = model_rnd.state_dict()

    model_trained = model_trained.to(device)
    model_rnd = model_rnd.to(device)

    model_trained.eval()
    model_rnd.eval()

    search_space = 100
    objects = sorted(list(cls_dict.keys()))
    hidden_size = len(objects)

    input_codes = []
    target_classes = []
    for label in objects:
        input_codes += [torch.rand(search_space, 10, 8, 8, device=device)]
        target_classes += [cls_dict[label]]*search_space

    input_codes_trained = torch.cat(input_codes, dim=0).to(device=device)
    input_codes_rnd = copy.deepcopy(input_codes_trained)
    input_codes_trained.requires_grad = True
    input_codes_rnd.requires_grad = True

    target_classes = torch.Tensor(target_classes).long().to(device)
    target_labels = np.array(data.get_labels(target_classes))

    optim_rnd = torch.optim.SGD([input_codes_trained], lr=1e-5)
    optim_trained = torch.optim.SGD([input_codes_rnd], lr=1e-5)

    embeddings_trained = []
    embeddings_rnd = []
    raw_outputs_trained = []
    raw_outputs_rnd = []

    n_epochs = 10000

    for i in range(n_epochs):
        model_trained.load_state_dict(torch.load(model_state_trained))
        model_rnd.load_state_dict(model_state_rnd)

        # Take each testing batch and process

        # randomly switch in zero rows to vary the number of grasps being identified
        padded_start = 9  # np.random.randint(1, 11)

        # set hidden layer
        hidden_trained = torch.full((search_space*hidden_size, hidden_size), 1 / hidden_size).to(device)
        hidden_rnd = torch.full((search_space*hidden_size, hidden_size), 1 / hidden_size).to(device)

        """ iterate through each grasp and run the model """
        output_trained = model_trained(input_codes_trained[:, 0, ...], hidden_trained)
        hidden_trained = nn.functional.softmax(output_trained, dim=-1)

        output_rnd = model_rnd(input_codes_rnd[:, 0, ...], hidden_rnd)
        hidden_rnd = nn.functional.softmax(output_rnd, dim=-1)

        for j in range(1, padded_start + 1):
            output_trained = model_trained(input_codes_trained[:, j, ...], hidden_trained)
            hidden_trained = nn.functional.softmax(output_trained, dim=-1)

            output_rnd = model_rnd(input_codes_rnd[:, j, ...], hidden_rnd)
            hidden_rnd = nn.functional.softmax(output_rnd, dim=-1)

        loss_trained = nn.CrossEntropyLoss()(output_trained, target_classes).to(device)
        loss_rnd = nn.CrossEntropyLoss()(output_rnd, target_classes).to(device)

        loss_trained.backward()
        optim_trained.step()

        loss_rnd.backward()
        optim_rnd.step()

        cls_max_trained = []
        cls_max_rnd = []
        for label in objects:
            indices = target_labels == label
            cls = cls_dict[label]

            cls_max_trained.append(output_trained[indices].max(0).values[cls].cpu().detach().item())
            cls_max_rnd.append(output_rnd[indices].max(0).values[cls].cpu().detach().item())

        if np.all(np.array(cls_max_trained) > 100.) or i >= n_epochs-1:
            embedding_trained = model_trained.get_embed().cpu().numpy()
            embedding_rnd = model_rnd.get_embed().cpu().numpy()

            embeddings_trained.extend(embedding_trained.squeeze())
            embeddings_rnd.extend(embedding_rnd.squeeze())
            raw_outputs_trained = output_trained.detach().squeeze().cpu().numpy()
            raw_outputs_rnd = output_rnd.detach().squeeze().cpu().numpy()
            break

        # print(f"epoch: {i} --- "
        #       f"trained: {''.join([f'{x:.2f} ' for x in cls_max_trained])},    "
        #       f"rnd: {''.join([f'{x:.2f} ' for x in cls_max_rnd])}")
        print(f"epoch: {i} --- "
              f"trained: {hidden_trained.max(0).values},    "
              f"rnd: {hidden_rnd.max(0).values}")
        print()

    embeddings_rnd = np.stack(embeddings_rnd)
    embeddings_trained = np.stack(embeddings_trained)
    plot_embeddings(outputs_trained=raw_outputs_trained,
                    outputs_rnd=raw_outputs_rnd,
                    true_labels=target_labels,
                    all_embeds_trained=embeddings_trained,
                    all_embeds_rnd=embeddings_rnd,
                    lbl_to_cls_dict=cls_dict,
                    save_folder=save_folder,
                    show=show,
                    save=save)


def attention_analysis(trained_model, data, batch_size, device='cpu', save_folder='./figures/', show=True, save=False):
    """Convert data into tensors"""

    trained_model = trained_model.to(device)
    trained_model.eval()

    # _, valid_data, _ = data

    batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size  # enforce even batch sizes
    half_batch = int(batch_size / 2)
    valid_batch_reminder = len(data) % half_batch
    n_valid_batches = int(len(data) / half_batch) if valid_batch_reminder == 0 else int(
        len(data) / half_batch) + 1

    hidden_size = 7
    n_grasps = 10

    valid_indices = list(range(len(data)))
    random.shuffle(valid_indices)

    true_labels = []
    raw_outputs_trained = []
    embeddings_trained = []

    for param in trained_model.parameters():
        param.requires_grad = False

    input_attentions = {obj: [] for obj in data.labels}
    recurrent_attentions = {obj: [] for obj in data.labels}
    for i in range(n_valid_batches):

        # Take each testing batch and process
        batch_start = i * half_batch
        batch_end = i * half_batch + half_batch \
            if i * half_batch + half_batch < len(data) \
            else len(data)

        X, y, y_labels = data[valid_indices[batch_start:batch_end]]

        X = X.reshape(-1, 10, 19).to(device)
        y_labels = y_labels.squeeze()

        true_labels.extend(y_labels.squeeze().tolist())

        # randomly switch in zero rows to vary the number of grasps being identified
        padded_start = 9  # use 10 grasps
        X_pad = X[:, :padded_start + 1, :]
        hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

        X_pad.requires_grad = True
        hidden.requires_grad = True

        """ iterate through each grasp and run the model """
        for j in range(0, padded_start + 1):
            output = trained_model(X_pad[:, j, :], hidden)

            score, _ = torch.max(output, 1)
            score.backward()

            slc_inputs = torch.abs(X_pad.grad)
            slc_hidden = torch.abs(hidden.grad)

            slc = (slc - slc.min()) / (slc.max() - slc.min())

            hidden = nn.functional.softmax(output, dim=-1)
            # calculate accuracy of classification
            _, preds = output.detach().max(dim=1)

        embedding_trained = trained_model.get_embed().cpu().numpy()

        raw_outputs_trained.extend(output.detach().squeeze().cpu().numpy())


        embeddings_trained.extend(embedding_trained.squeeze())


    true_labels = np.array(true_labels)
    all_embeds_trained = np.stack(embeddings_trained)
    all_outputs_trained = np.stack(raw_outputs_trained)

    lbl_to_cls_dict = data.label_to_cls

