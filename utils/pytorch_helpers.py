import numpy as np
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
import copy
import random
# from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import preprocessing
from utils import silhouette
import scipy.io
from utils.loss_plotting import plot_saliencies


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


def encode_labels(labels, classes):
    # encoded_label_frame = torch.zeros((len(labels), 1, 7), dtype=torch.float)  #
    encoded_label_frame = torch.zeros((len(labels)), dtype=torch.long)  #
    i = 0
    for label in labels:
        # encoded_label_frame[i, 0, classes.index(label)] = 1  # = 1
        encoded_label_frame[i] = classes.index(label)
        i += 1
    return encoded_label_frame


def decode_labels(preds, classes):
    # encoded_label_frame = torch.zeros((len(labels), 1, 7), dtype=torch.float)  #
    decoded_label_frame = []  #
    for i, _ in enumerate(preds):
        decoded_label_frame.append(classes[preds[i]])
    return decoded_label_frame


def gamma_loss(x_entropy, frame_out, labels):
    gamma = 0.9
    rows = frame_out.size(1)
    l_n = torch.zeros(rows, 1)
    for row in range(rows):
        l_n[row] = x_entropy(frame_out[:, row, :], labels) * pow(gamma, rows - row)

    loss_out = torch.sum(l_n) / rows
    return loss_out


def save_params(filename, loss, params):
    model_file = f'{filename}_model_state.pt'
    torch.save(params, model_file)
    # open file for writing, "w" is writing
    w = csv.writer(open(f'{filename}_losses.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in loss.items():
        # write every key and value to file
        if torch.is_tensor(val):
            val = [x.numpy() for x in val]
        w.writerow([key, val])


def plot_model(best_loss, train_loss, test_loss, type_train, type_test, type):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    x = list(range(1, len(test_loss) + 1))
    ax1.plot(x, train_loss, label="Training loss")
    ax1.plot(x, test_loss, label="Testing loss")
    ax1.plot(best_loss['epoch'], best_loss['train_loss'])
    ax1.plot(best_loss['epoch'], best_loss['test_loss'])
    ax1.set_xlabel('epoch #')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(type_train, label=f"Training {type}")
    ax2.plot(type_test, label=f"Testing {type}")
    ax2.plot(best_loss['epoch'], best_loss['train_acc'])
    ax2.plot(best_loss['epoch'], best_loss['test_acc'])
    ax2.set_xlabel('epoch #')
    ax2.set_ylabel('Accuracy')
    ax2.legend()


def early_stopping(loss_dict, patience, max_patience, test_loss, train_loss, train_acc, test_acc, epoch, model,
                   best_params):
    early_stop = False
    if loss_dict['test_acc'] is None or test_acc > loss_dict['test_acc']:

        loss_dict = {'train_loss': train_loss, 'test_loss': test_loss, 'train_acc': train_acc,
                     'test_acc': test_acc, 'epoch': epoch}
        patience = 0
        best_params = copy.copy(model.state_dict())
    else:
        patience += 1
    if patience >= max_patience:
        print(f'Early stopping: training terminated at epoch {epoch} due to es, '
              f'patience exceeded at {max_patience}')
        print(f'Best accuracies: Training: {loss_dict["train_acc"]} \t Testing: {loss_dict["test_acc"]}')
        early_stop = True
    else:
        early_stop = False
    return early_stop, loss_dict, patience, best_params


def model_init(model, n_grasps=None):
    model_name = model.__class__.__name__
    if n_grasps is not None:
        print(f'{model_name}_{n_grasps}')
    else:
        print(model_name)

    device = get_device()
    print(device)
    model.to(device)

    # Epochs
    train_loss_out = []
    test_loss_out = []
    train_acc_out = []
    test_acc_out = []

    patience = 0
    best_loss_dict = {'train_loss': None, 'test_loss': None, 'train_acc': None,
                      'test_acc': None, 'epoch': None}
    best_params = None
    return model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
           best_params


def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")


def salience_std(sal):
    grasp_std = {}
    grasp_mean = {}
    norm_data = {}
    data_max = {}
    for outers in sal:
        for inners in sal[outers]:
            inner_max = max(sal[outers][inners][0:-1])
            if outers not in data_max:
                data_max[outers] = 1 # 0.0
            if inner_max > data_max[outers]:
                data_max[outers] = 1 # inner_max
    for val in sal:
        for val_inner in sal[val]:
            data = np.array(sal[val][val_inner])
            data = data / data_max[val]
            data_std = np.var(data[0:-1])
            data_mean = np.mean(data[0:-1])
            if val in grasp_std:
                grasp_std[val].append(data_std)
                grasp_mean[val].append(data_mean)
            else:
                grasp_std[val] = [data_std]  # [grasp_var ** 0.5]
                grasp_mean[val] = [data_mean]
                # norm_data[val] = {}
            # norm_data[val][val_inner] = torch.mean(torch.from_numpy((data - data_mean) / data_std))

    return grasp_std, grasp_mean


def populate_sal_dicts(sal_dict, sal_vals, obj_labels, idx):
    if obj_labels[idx].item() in sal_dict:
        sal_dict[obj_labels[idx].item()].append(sal_vals[idx])
    else:
        sal_dict[obj_labels[idx].item()] = [sal_vals[idx]]
    return sal_dict


def train_RNN(model, train_loader, test_loader, optimizer, criterion, classes, batch_size, n_epochs=50,
              max_patience=25, save_folder='./', save=True, show=True):
    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
    best_params = model_init(model)
    hidden_size = 7

    for epoch in range(1, n_epochs + 1):
        train_loss, test_loss, train_accuracy, test_accuracy = 0.0, 0.0, 0.0, 0.0
        cycle = 0
        confusion_ints = torch.zeros((7, 7)).to(device)
        grasp_accuracy = torch.zeros((10, 2)).to(device)
        model.train()
        for data in train_loader:
            # Take each training batch and process
            frame = data["data"].to(device)  # .reshape(32, 10, 19)
            frame_labels = data["labels"]
            frame_loss = 0

            # randomly switch in zero rows to vary the number of grasps being identified
            padded_start = np.random.randint(1, 11)
            frame[:, padded_start:, :] = 0
            nul_rows = frame.sum(dim=2) != 0
            frame = frame[:, :padded_start, :]
            enc_lab = encode_labels(frame_labels, classes).to(device)
            # convert frame_labels to numeric and allocate to tensor to silhouette score
            le = preprocessing.LabelEncoder()
            frame_labels_num = le.fit_transform(frame_labels)

            # set the first hidden layer as a vanilla prediction or zeros
            if model_name == 'IterativeRNN2':
                hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)
            elif model_name == 'LSTM':  # input a vanilla starting layer output
                hidden = (torch.zeros(1, 64).to(device), torch.zeros(1, 64).to(device))
            else:
                hidden = torch.zeros(frame.size(0), hidden_size).to(device)
            # optimizer.zero_grad()

            if model_name == 'IterativeRCNN':
                frame = frame.reshape(frame.size(0), -1, 1, 19)
                hidden = hidden.reshape(frame.size(0), 1, -1)
            optimizer.zero_grad()

            # iterate through each grasp and run the model
            for i in range(padded_start):

                if model_name == 'SilhouetteRNN':
                    output, hidden, embeddings = model(frame[:, i, :], hidden)
                    loss1 = criterion(output, enc_lab)
                    # calculate the silhouette score at the bottleneck and add it to the loss value
                    loss2 = silhouette.silhouette.score(embeddings, enc_lab, loss=True)
                    loss = loss1 + 2 * loss2
                else:
                    output, hidden = model(frame[:, i, :], hidden)
                    loss = criterion(output, enc_lab)

                # if model_name == 'IterativeRCNN':
                #     hidden = hidden.reshape(frame.size(0), 1, hidden.size(-1))

                frame_loss += loss  # * np.exp(- i/11)  #loss_weights[i]  #
            frame_loss.backward()
            optimizer.step()
            output = nn.functional.softmax(output, dim=-1)

            _, inds = output.max(dim=1)
            frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
            train_accuracy += frame_accuracy
            grasp_accuracy[padded_start - 1, 1] += 1
            grasp_accuracy[padded_start - 1, 0] += frame_accuracy
            train_loss += frame_loss / padded_start
            cycle += 1

        train_loss = train_loss.detach().cpu() / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)
        train_loss_out.append(train_loss)
        train_acc_out.append(train_accuracy)
        grasp_accuracy[:, 1] = grasp_accuracy[:, 0] / grasp_accuracy[:, 1]
        grasp_accuracy[:, 0] = torch.linspace(1, 10, 10, dtype=int)

        grasp_accuracy = torch.zeros((10, 2)).to(device)
        model.eval()
        for data in test_loader:
            # Take each test batch and run the model
            frame = data["data"].to(device)  # .reshape(32, 10, 19)
            frame_labels = data["labels"]
            frame_loss = 0
            # randomly switch in zero rows to vary the number of grasps being identified
            padded_start = np.random.randint(1, 11)
            frame[:, padded_start:, :] = 0
            nul_rows = frame.sum(dim=2) != 0
            frame = frame[:, :padded_start, :]
            enc_lab = encode_labels(frame_labels, classes).to(device)

            # set the first hidden layer as a vanilla prediction or zeros
            if model_name == 'IterativeRNN2':
                hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)
            elif model_name == 'LSTM':  # input a vanilla starting layer output
                hidden = (torch.zeros(1, 64).to(device), torch.zeros(1, 64).to(device))
            else:
                hidden = torch.zeros(frame.size(0), hidden_size).to(device)

            if model_name == 'IterativeRCNN':
                frame = frame.reshape(frame.size(0), -1, 1, 19)
                hidden = hidden.reshape(frame.size(0), 1, -1)

            # Run the model through each grasp
            for i in range(padded_start):
                if model_name == 'SilhouetteRNN':
                    output, hidden, embeddings = model(frame[:, i, :], hidden)
                    loss1 = criterion(output, enc_lab)
                    # calculate the silhouette score at the bottleneck and add it to the loss value
                    loss2 = silhouette.silhouette.score(embeddings, enc_lab,
                                                        loss=True)  # torch.as_tensor(frame_labels_num)
                    loss3 = loss1 + 2 * loss2
                else:
                    output, hidden = model(frame[:, i, :], hidden)
                    loss3 = criterion(output, enc_lab)

                if model_name == 'IterativeRCNN':
                    hidden = hidden.reshape(frame.size(0), 1, hidden.size(-1))

                frame_loss += loss3
            test_loss += frame_loss / padded_start
            _, inds = output.max(dim=1)
            frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
            test_accuracy += frame_accuracy
            grasp_accuracy[padded_start - 1, 1] += 1
            grasp_accuracy[padded_start - 1, 0] += frame_accuracy
            for n, _ in enumerate(enc_lab):
                row = enc_lab[n]
                col = inds[n]
                confusion_ints[row, col] += 1

        test_accuracy = test_accuracy / len(test_loader)
        test_loss = test_loss.detach().cpu() / len(test_loader)
        test_loss_out.append(test_loss)
        test_acc_out.append(test_accuracy)
        confusion_perc = confusion_ints / torch.sum(confusion_ints, dim=1)
        grasp_accuracy[:, 1] = grasp_accuracy[:, 0] / grasp_accuracy[:, 1]
        grasp_accuracy[:, 0] = torch.linspace(1, 10, 10, dtype=int)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTesting loss: {:.4f} \t Training accuracy {:.2f} '
              '\t Testing accuracy {:.2f}'
              .format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))
        # empty cache to prevent overusing the memory
        torch.cuda.empty_cache()
        # Early Stopping Logic
        early_stop, best_loss_dict, patience, best_params = early_stopping(best_loss_dict, patience, max_patience,
                                                                           test_loss, train_loss, train_accuracy,
                                                                           test_accuracy, epoch, model, best_params)
        if early_stop:
            break
        loss_dict = {'training': train_loss_out, 'testing': test_loss_out, 'training_accuracy': train_acc_out,
                     'testing_accuracy': test_acc_out}
    if save and best_params is not None:
        model_file = f'{save_folder}{model_name}_weighted_loss'
        save_params(model_file, loss_dict, best_params)

    if show:
        # plot model losses
        plot_model(best_loss_dict, train_loss_out, test_loss_out, train_acc_out, test_acc_out, type="accuracy")

    return best_params, best_loss_dict


def learn_iter_model(model, train_loader, test_loader, optimizer, criterion, classes, n_epochs=50,
                     max_patience=10, save_folder='./', save=True, show=True):
    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
    best_params = model_init(model)

    try:
        for epoch in range(1, n_epochs + 1):
            # monitor training loss
            train_loss, test_loss, train_accuracy, test_accuracy = 0.0, 0.0, 0.0, 0.0
            cycle = 0
            confusion_ints = torch.zeros((7, 7)).to(device)
            # Training
            model.train()
            for data in train_loader:
                frame = data["data"].to(device)  # .reshape(32, 10, 19)
                frame_labels = data["labels"]

                # randomly switch in zero rows to vary the number of grasps being identified
                padded_start = np.random.randint(1, 11)
                frame[:, padded_start:, :] = 0
                enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
                pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

                for r in range(padded_start):
                    optimizer.zero_grad()
                    final_row, output = model(frame[:, r, :], pred_in)
                    pred_in = final_row.detach()
                    grasp_loss = gamma_loss(criterion, output, enc_lab)
                    if r == 0:
                        loss = grasp_loss
                    else:
                        loss += grasp_loss
                    train_loss += grasp_loss.item()

                loss.backward()  # loss +
                optimizer.step()
                _, inds = final_row.max(dim=1)
                frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
                train_accuracy += frame_accuracy
                cycle += 1
            train_loss = train_loss / len(train_loader)
            train_accuracy = train_accuracy / len(train_loader)
            train_loss_out.append(train_loss)
            train_acc_out.append(train_accuracy)

            model.eval()
            for data in test_loader:
                # take the next frame from the data_loader and process it through the model
                frame = data["data"].to(device)
                frame_labels = data["labels"]
                # randomly switch in zero rows to vary the number of grasps being identified
                padded_start = np.random.randint(1, 11)
                frame[:, padded_start:, :] = 0
                enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
                # set the initial guess as a flat probability for each object
                pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

                for r in range(padded_start):
                    final_row, output = model(frame[:, r, :], pred_in)
                    pred_in = final_row.detach()
                    loss2 = gamma_loss(criterion, output, enc_lab)
                    test_loss += loss2.item()

                _, inds = final_row.max(dim=1)
                frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
                test_accuracy += frame_accuracy

                for n, _ in enumerate(enc_lab):
                    row = enc_lab[n]
                    col = inds[n]
                    confusion_ints[row, col] += 1

            test_accuracy = test_accuracy / len(test_loader)
            test_loss = test_loss / len(test_loader)
            test_loss_out.append(test_loss)
            test_acc_out.append(test_accuracy)
            confusion_perc = confusion_ints / torch.sum(confusion_ints, dim=1)

            print('Epoch: {} \tTraining Loss: {:.4f} \tTesting loss: {:.4f} \t Training accuracy {:.2f} '
                  '\t Testing accuracy {:.2f}'
                  .format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))
            # empty cache to prevent overusing the memory
            torch.cuda.empty_cache()

            # Early Stopping Logic
            early_stop, best_loss_dict, patience, best_params = early_stopping(best_loss_dict, patience, max_patience,
                                                                               test_loss, train_loss, train_accuracy,
                                                                               test_accuracy, epoch, model, best_params)
            if early_stop:
                break
            loss_dict = {'training': train_loss_out, 'testing': test_loss_out, 'training_accuracy': train_acc_out,
                         'testing_accuracy': test_acc_out}

    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)  # __str__ allows args to be printed directly,
        torch.cuda.memory_snapshot()
        model_file = f'{save_folder}{model_name}_dropout_model_state_failed.pt'
        torch.save(best_params, model_file)

    print('Best parameters at:'
          'Epoch: {} \tTraining Loss: {:.4f} \tTesting loss: {:.4f} \tTraining accuracy: {:.2f} '
          '\tTesting accuracy {:.2f}'
          .format(best_loss_dict['epoch'], best_loss_dict['train_loss'], best_loss_dict['test_loss'],
                  best_loss_dict['train_acc'], best_loss_dict['test_acc']))

    if save and best_params is not None:
        model_file = f'{save_folder}{model_name}_dropout'
        save_params(model_file, loss_dict, best_params)

    if show:
        # plot model losses
        plot_model(best_loss_dict, train_loss_out, test_loss_out, train_acc_out, test_acc_out, type="accuracy")

    return best_params, best_loss_dict


def test_iter_model(model, test_loader, classes, criterion):
    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
    best_params = model_init(model)

    # Epochs
    test_loss = 0.0
    test_accuracy = 0.0
    grasp_accuracy = torch.zeros((10, 2)).to(device)
    confusion_ints = torch.zeros((10, 7, 7)).to(device)
    grasp_true_labels = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
    grasp_pred_labels = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
    true_labels = []
    pred_labels = []
    hidden_size = 7
    sm = nn.Softmax(dim=1)
    saliencies_hidden = {}
    saliencies_frm = {}
    grasp_sal_hid_dd = {}
    grasp_sal_frm_dd = {}
    grasp_sal_hid = torch.zeros((7, 10))
    grasp_sal_frm = torch.zeros((7, 10))
    grasp_sal_count = torch.zeros((7, 10))

    for data in test_loader:
        # take the next frame from the data_loader and process it through the model
        frame = data["data"].to(device)
        frame_labels = data["labels"]
        # randomly switch in zero rows to vary the number of grasps being identified
        padded_rows_start = np.random.randint(1, 11)
        frame[:, padded_rows_start:, :] = 0
        enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
        # set the initial guess as a flat probability for each object
        pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

        # run the model and calculate loss
        if model_name == 'IterativeRNN' or 'IterativeRCNN' or 'SilhouetteRNN' or 'IterativeRNN2' or 'LSTM':
            # set the first hidden layer as a vanilla prediction or zeros
            if model_name == 'IterativeRNN2':
                hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)
            elif model_name == 'LSTM':  # input a vanilla starting layer output
                hidden = (torch.zeros(1, 64).to(device), torch.zeros(1, 64).to(device))
            else:
                hidden = torch.zeros(frame.size(0), hidden_size).to(device)

            if model_name == 'IterativeRCNN':
                frame = frame.reshape(frame.size(0), -1, 1, 19)
                hidden = hidden.reshape(frame.size(0), 1, -1)

            for i in range(padded_rows_start):
                # optimizer.zero_grad()
                if model_name == 'SilhouetteRNN':
                    output, hidden, embeddings = model(frame[:, i, :], hidden)
                    loss1 = criterion(output, enc_lab)
                    # calculate the silhouette score at the bottleneck and add it to the loss value
                    loss3 = silhouette.silhouette.score(embeddings, enc_lab, loss=True)
                    loss2 = loss1 + 2 * loss3
                # LUCA ADD: do the processing here
                if model_name == 'IterativeRNN2':
                    frm = copy.deepcopy(frame[:, i, :])
                    frm.requires_grad_()
                    hidden.requires_grad_()

                    output = model(frm, hidden)

                    # loss = torch.ones_like(pred_back) - output
                    # loss.backward()

                    # sm_out = sm(output)
                    score_max_index = output.argmax(1)  # class output across batches (dim=batch size)
                    score_max = output[range(output.shape[0]),
                                score_max_index.data].mean()  # make sure this vector is dim=batch size, AND NOT A MATRIX
                    score_max.backward()

                    # frm_saliency = torch.mean(frm.grad.data.abs(), dim=1).detach().cpu().numpy()  # dim=batch size vectors
                    frm_saliency, _ = torch.max(frm.grad.data.abs(), dim=1)
                    frm_saliency = frm_saliency.detach().cpu().numpy()  # variant n.1
                    # frm_saliency = frm.grad.data.var()  #  variant n.2

                    hidden_saliency, _ = torch.max(hidden.grad.data.abs(), dim=1)  # dim=batch size vectors
                    hidden_saliency = hidden_saliency.detach().cpu().numpy()
                    # example on how to save stuff
                    # tod: sort this into a single dict populating function rather than called each time
                    # saliences_frm = populate_sal_dict(frm_saliency, enc_lab)
                    for i_elem, _ in enumerate(frm_saliency):
                        # record general saliencies
                        # frame salience
                        saliencies_frm = populate_sal_dicts(saliencies_frm, frm_saliency, enc_lab, i_elem)
                        saliencies_hidden = populate_sal_dicts(saliencies_hidden, hidden_saliency, enc_lab, i_elem)

                        # nested dict for the grasps within the objects
                        if enc_lab[i_elem].item() in grasp_sal_frm_dd:
                            if i in grasp_sal_frm_dd[enc_lab[i_elem].item()]:
                                grasp_sal_frm_dd[enc_lab[i_elem].item()][i].append(frm_saliency[i_elem])
                            else:
                                grasp_sal_frm_dd[enc_lab[i_elem].item()][i] = [frm_saliency[i_elem]]
                        else:
                            grasp_sal_frm_dd[enc_lab[i_elem].item()] = {}
                            grasp_sal_frm_dd[enc_lab[i_elem].item()][i] = [frm_saliency[i_elem]]
                        if enc_lab[i_elem].item() in grasp_sal_hid_dd:
                            if i in grasp_sal_hid_dd[enc_lab[i_elem].item()]:
                                grasp_sal_hid_dd[enc_lab[i_elem].item()][i].append(hidden_saliency[i_elem])
                            else:
                                grasp_sal_hid_dd[enc_lab[i_elem].item()][i] = [hidden_saliency[i_elem]]
                        else:
                            grasp_sal_hid_dd[enc_lab[i_elem].item()] = {}
                            grasp_sal_hid_dd[enc_lab[i_elem].item()][i] = [hidden_saliency[i_elem]]

                    #  tod: get the grasp saliencies to see how they vary through the iterations

                    for i_elem, _ in enumerate(output):
                        grasp_sal_hid[enc_lab[i_elem].item(), i] += hidden_saliency[i_elem]
                        grasp_sal_frm[enc_lab[i_elem].item(), i] += frm_saliency[i_elem]
                        grasp_sal_count[enc_lab[i_elem].item(), i] += 1

                    loss2 = criterion(hidden, enc_lab)

                    # hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)
                    hidden = copy.copy(output)
                    output.detach()

                else:
                    output, hidden, sal = model(frame[:, i, :], hidden)

                    loss2 = criterion(hidden, enc_lab)
                    # examine the confidence of each final grasp by taking the maximum value of the softmax output
                    dist = sm(hidden)

                if model_name == 'IterativeRCNN':
                    hidden = hidden.reshape(frame.size(0), 1, hidden.size(-1))
            # hidden.backward(torch.ones_like(hidden), retain_graph=True)
            # hidden.backward(torch.ones_like(hidden))  # clear previous grads
            last_frame = copy.copy(output)
        else:
            last_frame, output = model(frame, pred_in)
            loss2 = gamma_loss(criterion, output, enc_lab)  # criterion(outputs, enc_lab)

        test_loss += loss2.item()

        # calculate accuracy of classification
        _, inds = last_frame.max(dim=1)
        frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
        test_accuracy += frame_accuracy
        grasp_accuracy[padded_rows_start - 1, 1] += 1
        grasp_accuracy[padded_rows_start - 1, 0] += frame_accuracy

        # concatenate the mean salience for each variable
        # frame_salience = torch.cat((frame_salience, salience / len(inds)))

        # use indices of objects to form confusion matrix
        for n, _ in enumerate(enc_lab):
            row = enc_lab[n]
            col = inds[n]
            confusion_ints[padded_rows_start - 1, row, col] += 1
            # add the prediction and true to the grasp_labels dict
            grasp_num = get_nth_key(grasp_pred_labels, padded_rows_start - 1)
            grasp_true_labels[grasp_num].append(classes[row])
            grasp_pred_labels[grasp_num].append(classes[col])

        pred_labels.extend(decode_labels(inds, classes))
        true_labels.extend(frame_labels)

    test_accuracy = test_accuracy / len(test_loader)
    test_loss = test_loss / len(test_loader)
    test_loss_out.append(test_loss)
    test_acc_out.append(test_accuracy)
    grasp_accuracy[:, 1] = grasp_accuracy[:, 0] / grasp_accuracy[:, 1]
    grasp_accuracy[:, 0] = torch.linspace(1, 10, 10, dtype=int)
    print(f'Grasp accuracy: {grasp_accuracy}')
    confusion_perc = confusion_ints / torch.sum(confusion_ints, dim=0)
    print('Frame Saliencies \t Hidden Saliencies')
    for i, _ in enumerate(saliencies_frm):
        print(
            f'{sum(saliencies_frm[i]) / len(saliencies_frm[i])} \t {sum(saliencies_hidden[i]) / len(saliencies_hidden[i])}')
    # scale the grasp saliencies and print
    grasp_sal_frm = grasp_sal_frm / grasp_sal_count
    grasp_sal_hid = grasp_sal_hid / grasp_sal_count

    # calculate variance of the grasp saliences
    grasp_frm_std, grasp_frm_norm = salience_std(grasp_sal_frm_dd)
    grasp_hid_std, grasp_hid_norm = salience_std(grasp_sal_hid_dd)
    plot_saliencies(grasp_frm_norm, grasp_hid_norm, grasp_frm_std, grasp_hid_std, classes)

    torch.set_printoptions(precision=2)
    print('Frame Saliencies \t Hidden Saliencies')
    print(f'\n Frame Input Grasp Saliencies \n {grasp_sal_frm} \n Hidden Layer Grasp Saliencies \n {grasp_sal_hid}')
    return true_labels, pred_labels, grasp_true_labels, grasp_pred_labels


def learn_model(model, train_loader, test_loader, optimizer, criterion, n_grasps, n_epochs=50, max_patience=10,
                save_folder='./', save=True, show=True):
    model_name, device, train_loss_out, test_loss_out, train_sil_out, test_sil_out, patience, best_loss_dict, \
    best_params = model_init(model, n_grasps)

    best_loss = None
    best_params = None
    try:
        for epoch in range(1, n_epochs + 1):
            # monitor training loss
            train_loss = 0.0
            test_loss = 0.0
            train_sil = 0.0
            test_sil = 0.0
            cycle = 0

            # Training
            model.train()
            for data in train_loader:
                frame = data["data"].to(device)
                frame_labels = data["labels"]

                optimizer.zero_grad()
                outputs, embeddings = model(frame)

                # check if NaNs have appeared and when
                nan_check = torch.reshape(embeddings, (-1,))
                if sum(torch.isnan(nan_check)) > 0:
                    print(f"NaN present at cycle {cycle}")

                loss = criterion(outputs, frame)

                # convert frame_labels to numeric and allocate to tensor to silhouette score
                le = preprocessing.LabelEncoder()
                frame_labels_num = le.fit_transform(frame_labels)

                # calculate the silhouette score at the bottleneck and add it to the loss value
                silhouette_avg = silhouette.silhouette.score(embeddings, torch.as_tensor(frame_labels_num), loss=True)
                # silhouette_avg = silhouette_score(embeddings.cpu().detach().numpy(), frame_labels_num)
                # silhouette_avg = torch.from_numpy(np.array(silhouette_avg)) * -1
                loss = (loss + 0.025 * silhouette_avg)

                loss.backward()  # loss +
                optimizer.step()
                train_loss += loss.item()
                train_sil += silhouette_avg
                cycle += 1

            train_loss = train_loss / len(train_loader)
            train_loss_out.append(train_loss)
            train_sil = train_sil.detach() / len(train_loader)
            train_sil_out.append(train_sil)

            model.eval()
            for data in test_loader:
                # take the next frame from the data_loader and process it through the model
                frame = data["data"].to(device)
                frame_labels = data["labels"]

                # convert frame_labels to numeric and allocate to tensor to silhouette score
                le = preprocessing.LabelEncoder()
                frame_labels_num = le.fit_transform(frame_labels)
                outputs, embeddings = model(frame)

                # calculate the silhouette score at the bottleneck and add it to the loss value
                # silhouette_avg = silhouette_score(embeddings.cpu().detach().numpy(), frame_labels)
                # silhouette_avg = torch.from_numpy(np.array(silhouette_avg)) * -1
                silhouette_avg = silhouette.silhouette.score(embeddings, torch.as_tensor(frame_labels_num), loss=True)
                loss2 = criterion(outputs, frame)
                loss2 = (loss2 + 0.025 * silhouette_avg)  # loss2 + 0.02 *
                test_loss += loss2.item()
                test_sil += silhouette_avg

            test_loss = test_loss / len(test_loader)
            test_loss_out.append(test_loss)
            test_sil = test_sil.detach() / len(test_loader)
            test_sil_out.append(test_sil)

            # EARLY STOPPING LOGIC
            if -test_sil > 0.99:
                best_params = copy.copy(model.state_dict())
                best_loss_dict = {'train_loss': train_loss, 'test_loss': test_loss, 'train_sil': train_sil,
                                  'test_sil': test_sil}
                print('Early stopping: Silhouette score exceeded 0.99')
                break
            elif best_loss_dict['test_loss'] is None or test_loss < best_loss_dict['test_loss']:

                best_loss_dict = {'train_loss': train_loss, 'test_loss': test_loss, 'train_sil': train_sil,
                                  'test_sil': test_sil}
                patience = 0
                best_params = copy.copy(model.state_dict())
            else:
                patience += 1
                if patience >= max_patience:
                    print(f'Early stopping: training terminated at epoch {epoch} due to es, '
                          f'patience exceeded at {max_patience}')
                    break

            loss_dict = {'training': train_loss_out, 'testing': test_loss_out, 'training_silhouette': train_sil_out,
                         'testing_silhouette': test_sil_out}
            # luca: we observe loss*1e3 just for convenience. the loss scaling isn't necessary above
            print('Epoch: {} \tTraining Loss: {:.8f} \tTesting loss: {:.8f} \tTraining silhouette score: {:.4f} '
                  '\tTesting silhouette score {:.4f}'
                  .format(epoch, train_loss * 1e3, test_loss * 1e3, -train_sil, -test_sil))
            # empty cache to prevent overusing the memory
            torch.cuda.empty_cache()

    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)  # __str__ allows args to be printed directly,
        torch.cuda.memory_snapshot()
        model_file = f'{save_folder}{model_name}_{n_grasps}grasps_model_state_failed.pt'
        torch.save(best_params, model_file)
        loss_dict = {'training': train_loss_out, 'testing': test_loss_out, 'training_silhouette': train_sil_out}
        # open file for writing, "w" is writing
        w = csv.writer(open(f'{save_folder}{model.__class__.__name__}_{n_grasps}_losses_failed.csv', 'w'))
        # loop over dictionary keys and values
        for key, val in loss_dict.items():
            # write every key and value to file
            w.writerow([key, val])
        return

    print('Best parameters at:'
          'Epoch: {} \tTraining Loss: {:.8f} \tTesting loss: {:.8f} \tTraining silhouette score: {:.4f} '
          '\tTesting silhouette score {:.4f}'
          .format(epoch, best_loss_dict['train_loss'] * 1e3, best_loss_dict['test_loss'] * 1e3,
                  -best_loss_dict['train_sil'], -best_loss_dict['test_sil']))

    if save and best_params is not None:
        model_file = f'{save_folder}{model_name}_{n_grasps}grasps'
        save_params(model_file, best_loss_dict, best_params)

    if show:
        # plot model losses
        plot_model(best_loss_dict, train_loss_out, test_loss_out, train_sil_out, test_sil_out, type="silhouette")

    return best_params, loss_dict


def test_model(model, train_loader, test_loader, classes, n_grasps, show=True, compare=False):
    model_name = model.__class__.__name__
    # print(model_name)

    device = get_device()
    print(device)
    model.to(device)
    encoded_train_out = torch.FloatTensor()
    train_labels_out = []
    encoded_test_out = torch.FloatTensor()
    test_labels_out = []
    model.eval()
    test_sil = 0.0

    if compare:
        plt.figure()
        rows = 10
        columns = 2
        n_plots = int(rows * columns)
        plt.suptitle("Comparison of input and output data from model")
        plt.GridSpec(rows, columns, wspace=0.25, hspace=1)
        plt.show()

    for data in train_loader:
        frame = data["data"].to(device)
        labels = data["labels"]
        outputs, embeddings = model(frame)
        encoded_train_out = torch.cat((encoded_train_out, embeddings.cpu()), 0)
        train_labels_out.extend(labels)

        if compare:
            y_max = torch.max(outputs, frame).max()
            y_max = y_max.cpu().detach().numpy()
            for i in range(int(n_plots / 2)):
                ticks = np.linspace(1, 19, 19)
                exec(f"plt.subplot(grid{[2 * i]})")
                plt.cla()
                plt.bar(ticks, frame.cpu()[0, 0, i, :])
                plt.ylim((0, y_max))
                plt.title(labels[i])
                exec(f"plt.subplot(grid{[2 * i + 1]})")
                plt.cla()
                plt.bar(ticks, outputs.cpu().detach().numpy()[0, 0, i, :])
                plt.ylim((0, y_max))
                plt.title(labels[i])
    for data in test_loader:
        frame = data["data"].to(device)
        labels = data["labels"]
        outputs, embeddings = model(frame)
        encoded_test_out = torch.cat((encoded_test_out, embeddings.cpu()), 0)
        test_labels_out.extend(labels)
        # convert frame_labels to numeric and allocate to tensor to silhouette score
        le = preprocessing.LabelEncoder()
        frame_labels_num = le.fit_transform(labels)
        silhouette_avg = silhouette.silhouette.score(embeddings, torch.as_tensor(frame_labels_num), loss=True)
        test_sil += silhouette_avg
    test_sil = test_sil / len(test_loader)

    if show:
        # plot encoded data
        torch.Tensor.ndim = property(lambda self: len(self.shape))
        x = encoded_test_out[:, 0].cpu().detach().numpy()
        y = encoded_test_out[:, 1].cpu().detach().numpy()
        scipy.io.savemat(f'./data/{model_name}_{n_grasps}_bottleneck.mat',
                         {'Bottleneck_data': encoded_test_out.cpu().detach().numpy(),
                          'Labels': test_labels_out})
        plt.figure()

        if encoded_test_out.shape[1] == 3:
            z = encoded_test_out[:, 2].cpu()
            z = z.detach().numpy()
            ax = plt.axes(projection='3d')
        for label in classes:
            label_indices = []
            for idx, _ in enumerate(test_labels_out):
                if test_labels_out[idx] == label:
                    label_indices.append(idx)
            if encoded_test_out.shape[1] == 2:
                plt.scatter(x[label_indices], y[label_indices], label=label)
            elif encoded_test_out.shape[1] == 3:
                ax.scatter3D(x[label_indices], y[label_indices], z[label_indices], label=label, s=2)
                ax.set_zlabel('Component 3')

        # present overall silhouette score
        # convert frame_labels to numeric and allocate to tensor to silhouette score
        le = preprocessing.LabelEncoder()
        frame_labels_num = le.fit_transform(test_labels_out)
        silhouette_avg = silhouette.silhouette.score(encoded_test_out, torch.as_tensor(frame_labels_num), loss=False)
        print(f"Silhouette score: {silhouette_avg}")

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.suptitle(f"{n_grasps} Grasps Bottleneck Data")
        # plt.show()

        return encoded_train_out, train_labels_out, encoded_test_out, test_labels_out, -test_sil
