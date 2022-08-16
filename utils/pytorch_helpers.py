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
    for i in range(len(preds)):
        decoded_label_frame.append(classes[preds[i]])
    return decoded_label_frame


def learn_iter_model(model, train_loader, test_loader, optimizer, criterion, n_grasps, classes, n_epochs=50,
                     max_patience=10, save_folder='./', save=True, show=True):
    model_name = model.__class__.__name__
    print(f'{model_name} {n_grasps} grasps')

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

    try:
        for epoch in range(1, n_epochs + 1):
            # monitor training loss
            train_loss = 0.0
            test_loss = 0.0
            cycle = 0
            train_accuracy = 0.0
            test_accuracy = 0.0
            confusion_ints = torch.zeros((7, 7)).to(device)

            # Training
            model.train()
            for data in train_loader:
                frame = data["data"].to(device)  # .reshape(32, 10, 19)
                frame_labels = data["labels"]

                # randomly switch in zero rows to vary the number of grasps being identified
                padded_rows = np.random.randint(1, n_grasps)
                frame[:, padded_rows:, :] = 0
                enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
                pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

                optimizer.zero_grad()
                outputs = model(frame, pred_in)

                loss = criterion(outputs, enc_lab)

                loss.backward()  # loss +
                optimizer.step()
                train_loss += loss.item()
                _, inds = outputs.max(dim=1)
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
                padded_rows = np.random.randint(1, n_grasps+1)
                frame[:, padded_rows:, :] = 0
                enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
                # set the initial guess as a flat probability for each object
                pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

                outputs = model(frame, pred_in)
                loss2 = criterion(outputs, enc_lab)

                test_loss += loss2.item()
                _, inds = outputs.max(dim=1)
                frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
                test_accuracy += frame_accuracy
                for n in range(len(enc_lab)):
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
            if best_loss_dict['test_loss'] is None or test_loss < best_loss_dict['test_loss']:

                best_loss_dict = {'train_loss': train_loss, 'test_loss': test_loss, 'train_acc': train_accuracy,
                                  'test_acc': test_accuracy, 'epoch': epoch}
                patience = 0
                best_params = copy.copy(model.state_dict())
            else:
                patience += 1
                if patience >= max_patience:
                    print(f'Early stopping: training terminated at epoch {epoch} due to es, '
                          f'patience exceeded at {max_patience}')
                    break

            loss_dict = {'training': train_loss_out, 'testing': test_loss_out, 'training_silhouette': train_acc_out,
                         'testing_silhouette': test_acc_out}

    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)  # __str__ allows args to be printed directly,
        torch.cuda.memory_snapshot()
        model_file = f'{save_folder}{model_name}_{n_grasps}grasps_model_state_failed.pt'
        torch.save(best_params, model_file)

    print('Best parameters at:'
          'Epoch: {} \tTraining Loss: {:.4f} \tTesting loss: {:.4f} \tTraining accuracy: {:.2f} '
          '\tTesting accuracy {:.2f}'
          .format(best_loss_dict['epoch'], best_loss_dict['train_loss'], best_loss_dict['test_loss'],
                  best_loss_dict['train_acc'], best_loss_dict['test_acc']))

    if save and best_params is not None:
        model_file = f'{save_folder}{model_name}_{n_grasps}grasps_model_state.pt'
        torch.save(best_params, model_file)
        # open file for writing, "w" is writing
        w = csv.writer(open(f'{save_folder}{model.__class__.__name__}_{n_grasps}_losses.csv', 'w'))
        # loop over dictionary keys and values
        for key, val in loss_dict.items():
            # write every key and value to file
            w.writerow([key, val])

    if show:
        # plot model losses
        fig, [ax1, ax2] = plt.subplots(1, 2)
        x = list(range(1, len(test_loss_out) + 1))
        ax1.plot(x, train_loss_out, label="Training loss")
        ax1.plot(x, test_loss_out, label="Testing loss")
        ax1.plot(best_loss_dict['epoch'], best_loss_dict['train_loss'])
        ax1.plot(best_loss_dict['epoch'], best_loss_dict['test_loss'])
        ax1.set_xlabel('epoch #')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(train_acc_out, label="Training accuracy")
        ax2.plot(test_acc_out, label="Testing accuracy")
        ax2.plot(best_loss_dict['epoch'], best_loss_dict['train_acc'])
        ax2.plot(best_loss_dict['epoch'], best_loss_dict['test_acc'])
        ax2.set_xlabel('epoch #')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

    return best_params, best_loss_dict


def test_iter_model(model, test_loader, classes, criterion):
    model_name = model.__class__.__name__
    print(f'{model_name}')

    device = get_device()
    print(device)
    model.to(device)
    # Epochs
    test_loss_out = []
    test_acc_out = []
    test_loss = 0.0
    test_accuracy = 0.0
    grasp_accuracy = torch.zeros((9, 2)).to(device)
    confusion_ints = torch.zeros((7, 7)).to(device)
    confusion_perc = torch.zeros((7, 7)).to(device)
    true_labels = []
    pred_labels = []

    for data in test_loader:
        # take the next frame from the data_loader and process it through the model
        frame = data["data"].to(device)
        frame_labels = data["labels"]
        # randomly switch in zero rows to vary the number of grasps being identified
        padded_rows_start = np.random.randint(1, 10)
        frame[:, padded_rows_start:, :] = 0
        enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
        # set the initial guess as a flat probability for each object
        pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

        # run the model and calculate loss
        outputs = model(frame, pred_in)
        loss2 = criterion(outputs, enc_lab)

        test_loss += loss2.item()

        # calculate accuracy of classification
        _, inds = outputs.max(dim=1)
        frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
        test_accuracy += frame_accuracy
        grasp_accuracy[padded_rows_start-1, 0] += 1
        grasp_accuracy[padded_rows_start-1, 1] += frame_accuracy

        # use indices of objects to form confusion matrix
        for n in range(len(enc_lab)):
            row = enc_lab[n]
            col = inds[n]
            confusion_ints[row, col] += 1
        pred_labels.extend(decode_labels(inds, classes))
        true_labels.extend(frame_labels)

    test_accuracy = test_accuracy / len(test_loader)
    test_loss = test_loss / len(test_loader)
    test_loss_out.append(test_loss)
    test_acc_out.append(test_accuracy)
    grasp_accuracy[:, 1] = grasp_accuracy[:, 1] / grasp_accuracy[:, 0]
    print(f'Grasp accuracy: {grasp_accuracy}')
    confusion_perc = confusion_ints / torch.sum(confusion_ints, dim=0)

    return true_labels, pred_labels


def learn_model(model, train_loader, test_loader, optimizer, criterion, n_grasps, n_epochs=50, max_patience=10,
                save_folder='./', save=True, show=True):
    model_name = model.__class__.__name__
    print(f'{model_name} {n_grasps} grasps')

    device = get_device()
    print(device)
    model.to(device)

    # Epochs
    train_loss_out = []
    test_loss_out = []
    train_sil_out = []
    test_sil_out = []
    best_loss_dict = {'test_loss': None}

    patience = 0
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
        model_file = f'{save_folder}{model_name}_{n_grasps}grasps_model_state.pt'
        torch.save(best_params, model_file)
        # open file for writing, "w" is writing
        w = csv.writer(open(f'{save_folder}{model.__class__.__name__}_{n_grasps}_losses.csv', 'w'))
        # loop over dictionary keys and values
        for key, val in loss_dict.items():
            # write every key and value to file
            w.writerow([key, val])

    if show:
        # plot model losses
        x = list(range(1, len(test_loss_out) + 1))
        plt.plot(x, train_loss_out, label=model_name + "Training loss")
        plt.plot(x, test_loss_out, label=model_name + "Testing loss")
        plt.xlabel('epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

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
            for idx in range(len(test_labels_out)):
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
