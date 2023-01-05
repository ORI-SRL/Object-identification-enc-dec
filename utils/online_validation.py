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
import serial
import time
from utils.ml_classifiers import *
from utils.pytorch_helpers import *
import pickle
import math


# from utils.networks import IterativeRNN2


def write_read(arduino, stack):
    data = arduino.readline()
    # data = data.decode()
    stack.append(data)
    return stack


def format_rows(data, norm_vals, start_idx, norm=True):
    """Take only the first and last rows to find the delta between them but check for a DC shift"""
    all_data_list = []  # np.empty((0, 19))
    final_idx = len(data)
    if len(data[start_idx + 1:]) != 0:
        data = data[start_idx + 1:]
    else:
        start_idx = 0
        data = data[start_idx + 1:]
    for row in enumerate(data):
        if len(data[row[0]]) > 100:
            all_data_list.append(list(map(int, data[row[0]].split(";")[-20:-1])))
        # else:
        # final_idx -= 1
    all_data_arr = np.array(all_data_list)
    plt.plot(all_data_arr)
    plt.show()
    plt.cla()
    init_row_str = data[1].split(";")[-20:-1]
    end_row_str = data[-1].split(";")[-20:-1]
    init_row = torch.FloatTensor(np.double(init_row_str))
    end_row = torch.FloatTensor(np.double(end_row_str))
    datarow = end_row - init_row  # - math.floor(sum(end_row - init_row) / len(end_row))
    # rejig the order in case the mapping is wrong
    # initial order -> 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 1, 0, 2, 3, 18
    data_row_copy = copy.copy(datarow)
    # datarow[14] = data_row_copy[16]
    # datarow[15] = data_row_copy[17]
    # datarow[16] = data_row_copy[14]
    # datarow[17] = data_row_copy[15]
    if norm:
        out_row = (datarow - norm_vals[1, :]) / (norm_vals[0, :] - norm_vals[1, :])
        out_row[out_row < 0] = 0
    else:
        out_row = datarow
    return out_row, datarow, final_idx


def grasp_obj():
    print("pressed")


def setup_gui(classes):
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    fig.set_size_inches(12.8, 8.4)
    filler = np.zeros(7)
    # blank bar chart to begin with
    plt.subplots_adjust(bottom=0.5)
    # plot_wrapper = fig.add_axes([0.1, 0.5, 0.8, 0.45])
    plt.bar(range(7), filler, tick_label=classes, label='plot')
    plt.ylim(0, 1)

    # text box
    text_wrapper_l = fig.add_axes([0.1, 0.2, 0.3, 0.2])
    text_wrapper_l.set_axis_off()
    # ax.text(.4,.4,"Rounded Box", transform=ax.transAxes, bbox=dict(boxstyle='round', fc='w', ec='k'))
    text_box_l = text_wrapper_l.text(0.5, 0.5, "Press 'grasp' to take first prediction",
                                     horizontalalignment='center', verticalalignment='center',
                                     bbox=dict(boxstyle='round', fc='w', ec='k'))
    text_wrapper_r = fig.add_axes([0.6, 0.2, 0.3, 0.2])
    text_wrapper_l.set_axis_off()
    text_box_r = text_wrapper_r.text(0.5, 0.5, "Press 'grasp' to take first prediction",
                                     horizontalalignment='center', verticalalignment='center',
                                     bbox=dict(boxstyle='round', fc='w', ec='k'))
    text_wrapper_r.set_axis_off()
    # wgt.TextBox(text_wrapper, label="", initial="Press 'grasp' to take first prediction")
    # text_wrapper.text(0.5, 0.5, "Press 'grasp' to take first prediction",
    # horizontalalignment='center', verticalalignment='center')
    #
    # button box
    button_wrapper = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    button = wgt.Button(button_wrapper, "Grasp")
    # plt.show()
    # button.on_clicked(grasp_obj)
    return fig, text_box_l, text_box_r


def present_grasp_result(probabilities, classes, fig, left_text, right_text):
    probabilities = probabilities.detach().numpy()
    idx = np.argmax(probabilities)
    max_obj = classes[idx]
    max_prob = probabilities[idx]
    if max_prob > 0.85:
        decision_txt = "No"
    else:
        decision_txt = "Yes"
    # odo: convert to vbox equivalent and populate lower box with text rather than floating text box and potentially
    #  a button to trigger the grasp rather than the 'input' line
    fig.axes[0].cla()
    fig.axes[0].bar(range(len(classes)), probabilities, tick_label=classes, color='r')
    fig.axes[0].set_ylim(0, 1)

    left_text.set_text("\nPredicted Object:\n\nProbability\n\nGrasp Again?\n")
    right_text.set_text("\n{}\n\n{:.2f}\n\n{}\n".format(max_obj, max_prob, decision_txt))

    plt.show()


def online_loop(model, save_folder, norm_vals, classes):
    data_arduino = serial.Serial(port='COM13', baudrate=9600, timeout=.1)
    # valve_arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
    data_stack = []

    # load pre-trained model state
    model_state = f'{save_folder}{model.__class__.__name__}_dropout_model_state.pt'
    if exists(model_state):
        model.load_state_dict(torch.load(model_state))
    model.eval()

    # set up model initial conditions
    # device = get_device()
    hidden = torch.tensor([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])  # .to(device)
    sm = nn.Softmax(dim=0)
    # fig, ax = plt.subplots()
    probs = torch.zeros(7)
    identifying = True
    fig, left_text, right_text = setup_gui(classes, arduino=None)

    while identifying:
        grasp_init = input("Take grasp? Y or N")  # "y"
        data_arduino.flush()
        data_arduino.readline()
        if grasp_init == "Y" or grasp_init == "y":
            start_time = time.time()
            grasp_time = time.time()
            # Wait to allow the user to pass the object to the hand and record the start values
            print("Present object to hand")
            while grasp_time - start_time < 5:
                data_stack = write_read(data_arduino, data_stack)
                grasp_time = time.time()

            # reset start time and send message to valve arduino to close the hand around the object to hold for 20
            # seconds
            start_time = time.time()
            # valve_arduino.write(bytes("1", 'utf-8'))
            print("Hold object for 20 secs")
            while grasp_time - start_time < 20:
                data_stack = write_read(data_arduino, data_stack)
                grasp_time = time.time()
            # valve_arduino.write(bytes("0", 'utf-8'))
            print("Release object")
            input_data = format_rows(data_stack, norm_vals, norm=True)  # .to(device)
            hidden = model(input_data, hidden)
            probs = sm(hidden)
            present_grasp_result(probs, classes, fig, left_text, right_text)
            if max(probs) > 0.85:
                identifying = False
                print('Successfully identified object')
                print('Finished test')
        elif grasp_init == "N" or grasp_init == "n":
            identifying = False
            print('Finished test')


def gather_grasps(data_folder, classes, norm_vals):
    data_arduino = serial.Serial(port='COM13', baudrate=9600, timeout=.1)
    # valve_arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
    save_folder = f"{data_folder}val_test_grasps"
    data_stack = []
    data_stack_dec = []
    grasping = True
    grasp_num = 6
    new_gather = {}
    delta_values = []
    object_name = "weights"  # input("Which object is being grasped?")
    while grasping:
        grasp_init = "Y"  # input("Take grasp? Y or N")  # "y"
        data_arduino.flush()
        data_arduino.readline()
        if grasp_init == "Y" or grasp_init == "y":
            data_arduino.flush()
            data_arduino.readline()
            data_arduino.readline()
            start_time = time.time()
            grasp_time = time.time()
            # Wait to allow the user to pass the object to the hand and record the start values
            print("Present object to hand")
            while grasp_time - start_time < 5:
                data_stack = write_read(data_arduino, data_stack)
                grasp_time = time.time()

            # reset start time and send message to valve arduino to close the hand around the object to hold for 20
            # seconds
            start_time = time.time()
            # valve_arduino.write(bytes("1", 'utf-8'))
            print("Hold object for 20 secs")
            while grasp_time - start_time < 20:
                data_stack = write_read(data_arduino, data_stack)
                grasp_time = time.time()
            # odo: concat grasps to be n x t x 19
            with open(f"{data_folder}val_test_grasps/{object_name}{grasp_num}", "wb") as fp:
                pickle.dump(data_stack, fp)

            '''if object_name in new_gather:
                new_gather[object_name].append(data_stack)
            else:
                new_gather[object_name] = [data_stack]'''
            for row in enumerate(data_stack):
                r_temp = data_stack[row[0]].decode()
                if r_temp != "":
                    data_stack_dec.append(data_stack[row[0]].decode())
            data_delta = format_rows(data_stack_dec, norm_vals, start_idx=0)
            delta_values.append(data_delta)
            grasp_num += 1
            print(grasp_num, " grasps have been taken.")
            data_arduino.flush()

        elif grasp_init == "N" or grasp_init == "n":
            grasping = False
            with open(f"{data_folder}val_test_grasps/{object_name}_deltas2", "wb") as fp:
                pickle.dump(delta_values, fp)


def test_new_grasps(model, data_folder, save_folder, classes, norm_vals):
    # load model state
    model_state = f'{save_folder}{model.__class__.__name__}_dropout_model_state.pt'
    if exists(model_state):
        model.load_state_dict(torch.load(model_state))
    model.eval()
    model_name = model.__class__.__name__

    # predicted and true labels are used for the confusion matrix
    pred_labels = []
    true_labels = []
    new_norms = torch.tensor(
        [[32, 10, 35, 48, 127, 52, 27, 18, 33, 16, 32, 30, 15, 64, 48, 69, 48, 69, 170],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # torch.zeros(19)
    # fig, left_text, right_text = setup_gui(classes)
    sm = nn.Softmax(dim=1)

    # loop over the objects, load the raw file to extract the delta between first and last row
    for object_name in classes:
        obj_deltas = []
        save_deltas = []
        start_idx = 0
        for f in range(20):
            raw_file = f"{data_folder}val_test_grasps/{object_name}{f}"
            with open(raw_file, "rb") as fp:  # Unpickling
                raw_data = pickle.load(fp)
            del_row, max_row, start_idx = format_rows(raw_data, norm_vals, start_idx, norm=True)
            obj_deltas.append(del_row)
            save_deltas.append(max_row)
            # new_norms = torch.max(torch.stack((new_norms, max_row)), dim=0).values

        # set the initial hidden layer to be a vanilla uniform distribution
        hidden = torch.tensor([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])  # .to(device)
        hidden = torch.stack((hidden, hidden))

        # data_file = f"{data_folder}val_test_grasps/{object_name}_deltas"
        true_val = classes.index(object_name)
        with open(f"{data_folder}val_test_grasps/{object_name}_deltas", "wb") as fp:  # Pickling
            pickle.dump(save_deltas, fp)
        input_data = obj_deltas
        # if object_name == 'sponge':
        #     input_data.extend(input_data)
        if len(input_data) > 20:
            input_data = input_data[0:20]
        input_data = torch.stack(input_data)
        input_data = torch.reshape(input_data, (2, -1, 19))
        for k in range(input_data.size(dim=1)):  # enumerate(input_data):
            idx = k  # [0]

            hidden = model(input_data[:, idx, :], hidden)
            probs = sm(hidden)
            if idx % 9 == 0 and idx != 0:
                # hidden = torch.tensor([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])
                pred_labels.extend(list(np.argmax(probs.detach().numpy(), axis=1)))
                true_labels.append(true_val)
                true_labels.append(true_val)
                break
            # present_grasp_result(probs, classes, fig, left_text, right_text)
    plot_confusion(pred_labels, true_labels, model_name, 1, iter=True)
    plt.show()
    print('done')


def organise_tuning_data(old_data_file, old_labels_file, new_data_file, new_labels_file, num_epochs, batch_size,
                         sensor_maxima=None):
    """ load datasets - old is the original data and new is the tuning set"""
    old_data = np.load(old_data_file)
    old_labels = np.load(old_labels_file)
    new_data = np.load(new_data_file)
    new_labels = np.load(new_labels_file)
    num_batches = int(np.floor(len(new_data) / batch_size))
    """ In the training set, concat the data and find the maxima of each sensor value"""
    data_cat = np.concatenate((np.reshape(new_data, (-1, 19)), np.reshape(old_data, (-1, 19))), axis=0)
    if sensor_maxima is None:
        sensor_maxima = data_cat.max(axis=0)

    data_out = None
    labels_out = None

    """ Predetermine the data in each epoch to be processed by the model"""
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(f'Epochs completed: {epoch}')
        epoch_data_out = np.empty((1, 0, 10, 19))
        epoch_labels_out = np.empty((1, 0))

        # set the indices for the randomly shuffled data
        indices_data_1 = list(range(len(old_data)))
        np.random.shuffle(indices_data_1)
        indices_data_2 = list(range(len(new_data)))
        np.random.shuffle(indices_data_2)
        for i in range(num_batches):
            # select data to be in each batch, evenly distributed between old and new
            data1_batch = old_data[indices_data_1[i * batch_size:i * batch_size + batch_size], :]
            data1_labels = old_labels[indices_data_1[i * batch_size:i * batch_size + batch_size]]
            data2_batch = new_data[indices_data_2[i * batch_size:i * batch_size + batch_size], :]
            data2_labels = new_labels[indices_data_2[i * batch_size:i * batch_size + batch_size]]

            # normalise by the training maxima and reshape the data
            data1_batch = np.reshape(data1_batch, (1, -1, 10, 19)) / sensor_maxima
            data2_batch = np.reshape(data2_batch, (1, -1, 10, 19)) / sensor_maxima
            data1_labels = np.reshape(data1_labels, (1, -1))
            data2_labels = np.reshape(data2_labels, (1, -1))

            # stack the data into batches
            epoch_data_out = np.append(epoch_data_out, data1_batch, axis=1)
            epoch_data_out = np.append(epoch_data_out, data2_batch, axis=1)
            epoch_labels_out = np.append(epoch_labels_out, data1_labels, axis=1)
            epoch_labels_out = np.append(epoch_labels_out, data2_labels, axis=1)
        if 'data_out' is None:
            data_out = epoch_data_out
            labels_out = epoch_labels_out
        else:
            data_out = np.append(data_out, epoch_data_out, axis=0)
            labels_out = np.append(labels_out, epoch_labels_out, axis=0)

    return sensor_maxima, data_out, labels_out


def tune_RNN_network(model, train_loader, train_labels, test_loader, test_labels, optimizer, criterion, classes,
                     batch_size, n_epochs=50, max_patience=25, save_folder='./', save=True, show=True):
    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
        best_params = model_init(model)
    hidden_size = 7

    """Convert data into tensors"""
    train_loader = torch.tensor(train_loader)
    test_loader = torch.tensor(test_loader)
    """Calculate how many batches there are"""
    n_train_batches = int(train_loader.size(1) / batch_size)
    n_test_batches = int(test_loader.size(1) / batch_size)
    for epoch in range(n_epochs):
        train_loss, test_loss, train_accuracy, test_accuracy = 0.0, 0.0, 0.0, 0.0
        cycle = 0
        confusion_ints = torch.zeros((7, 7)).to(device)
        grasp_accuracy = torch.zeros((10, 2)).to(device)
        model.train()
        for i in range(n_train_batches):
            # Take each training batch and process
            frame = train_loader[epoch, i * batch_size:i * batch_size + batch_size, :, :].reshape(-1, 10, 19).to(device)
            frame_labels = train_labels[epoch, i * batch_size:i * batch_size + batch_size]
            frame_loss = 0

            # randomly switch in zero rows to vary the number of grasps being identified
            padded_start = np.random.randint(1, 11)
            frame[:, padded_start:, :] = 0
            nul_rows = frame.sum(dim=2) != 0
            frame = frame[:, :padded_start, :]

            # convert frame_labels to numeric and allocate to tensor
            enc_lab = encode_labels(frame_labels, classes).to(device)

            # set hidden layer
            hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)

            optimizer.zero_grad()
            """ iterate through each grasp and run the model """
            for j in range(padded_start):
                output = model(frame[:, j, :].float(), hidden)
                loss = criterion(output, enc_lab)
                hidden = copy.copy(output)
                frame_loss += loss  # sum loss at every loop
            frame_loss.backward()
            optimizer.step()
            # output = nn.functional.softmax(output, dim=-1)

            _, inds = output.max(dim=1)
            frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / len(inds)
            train_accuracy += frame_accuracy
            grasp_accuracy[padded_start - 1, 1] += 1
            grasp_accuracy[padded_start - 1, 0] += frame_accuracy
            train_loss += frame_loss / padded_start
            cycle += 1
        train_loss = train_loss.detach().cpu() / n_train_batches
        train_accuracy = train_accuracy / n_train_batches
        train_loss_out.append(train_loss)
        train_acc_out.append(train_accuracy)
        grasp_accuracy[:, 1] = grasp_accuracy[:, 0] / grasp_accuracy[:, 1]
        grasp_accuracy[:, 0] = torch.linspace(1, 10, 10, dtype=int)

        grasp_accuracy = torch.zeros((10, 2)).to(device)
        model.eval()
        for i in range(n_test_batches):
            # Take each test batch and run the model
            frame_in = test_loader[epoch, i * batch_size:i * batch_size + batch_size, :, :].reshape(-1, 10, 19).to(
                device)
            frame_labels = test_labels[epoch, i * batch_size:i * batch_size + batch_size]
            frame_loss = 0
            # randomly switch in zero rows to vary the number of grasps being identified
            padded_start = np.random.randint(1, 11)
            frame_in[:, padded_start:, :] = 0
            nul_rows = frame_in.sum(dim=2) != 0

            # take only the rows that are non-zero
            frame = frame_in[:, :padded_start, :]
            # convert frame_labels to numeric and allocate to tensor
            enc_lab = encode_labels(frame_labels, classes).to(device)

            # set hidden layer
            hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)

            # set the first hidden layer as a vanilla prediction or zeros
            hidden = torch.zeros(frame.size(0), hidden_size).to(device)

            """ Run the model through each grasp """
            for j in range(padded_start):
                output = model(frame[:, j, :].float(), hidden)
                loss3 = criterion(output, enc_lab)
                hidden = copy.copy(output)
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

        # calculate the testing accuracy and losses and divide by the number of batches
        test_accuracy = test_accuracy / n_test_batches
        test_loss = test_loss.detach().cpu() / n_test_batches
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
        if epoch == 198:
            epoch = 0
    if save and best_params is not None:
        model_file = f'{save_folder}{model_name}_dropout631'
        save_params(model_file, loss_dict, best_params)

    if show:
        # plot model losses
        plot_model(best_loss_dict, train_loss_out, test_loss_out, train_acc_out, test_acc_out, type="accuracy")

    return model, best_params, best_loss_dict


def test_tuned_model(model, val_data, val_labels, n_epochs, batch_size, classes, criterion):
    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
        best_params = model_init(model)

    # set zero values for all initial parameters
    test_loss, test_accuracy = 0.0, 0.0
    grasp_accuracy = torch.zeros((10, 2)).to(device)
    confusion_ints = torch.zeros((10, 7, 7)).to(device)
    grasp_true_labels = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
    grasp_pred_labels = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
    true_labels = []
    pred_labels = []
    hidden_size = 7
    sm = nn.Softmax(dim=1)

    val_loader = torch.tensor(val_data) # convert data to tensors
    n_val_batches = int(val_loader.size(1) / batch_size)
    model.eval()
    for epoch in range(10): # just use 10 epochs to save time
        for i in range(n_val_batches):
            # take the next frame from the data_loader and process it through the model
            frame = val_loader[epoch, i * batch_size:i * batch_size + batch_size, :, :].reshape(-1, 10, 19).to(device)
            frame_labels = val_labels[epoch, i * batch_size:i * batch_size + batch_size]
            frame_loss = 0
            # randomly switch in zero rows to vary the number of grasps being identified
            padded_rows_start = np.random.randint(1, 11)
            frame[:, padded_rows_start:, :] = 0
            enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
            # set the initial guess as a flat probability for each object
            pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

            # run the model and calculate loss

            hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)

            for i in range(padded_rows_start):
                # frm = copy.deepcopy(frame[:, i, :])
                # frm.requires_grad_()
                # hidden.requires_grad_()

                output = model(frame[:, i, :], hidden)
                score_max_index = output.argmax(1)  # class output across batches (dim=batch size)
                # score_max = output[range(output.shape[0]),
                #    score_max_index.data].mean()  # make sure this vector is dim=batch size, AND NOT A MATRIX
                # score_max.backward()

                loss2 = criterion(hidden, enc_lab)
                hidden = copy.copy(output)
                output.detach()

            last_frame = copy.copy(output)
            test_loss += loss2.item()

            # calculate accuracy of classification
            _, inds = last_frame.max(dim=1)
            frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / batch_size
            test_accuracy += frame_accuracy
            grasp_accuracy[padded_rows_start - 1, 1] += 1
            grasp_accuracy[padded_rows_start - 1, 0] += frame_accuracy

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

    test_accuracy = test_accuracy / n_val_batches / 10
    test_loss = test_loss / n_val_batches / 10
    test_loss_out.append(test_loss)
    test_acc_out.append(test_accuracy)
    grasp_accuracy[:, 1] = grasp_accuracy[:, 0] / grasp_accuracy[:, 1]
    grasp_accuracy[:, 0] = torch.linspace(1, 10, 10, dtype=int)
    print(f'Grasp accuracy: {grasp_accuracy}')
    confusion_perc = confusion_ints / torch.sum(confusion_ints, dim=0)

    return true_labels, pred_labels, grasp_true_labels, grasp_pred_labels


def online_grasp_w_early_stop(model, val_data, val_labels, n_epochs, batch_size, classes, criterion):
    """unfinished but nearly ready for plot adding"""
    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
        best_params = model_init(model)
    grasp_true_labels = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
    grasp_pred_labels = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": []}
    true_labels = []
    pred_labels = []
    hidden_size = 7
    sm = nn.Softmax(dim=1)
    val_loader = torch.tensor(val_data)
    n_val_batches = int(val_loader.size(1) / batch_size)
    model.eval()
    test_accuracy = 0
    grasp_accuracy = torch.zeros((10, 2)).to(device)
    confusion_ints = torch.zeros((10, 7, 7)).to(device)
    confs = torch.zeros((10, 2)).to(device)
    conf_sums = torch.zeros((10, 2)).to(device)

    for i in range(val_loader.size(dim=1)):
        # take the next frame from the data_loader and process it through the model
        frame = val_loader[0, i, :, :].reshape(-1, 10, 19).to(device)
        frame_labels = val_labels[0, i]
        frame_loss = 0

        enc_lab = encode_labels(frame_labels, classes).to(device)  # .softmax(dim=-1)
        # set the initial guess as a flat probability for each object
        pred_in = torch.full((frame.size(0), 7), 1 / 7).to(device)

        # run the model and calculate loss
        hidden = torch.full((frame.size(0), hidden_size), 1 / 7).to(device)

        grasps_taken = 0

        for j in range(np.size(frame, 1)):
            output = model(frame[:, j, :], hidden)
            probs_out = sm(output)
            score_max_index = probs_out.argmax(1)  # class output across batches (dim=batch size)
            score_max = probs_out.max(dim=1)

            hidden = copy.copy(output)
            output.detach()
            confs[enc_lab, grasps_taken] += score_max
            conf_sums[enc_lab, grasps_taken] += 1

            grasps_taken += 1
            if score_max > 0.95:
                break

        last_frame = copy.copy(output)

        # calculate accuracy of classification
        _, inds = last_frame.max(dim=1)
        frame_accuracy = torch.sum(inds == enc_lab).cpu().numpy() / batch_size
        test_accuracy += frame_accuracy
        grasp_accuracy[grasps_taken - 1, 1] += 1
        grasp_accuracy[grasps_taken - 1, 0] += frame_accuracy

        # use indices of objects to form confusion matrix
        for n, _ in enumerate(enc_lab):
            row = enc_lab[n]
            col = inds[n]
            confusion_ints[grasps_taken - 1, row, col] += 1
            # add the prediction and true to the grasp_labels dict
            grasp_num = get_nth_key(grasp_pred_labels, grasps_taken - 1)
            grasp_true_labels[grasp_num].append(classes[row])
            grasp_pred_labels[grasp_num].append(classes[col])

        pred_labels.extend(decode_labels(inds, classes))
        true_labels.extend(frame_labels)
    confidences = confs / conf_sums
    x = range(1, 11)
    for row in enumerate(classes):
        plt.plot(confidences[row, :], '-', label=f'{classes[row]}')
