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
    data_arduino = ""  # serial.Serial(port='COM13', baudrate=9600, timeout=.1)
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
    fig, left_text, right_text = setup_gui(classes)

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
    data_arduino = ""  # serial.Serial(port='COM13', baudrate=9600, timeout=.1)
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
        if data_out is None:
            data_out = epoch_data_out
            labels_out = epoch_labels_out
        else:
            data_out = np.append(data_out, epoch_data_out, axis=0)
            labels_out = np.append(labels_out, epoch_labels_out, axis=0)

    return sensor_maxima, data_out, labels_out


def tune_RNN_network(model, optimizer, criterion, batch_size, blocked_sensor=None, old_data=None, new_data=None,
                     n_epochs=50, max_patience=25, save_folder='./', oldnew=True, save=True, show=True):
    model_name, device, train_loss_out, valid_loss_out, train_acc_out, valid_acc_out, patience, best_loss_dict, \
        best_params = model_init(model)
    hidden_size = 7

    """Convert data into tensors"""
    old_train_data, old_valid_data, _ = old_data
    new_train_data, new_valid_data, _ = new_data

    """Calculate how many batches there are"""

    batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size  # enforce even batch sizes
    half_batch = int(batch_size / 2)

    train_batch_reminder = len(new_train_data) % half_batch if oldnew else len(old_train_data) % half_batch
    valid_batch_reminder = len(new_valid_data) % half_batch if oldnew else len(old_valid_data) % half_batch

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
            # X_cat[X_cat < 1] = 0

            y_cat = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_old.to(device)
            batch_ints = list(range(len(y_cat)))
            random.shuffle(batch_ints)
            X = X_cat[batch_ints, :, :]
            y = y_cat[batch_ints]

            """Allocate a sensor to be blocked in retraining"""
            if blocked_sensor is not None:
                # X[:, :, blocked_sensor] = 0
                drop = blocked_sensor  # np.random.randint(blocked_sensor)
                X[:, :, random.sample(range(X.shape[-1]), k=drop)] = 0.

            # randomly pick a number of grasps to train
            padded_ints = list(range(n_grasps))
            random.shuffle(padded_ints)

            # randomly pick a grasp order
            grasps_order = list(range(n_grasps))
            random.shuffle(grasps_order)

            model.train()
            for k in range(n_grasps):
                frame_loss = 0

                # randomly switch in zero rows to vary the number of grasps being identified
                random.shuffle(grasps_order)

                padded_start = padded_ints[k]  # np.random.randint(1, 11)
                X_pad = X[:, grasps_order[:padded_start + 1], :]

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

        grasp_accuracies = {i: 0. for i in range(n_grasps)}  # setup for accuracy at each grasp number
        grasp_predictions = {i: [] for i in range(n_grasps)}
        grasp_true_labels = {i: [] for i in range(n_grasps)}

        accuracies = np.zeros(n_grasps).astype(float)  # setup for accuracy at each grasp number

        model.eval()
        for i in range(n_valid_batches):

            # Take each validation batch and process

            if oldnew:
                batch_start = i * half_batch
                batch_end = i * half_batch + half_batch if i * half_batch + half_batch < len(new_valid_data) \
                    else len(new_valid_data)
            else:
                batch_start = i * batch_size
                batch_end = i * batch_size + batch_size if i * batch_size + batch_size < len(old_train_data) \
                    else len(old_train_data)

            X_old, y_old, y_labels_old = old_valid_data[old_valid_indices[batch_start:batch_end]]
            X_new, y_new, y_labels_new = new_valid_data[new_valid_indices[batch_start:batch_end]]

            X_cat = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) if oldnew else \
                X_old.reshape(-1, 10, 19).to(device)
            # noise = torch.normal(0, 0.2, X_cat.shape).to(device)
            # X_cat += noise
            # X_cat[X_cat < 1] = 0
            y_cat = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_old.to(device)
            y_labels = np.concatenate([y_labels_old, y_labels_new]) if oldnew else y_labels_old

            batch_ints = list(range(len(y_cat)))
            random.shuffle(batch_ints)
            X = X_cat[batch_ints, :, :]
            y = y_cat[batch_ints]

            y_labels = y_labels[batch_ints]

            all_no_of_grasps = list(range(n_grasps))
            random.shuffle(all_no_of_grasps)

            # randomly pick a grasp order
            grasps_order = list(range(n_grasps))
            random.shuffle(grasps_order)

            if blocked_sensor is not None:
                # X[:, :, blocked_sensor] = 0
                drop = blocked_sensor  # np.random.randint(blocked_sensor)
                X[:, :, random.sample(range(X.shape[-1]), k=drop)] = 0.

            padded_ints = list(range(n_grasps))
            random.shuffle(padded_ints)

            for k in range(n_grasps):

                # randomly switch in zero rows to vary the number of grasps being identified
                random.shuffle(grasps_order)

                padded_start = padded_ints[k]  # np.random.randint(1, 11)
                X_pad = X[:, grasps_order[:padded_start + 1], :]

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
                grasp_accuracies[padded_start] += frame_accuracy
                grasp_predictions[padded_start].extend(old_train_data.get_labels(preds.cpu().numpy()))
                grasp_true_labels[padded_start].extend(y_labels.flatten().tolist())

        # calculate the testing accuracy and losses and divide by the number of batches
        valid_accuracy = valid_accuracy / (n_valid_batches * n_grasps)
        for i in range(n_grasps): accuracies[i] /= n_valid_batches
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
            best_loss_dict = {'train_loss': train_loss,
                              'valid_loss': valid_loss,
                              'train_acc': train_accuracy,
                              'valid_acc': valid_accuracy,
                              'grasp_accuracies': grasp_accuracies,
                              'predictions': grasp_predictions,
                              'true_labels': grasp_true_labels,
                              'epoch': epoch, }
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
        model_file = f'{save_folder}{model_name}_dropout_sens_drop'
        save_params(model_file, best_loss_dict, best_params)

    if show:
        # plot model losses
        plot_model(best_loss_dict, train_loss_out, valid_loss_out, train_acc_out, valid_acc_out, type="accuracy")

    print(f'Grasp accuracy: {grasp_accuracy}')
    return model, best_params, best_loss_dict


def test_tuned_model(model, n_epochs, criterion, batch_size, blocked_sensor=None, old_data=None, new_data=None,
                     oldnew=False, show_confusion=True):

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

    obj_accuracy = np.zeros((7, 2)).astype(float)

    """Extract data"""
    _, _, old_test_data = old_data
    _, _, new_test_data = new_data

    """Calculate how many batches there are"""
    half_batch = int(batch_size / 2)
    batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size  # enforce even batch sizes

    if oldnew:
        test_batch_reminder = len(new_test_data) % half_batch

        n_test_batches = int(len(new_test_data) / half_batch) if test_batch_reminder == 0 else int(
            len(new_test_data) / half_batch) + 1
    else:
        test_batch_reminder = len(old_test_data) % batch_size

        n_test_batches = int(len(old_test_data) / batch_size) if test_batch_reminder == 0 else int(
            len(old_test_data) / batch_size) + 1

    old_test_indices = list(range(len(old_test_data)))
    new_test_indices = list(range(len(new_test_data)))

    model.eval()
    for _ in range(5):
        random.shuffle(old_test_indices)
        random.shuffle(new_test_indices)

        for i in range(n_test_batches):

            # Take each testing batch and process

            if oldnew:
                batch_start = i * half_batch
                batch_end = i * half_batch + half_batch \
                    if i * half_batch + half_batch < len(new_test_data) \
                    else len(new_test_data)
            else:
                batch_start = i * batch_size
                batch_end = i * batch_size + batch_size \
                    if i * batch_size + batch_size < len(old_test_data) \
                    else len(old_test_data)

            X_old, y_old, y_labels_old = old_test_data[old_test_indices[batch_start:batch_end]]
            X_new, y_new, y_labels_new = new_test_data[new_test_indices[batch_start:batch_end]]

            X = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) \
                if oldnew else X_old.reshape(-1, 10, 19).to(device)
            # X[X < 1] = 0
            y = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_old.to(device)
            y_labels = np.concatenate([y_labels_old, y_labels_new]) if oldnew else y_labels_old

            # randomly pick a grasp order
            grasps_order = list(range(n_grasps))
            random.shuffle(grasps_order)

            if blocked_sensor is not None:
                # X[:, :, blocked_sensor] = 0
                drop = blocked_sensor  # np.random.randint(blocked_sensor)
                X[:, :, random.sample(range(X.shape[-1]), k=drop)] = 0.
            true_labels.extend(y_labels.squeeze().tolist())

            padded_ints = list(range(n_grasps))
            random.shuffle(padded_ints)

            for k in range(n_grasps):
                # randomly switch the number and order of rows to vary the grasps being identified
                random.shuffle(grasps_order)

                padded_start = padded_ints[k]  # np.random.randint(1, 11)
                X_pad = X[:, grasps_order[:padded_start + 1], :]

                # set hidden layer
                hidden = torch.full((X_pad.size(0), hidden_size), 1 / hidden_size).to(device)

                """ iterate through each grasp and run the model """
                output = model(X_pad[:, 0, :], hidden)
                hidden = nn.functional.softmax(output, dim=-1)
                for j in range(1, padded_start + 1):
                    output = model(X_pad[:, j, :], hidden)
                    hidden = nn.functional.softmax(output, dim=-1)

                loss2 = criterion(output, y.squeeze())
                test_loss += loss2.item()

                # calculate accuracy of classification
                _, preds = output.detach().max(dim=1)
                frame_accuracy = torch.sum(preds == y.flatten()).cpu().numpy() / len(preds)

                test_accuracy += frame_accuracy
                grasp_accuracy[padded_start, 1] += frame_accuracy
                for idx, ob in enumerate(y.flatten()):
                    if preds[idx] == ob:
                        obj_accuracy[ob, 1] += 1
                    obj_accuracy[ob, 0] += 1

                # add the prediction and true to the grasp_labels dict
                pred_labels_tmp = old_test_data.get_labels(preds.cpu().numpy())
                pred_labels.extend(pred_labels_tmp)
                grasp_pred_labels[str(padded_start + 1)].extend(pred_labels_tmp)

    test_accuracy = test_accuracy / (n_epochs * n_grasps)
    test_loss = test_loss / (n_epochs * n_grasps)
    test_loss_out.append(test_loss)
    test_acc_out.append(test_accuracy)

    grasp_accuracy[:, 1] = grasp_accuracy[:, 1] / n_test_batches / 5
    print(f'Grasp accuracy: \n{grasp_accuracy}')
    print(f'Average accuracy = \n{np.mean(grasp_accuracy[:, 1])}')

    obj_accuracy[:, 1] = obj_accuracy[:, 1] / obj_accuracy[:, 0]
    print(f'Object accuracy: {obj_accuracy}')

    if show_confusion:
        for grasp_no in range(n_grasps):
            unique_labels = new_test_data.labels
            cm = confusion_matrix(true_labels, grasp_pred_labels[str(grasp_no + 1)], labels=unique_labels)
            # cm_display = ConfusionMatrixDisplay(cm, display_labels=unique_labels).plot()
            cm = cm.astype('float64')
            for row in range(len(unique_labels)):
                cm[row, :] = cm[row, :] / cm[row, :].sum()
            fig = plt.figure()
            plt.title(f'{grasp_no + 1} grasps - {model.__class__.__name__}')
            fig.set_size_inches(8, 5)
            sns.set(font_scale=1.2)
            cm_display_percentages = sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues',
                                                 xticklabels=unique_labels,
                                                 yticklabels=unique_labels, vmin=0, vmax=1).plot()

            plt.show()

    return grasp_pred_labels


def online_grasp_w_early_stop(model, n_epochs, batch_size, classes, criterion, old_data=None, new_data=None,
                              oldnew=True, n_grasps=10):
    model_name, device, train_loss_out, test_loss_out, train_acc_out, test_acc_out, patience, best_loss_dict, \
        best_params = model_init(model)
    test_accuracy = 0
    grasp_accuracy = torch.zeros((10, 2)).to(device)
    confusion_ints = torch.zeros((10, 7, 7)).to(device)
    confs = torch.zeros((7, 10)).to(device)
    conf_sums = torch.zeros((7, 10)).to(device)
    conf_std = [[[] for temp in range(10)] for _ in classes]
    true_labels = []
    hidden_size = 7
    sm = nn.Softmax(dim=1)

    conf_obj_examine = 6

    """Extract data"""
    _, _, old_test_data = old_data
    _, _, new_test_data = new_data
    batch_size = batch_size - 1 if batch_size % 2 != 0 else batch_size  # enforce even batch sizes
    half_batch = batch_size / 2

    test_batch_reminder = len(old_test_data) % half_batch if oldnew else len(new_test_data) % half_batch

    n_test_batches = int(len(new_test_data) / half_batch) if test_batch_reminder == 0 else int(
        len(new_test_data) / half_batch) + 1

    old_test_indices = list(range(len(old_test_data)))
    new_test_indices = list(range(len(new_test_data)))

    model.eval()

    random.shuffle(old_test_indices)
    random.shuffle(new_test_indices)

    for x in range(10):
        random.shuffle(old_test_indices)
        random.shuffle(new_test_indices)
        for i in range(n_test_batches):
            batch_start = i * batch_size
            batch_end = i * batch_size + batch_size \
                if i * batch_size + batch_size < len(new_test_data) \
                else len(new_test_data)

            X_old, y_old, y_labels_old = old_test_data[old_test_indices[batch_start:batch_end]]
            X_new, y_new, y_labels_new = new_test_data[new_test_indices[batch_start:batch_end]]

            X = torch.cat([X_old.reshape(-1, 10, 19), X_new.reshape(-1, 10, 19)], dim=0).to(device) if oldnew else \
                X_new.reshape(-1, 10, 19).to(device)
            """Vary the noise to find a balance between perfect results and more realistic variation"""
            noise = 0  # torch.normal(0, 0.01, X.shape).to(device)
            X += noise
            # X[X < 1] = 0
            y = torch.cat([y_old, y_new], dim=0).to(device) if oldnew else y_new.to(device)
            y_labels = np.concatenate([y_labels_old, y_labels_new]) if oldnew else y_labels_new

            all_no_of_grasps = list(range(n_grasps))
            random.shuffle(all_no_of_grasps)

            # randomly pick a grasp order
            grasps_order = list(range(n_grasps))
            random.shuffle(grasps_order)

            for r in range(X.size(0)):
                obj_stack = np.zeros((0, 7))
                X_frame = X[r, grasps_order, :].reshape((1, 10, -1))
                true_labels.extend(y_labels.squeeze().tolist())

                # run the model and calculate loss
                hidden = torch.full((X_frame.size(0), hidden_size), 1 / 7).to(device)

                grasps_taken = 0

                for j in range(X.size(1)):
                    output = model(X_frame[:, j:j + 1, :], hidden)
                    probs_out = sm(output)
                    # score_max_index = probs_out.argmax(1)  # class output across batches (dim=batch size)
                    score_max, score_max_index = probs_out.max(dim=1)

                    hidden = nn.functional.softmax(output, dim=-1)
                    confs[y[r], grasps_taken] += score_max
                    conf_sums[y[r], grasps_taken] += 1
                    conf_std[y[r]][grasps_taken].append(score_max)

                    if y[r] == conf_obj_examine:
                        obj_stack = np.append(obj_stack, probs_out.detach().cpu().numpy() * 100, axis=0)
                    grasps_taken += 1
                    if score_max > 0.99:
                        break

                last_frame = probs_out

                # calculate accuracy of classification
                _, inds = last_frame.max(dim=1)
                frame_accuracy = 1 if inds == y[r] else 0
                # frame_accuracy = torch.sum(inds == y.flatten()).cpu().numpy() / batch_size
                test_accuracy += frame_accuracy
                grasp_accuracy[grasps_taken - 1, 1] += 1
                grasp_accuracy[grasps_taken - 1, 0] += frame_accuracy

                if conf_obj_examine == y[r] and np.size(obj_stack, 0) > 1:
                    n_rows = np.size(obj_stack, 0)
                    # fig, axs = plt.subplots(n_rows, 1)
                    fig, ax = plt.subplots(1, 1)
                    y_shift = 0
                    y_data = []
                    for row in range(n_rows):
                        cyl_list = []
                        # obj_stack[row, :] += y_shift
                        for idx, ent in enumerate(obj_stack[row, :]):
                            temp_list = [idx] * int(np.ceil(ent))
                            cyl_list.extend(temp_list)
                        s = pd.Series(cyl_list)
                        curve = s.plot.density(color='black')
                        x_line = ax.get_lines()[row].get_xdata()
                        res = []
                        for idx in range(0, len(x_line)):
                            if 0 < x_line[idx] < 6:
                                res.append(idx)
                        y_line = ax.get_lines()[row].get_ydata()[res] + y_shift
                        y_data.append(y_line)
                        y_shift = max(y_line) + 0.3
                    fig, ax = plt.subplots(1, 1)
                    for line in y_data:
                        xx = np.linspace(0, 6, len(line))
                        ax.plot(xx, line)
                    ax.set_xlim((-0.25, 6.25))
                    ax.set_ylim((0, line.max() + 0.3))
                    plt.xticks(ticks=range(0, 7), labels=classes)
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    fig.text(0.085, 0.5, 'Belief Distribution', va='center', rotation='vertical')
                    fig.set_size_inches((9, 6))
                    # plt.show()

    grasp_accuracy[:, 0] = grasp_accuracy[:, 0] / grasp_accuracy[:, 1]
    print(f'Grasp accuracy: {grasp_accuracy}')
    # extract the confidences for each grasp number and object then ensure there are no nan values and
    # calculate the std values
    confidences = (confs / conf_sums * 100).detach().cpu().numpy()
    # confidences[np.isnan(confidences)] = 100
    for obj_idx, obj in enumerate(conf_std):
        for g_idx, g in enumerate(obj):
            if g:
                conf_std[obj_idx][g_idx] = torch.std(torch.stack(conf_std[obj_idx][g_idx]))
            else:
                break
    std_list = [[] for ob in classes]
    for ob, _ in enumerate(classes):
        for g in range(10):
            if torch.is_tensor(conf_std[ob][g]):
                if torch.isnan(conf_std[ob][g]):
                    std_list[ob].append(0)
                else:
                    std_list[ob].append(conf_std[ob][g].detach().cpu().numpy())
            else:
                break

    x = range(1, 11)
    x_start = 0
    x_bar_ticks, x_bar_ticks_minor, x_bar_labels = [], [], []
    max_grasps = 5
    fig_line, ax_line = plt.subplots(1, 1)
    fig_bar, ax_bar = plt.subplots(1, 1)
    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                   '#17becf']
    for row, class_name in enumerate(classes):
        y_plt = confidences[row, ~np.isnan(confidences[row, :])]
        x_plt = [*range(len(y_plt))]
        x_bar = [x_start + _ * 0.2 for _ in x_plt]
        x_start = max(x_bar) + 0.4
        x_bar_ticks.extend(x_bar)
        x_bar_ticks_minor.append((max(x_bar) + min(x_bar)) / 2)
        x_bar_labels.extend([_ + 1 for _ in x_plt])
        # x_bar = [row + _ for _ in x_tix]
        # ax.plot(x_plt, y_plt, '-', label=f'{class_name}')
        ax_bar.plot(x_bar, y_plt, linewidth=3, color=color_cycle[row]) if len(x_plt) > 1 else \
            ax_bar.scatter(x_bar, y_plt, marker='x', linewidth=3, color=color_cycle[row])  # label=f'{class_name}'
        ax_bar.bar(x_bar, y_plt, width=0.15, alpha=0.3, label=classes[row], color=color_cycle[row])
    conf_99 = np.array([[-0.5, 99], [np.max(x_bar) + 0.5, 99]])
    ax_bar.plot(conf_99[:, 0], conf_99[:, 1], 'k--', alpha=0.3)
    plt.show()
    plt.xticks(ticks=x_bar_ticks, labels=x_bar_labels)
    ax_bar.set_ylabel('Belief / %')
    ax_bar.set_xlim((-0.5, np.max(x_bar) + 0.5))
    # ax_bar.set_xlabel('Objects')
    fig_bar.set_size_inches((8, 4))
    ax_bar.legend()
