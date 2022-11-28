import numpy as np
import os
from os.path import exists
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch
import serial
import time
from utils.pytorch_helpers import get_device
from utils.ml_classifiers import *
import pickle


# from utils.networks import IterativeRNN2


def write_read(arduino, stack):
    arduino.readline()
    arduino.readline()
    arduino.readline()
    data = arduino.readline()
    data = data.decode()
    stack.append(data)
    return stack


def format_rows(data, norm_vals):
    """Take only the first and last rows to find the delta between them but check for a DC shift"""
    init_row_str = data[0].split(";")[20:-1]
    end_row_str = data[-1].split(";")[20:-1]
    init_row = torch.FloatTensor(np.double(init_row_str))
    end_row = torch.FloatTensor(np.double(end_row_str))
    datarow = end_row - init_row
    norm_row = (datarow - norm_vals[1, :]) / (norm_vals[0, :] - norm_vals[1, :])
    norm_row[norm_row < 0] = 0
    return norm_row


def grasp_obj(self, arduino):
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
    device = get_device()
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
            input_data = format_rows(data_stack, norm_vals)  # .to(device)
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
    grasping = True
    grasp_num = 6
    new_gather = {}
    delta_values = []
    object_name = input("Which object is being grasped?")
    while grasping:
        grasp_init = input("Take grasp? Y or N")  # "y"
        data_arduino.flush()
        data_arduino.readline()
        if grasp_init == "Y" or grasp_init == "y":
            data_arduino.flush()
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
            data_delta = format_rows(data_stack, norm_vals)
            delta_values.append(data_delta)
            grasp_num += 1
            print(grasp_num, " grasps have been taken.")
            data_arduino.flush()

        elif grasp_init == "N" or grasp_init == "n":
            grasping = False
            with open(f"{data_folder}val_test_grasps/{object_name}_deltas2", "wb") as fp:
                pickle.dump(delta_values, fp)


def test_new_grasps(model, data_folder, save_folder, classes):
    model_state = f'{save_folder}{model.__class__.__name__}_dropout_model_state.pt'
    if exists(model_state):
        model.load_state_dict(torch.load(model_state))
    model.eval()
    model_name = model.__class__.__name__
    # classes = ['apple', 'bottle', 'cards', 'cube', 'cup', 'cylinder', 'sponge']
    pred_labels = []
    true_labels = []

    # fig, left_text, right_text = setup_gui(classes)
    device = get_device()
    sm = nn.Softmax(dim=1)
    for object_name in classes:
        # object_name = input("Object Name?")
        hidden = torch.tensor([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])  # .to(device)
        hidden = torch.stack((hidden, hidden))
        data_file = f"{data_folder}val_test_grasps/{object_name}_deltas"
        true_val = classes.index(object_name)
        with open(data_file, "rb") as fp:  # Unpickling
            input_data = pickle.load(fp)
        if object_name == 'sponge':
            input_data.extend(input_data)
        if len(input_data) > 20:
            input_data = input_data[0:20]
        input_data = torch.stack(input_data)
        input_data = torch.reshape(input_data, (2, -1, 19))
        for k in range(input_data.size(dim=1)):  # enumerate(input_data):
            idx = k  # [0]

            hidden = model(input_data[:, idx, :], hidden)
            probs = sm(hidden)
            if idx % 4 == 0 and idx != 0:
                # hidden = torch.tensor([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])
                pred_labels.extend(list(np.argmax(probs.detach().numpy(), axis=1)))
                true_labels.append(true_val)
                true_labels.append(true_val)
                break
            # present_grasp_result(probs, classes, fig, left_text, right_text)
    plot_confusion(pred_labels, true_labels, model_name, 1, iter=True)
    print('done')
