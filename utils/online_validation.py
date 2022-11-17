import numpy as np
import os
from os.path import exists
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.widgets as wgt
import serial
import time
from utils.pytorch_helpers import get_device


# from utils.networks import IterativeRNN2


def write_read(arduino):
    data = arduino.readline()
    data = data.decode()
    return data


def format_rows(data, norm_vals):
    """Take only the first and last rows to find the delta between them"""
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


def setup_gui(classes, arduino):
    fig, ax = plt.subplots()
    filler = np.zeros(7)
    # blank bar chart to begin with
    plt.subplots_adjust(bottom=0.5)
    # plot_wrapper = fig.add_axes([0.1, 0.5, 0.8, 0.45])
    plt.bar(range(7), filler, tick_label=classes, label='plot')
    plt.ylim(0, 1)

    # text box
    text_wrapper_l = fig.add_axes([0.1, 0.2, 0.3, 0.2])
    text_wrapper_l.set_axis_off()
    text_box_l = text_wrapper_l.text(0.5, 0.5, "Press 'grasp' to take first prediction",
                                     horizontalalignment='center', verticalalignment='center')
    text_wrapper_r = fig.add_axes([0.6, 0.2, 0.3, 0.2])
    text_wrapper_l.set_axis_off()
    text_box_r = text_wrapper_r.text(0.5, 0.5, "Press 'grasp' to take first prediction",
                                     horizontalalignment='center', verticalalignment='center')
    text_wrapper_r.set_axis_off()
    # wgt.TextBox(text_wrapper, label="", initial="Press 'grasp' to take first prediction")
    # text_wrapper.text(0.5, 0.5, "Press 'grasp' to take first prediction",
    # horizontalalignment='center', verticalalignment='center')
    #
    # button box
    button_wrapper = fig.add_axes([0.4, 0.05, 0.2, 0.075])
    button = wgt.Button(button_wrapper, "Grasp")
    # button.on_clicked(grasp_obj)
    return fig, text_box_l, text_box_r


def present_grasp_result(probabilities, classes, fig, left_text, right_text):
    probabilities = probabilities.detach().numpy()
    idx = np.argmax(probabilities)
    max_obj = classes[idx]
    max_prob = probabilities[idx]
    textstr = '\n'.join((
        f'$Predicted Object:$ {max_obj}',
        r'$Probability: %.2f$' % (max_prob)))
    # odo: convert to vbox equivalent and populate lower box with text rather than floating text box and potentially
    #  a button to trigger the grasp rather than the 'input' line
    fig.axes[0].cla()
    fig.axes[0].bar(range(len(classes)), probabilities, tick_label=classes, color='r')
    fig.axes[0].set_ylim(0, 1)

    left_text.set_text("Predicted Object: /n Probability /n")
    right_text.set_text(f"{max_obj} /n {max_prob} /n")

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # fig.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    #          verticalalignment='top', bbox=props)
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
        grasp_init = "y"  # input("Take grasp? Y or N")
        data_arduino.flush()
        data_arduino.readline()
        if grasp_init == "Y" or grasp_init == "y":
            start_time = time.time()
            grasp_time = time.time()
            # Wait to allow the user to pass the object to the hand and record the start values
            print("Present object to hand")
            while grasp_time - start_time < 5:
                data_row = write_read(data_arduino)
                data_stack.append(data_row)
                grasp_time = time.time()

            # reset start time and send message to valve arduino to close the hand around the object to hold for 20
            # seconds
            start_time = time.time()
            # valve_arduino.write(bytes("1", 'utf-8'))
            print("Release object for 20 secs")
            while grasp_time - start_time < 20:
                data_row = write_read(data_arduino)
                data_stack.append(data_row)
                grasp_time = time.time()
            # valve_arduino.write(bytes("0", 'utf-8'))

            input_data = format_rows(data_stack, norm_vals)  # .to(device)
            hidden = model(input_data, hidden)
            probs = sm(hidden)
            present_grasp_result(probs, classes, fig, left_text, right_text)
        if max(probs) > 0.75:
            identifying = False
