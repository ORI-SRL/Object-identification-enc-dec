import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import serial
import time

arduino = serial.Serial(port='COM13', baudrate=9600, timeout=.1)


def write_read(x):
    # arduino.write(bytes(x, 'utf-8'))
    # time.sleep(0.05)
    data = arduino.readline()
    print(data)


while True:
    num = input("Enter a number: ")  # Taking input from user
    value = write_read(num)
    print(value)  # printing the value


