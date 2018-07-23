import tensorflow as tf
import pandas as pd
import numpy as np


"""
reference:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/neural_network_raw.ipynb
https://becominghuman.ai/creating-your-own-neural-network-using-tensorflow-fa8ca7cc4d0e
https://github.com/michaelwayman/python-ann/blob/master/neural_network/neural_network.py

real
https://github.com/nikhilroxtomar/Iris-Data-Set-Classification-using-TensorFlow-MLP/blob/master/iris.py
"""


class NeuralNet(object):

    def __init__(self, file, num_entradas, num_ocultos, num_salidas, num_classes):

        self.data_file = file
        self.n_input = num_entradas
        self.n_hidden = num_ocultos
        self.n_output = num_salidas
        self.n_class = num_classes

    def entrenar(self):
        pass

