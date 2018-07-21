import pandas

"""
reference:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/neural_network_raw.ipynb
https://becominghuman.ai/creating-your-own-neural-network-using-tensorflow-fa8ca7cc4d0e
https://github.com/michaelwayman/python-ann/blob/master/neural_network/neural_network.py
"""


class NeuralNet(object):

    def __init__(self, num_entradas, num_ocultos, num_salidas, num_classes):
        """

        :param num_entradas:
        :param num_ocultos:
        :param num_salidas:
        :param num_classes:
        """
        self.n_input = num_entradas
        self.n_hidden = num_ocultos
        self.n_output = num_salidas
        self.n_class = num_classes

    def entrenar(self):
        pass

