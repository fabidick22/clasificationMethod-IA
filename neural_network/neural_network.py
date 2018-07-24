import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection


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

        res = datasets.fetch_kddcup99()
        self.data_file = res
        self.n_input = num_entradas
        self.n_hidden = num_ocultos
        self.n_output = num_salidas
        self.n_class = num_classes


    def entrenar(self):
        pass

    def breakDataSet(self):
        target= self.data_file.target
        data= self.data_file.data
        dataSetBreak={"dataTrain":None,"dataTest":None,"classTrain":None,"classTest":None}
        x_train, x_test, y_train, y_test = model_selection\
            .train_test_split(data, target, test_size=0.2, random_state=42)
        dataSetBreak["dataTrain"]=x_train
        dataSetBreak["dataTest"]=x_test
        dataSetBreak["classTrain"]=y_train
        dataSetBreak["classTest"]=y_test
        return dataSetBreak

