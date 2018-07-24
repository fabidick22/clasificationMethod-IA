import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
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

    def __init__(self, file, num_entradas, num_ocultos, num_salidas, num_classes, epoch=200):

        res = datasets.fetch_kddcup99()
        self.data_file = res
        self.n_input = num_entradas
        self.n_hidden = num_ocultos
        self.n_output = num_salidas
        self.n_class = num_classes
        self.epoch = epoch
        self.dataSetBreak = {"dataTrain": None,
                             "dataTest": None,
                             "classTrain": None,
                             "classTest": None}

    def entrenar(self, classifierModel):
        """
        entrenamos la red neuronal con los datos de entrenamiento
        :return:
        """
        classifier = classifierModel
        classifier.fit(self.dataSetBreak["dataTrain"], self.dataSetBreak["classTrain"], steps=self.epoch)

        # usamos el modelo entrenado para realizar predicciones con los datos de test
        predictions = list(classifier.predict(self.dataSetBreak["dataTest"], as_iterable=True))
        return predictions


    def crete_model_net(self):
        self.break_datadet()
        if self.n_class is None:
            self.n_class = len(set(self.data_file.target))
        f_columnas = tf.contrib.learn.infer_real_valued_columns_from_input(self.dataSetBreak["dataTrain"])
        # hidden_units= [10, 20, 10] se va a crear un modelo de 3 capas con 10, 20 y 10 unidades respectivamente
        return tf.contrib.learn.DNNClassifier(feature_columns=f_columnas, hidden_units=self.n_hidden,
                                                    n_classes=self.n_class)

    def break_datadet(self):
        target = self.data_file.target
        data = self.data_file.data
        x_train, x_test, y_train, y_test = model_selection\
            .train_test_split(data, target, test_size=0.2, random_state=42)
        self.dataSetBreak["dataTrain"]=x_train
        self.dataSetBreak["dataTest"]=x_test
        self.dataSetBreak["classTrain"]=y_train
        self.dataSetBreak["classTest"]=y_test

    def runClasificador(self):
        clasificador = self.crete_model_net()
        predictions = self.entrenar(clasificador)

        # comprobamos como de buenas han sido las predicciones y calculamos la precisión de nuestro modelo
        score = metrics.accuracy_score(self.dataSetBreak["classTest"], predictions)
        print('Accuracy: {0:f}'.format(score))
