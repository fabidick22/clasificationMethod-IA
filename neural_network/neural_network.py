#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix




"""
reference:
https://github.com/Belval/ML-IDS
https://github.com/khakhulin/IntrusionDetection
"""


class NeuralNet(object):

    def __init__(self, filePath, num_entradas, num_ocultos, num_salidas, num_classes, epoch=20, testSize=0.3):

        self.data_file = pd.read_csv(filePath)
        self.n_input = num_entradas
        self.n_hidden = num_ocultos
        self.n_output = num_salidas
        self.n_class = num_classes
        self.n_epoch = epoch
        self.test_size = testSize
        self.dataSetBreak = {"dataTrain": None,
                             "dataTest": None,
                             "classTrain": None,
                             "classTest": None}

    def entrenar(self):
        """
        entrenamos la red neuronal con los datos de entrenamiento
        :return:
        """
        # iniciar la ANN
        classifier = Sequential()

        # agregando la capa de entrada y la primera capa oculta
        classifier.add(Dense(output_dim=60, init='uniform', activation='relu', input_dim=118))

        # agregando la segunda capa oculta
        classifier.add(Dense(output_dim=60, init='uniform', activation='relu'))

        # agregando la tercera capa oculta
        classifier.add(Dense(output_dim=60, init='uniform', activation='relu'))

        # agregando capa de salida
        classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

        # Compiling the ANN
        # copilando ANN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # ajustar ANN con los datos de entrenamiento
        classifier.fit(self.dataSetBreak["x_train"], self.dataSetBreak["y_train"], batch_size=10, nb_epoch=self.n_epoch)

        # predecir los resultados con los datos de test
        y_pred = classifier.predict(self.dataSetBreak["x_test"])
        y_pred = (y_pred > 0.5)

        return y_pred


    def crete_model_net(self):
        self.break_datadet()
        if self.n_class is None:
            self.n_class = len(set(self.data_file.target))

        # f = [tf.feature_column.numeric_column("x", shape=[4])]
        # f_columnas = tf.contrib.learn.infer_real_valued_columns_from_input(self.dataSetBreak["dataTrain"])
        feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
        # hidden_units= [10, 20, 10] se va a crear un modelo de 3 capas con 10, 20 y 10 unidades respectivamente
        clasi = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=self.n_hidden,
                                               n_classes=self.n_class)
        return clasi

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def break_datadet(self):
        # separar los datos de la clase
        x = self.data_file.iloc[:, :-1].values
        y = self.data_file.iloc[:, 41].values

        # encodificiacion de los datos
        labelencoder_x_1 = LabelEncoder()
        labelencoder_x_2 = LabelEncoder()
        labelencoder_x_3 = LabelEncoder()
        x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
        x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
        x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])
        onehotencoder_1 = OneHotEncoder(categorical_features=[1])
        x = onehotencoder_1.fit_transform(x).toarray()
        onehotencoder_2 = OneHotEncoder(categorical_features=[4])
        x = onehotencoder_2.fit_transform(x).toarray()
        onehotencoder_3 = OneHotEncoder(categorical_features=[70])
        x = onehotencoder_3.fit_transform(x).toarray()
        labelencoder_y = LabelEncoder()
        y = labelencoder_y.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=0)

        # calado de caracteristicas
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        self.dataSetBreak["x_train"] = x_train
        self.dataSetBreak["x_test"] = x_test
        self.dataSetBreak["y_train"] = y_train
        self.dataSetBreak["y_test"] = y_test

    def runClasificador(self):
        # cambiar multiclase a una clase binaria separando los datos normales de un ataque
        self.data_file['normal.'] = self.data_file['normal.'].replace(
            ['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.',
             'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.',
             'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

        self.break_datadet()
        y_predi = self.entrenar()
        cm = confusion_matrix(self.dataSetBreak["y_test"], y_predi)

        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        a = ["1:Normal", "0:Attack"]
        self.plot_confusion_matrix(cm, classes=a, title='Matrix de confusión, sin normalizar')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cm, classes=a, normalize=True,
                                   title='Matrix de confusión normalizada ')
        plt.show()

        # the performance of the classification model
        print("La exactitud es: " + str((cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])))
        print("Ratio de falsos positivos: " + str(cm[1, 0] / (cm[0, 0] + cm[1, 0])))
        precision = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        print("Precision es: " + str(precision))

