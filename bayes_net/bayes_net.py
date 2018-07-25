#!/usr/bin/env python3
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


class BayesNet(object):

    def __init__(self, filePath, k, num_classes, testSize=0.3):

        self.data_file = pd.read_csv(filePath)
        self.k = k
        self.n_class = num_classes
        self.test_size = testSize
        self.dataSetBreak = {"dataTrain": None,
                             "dataTest": None,
                             "classTrain": None,
                             "classTest": None}

    def entrenar(self):
        """
        funcion para realizar clasificacion y prediccion de resultados de test
        :return: regresa la prediccion de y y el clasificado para
        """
        classifier = GaussianNB()
        classifier.fit(self.dataSetBreak["x_train"], self.dataSetBreak["y_train"])

        # prediccion de resultados con datos de test
        y_pred = classifier.predict(self.dataSetBreak["x_test"])

        return y_pred, classifier

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
        """
        funcion para codificar los datos y separa datos para entrenamiento y preubas
        :return:
        """
        x = self.data_file.iloc[:, :-1].values
        y = self.data_file.iloc[:, 41].values

        # encoding categorical data
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

        # splitting the dataset into the training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=0)

        # feature scaling
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        self.dataSetBreak["x_train"] = x_train
        self.dataSetBreak["x_test"] = x_test
        self.dataSetBreak["y_train"] = y_train
        self.dataSetBreak["y_test"] = y_test

    def run_bayes_net(self):
        """
        funcion inicial para ejecutar_todo el algoritmo
        :return:
        """
        self.data_file['normal.'] = self.data_file['normal.'].replace(
            ['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.',
             'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.',
             'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

        self.break_datadet()
        y_predi, classi = self.entrenar()
        cm = confusion_matrix(self.dataSetBreak["y_test"], y_predi)

        # graficar la matrix
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

