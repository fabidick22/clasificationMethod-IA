# encoding: utf-8

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix




"""
reference:
real
https://github.com/nikhilroxtomar/Iris-Data-Set-Classification-using-TensorFlow-MLP/blob/master/iris.py
https://github.com/chadlimedamine/kdd-cup-99-Analysis-machine-learning-python/blob/master/kdd_binary_classification_ANN.py
https://github.com/Belval/ML-IDS
https://github.com/khakhulin/IntrusionDetection
"""


class NeuralNet(object):

    def __init__(self, filePath, num_entradas, num_ocultos, num_salidas, num_classes, epoch=200):

        self.data_file = pd.read_csv(filePath)
        self.n_input = num_entradas
        self.n_hidden = num_ocultos
        self.n_output = num_salidas
        self.n_class = num_classes
        self.n_epoch = epoch
        self.dataSetBreak = {"dataTrain": None,
                             "dataTest": None,
                             "classTrain": None,
                             "classTest": None}

    def entrenar(self):
        """
        entrenamos la red neuronal con los datos de entrenamiento
        :return:
        """
        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and the first hidden layer
        classifier.add(Dense(output_dim=60, init='uniform', activation='relu', input_dim=118))

        # Adding a second hidden layer
        classifier.add(Dense(output_dim=60, init='uniform', activation='relu'))

        # Adding a third hidden layer
        classifier.add(Dense(output_dim=60, init='uniform', activation='relu'))

        # Adding the output layer
        classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

        # Compiling the ANN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fitting the ANN to the Training set
        classifier.fit(self.dataSetBreak["dataTrain"], self.dataSetBreak["classTrain"], batch_size=10, nb_epoch=2)

        # Predicting the Test set results
        y_pred = classifier.predict(self.dataSetBreak["dataTest"])
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

    def break_datadet(self):
        x = self.data_file.iloc[:, :-1].values
        y = self.data_file.iloc[:, 41].values

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
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        # feature scaling
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        self.dataSetBreak["dataTrain"] = x_train
        self.dataSetBreak["dataTest"] = x_test
        self.dataSetBreak["classTrain"] = y_train
        self.dataSetBreak["classTest"] = y_test

    def runClasificador(self):
        # importing the dataset
        # change Multi-class to binary-class
        self.data_file['normal.'] = self.data_file['normal.'].replace(
            ['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.',
             'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.',
             'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

        y_predi = self.entrenar()
        cm = confusion_matrix(self.dataSetBreak["classTest"], y_predi)

        # the performance of the classification model
        print("la precisión es: " + str((cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])))
        recall = cm[1, 1] / (cm[0, 1] + cm[1, 1])
        # print("Recall is : " + str(recall))
        print("Tasa de Falsos Positivos: " + str(cm[1, 0] / (cm[0, 0] + cm[1, 0])))
        precision = cm[1, 1] / (cm[1, 0] + cm[1, 1])
