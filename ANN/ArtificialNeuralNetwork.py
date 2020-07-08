#Aritficial Neural Network

#importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#seperating dependent and independent variables
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#adjusting categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lx = LabelEncoder()
lx1 = LabelEncoder()
x[:,1] = lx.fit_transform(x[:,1])
x[:,2] = lx1.fit_transform(x[:,2])
ohc = OneHotEncoder(categorical_features=[1,2])
x = ohc.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

import keras
from keras.models import Sequential 
from keras.layers import Dense

#classifying neural network
classifier = Sequential()
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu',input_dim = 12))
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#applying stochastic gradiant descent on network
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
#training (fitting ann to training set)
classifier.fit(x_train,y_train,batch_size = 10 ,  nb_epoch = 100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)