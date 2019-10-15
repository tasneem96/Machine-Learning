# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as mpp

#read dataset
dataset=pd.read_csv('linear regression .csv')

# devide X,Y variable from dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#split the dataset acording to tesing and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#implement classifire based on simple linear regression

from sklearn.linear_model import LinearRegression
simplelinearregression=LinearRegression()
simplelinearregression.fit(X_train,Y_train)

#predict the value of y in respect of X_test
y_predict=simplelinearregression.predict(X_test)

#plot the graph
mpp.scatter(X_train,Y_train,color='red')
mpp.plot(X_train,simplelinearregression.predict(X_train))
mpp.show()