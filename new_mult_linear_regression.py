# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:52:47 2020

@author: LENOVO
"""

import numpy as np
import pandas as pd
from sklearn import linear_model

data_frame = pd.read_csv("multivariable_regression.csv")

import math
m=data_frame.bedrooms.median()
data_frame.bedrooms=data_frame.bedrooms.fillna(m)

new_data_frame = data_frame.drop('price',axis='columns')

reg = linear_model.LinearRegression()
reg.fit(new_data_frame,data_frame.price)

reg.predict([[3000,3,40]])
