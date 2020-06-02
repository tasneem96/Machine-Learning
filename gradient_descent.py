# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 23:38:37 2020

@author: LENOVO
"""

import numpy as np 
import matplotlib.pyplot as plt 

def gradient_descent (x,y):
    m_curr = 0
    b_curr = 0
    iterations = 100
    n= len(x)
    learning_rate =.008
    plt.scatter (x,y,color="red",marker="*")
    
    for i in range(iterations):
        y_prediction = m_curr*x + b_curr
        cost = (1/n)*sum(val**2 for val in (y-y_prediction))
        plt.plot(x,y_prediction,color="green")
        md=-(2/n)*sum(x*(y-y_prediction))
        bd=-(2/n)*sum(y-y_prediction)
        
        m_curr=m_curr-learning_rate*md
        b_curr=m_curr-learning_rate*bd
        
        print("M {} , B {} , Cost {}, Iterations {}".format(m_curr,b_curr,cost,i))
        
x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
gradient_descent(x,y)
    
        
