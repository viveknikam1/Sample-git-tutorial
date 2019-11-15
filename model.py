# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:59:12 2019

@author: Vivek Nikam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
df = pd.read_csv('winequality-red.csv')
columns = list(df.columns)
X = df.drop('quality',axis=1)
y = df['quality']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(X_train,y_train)
model_LR.score(X_test,y_test)