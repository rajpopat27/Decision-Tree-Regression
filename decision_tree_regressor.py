# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:06:56 2018

@author: RAJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(X,y)

y_pred = tree_regressor.predict(6.5)


X_grid = np.arange(min(X), max(X), 0.01).reshape(-1,1)
plt.scatter(X,y,color='blue')
plt.plot(X_grid,tree_regressor.predict(X_grid),color='red')
plt.show()

    