import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from tools import *

X, y, Xtest, ytest, Xval, yval = load_mat(expath(5) + 'ex5data1.mat', ['X', 'y', 'Xtest', 'ytest', 'Xval', 'yval'])

plt.plot(X, y, 'r*')
# plt.show()

m = len(y)
n = len(X[0])

losses = list()
model = LinearRegression(n)
loss = model.fit_scipy(X,y)
print(loss)
theta = model.theta

predict_y = model.predict(X, m)
plot_decision_boundary(X, predict_y)

def plot_decision_boundary(X, y):
    plt.plot(X, y, 'b-')
    plt.show()