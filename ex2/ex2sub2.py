import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures

from tools import *


def plot_decision_boundary(X, y, model):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos][:,0], X[pos][:,1], 'k+')
    plt.plot(X[neg][:,0], X[neg][:,1], 'bo')

    x_min, x_max = X[:, 0].min() * 1.2, X[:, 0].max() * 1.2
    y_min, y_max = X[:, 1].min() * 1.2, X[:, 1].max() * 1.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    
    XX = np.array([xx.ravel(), yy.ravel()]).T
    XX = PolynomialFeatures(degree = 6, include_bias = False).fit_transform(XX)
    Z = np.array([model.test(i) for i in XX]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha = 0.2, levels = 1)
    plt.show()

data = load_txt('ex2data2.txt', ['test1', 'test2', 'accepted'])

data = data.to_numpy()
X_raw = data[:,:-1]
X = PolynomialFeatures(degree = 6, include_bias = False).fit_transform(X_raw)
y = data[:,-1]

model = LogisticRegression(len(X[0]))
loss = model.train(X, y, lam = 1)
print('loss =', loss)
theta = model.theta
print('theta =', theta)

plot_decision_boundary(X_raw, y, model)
