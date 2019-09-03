import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import scipy.optimize as op

from tools import *

data = load_txt('ex2data1.txt', ['exam1', 'exam2', 'admit'])

data = data.to_numpy()
X = data[:,:-1]
y = data[:,-1]

model = LogisticRegression(len(X[0]))
loss = model.train_scipy(X,y)
theta = model.theta
print('theta =', theta)
print('loss =', loss)
print('test =', model.test([45,85]))

def plot_decision_boundary(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos][:,0], X[pos][:,1], 'k+')
    plt.plot(X[neg][:,0], X[neg][:,1], 'yo')

    # denormalize
    mean1 = np.mean(X[:,0])
    std1 = np.std(X[:,0])
    mean2 = np.mean(X[:,1])
    std2 = np.std(X[:,1])
    x1s = list()
    x2s = list()
    for x1 in X[:,0]:
        x11 = (x1-mean1)/std1
        x22 = -(x11*theta[1]+theta[0])/theta[2]
        x2 = x22 * std2 + mean2
        x1s.append(x1)
        x2s.append(x2)
    plt.plot(x1s,x2s,'b-')
    plt.show()
