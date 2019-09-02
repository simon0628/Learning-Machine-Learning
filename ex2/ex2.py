import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import scipy.optimize as op

from tools import *

data = load_txt('ex2data1.txt', ['exam1', 'exam2', 'admit'])

data = data.to_numpy()
X = data[:,:-1]
y = data[:,-1]
m = len(y)
n = len(X[0])

mean = X.mean(axis = 0)
std = X.std(axis = 0)
X = (X - mean)/ std

X = np.concatenate((np.ones((m,1)),X),axis=1)
theta = np.zeros((n+1,1))
iterations = 1500
alpha = 0.1

losses = list()
model = LogisticRegression(n)
# for i in range(iterations):
#     if i % 100 == 0:
#         print(i)
#     loss = model.loss(theta, X, y)
#     losses.append(loss)
#     model.grad_des(theta, alpha, m, X, y)

result = op.minimize(fun = model.loss, 
            x0 = theta, 
            args = (X, y))

theta = result.x
loss = result.fun

predict1 = model.sigmoid(theta, np.concatenate((np.ones(1),([45, 85] - mean)/std)))
print(predict1)
# predict_y = X.dot(theta)
# plt.plot(data[:,0], y, 'r*')
# plt.plot(data[:,0], predict_y, 'b-')
# iteration = range(len(losses))
# plt.plot(iteration, losses, 'g-')
# plt.show()

