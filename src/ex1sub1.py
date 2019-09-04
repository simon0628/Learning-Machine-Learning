import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from tools import *

data = load_txt(expath(1) + 'ex1data1.txt', ['population', 'profit'])

data = data.to_numpy()
X = data[:,0]
y = data[:,1]
m = len(y)

X = np.concatenate((np.ones((m,1)),data[:,0].reshape(m,1)),axis=1)
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

losses = list()
model = LinearRegression()
for i in range(iterations):
    losses.append(model.loss(m, theta, X, y))
    model.grad_des(theta, alpha, m, X, y)

predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)

predict_y = X.dot(theta)
plt.plot(data[:,0], y, 'r*')
plt.plot(data[:,0], predict_y, 'b-')
plt.show()

