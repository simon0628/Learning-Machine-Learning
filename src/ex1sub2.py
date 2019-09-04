import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from tools import *

data = load_txt(expath(1) + 'ex1data2.txt', ['size', 'bedroom', 'price'])

data = data.to_numpy()
X = data[:,:-1]
y = data[:,-1]
m = len(y)
n = len(X[0])

# X = normalize(X)
X = (X - X.mean(axis = 0))/ X.std(axis = 0)
X = np.concatenate((np.ones((m,1)),X),axis=1)
theta = np.zeros((n+1,1))
iterations = 50
alpha = 0.1

losses = list()
model = LinearRegression()
for i in range(iterations):
    loss = model.loss(m, theta, X, y)
    losses.append(loss)
    # if loss < 2.5e9:
    #     break
    model.grad_des(theta, alpha, m, X, y)

# predict_y = X.dot(theta)
# plt.plot(data[:,0], y, 'r*')
# plt.plot(data[:,0], predict_y, 'b-')
iteration = range(len(losses))
plt.plot(iteration, losses, 'g-')
plt.show()

