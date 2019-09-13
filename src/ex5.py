import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from tools import *

X, y, Xtest, ytest, Xval, yval = load_mat(expath(5) + 'ex5data1.mat', ['X', 'y', 'Xtest', 'ytest', 'Xval', 'yval'])

# plt.plot(X, y, 'r*')
# plt.show()

m = len(y)
n = len(X[0])

errors_train = list()
errors_val = list()
for i in range(2, m-1):
    model = LinearRegression(n)
    loss = model.fit_scipy(X[1:i,:],y[1:i])

    # error_train = model.loss(model.theta, model.preprocess(X[1:,:],len(y)-1), y[1:], 0)
    error_val = model.loss(model.theta, model.preprocess(Xval,len(yval)), yval, 0)
    errors_train.append(loss)
    errors_val.append(error_val)

iteration = range(2, m-1)
plt.plot(iteration, errors_train, 'b-')
plt.plot(iteration, errors_val, 'g-')
plt.show()

# theta = model.theta

# predict_y = model.predict(X, m)
# plot_decision_boundary(X, predict_y)

def plot_decision_boundary(X, y):
    plt.plot(X, y, 'b-')
    plt.show()