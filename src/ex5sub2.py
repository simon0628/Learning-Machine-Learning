import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

from tools import *

X, y, Xtest, ytest, Xval, yval = load_mat(expath(5) + 'ex5data1.mat', ['X', 'y', 'Xtest', 'ytest', 'Xval', 'yval'])

def polyize(X, degree = 2):
    res = X
    for i in range(2, degree+1):
        res = np.concatenate((res, np.power(X,i)), axis = 1)
    return res

degree = 8
X = polyize(X, degree)
Xval = polyize(Xval, degree)

m = len(y)
n = len(X[0])

errors_train = list()
errors_val = list()
for i in range(2, m-1):
    model = LinearRegression(n)
    loss = model.fit_scipy(X[1:i,:],y[1:i], lam = 0)

    error_val = model.loss(model.theta, model.preprocess(X[1:i,:],len(y[1:i])), y[1:i], 0)
    errors_train.append(error_val)

    error_val = model.loss(model.theta, model.preprocess(Xval,len(yval)), yval, 0)
    errors_val.append(error_val)

iteration = range(2, m-1)
plt.plot(iteration, errors_train, 'b-')
plt.plot(iteration, errors_val, 'g-')
plt.show()

# theta = model.theta

X_plot = np.linspace(min(X[:,0])*1.2, max(X[:,0])*1.2)
X_plot = X_plot.reshape((len(X_plot), 1))
X_plot = polyize(X_plot, degree)
predict_y = model.predict(X_plot, len(X_plot))

plt.plot(X[:,0], y, 'r*')
plt.plot(X_plot[:,0], predict_y, 'b-')
plt.show()