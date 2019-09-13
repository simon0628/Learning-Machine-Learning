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
lams = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
for lam in lams:
    model = LinearRegression(n)
    loss = model.fit_scipy(X,y, lam = lam)

    error_val = model.loss(model.theta, model.preprocess(X,len(y)), y, 0)
    errors_train.append(error_val)

    error_val = model.loss(model.theta, model.preprocess(Xval,len(yval)), yval, 0)
    errors_val.append(error_val)

# iteration = range(2, m-1)
plt.plot(lams, errors_train, 'b-')
plt.plot(lams, errors_val, 'g-')
# plt.title('lambda=' + str(lam))
plt.show()

# theta = model.theta


model = LinearRegression(n)
loss = model.fit_scipy(X,y, lam = 3)
print('train loss =', loss)

Xtest = polyize(Xtest, degree)
loss = model.loss(model.theta, model.preprocess(Xtest,len(ytest)), ytest, 0)
print('test loss =', loss)

X_plot = np.linspace(min(X[:,0])*1.2, max(X[:,0])*1.2)
X_plot = X_plot.reshape((len(X_plot), 1))
X_plot = polyize(X_plot, degree)
y_plot = model.predict(X_plot, len(X_plot))

plt.plot(X[:,0], y, 'r*')
plt.plot(X_plot[:,0], y_plot, 'b-')
plt.show()