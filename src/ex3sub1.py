import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures

from tools import *


def display_data(X,y, size):
    choices = np.random.randint(len(y), size = size)
    X = X[choices]
    y = y[choices]
    plt.figure(figsize=(20,8))
    for index, (image, label) in enumerate(zip(X[0:size], y[0:size])):
        plt.subplot(size/5, 5, index + 1)
        plt.imshow(np.reshape(image, (20,20)), cmap=plt.cm.gray)
        plt.title('Training: %i\n' % label, fontsize = 20)
    plt.show()


[X, y] = load_mat("../ex3/ex3data1.mat",['X','y'])

display_data(X,y, 10)
# data = data.to_numpy()
# X_raw = data[:,:-1]
# X = PolynomialFeatures(degree = 6, include_bias = False).fit_transform(X_raw)
# y = data[:,-1]

# model = LogisticRegression(len(X[0]))
# loss = model.train(X, y, lam = 1)
# print('loss =', loss)
# theta = model.theta
# print('theta =', theta)

# plot_decision_boundary(X_raw, y, model)
