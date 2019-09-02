import pandas as pd
import numpy as np

def load_txt(filename, names):
    data = pd.read_csv(filename, sep=",", header=None, names = names)
    return data

def normalize(X):
    X = (X - X.mean(axis = 0))/ X.std(axis = 0)
    return X
    
class LinearRegression:
    def h(self, theta, x):
        total = 0
        for i in range(len(x)):
            total += theta[i] * x[i]
        return total

    def loss(self, m, theta, x, y):
        total = 0
        for i in range(m):
            diff = self.h(theta, x[i])-y[i]
            total += diff * diff
        return total / (2*m)

    def grad_des(self, theta, alpha, m, x, y):
        tmp_thetas = list()
        for j in range(len(theta)):
            tmp_theta = theta[j] - alpha/m * sum([(self.h(theta, x[i])-y[i])*x[i][j] for i in range(m)])
            tmp_thetas.append(tmp_theta)
        for j in range(len(theta)):
            theta[j] = tmp_thetas[j]