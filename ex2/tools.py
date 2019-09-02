import pandas as pd
import numpy as np
import scipy.optimize as op

def load_txt(filename, names):
    data = pd.read_csv(filename, sep=",", header=None, names = names)
    return data
    
class LinearRegression:
    def h(self, theta, x):
        return x.dot(theta)

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


class LogisticRegression:

    def __init__(self, n):
        # n: feature number
        self.theta = np.zeros((n+1, 1))
        self.mean = None
        self.std = None

    def normalize(self, x):
        self.mean = x.mean(axis = 0)
        self.std = x.std(axis = 0)
        x = (x - self.mean)/ self.std
        return x

    def train(self, x, y, max_iter = 1500, alpha = 0.1): 

        x = self.normalize(x)
        x = np.concatenate((np.ones((len(y),1)),x), axis=1)

        losses = list()
        for _ in range(max_iter):  
            loss = self.loss(self.theta, x, y)
            losses.append(loss)
            self.grad_des(self.theta, alpha, x, y)
        return losses

    def train_scipy(self, x, y):

        x = self.normalize(x)
        x = np.concatenate((np.ones((len(y),1)),x), axis=1)

        result = op.minimize(fun = self.loss, 
                                x0 = self.theta, 
                                args = (x, y))

        self.theta = result.x
        loss = result.fun
        return loss

    def test(self, x):
        return self.h(self.theta, np.concatenate((np.ones(1),(x - self.mean)/self.std)))

    def sigmoid(self, theta, x):
        return 1/(1+np.exp(-x.dot(theta)))

    def h(self, theta, x):
        return self.sigmoid(theta, x)

    def loss(self, theta, x, y):
        total = 0
        m = len(y)
        for i in range(m):
            htheta = self.h(theta, x[i])
            total += (-y[i] * np.log(htheta) - (1-y[i]) * np.log(1-htheta))
        return total / m

    def grad_des(self, theta, alpha, x, y):
        m = len(y)
        tmp_thetas = list()
        for j in range(len(theta)):
            tmp_theta = theta[j] - alpha/m * sum([(self.h(theta, x[i])-y[i])*x[i][j] for i in range(m)])
            tmp_thetas.append(tmp_theta)
        for j in range(len(theta)):
            theta[j] = tmp_thetas[j]