import pandas as pd
import numpy as np
import scipy.optimize as op
import scipy.io as scio

def expath(n):
    return '../ex' + str(n) + '/'

def load_txt(filename, names):
    data = pd.read_csv(filename, sep=",", header=None, names = names)
    return data

def load_mat(filename, names):
    data = scio.loadmat(filename)
    return [data[name] for name in names]


def sigmoid(z):    
    return 1/(1+np.exp(-z))

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

    def preprocess(self, x, y):
        x = self.normalize(x)
        x = np.concatenate((np.ones((len(y),1)),x), axis=1)
        return x

    def train(self, x, y, max_iter = 1500, alpha = 0.1, lam = 0): 
        x = self.preprocess(x, y)
        last_loss = self.loss(self.theta, x, y, lam)

        for _ in range(max_iter):  
            self.grad_des(alpha, x, y, lam)
            
            loss = self.loss(self.theta, x, y, lam)
            # print(loss)
            if np.abs(last_loss - loss) < 1e-6:
                break
            last_loss = loss
        return loss

    def train_scipy(self, x, y):
        x = self.preprocess(x, y)

        result = op.minimize(fun = self.loss, 
                                x0 = self.theta, 
                                args = (x, y))

        self.theta = result.x
        loss = result.fun
        return loss

    def test(self, x):
        return self.h(self.theta, np.concatenate((np.ones(1),(x - self.mean)/self.std)))

    def h(self, theta, x):
        return sigmoid(x.dot(theta))

    def loss(self, theta, x, y, lam = 0):
        m = len(y)

        htheta = self.h(self.theta, x)
        a = np.log(htheta)
        b = np.log(np.ones(htheta.shape)-htheta)
        cross_entropy = (-y.T.dot(a)) - (np.ones(y.shape)-y).T.dot(b)

        regular = lam / (2*m) * sum([t.dot(t) for t in self.theta[1:]])
        return cross_entropy / m + regular

    def grad_des(self, alpha, x, y, lam = 0):
        m = len(y)

        beta = self.h(self.theta, x).reshape(y.shape) - y
        deri = (x.T.dot(beta)/m).reshape(self.theta.shape)
        theta_reg = np.concatenate(([self.theta[0]], (1-lam/m) * np.array(self.theta[1:])))

        self.theta = theta_reg - (alpha * deri)
