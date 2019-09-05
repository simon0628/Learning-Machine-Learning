import pandas as pd
import numpy as np
import scipy.optimize as op
import scipy.io as scio
# from tqdm import tqdm

def expath(n):
    return '../data/ex' + str(n) + '/'

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
        self.zero_std_feature = None

    def prepare_arg(self, x):
        self.std = x.std(axis = 0)

        # remove all the features that have zero std
        self.zero_std_feature = np.where(self.std == 0)[0]
        # print('remove features:', self.zero_std_feature)
        x = np.delete(x, self.zero_std_feature, axis = 1)
        self.std = np.delete(self.std, self.zero_std_feature)
        self.theta = np.delete(self.theta, self.zero_std_feature)

        self.mean = x.mean(axis = 0)

    def preprocess(self, x, m):
        if m == 1: # single x sample
            x = np.delete(x, self.zero_std_feature)
            x = (x - self.mean)/ self.std
            x = np.concatenate((np.ones(1),x))
        else: # batch x samples
            x = np.delete(x, self.zero_std_feature, axis = 1)
            x = (x - self.mean)/ self.std
            x = np.concatenate((np.ones((m,1)),x), axis=1)
        return x


    def fit(self, x, y, max_iter = 1500, alpha = 0.1, lam = 1): 
        self.prepare_arg(x)
        x = self.preprocess(x,len(y))

        last_loss = self.loss(self.theta, x, y, lam)
        for _ in range(max_iter):  
            self.grad_des(alpha, x, y, lam)
            
            loss = self.loss(self.theta, x, y, lam)
            # print(loss)
            # stop when the loss is steady
            if np.abs(last_loss - loss) < 1e-6:
                break
            last_loss = loss
        return loss

    def fit_scipy(self, x, y, max_iter = 1500, lam = 1):
        self.prepare_arg(x)
        x = self.preprocess(x,len(y))

        result = op.minimize(fun = self.loss, 
                                x0 = self.theta, 
                                args = (x, y, lam),
                                tol=1e-3,
                                options={'maxiter': max_iter})
        # print(result)
        self.theta = result.x
        loss = result.fun
        return loss

    def predict(self, x, m = 1):
        x = self.preprocess(x,m)
        return self.h(self.theta, x)

    def h(self, theta, x):
        return sigmoid(x.dot(theta))

    # CAUTION: don't use self.theta in loss function
    # otherwise the op.minimize function won't work (it passes in temp theta)
    def loss(self, theta, x, y, lam = 1):
        # vectorized loss function
        m = len(y)

        htheta = self.h(theta, x)

        # CAUTION: if a is too small (i.e. <1e-8 and reduced to 0), log(a) will crash
        aa = np.where(htheta == 0, 1e-6, htheta)
        a = np.log(aa)

        bb = np.ones(htheta.shape)-htheta
        bb = np.where(bb == 0, 1e-6, bb)
        b = np.log(bb)

        cross_entropy = (-y.T.dot(a)) - (np.ones(y.shape)-y).T.dot(b)
        regular = lam / (2*m) * sum([t*t for t in theta[1:]])
        return cross_entropy / m + regular

    def grad_des(self, alpha, x, y, lam = 1):
        m = len(y)

        beta = self.h(self.theta, x).reshape(y.shape) - y
        deri = (x.T.dot(beta)/m).reshape(self.theta.shape)
        theta_reg = np.concatenate(([self.theta[0]], (1-lam/m) * np.array(self.theta[1:])))

        self.theta = theta_reg - (alpha * deri)

class NN:
    def __init__(self, n, layer, k = 1):
        # n: feature number
        # k: label number
        self.theta = np.zeros((n+1, k))
        self.n = n
        self.k = k
        self.layer = layer
        self.l = len(layer)

        self.mean = None
        self.std = None
        self.zero_std_feature = None

    def add_bias(self, x):
        x = np.concatenate((np.ones(1),x))
        return x

    def fit(self, x, y, max_iter = 1500, alpha = 0.1, lam = 1): 
        x = self.add_bias(x)

        last_loss = self.loss(self.theta, x, y, lam)
        for _ in range(max_iter):  
            self.grad_des(alpha, x, y, lam)
            
            loss = self.loss(self.theta, x, y, lam)
            # print(loss)
            # stop when the loss is steady
            if np.abs(last_loss - loss) < 1e-6:
                break
            last_loss = loss
        return loss

    def fit_scipy(self, x, y, max_iter = 1500, lam = 1):
        x = self.add_bias(x)

        result = op.minimize(fun = self.loss, 
                                x0 = self.theta, 
                                args = (x, y, lam),
                                tol=1e-3,
                                options={'maxiter': max_iter})
        # print(result)
        self.theta = result.x
        loss = result.fun
        return loss

    def forward_prop(self, theta, x):
        a = list()
        a.append(self.add_bias(x))
        for i in range(1, self.l):
            a.append(self.add_bias(self.h(self.theta[i], a[i-1])))
        return a[:-1]

    def predict(self, x):
        return self.forward_prop(self.theta, x)

    def h(self, theta, x):
        return sigmoid(x.dot(theta))

    # CAUTION: don't use self.theta in loss function
    # otherwise the op.minimize function won't work (it passes in temp theta)
    def loss(self, theta, x, y, lam = 1):
        # vectorized loss function
        m = len(y)

        htheta = self.h(theta, x)

        # CAUTION: if a is too small (i.e. <1e-8 and reduced to 0), log(a) will crash
        aa = np.where(htheta == 0, 1e-6, htheta)
        a = np.log(aa)

        bb = np.ones(htheta.shape)-htheta
        bb = np.where(bb == 0, 1e-6, bb)
        b = np.log(bb)

        cross_entropy = (-y.T.dot(a)) - (np.ones(y.shape)-y).T.dot(b)
        regular = lam / (2*m) * sum([t*t for t in theta[1:]])
        return cross_entropy / m + regular

    def grad_des(self, alpha, x, y, lam = 1):
        m = len(y)

        beta = self.h(self.theta, x).reshape(y.shape) - y
        deri = (x.T.dot(beta)/m).reshape(self.theta.shape)
        theta_reg = np.concatenate(([self.theta[0]], (1-lam/m) * np.array(self.theta[1:])))

        self.theta = theta_reg - (alpha * deri)
