import pandas as pd
import numpy as np
import scipy.optimize as op
import scipy.io as scio
from tqdm import tqdm

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
    def __init__(self, n):
        # n: feature number
        self.theta = np.zeros((n+1, 1))
        self.mean = None
        self.std = None
        self.zero_std_feature = None

    def h(self, theta, x):
        return x.dot(theta)

    def loss(self, theta, x, y, lam = 0):
        m = len(y)
        sub = self.h(theta, x)
        sub = sub.reshape(y.shape) - y

        regular = lam / (2*m) * theta[1:].T.dot(theta[1:])
        result = sum([i*i for i in sub]) / (2*m) + regular
        print(result)
        return result

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

    def grad_des(self, alpha, x, y, lam = 0):
        m = len(y)

        beta = self.h(self.theta, x).reshape(y.shape) - y
        deri = (x.T.dot(beta)/m).reshape(self.theta.shape)
        theta_reg = np.concatenate(([self.theta[0]], (1-lam/m) * np.array(self.theta[1:])))

        self.theta = theta_reg - (alpha * deri)

    def fit(self, x, y, max_iter = 1500, alpha = 0.1, lam = 1): 
        self.prepare_arg(x)
        x = self.preprocess(x,len(y))

        last_loss = self.loss(self.theta, x, y, lam)
        for _ in range(max_iter):  
            self.grad_des(alpha, x, y, lam)
            # print(last_loss)
            
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
        # l: layer number
        self.n = n
        self.k = k
        self.layer = layer
        self.l = len(layer)

        self.theta = list() # theta-dim: l * (last_layer_dim * next_layer_dim)
        for i in range(self.l-1):
            self.theta.append(self.rand_init([layer[i]+1, layer[i+1]]))

        self.a = None 
        self.z = None

    def rand_init(self, shape, epi = 0.12):
        return np.random.random(shape)*2*epi-epi

    def add_bias(self, x):
        x = np.concatenate((np.ones(1),x))
        return x

    def fit(self, x, y, max_iter = 1500, alpha = 0.1, lam = 1): 
        # x = self.add_bias(x)

        # loss = self.loss(self.theta, x, y, lam) # contains forward prop

        for _ in tqdm(range(max_iter)):  
            self.grad_des(alpha, x, y, lam)
            
            # loss = self.loss(self.theta, x, y, lam)
            # print(loss)
            # stop when the loss is steady
            # if np.abs(last_loss - loss) < 1e-6:
            #     break
            # last_loss = loss
        loss = self.loss(self.theta, x, y, lam)
        return loss

    def predict(self, x):
        return self.forward_prop(self.theta, x)

    def h(self, theta, x):
        return sigmoid(theta.dot(x))

    def forward_prop(self, theta, x, save_az = False):
        aa = self.add_bias(x) # a(1)
        if save_az:
            self.a[0].append(aa)

        for i in range(1, self.l-1):
            zz = theta[i-1].T.dot(aa)
            x = sigmoid(zz)
            aa = self.add_bias(x)

            if save_az:
                self.a[i].append(aa)
                self.z[i-1].append(zz)

        # a(end)
        i = self.l-1
        zz = theta[i-1].T.dot(aa)
        aa = sigmoid(zz)
        if save_az:
            self.a[i].append(aa)
            self.z[i-1].append(zz)

        return aa

    def sigmoid_grad(self, z):
        return sigmoid(z)*(1-sigmoid(z))

    def back_prop(self, y):
        # x = self.add_bias(x)

        deltas = list()

        delta = self.a[-1] - y
        deltas.append(np.dot(np.transpose(self.a[-2]),delta))
        for i in range(self.l-1, 1, -1):
            new_delta = list()
            for j in range(len(y)):
                g_grad = self.sigmoid_grad(self.z[i-2][j])
                tmp_delta = np.multiply(self.theta[i-1][1:,:].dot(delta[j]).dot(g_grad.T), g_grad)
                new_delta.append(tmp_delta)
      
            deltas.append(np.dot(np.transpose(self.a[i-2]),new_delta))
            delta = new_delta
        deltas.reverse()
        return deltas

    def grad_des(self, alpha, x, y, lam = 1):

        self.a = list() # a-dim: (l-1) * m * (current_layer_dim)
        self.z = list()

        for i in range(self.l-1):
            self.a.append(list())
            self.z.append(list())
        self.a.append(list())

        for i in range(len(y)):
            self.forward_prop(self.theta, x[i], True)
        
        delta = self.back_prop(y)
        self.theta -= np.array(delta)/len(y)
        # theta_reg = np.concatenate(([self.theta[0]], (1-lam/m) * np.array(self.theta[1:])))

        # self.theta = theta_reg - (alpha * deri)

    # CAUTION: don't use self.theta in loss function
    # otherwise the op.minimize function won't work (it passes in temp theta)
    def loss(self, theta, x, y, lam = 1):
        total = 0
        m,k = y.shape
        for i in range(m):
            htheta = np.array(self.forward_prop(theta, x[i]))

            # CAUTION: if a is too small (i.e. <1e-8 and reduced to 0), log(a) will crash
            aa = np.where(htheta == 0, 1e-6, htheta)
            a = np.log(aa)

            bb = np.ones(htheta.shape)-htheta
            bb = np.where(bb == 0, 1e-6, bb)
            b = np.log(bb)

            total += sum([(-y[i][j] * a[j] - (1-y[i][j]) * b[j]) for j in range(k)])
        regular = lam/(2*m) * sum([np.sum(i) for i in np.square(theta)])
        return total / m + regular
