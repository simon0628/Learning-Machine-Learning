#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from tools import *
# from tqdm import tqdm
#%%
def deonehot(y):
    loc = np.argmax(y)
    return loc

def onehot(y):
    m = len(y)
    label2num = dict() # {label: num} e.g. {'cat':0, 'dog':1}
    num2label = dict() # {num: label} e.g. {0:'cat', 1:'dog'}
    cnt = 0
    for yy in np.unique(y):
        label2num[yy] = cnt
        num2label[cnt] = yy
        cnt += 1
    y_onehot = np.zeros((m, cnt))

    for i in range(m):
        y_onehot[i][label2num[y[i][0]]] = 1
    return cnt, num2label, y_onehot

[X, y] = load_mat(expath(3) + "ex3data1.mat",['X','y'])
theta1, theta2 = load_mat(expath(3) + "ex3weights.mat",['Theta1','Theta2'])
K, y_dict, y_onehot = onehot(y)

layer_def = [400, 25, 10]

model = NN(len(X[0]), layer_def)

#%%
model.fit(X, y_onehot, 1500)
#%%
y_preds = list()
match = list()
for i in range(len(X)):
    a = model.predict(X[i])
    y_pred = y_dict[deonehot(a)]
    # loss = model.loss(theta, X, y_onehot) 
    # print(loss)
    y_preds.append(y_pred)
    match.append(1 if (y_pred == y[i]) else 0)

#%%

precision = sum(match) / len(match)
print('precision = ', precision)


#%%
